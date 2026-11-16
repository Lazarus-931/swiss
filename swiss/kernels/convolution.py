from dataclasses import dataclass
from functools import cache
from typing import Callable

import numpy as np
from carbon import language as cl
from carbon.compiler import Layout, View
from carbon.kernels import shared_workspace
from carbon.kernels.common import operand_buffers
from carbon.kernels.gemm import dense_matmul, epilogue as epilogues, matmul

from swiss.kernels.spatial import depth_to_space, unfold

WINOGRAD_BYTES = 128 << 20
UNFOLD_BYTES = 128 << 20
CORNER = (
    (4, 0, -5, 0, 1, 0),
    (0, -4, -4, 1, 1, 0),
    (0, 4, -4, -1, 1, 0),
    (0, -2, -1, 2, 1, 0),
    (0, 2, -1, -2, 1, 0),
    (0, 4, 0, -5, 0, 1),
)
FILTER = (
    (1 / 4, 0, 0),
    (-1 / 6, -1 / 6, -1 / 6),
    (-1 / 6, 1 / 6, -1 / 6),
    (1 / 24, 1 / 12, 1 / 6),
    (1 / 24, -1 / 12, 1 / 6),
    (0, 0, 1),
)
GATHER = (
    (1, 1, 1, 1, 1, 0),
    (0, 1, -1, 2, -2, 0),
    (0, 1, 1, 4, 4, 0),
    (0, 1, -1, 8, -8, 1),
)


def _single():
    return Layout.blocked((256,), threads=256)


def _combine(matrix, values):
    rows = []
    for row in matrix:
        total = None
        for coefficient, value in zip(row, values):
            if coefficient == 0:
                continue
            if total is None:
                total = value if coefficient == 1 else value * float(coefficient)
            elif coefficient == 1:
                total = total + value
            elif coefficient == -1:
                total = total - value
            else:
                total = total + value * float(coefficient)
        rows.append(total)
    return rows


def winograd_weights(weight):
    """The (36, filters, channels) f32 Winograd transform G g Gᵀ of (filters, 3, 3,
    channels) weights, in f64."""
    filters, _, _, channels = weight.shape
    g = np.asarray(FILTER, np.float64)
    transformed = np.einsum("ia,fabc,jb->ijfc", g, np.asarray(weight, np.float64), g)
    return transformed.reshape(36, filters, channels).astype(np.float32)


@cache
def make_winograd_input(dtype=cl.bf16):

    def transform(image, fields, first, count, element, live, H, W, C, TW):
        tile, channel = first + element // C, element % C
        top, left = tile // TW * 4, tile % TW * 4
        field = []
        for a in range(6):
            for b in range(6):
                inside = (
                    (top + a >= 1) & (top + a <= H) & (left + b >= 1) & (left + b <= W)
                )
                at = ((top + a - 1) * W + left + b - 1) * C + channel
                field.append(
                    cl.cast(cl.load(image, at, live & inside, other=0.0), cl.f32),
                )
        columns = [
            _combine(CORNER, [field[6 * a + b] for a in range(6)]) for b in range(6)
        ]
        for i in range(6):
            for j, value in enumerate(
                _combine(CORNER, [columns[b][i] for b in range(6)]),
            ):
                cl.store(
                    fields,
                    ((i * 6 + j) * count + element // C) * C + channel,
                    value,
                    live,
                )

    @cl.kernel
    def winograd_input(
        image: cl.buffer(dtype),
        fields: cl.buffer(cl.f32),
        first: cl.u32,
        count: cl.u32,
        H: cl.constexpr,
        W: cl.constexpr,
        C: cl.constexpr,
        TW: cl.constexpr,
    ):
        element = cl.program_id() * 256 + cl.arange(256, layout=_single())
        transform(
            image,
            fields,
            first,
            count,
            element,
            element < count * C,
            H,
            W,
            C,
            TW,
        )

    return winograd_input


@cache
def make_winograd_output(dtype=cl.bf16, *, biased=False, epilogue=None):

    def transform(
        products,
        bias,
        output,
        outs,
        first,
        count,
        element,
        live,
        H,
        W,
        F,
        TW,
    ):
        tile, filtered = first + element // F, element % F
        top, left = tile // TW * 4, tile % TW * 4
        grid = [
            cl.load(
                products,
                (index * count + element // F) * F + filtered,
                live,
                other=0.0,
            )
            for index in range(36)
        ]
        columns = [
            _combine(GATHER, [grid[6 * a + b] for a in range(6)]) for b in range(6)
        ]
        shift = (
            cl.cast(cl.load(bias[0], filtered, live, other=0.0), cl.f32)
            if biased
            else 0.0
        )
        for i in range(4):
            for j, value in enumerate(
                _combine(GATHER, [columns[b][i] for b in range(6)]),
            ):
                inside = live & (top + i < H) & (left + j < W)
                pixel = (top + i) * W + left + j
                at = pixel * F + filtered
                result = (
                    value + shift
                    if epilogue is None
                    else epilogue(value + shift, pixel, filtered, outs)
                )
                cl.store(output, at, cl.cast(result, dtype), inside)

    @cl.kernel
    def winograd_output(
        products: cl.buffer(cl.f32),
        bias: (cl.buffer(dtype),) if biased else (),
        output: cl.buffer(dtype),
        outs: operand_buffers(epilogue),
        first: cl.u32,
        count: cl.u32,
        H: cl.constexpr,
        W: cl.constexpr,
        F: cl.constexpr,
        TW: cl.constexpr,
    ):
        element = cl.program_id() * 256 + cl.arange(256, layout=_single())
        transform(
            products,
            bias,
            output,
            outs,
            first,
            count,
            element,
            element < count * F,
            H,
            W,
            F,
            TW,
        )

    return winograd_output


def winograd_conv(
    device,
    image,
    weights,
    output,
    *,
    height,
    width,
    channels,
    filters,
    bias=None,
    epilogue=None,
    operands=(),
    band=None,
    dtype=cl.bf16,
):
    """Dispatches for a 3 x 3, stride 1, padding 1 convolution of an NHWC image by
    Winograd F(4 x 4, 3 x 3)."""
    across, down = -(-width // 4), -(-height // 4)
    tiles = across * down
    band = min(
        tiles,
        WINOGRAD_BYTES // (36 * (channels + filters) * 4) if band is None else band,
    )
    if band < 1 or 36 * band * max(channels, filters) >= 2**32:
        raise ValueError(
            "winograd_conv needs tiles in its band and under 2^32 transformed values",
        )
    fields_bytes = 36 * band * channels * 4
    workspace = shared_workspace(device, fields_bytes + 36 * band * filters * 4)
    fields, products = (
        workspace.view(0, fields_bytes),
        workspace.view(fields_bytes, 36 * band * filters * 4),
    )
    constants = {"H": height, "W": width, "TW": across}
    forward = make_winograd_input(dtype).compile(
        constants={**constants, "C": channels},
        threads=256,
        device=device,
        max_grid=(2**32 - 1) // 256,
    )
    backward = make_winograd_output(
        dtype,
        biased=bias is not None,
        epilogue=epilogue,
    ).compile(
        constants={**constants, "F": filters},
        threads=256,
        device=device,
        max_grid=(2**32 - 1) // 256,
    )
    right = View(
        "f32",
        (36, channels, filters),
        (filters * channels, 1, channels),
        36 * filters * channels * 4,
    )
    steps = []
    for first in range(0, tiles, band):
        count = min(band, tiles - first)
        left = View(
            "f32",
            (36, count, channels),
            (count * channels, channels, 1),
            fields_bytes,
        )
        target = View(
            "f32",
            (36, count, filters),
            (count * filters, filters, 1),
            36 * band * filters * 4,
        )
        scalars = {"first": first, "count": count}
        steps.append(
            forward.dispatch(
                device,
                [image, fields],
                grid=(-(-count * channels // 256), 1, 1),
                scalars=scalars,
            ),
        )
        steps += matmul(device, fields, left, weights, right, products, target)
        steps.append(
            backward.dispatch(
                device,
                [products, *([] if bias is None else [bias]), output, *operands],
                grid=(-(-count * filters // 256), 1, 1),
                scalars=scalars,
            ),
        )
    return steps


def _after(bias, epilogue, operands, dtype, period=None):
    parts = ([] if bias is None else [epilogues.bias(dtype, period)]) + (
        [] if epilogue is None else [epilogue]
    )
    bound = ([] if bias is None else [bias]) + list(operands)
    return (
        {}
        if not parts
        else {
            "epilogue": parts[0] if len(parts) == 1 else epilogues.chain(*parts),
            "operands": tuple(bound),
        }
    )


@dataclass(frozen=True)
class _Shifted:
    function: Callable
    first: int

    def __call__(self, value, row, column, *operands):
        return self.function(value, row + self.first, column, *operands)


def _winograd(kernel, stride, padding):
    return kernel == 3 and stride == 1 and padding == 1


@dataclass(frozen=True, eq=False)
class ConvWeights:
    """A convolution's weight on the device in the form its route reads, from
    `conv_weights`."""

    buffer: object
    filters: int
    channels: int
    kernel: int
    stride: int
    padding: int
    dtype: object


def _stored(values, dtype):
    values = np.ascontiguousarray(values, np.float32)
    if dtype != cl.bf16:
        return values.astype(np.float16 if dtype == cl.f16 else np.float32).tobytes()
    words = values.view(np.uint32)
    rounded = ((words + 0x7FFF + ((words >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(values), np.uint16(0x7FC0), rounded).tobytes()


def conv_weights(device, weight, *, kernel, stride=1, padding=0, dtype=cl.bf16):
    """(filters, kernel, kernel, channels) weights on `device` in the form conv2d reads
    for this kernel and stride."""
    filters, rows, columns, channels = np.shape(weight)
    if rows != kernel or columns != kernel or stride < 1 or padding < 0:
        raise ValueError(
            "conv_weights takes a (filters, kernel, kernel, channels) weight, a stride "
            "and a padding",
        )
    if _winograd(kernel, stride, padding):
        data = winograd_weights(weight).tobytes()
    else:
        data = _stored(np.reshape(weight, (filters, -1)), dtype)
    buffer = device.buffer(max(16, -(-len(data) // 16) * 16))
    buffer.write(data + bytes(buffer.size - len(data)))
    return ConvWeights(buffer, filters, channels, kernel, stride, padding, dtype)


def conv2d(
    device,
    image,
    weights,
    output,
    *,
    height,
    width,
    bias=None,
    epilogue=None,
    operands=(),
    band=None,
):
    """Dispatches for the convolution of an NHWC image with `conv_weights`, plus `bias`,
    then `epilogue`."""
    w = weights
    if _winograd(w.kernel, w.stride, w.padding):
        return winograd_conv(
            device,
            image,
            w.buffer,
            output,
            height=height,
            width=width,
            channels=w.channels,
            filters=w.filters,
            bias=bias,
            epilogue=epilogue,
            operands=operands,
            band=band,
            dtype=w.dtype,
        )
    out_height = (height + 2 * w.padding - w.kernel) // w.stride + 1
    out_width = (width + 2 * w.padding - w.kernel) // w.stride + 1
    if out_height < 1 or out_width < 1:
        raise ValueError("conv2d needs a kernel that fits the padded image")
    pixels, depth, size = (
        out_height * out_width,
        w.kernel * w.kernel * w.channels,
        w.dtype.itemsize,
    )
    if w.kernel == 1 and w.stride == 1 and w.padding == 0:
        return dense_matmul(
            device,
            image,
            w.buffer,
            output,
            rows=pixels,
            n=w.filters,
            k=depth,
            dtype=w.dtype,
            **_after(bias, epilogue, operands, w.dtype),
        )
    band = min(pixels, max(1, UNFOLD_BYTES // (depth * size))) if band is None else band
    fields = shared_workspace(device, band * depth * size)
    steps = []
    for first in range(0, pixels, band):
        count = min(band, pixels - first)
        target = (
            output
            if count == pixels
            else output.view(first * w.filters * size, count * w.filters * size)
        )
        moved = (
            epilogue
            if epilogue is None or not first
            else epilogues.Epilogue(
                _Shifted(epilogue.function, first),
                epilogue.operands,
            )
        )
        steps += unfold(
            device,
            image,
            fields,
            height=height,
            width=width,
            channels=w.channels,
            kernel=w.kernel,
            stride=w.stride,
            padding=w.padding,
            first=first,
            rows=count,
            dtype=w.dtype,
        )
        steps += dense_matmul(
            device,
            fields,
            w.buffer,
            target,
            rows=count,
            n=w.filters,
            k=depth,
            dtype=w.dtype,
            **_after(bias, moved, operands, w.dtype),
        )
    return steps


def conv_transpose(
    device,
    image,
    weight,
    output,
    *,
    height,
    width,
    channels,
    filters,
    kernel=2,
    bias=None,
    epilogue=None,
    operands=(),
    dtype=cl.bf16,
):
    """Dispatches for a transposed convolution with kernel = stride, plus `bias`, then
    `epilogue`."""
    pixels, blocks = height * width, kernel * kernel
    if blocks * pixels * filters >= 2**32:
        raise ValueError(
            "conv_transpose takes under 2^32 output elements, for u32 indexing",
        )
    product = shared_workspace(device, blocks * pixels * filters * dtype.itemsize)
    return dense_matmul(
        device,
        image,
        weight,
        product,
        rows=pixels,
        n=blocks * filters,
        k=channels,
        dtype=dtype,
        **_after(bias, epilogue, operands, dtype, filters),
    ) + [
        depth_to_space(
            device,
            product,
            output,
            height=height,
            width=width,
            filters=filters,
            block=kernel,
            dtype=dtype,
        ),
    ]
