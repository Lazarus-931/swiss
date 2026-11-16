from functools import cache

from carbon import language as cl
from carbon.compiler import Layout

BLOCK = 1024


def block_layout():
    return Layout.blocked((BLOCK,), threads=256)


@cache
def make_unfold(dtype=cl.bf16, *, kernel, stride, padding):

    @cl.kernel
    def unfold(
        image: cl.buffer(dtype),
        output: cl.buffer(dtype),
        first: cl.u32,
        count: cl.u32,
        H: cl.constexpr,
        W: cl.constexpr,
        C: cl.constexpr,
        OW: cl.constexpr,
    ):
        element = cl.program_id() * BLOCK + cl.arange(BLOCK, layout=block_layout())
        width = kernel * kernel * C
        pixel, rest = first + element // width, element % width
        y = pixel // OW * stride + rest // (kernel * C)
        x = pixel % OW * stride + rest // C % kernel
        inside = (y >= padding) & (y < H + padding) & (x >= padding) & (x < W + padding)
        live = element < count * width
        value = cl.load(
            image,
            ((y - padding) * W + x - padding) * C + rest % C,
            live & inside,
            other=0.0,
        )
        cl.store(output, element, value, live)

    return unfold


def unfold(
    device,
    image,
    output,
    *,
    height,
    width,
    channels,
    kernel,
    stride=1,
    padding=0,
    first=0,
    rows=None,
    dtype=cl.bf16,
):
    """Rows first ..< first + rows of the zero-padded (dy, dx, channel) unfolding of an
    image, in raster order."""
    out_height = (height + 2 * padding - kernel) // stride + 1
    out_width = (width + 2 * padding - kernel) // stride + 1
    rows = out_height * out_width - first if rows is None else rows
    if (
        out_height < 1
        or out_width < 1
        or rows < 1
        or first + rows > out_height * out_width
    ):
        raise ValueError(
            "unfold needs a kernel that fits the padded image and rows inside its "
            "output",
        )
    elements = rows * kernel * kernel * channels
    if elements >= 2**32 or height * width * channels >= 2**32:
        raise ValueError("unfold takes under 2^32 elements, for u32 indexing")
    artifact = make_unfold(
        dtype,
        kernel=kernel,
        stride=stride,
        padding=padding,
    ).compile(
        constants={"H": height, "W": width, "C": channels, "OW": out_width},
        threads=256,
        device=device,
        max_grid=(2**32 - 1) // BLOCK,
    )
    return [
        artifact.dispatch(
            device,
            [image, output],
            grid=(-(-elements // BLOCK), 1, 1),
            scalars={"first": first, "count": rows},
        ),
    ]


@cache
def make_depth_to_space(dtype=cl.bf16, *, block=2):
    k = block

    @cl.kernel
    def depth_to_space(
        source: cl.buffer(dtype),
        output: cl.buffer(dtype),
        H: cl.constexpr,
        W: cl.constexpr,
        F: cl.constexpr,
    ):
        element = cl.program_id() * BLOCK + cl.arange(BLOCK, layout=block_layout())
        live = element < k * k * H * W * F
        y, x = element // (k * W * F), element // F % (k * W)
        at = ((y // k * W + x // k) * (k * k) + y % k * k + x % k) * F + element % F
        cl.store(output, element, cl.load(source, at, live, other=0.0), live)

    return depth_to_space


def depth_to_space(
    device,
    source,
    output,
    *,
    height,
    width,
    filters,
    block=2,
    dtype=cl.bf16,
):
    elements = block * block * height * width * filters
    kernel = make_depth_to_space(dtype, block=block).compile(
        constants={"H": height, "W": width, "F": filters},
        threads=256,
        device=device,
        max_grid=-(-elements // BLOCK),
    )
    return kernel.dispatch(device, [source, output], grid=(-(-elements // BLOCK), 1, 1))


def _largest(values):
    while len(values) > 1:
        values = [
            cl.maximum(values[i], values[i + 1]) if i + 1 < len(values) else values[i]
            for i in range(0, len(values), 2)
        ]
    return values[0]


@cache
def make_max_pool(dtype=cl.bf16, *, kernel=2):
    k = kernel

    @cl.kernel
    def max_pool(
        image: cl.buffer(dtype),
        output: cl.buffer(dtype),
        H: cl.constexpr,
        W: cl.constexpr,
        C: cl.constexpr,
    ):
        element = cl.program_id() * BLOCK + cl.arange(BLOCK, layout=block_layout())
        live = element < H // k * (W // k) * C
        y, x, c = element // (W // k * C) * k, element // C % (W // k) * k, element % C
        corners = [
            cl.cast(
                cl.load(image, ((y + dy) * W + x + dx) * C + c, live, other=0.0),
                cl.f32,
            )
            for dy in range(k)
            for dx in range(k)
        ]
        cl.store(output, element, cl.cast(_largest(corners), dtype), live)

    return max_pool


def max_pool(
    device,
    image,
    output,
    *,
    height,
    width,
    channels,
    kernel=2,
    dtype=cl.bf16,
):
    """Max pool of kernel x kernel blocks of stride `kernel` over a (height, width,
    channels) image."""
    if height % kernel or width % kernel or not height or not width:
        raise ValueError(
            "the max pool takes a height and width that its kernel divides",
        )
    elements = height // kernel * (width // kernel) * channels
    artifact = make_max_pool(dtype, kernel=kernel).compile(
        constants={"H": height, "W": width, "C": channels},
        threads=256,
        device=device,
        max_grid=-(-elements // BLOCK),
    )
    return [
        artifact.dispatch(device, [image, output], grid=(-(-elements // BLOCK), 1, 1)),
    ]
