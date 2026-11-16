"""Linear products, embeddings and the decode-chain feed."""

from dataclasses import dataclass
from functools import cache

import numpy as np
from carbon import language as cl
from carbon.compiler import CopyAtom, Layout
from carbon.kernels.gemm import (
    Affine,
    dense_matmul,
    epilogue,
    interleave_halves,
    quantized_matmul,
)
from carbon.kernels.gemm.epilogue import Glu
from carbon.kernels.gemm.quantized import dequantize


@dataclass(frozen=True)
class QuantizedWeight:
    """An affine-quantized (outputs, inputs) matrix: u32 codes, bf16 scales and
    biases."""

    packed: object
    scales: object
    biases: object
    outputs: int
    inputs: int
    bits: int = 4
    group: int = 64

    @classmethod
    def upload(cls, device, packed, scales, biases, *, bits=4, group=64):
        """Upload u32 codes, scales and biases."""
        outputs, words = packed.shape
        return cls(
            raw(device, packed),
            raw(device, scales),
            raw(device, biases),
            outputs,
            words * 32 // bits,
            bits,
            group,
        )


def scratch(device, weights):
    """bf16 storage for the largest quantized weight, or None when none is quantized."""
    sizes = [
        w.outputs * w.inputs * 2 for w in weights if isinstance(w, QuantizedWeight)
    ]
    return device.buffer(max(sizes)) if sizes else None


def product(
    device,
    x,
    weight,
    output,
    rows,
    scratch=None,
    activation=None,
    epilogue=None,
    operands=(),
):
    """Dispatches for output = x · weightᵀ in bf16; with `activation`, act(gate) ·
    up."""
    options = dict(
        rows=rows,
        n=weight.outputs,
        k=weight.inputs,
        dtype=cl.bf16,
        glu=None if activation is None else glu(activation),
        epilogue=epilogue,
        operands=operands,
    )
    if isinstance(weight, DenseWeight):
        return dense_matmul(device, x, weight.buffer, output, **options)
    return quantized_matmul(
        device,
        x,
        weight.packed,
        weight.scales,
        weight.biases,
        output,
        fmt=Affine(weight.bits, weight.group),
        scratch=scratch,
        **options,
    )


def raw(device, array):
    """Copy an array's bytes into a new device buffer."""
    data = np.ascontiguousarray(array).tobytes()
    buffer = device.buffer(max(16, -(-len(data) // 16) * 16))
    buffer.write(data)
    if buffer.size > len(data):
        buffer.write(bytes(buffer.size - len(data)), len(data))
    return buffer


def to_bf16(values):
    """Round float32 values to bf16 bit patterns, nearest even."""
    words = np.ascontiguousarray(values, np.float32).view(np.uint32)
    rounded = ((words + 0x7FFF + ((words >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(values), np.uint16(0x7FC0), rounded)


def from_bf16(bits):
    """float32 values of bf16 bit patterns."""
    return (np.ascontiguousarray(bits, np.uint16).astype(np.uint32) << 16).view(
        np.float32,
    )


WIDENS = (4, 6, 8)


def widen(packed, bits, target):
    """Affine codes of `bits` bits as codes of `target` bits."""
    if bits == target:
        return packed
    if target != 8 or bits not in WIDENS:
        raise ValueError(f"cannot widen {bits}-bit codes to {target} bits")
    data = np.ascontiguousarray(packed).view(np.uint8).reshape(packed.shape[0], -1)
    if bits == 4:
        codes = np.stack([data & 15, data >> 4], axis=-1)
    else:
        b0, b1, b2 = (data.reshape(packed.shape[0], -1, 3)[..., i] for i in range(3))
        codes = np.stack(
            [b0 & 63, b0 >> 6 | (b1 & 15) << 2, b1 >> 4 | (b2 & 3) << 4, b2 >> 2],
            axis=-1,
        )
    return np.ascontiguousarray(
        codes.reshape(packed.shape[0], -1).astype(np.uint8),
    ).view(np.uint32)


def quantize(matrix, bits, group, rows=8192):
    """Affine codes, bf16 scales and bf16 biases of a bf16 (outputs, inputs) matrix,
    packed as MLX packs them."""
    parts = [
        _quantize(matrix[start : start + rows], bits, group)
        for start in range(0, matrix.shape[0], rows)
    ]
    return tuple(np.concatenate(part) for part in zip(*parts))


def _quantize(matrix, bits, group):
    values = from_bf16(matrix).reshape(matrix.shape[0], -1, group)
    low, high = values.min(-1, keepdims=True), values.max(-1, keepdims=True)
    scales = from_bf16(to_bf16(np.maximum(high - low, 1e-8) / (2**bits - 1)))
    biases = from_bf16(to_bf16(low))
    codes = np.clip(np.rint((values - biases) / scales), 0, 2**bits - 1)
    codes = codes.astype(np.uint32).reshape(matrix.shape[0], -1, 32 // bits)
    packed = (codes << (np.arange(32 // bits, dtype=np.uint32) * bits)).sum(-1)
    return (
        packed.astype(np.uint32),
        to_bf16(scales[..., 0]),
        to_bf16(biases[..., 0]),
    )


@dataclass(frozen=True)
class DenseWeight:
    """A bf16 (outputs, inputs) matrix; paired [gate | up] rows are interleaved."""

    buffer: object
    outputs: int
    inputs: int

    @classmethod
    def upload(cls, device, matrix, paired=False):
        """Upload bf16 bit patterns of shape (outputs, inputs)."""
        if paired:
            matrix = interleave_halves(matrix)
        return cls(raw(device, matrix), *matrix.shape)


@dataclass(frozen=True)
class _Scaled:
    factor: float

    def __call__(self, value, row, column):
        return value * self.factor


def scaled(factor):
    """An epilogue: the product rounded to bf16, then times `factor`."""
    return epilogue.chain(
        epilogue.rounded(cl.bf16),
        epilogue.Epilogue(_Scaled(float(np.float32(factor)))),
    )


@cache
def glu(kind):
    """The pairing of a [gate | up] product for activation `kind`."""

    def pair(gate, up):
        gate = cl.cast(cl.cast(gate, cl.bf16), cl.f32)
        return activate(gate, kind) * cl.cast(cl.cast(up, cl.bf16), cl.f32)

    return Glu(pair)


EMBED_THREADS = 256
ACTIVATIONS = {"silu": epilogue.silu, "gelu_tanh": epilogue.gelu_tanh}


def activate(gate, kind):
    """The gated MLP activation of an f32 gate."""
    if kind not in ACTIVATIONS:
        raise ValueError(f"unknown activation {kind!r}")
    return ACTIVATIONS[kind](gate, None, None, ())


ELEMENT_BLOCK, ELEMENT_THREADS = 1024, 256


@cache
def _embedding_kernel(bits, group):
    fmt = Affine(bits, group)

    @cl.kernel
    def embed(
        ids: cl.buffer(cl.u32),
        packed: cl.buffer(cl.u32),
        scales: cl.buffer(cl.bf16),
        biases: cl.buffer(cl.bf16),
        output: cl.buffer(cl.bf16),
        WIDTH: cl.constexpr,
        SCALE: cl.constexpr,
        SCHEDULE: cl.constexpr,
    ):
        row = cl.program_id()
        column = cl.arange(WIDTH, layout=SCHEDULE)
        token = cl.load(ids, row, True, other=0)
        value = cl.cast(
            cl.cast(
                dequantize(packed, scales, biases, token, column, WIDTH, True, fmt),
                cl.bf16,
            ),
            cl.f32,
        )
        cl.store(output, row * WIDTH + column, cl.cast(value * SCALE, cl.bf16), True)

    return embed


@cache
def _dense_embedding_kernel():
    @cl.kernel
    def embed(
        ids: cl.buffer(cl.u32),
        table: cl.buffer(cl.bf16),
        output: cl.buffer(cl.bf16),
        WIDTH: cl.constexpr,
        SCALE: cl.constexpr,
        SCHEDULE: cl.constexpr,
    ):
        row = cl.program_id()
        column = cl.arange(WIDTH, layout=SCHEDULE)
        token = cl.load(ids, row, True, other=0)
        value = cl.cast(cl.load(table, token * WIDTH + column, True, other=0.0), cl.f32)
        cl.store(output, row * WIDTH + column, cl.cast(value * SCALE, cl.bf16), True)

    return embed


def embedding(device, ids, weight, output, rows, scale=1.0):
    """Dispatch: rows `ids` of an embedding table, times `scale` rounded to bf16."""
    scale = float(from_bf16(to_bf16(np.float32([scale])))[0])
    constants = {
        "WIDTH": weight.inputs,
        "SCALE": scale,
        "SCHEDULE": Layout.blocked((weight.inputs,), threads=EMBED_THREADS),
    }
    if isinstance(weight, QuantizedWeight):
        kernel, buffers = (
            _embedding_kernel(weight.bits, weight.group),
            [weight.packed, weight.scales, weight.biases],
        )
    else:
        kernel, buffers = _dense_embedding_kernel(), [weight.buffer]
    artifact = kernel.compile(constants=constants, threads=EMBED_THREADS, device=device)
    return artifact.dispatch(device, [ids, *buffers, output], grid=(rows, 1, 1))


@cache
def _ple_kernels():
    def bf16(value):
        return cl.cast(cl.cast(value, cl.bf16), cl.f32)

    @cl.kernel
    def ple_combine(
        normed: cl.buffer(cl.bf16),
        raw: cl.buffer(cl.bf16),
        output: cl.buffer(cl.bf16),
        SIZE: cl.constexpr,
        SCALE: cl.constexpr,
        SCHEDULE: cl.constexpr,
    ):
        index = cl.program_id() * SCHEDULE.size + cl.arange(
            SCHEDULE.size,
            layout=SCHEDULE,
        )
        active = index < SIZE
        total = bf16(
            cl.cast(cl.load(normed, index, active, other=0.0), cl.f32)
            + cl.cast(cl.load(raw, index, active, other=0.0), cl.f32),
        )
        cl.store(output, index, cl.cast(total * SCALE, cl.bf16), active)

    @cl.kernel
    def ple_gate(
        gate: cl.buffer(cl.bf16),
        inputs: cl.buffer(cl.bf16),
        output: cl.buffer(cl.bf16),
        SIZE: cl.constexpr,
        WIDTH: cl.constexpr,
        STRIDE: cl.constexpr,
        OFFSET: cl.constexpr,
        SCHEDULE: cl.constexpr,
    ):
        index = cl.program_id() * SCHEDULE.size + cl.arange(
            SCHEDULE.size,
            layout=SCHEDULE,
        )
        active = index < SIZE
        value = bf16(
            activate(
                cl.cast(cl.load(gate, index, active, other=0.0), cl.f32),
                "gelu_tanh",
            ),
        )
        other = cl.cast(
            cl.load(
                inputs,
                index // WIDTH * STRIDE + OFFSET + index % WIDTH,
                active,
                other=0.0,
            ),
            cl.f32,
        )
        cl.store(output, index, cl.cast(value * other, cl.bf16), active)

    return ple_combine, ple_gate


def _elementwise(size):
    return CopyAtom(cl.bf16, 4).layout((ELEMENT_BLOCK,), threads=ELEMENT_THREADS), -(
        -size // ELEMENT_BLOCK
    )


def ple_combine(device, normed, raw, output, size, scale):
    """Dispatch: output = (normed + raw) · scale, each step rounded to bf16."""
    schedule, grid = _elementwise(size)
    scale = float(from_bf16(to_bf16(np.float32([scale])))[0])
    artifact = _ple_kernels()[0].compile(
        constants={"SIZE": size, "SCALE": scale, "SCHEDULE": schedule},
        threads=ELEMENT_THREADS,
        device=device,
    )
    return artifact.dispatch(device, [normed, raw, output], grid=(grid, 1, 1))


def ple_gate(device, gate, inputs, output, rows, width, stride, offset):
    """Dispatch: output = gelu_tanh(gate) · a slice of `inputs`."""
    size = rows * width
    schedule, grid = _elementwise(size)
    artifact = _ple_kernels()[1].compile(
        constants={
            "SIZE": size,
            "WIDTH": width,
            "STRIDE": stride,
            "OFFSET": offset,
            "SCHEDULE": schedule,
        },
        threads=ELEMENT_THREADS,
        device=device,
    )
    return artifact.dispatch(device, [gate, inputs, output], grid=(grid, 1, 1))


@cache
def _place_kernel():
    @cl.kernel
    def place(
        source: cl.buffer(cl.bf16),
        output: cl.buffer(cl.bf16),
        SIZE: cl.constexpr,
        WIDTH: cl.constexpr,
        STRIDE: cl.constexpr,
        OFFSET: cl.constexpr,
        SCHEDULE: cl.constexpr,
    ):
        index = cl.program_id() * SCHEDULE.size + cl.arange(
            SCHEDULE.size,
            layout=SCHEDULE,
        )
        active = index < SIZE
        cl.store(
            output,
            index // WIDTH * STRIDE + OFFSET + index % WIDTH,
            cl.load(source, index, active, other=0.0),
            active,
        )

    return place


def place(device, source, output, rows, width, stride, offset):
    """Dispatch: `rows` contiguous bf16 rows of `width` into columns [offset, offset +
    width) of `output`, whose rows are `stride` wide."""
    schedule, grid = _elementwise(rows * width)
    artifact = _place_kernel().compile(
        constants={
            "SIZE": rows * width,
            "WIDTH": width,
            "STRIDE": stride,
            "OFFSET": offset,
            "SCHEDULE": schedule,
        },
        threads=ELEMENT_THREADS,
        device=device,
    )
    return artifact.dispatch(device, [source, output], grid=(grid, 1, 1))


@cache
def _feed_kernel():
    @cl.kernel
    def feed(
        token: cl.buffer(cl.u32),
        ids: cl.buffer(cl.u32),
        lengths: cl.buffer(cl.u32),
        history: cl.buffer(cl.u32),
        ROWS: cl.constexpr,
        CAPACITY: cl.constexpr,
        SCHEDULE: cl.constexpr,
    ):
        row = cl.arange(SCHEDULE.size, layout=SCHEDULE)
        active = row < ROWS
        value = cl.load(token, row, active, other=0)
        length = cl.load(lengths, row, active, other=0)
        cl.store(ids, row, value, active)
        cl.store(history, row * CAPACITY + length, value, active)
        cl.store(lengths, row, length + 1, active)

    return feed


def feed(device, token, ids, lengths, history, rows, capacity):
    """Dispatch: each row's chosen token becomes its next input and its length
    advances."""
    threads = 32 * -(-rows // 32)
    artifact = _feed_kernel().compile(
        constants={
            "ROWS": rows,
            "CAPACITY": capacity,
            "SCHEDULE": Layout.blocked((threads,), threads=threads),
        },
        threads=threads,
        device=device,
    )
    return artifact.dispatch(device, [token, ids, lengths, history], grid=(1, 1, 1))
