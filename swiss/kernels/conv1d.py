from dataclasses import dataclass
from functools import cache

from carbon import language as cl
from carbon.compiler import Layout
from carbon.kernels.common import Hook, applied, operand_buffers

CONV_BLOCK = 256
CONV_ROWS = 1024
CONV_NARROW = 90


Inputs = Hook


@dataclass(frozen=True)
class _Strided:
    stride: int

    def __call__(self, row, channel, live, source):
        return cl.cast(
            cl.load(source, row * self.stride + channel, live, other=0.0),
            cl.f32,
        )


def strided(stride, dtype=cl.bf16):
    """Inputs source[row · stride + channel] from one buffer of `dtype`: the first
    channels of rows `stride` apart."""
    return Inputs(_Strided(stride), (dtype,))


@cache
def make_conv_walk(*, channels, taps, batch, snapshot, inputs, epilogue):
    C = channels
    layout = Layout.blocked((CONV_BLOCK,), threads=CONV_BLOCK)

    def held(state, base, channel, active):
        return tuple(
            cl.cast(cl.load(state, base + tap * C + channel, active, other=0.0), cl.f32)
            for tap in range(taps - 1)
        )

    def kept(state, base, values, channel, active):
        for tap, value in enumerate(values):
            cl.store(state, base + tap * C + channel, cl.cast(value, cl.bf16), active)

    def shifted(values, value):
        return (values + (value,))[1:]

    def mixed(weight, values, value, channel, active):
        total = None
        for tap, x in enumerate(values + (value,)):
            w = cl.cast(
                cl.load(weight, channel * taps + tap, active, other=0.0),
                cl.f32,
            )
            total = w * x if total is None else total + w * x
        return total

    @cl.kernel
    def conv_walk(
        sources: operand_buffers(inputs),
        weight: cl.buffer(cl.bf16),
        state: cl.buffer(cl.bf16),
        slots: cl.buffer(cl.u32),
        output: cl.buffer(cl.bf16),
        outs: operand_buffers(epilogue),
        snapshots: (cl.buffer(cl.u32),) if snapshot else (),
        rows: cl.u32,
        BLOCKS: cl.constexpr,
    ):
        channel = cl.program_id() % BLOCKS * CONV_BLOCK + cl.arange(
            CONV_BLOCK,
            layout=layout,
        )
        active = channel < C
        start = cl.program_id() // BLOCKS if batch else 0
        base = cl.load(slots, start, True, other=0) * (taps - 1) * C
        values = held(state, base, channel, active)
        for step in range(1 if batch else rows):
            row = start + step
            value = inputs(row, channel, active, sources)
            total = mixed(weight, values, value, channel, active)
            cl.store(
                output,
                row * C + channel,
                cl.cast(applied(epilogue, total, row, channel, outs), cl.bf16),
                active,
            )
            values = shifted(values, value)
            if snapshot:
                kept(
                    state,
                    cl.load(snapshots[0], step, True, other=0) * (taps - 1) * C,
                    values,
                    channel,
                    active,
                )
        kept(state, base, values, channel, active)

    return conv_walk


@cache
def make_conv_rows(*, channels, taps, inputs, epilogue):
    C = channels
    layout = Layout.blocked((CONV_BLOCK,), threads=CONV_BLOCK)

    def window(sources, state, position, base, channel, active):
        held = position < taps - 1
        kept = cl.cast(
            cl.load(
                state,
                base + cl.select(held, position, 0) * C + channel,
                active & held,
                other=0.0,
            ),
            cl.f32,
        )
        row = cl.select(held, 0, position - (taps - 1))
        return cl.select(
            held,
            kept,
            inputs(row, channel, active & (position >= taps - 1), sources),
        )

    def convolved(sources, weight, state, row, base, channel, active):
        total = cl.full((CONV_BLOCK,), 0.0, cl.f32, layout=layout)
        for tap in range(taps):
            weighted = cl.cast(
                cl.load(weight, channel * taps + tap, active, other=0.0),
                cl.f32,
            )
            total = total + weighted * window(
                sources,
                state,
                row + tap,
                base,
                channel,
                active,
            )
        return total

    @cl.kernel
    def conv_rows(
        sources: operand_buffers(inputs),
        weight: cl.buffer(cl.bf16),
        state: cl.buffer(cl.bf16),
        slots: cl.buffer(cl.u32),
        output: cl.buffer(cl.bf16),
        outs: operand_buffers(epilogue),
        BLOCKS: cl.constexpr,
    ):
        row = cl.program_id() // BLOCKS
        channel = cl.program_id() % BLOCKS * CONV_BLOCK + cl.arange(
            CONV_BLOCK,
            layout=layout,
        )
        active = channel < C
        base = cl.load(slots, 0, True, other=0) * (taps - 1) * C
        total = convolved(sources, weight, state, row, base, channel, active)
        cl.store(
            output,
            row * C + channel,
            cl.cast(applied(epilogue, total, row, channel, outs), cl.bf16),
            active,
        )

    return conv_rows


@cache
def make_conv_shift(*, channels, taps, inputs):
    C = channels
    layout = Layout.blocked((CONV_BLOCK,), threads=CONV_BLOCK)

    @cl.kernel
    def conv_shift(
        sources: operand_buffers(inputs),
        state: cl.buffer(cl.bf16),
        slots: cl.buffer(cl.u32),
        rows: cl.u32,
    ):
        channel = cl.program_id() * CONV_BLOCK + cl.arange(CONV_BLOCK, layout=layout)
        active = channel < C
        base = cl.load(slots, 0, True, other=0) * (taps - 1) * C
        for held in range(taps - 1):
            position = rows + held
            stays = position < taps - 1
            kept = cl.cast(
                cl.load(
                    state,
                    base + cl.select(stays, position, 0) * C + channel,
                    active & stays,
                    other=0.0,
                ),
                cl.f32,
            )
            row = cl.select(stays, 0, position - (taps - 1))
            fresh = inputs(row, channel, active & (position >= taps - 1), sources)
            cl.store(
                state,
                base + held * C + channel,
                cl.cast(cl.select(stays, kept, fresh), cl.bf16),
                active,
            )

    return conv_shift


def causal_conv(
    device,
    weight,
    state,
    slots,
    output,
    *,
    rows,
    channels,
    taps,
    inputs,
    input_operands,
    epilogue=None,
    operands=(),
    batch=False,
    snapshots=None,
):
    """Return dispatches for output (rows, channels) = the causal depthwise convolution
    of the inputs."""
    if len(input_operands) != len(inputs.operands) or len(operands) != (
        0 if epilogue is None else len(epilogue.operands)
    ):
        raise ValueError("inputs and epilogue take one buffer per operand type")
    if batch and snapshots is not None:
        raise ValueError("snapshots keep the inputs after the rows of one sequence")
    blocks = -(-channels // CONV_BLOCK)
    sources, outs = list(input_operands), list(operands)
    if (
        batch
        or snapshots is not None
        or rows < CONV_ROWS
        or blocks * taps >= CONV_NARROW
    ):
        walk = make_conv_walk(
            channels=channels,
            taps=taps,
            batch=batch,
            snapshot=snapshots is not None,
            inputs=inputs,
            epilogue=epilogue,
        ).compile(constants={"BLOCKS": blocks}, threads=CONV_BLOCK, device=device)
        kept = [] if snapshots is None else [snapshots]
        return [
            walk.dispatch(
                device,
                [*sources, weight, state, slots, output, *outs, *kept],
                grid=(blocks * (rows if batch else 1), 1, 1),
                scalars={"rows": rows},
            ),
        ]
    convolve = make_conv_rows(
        channels=channels,
        taps=taps,
        inputs=inputs,
        epilogue=epilogue,
    ).compile(constants={"BLOCKS": blocks}, threads=CONV_BLOCK, device=device)
    shift = make_conv_shift(channels=channels, taps=taps, inputs=inputs).compile(
        constants={},
        threads=CONV_BLOCK,
        device=device,
    )
    return [
        convolve.dispatch(
            device,
            [*sources, weight, state, slots, output, *outs],
            grid=(rows * blocks, 1, 1),
        ),
        shift.dispatch(
            device,
            [*sources, state, slots],
            grid=(blocks, 1, 1),
            scalars={"rows": rows},
        ),
    ]
