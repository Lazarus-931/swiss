from functools import cache

from carbon import language as cl
from carbon.compiler import Layout
from carbon.kernels.common import Hook, operand_buffers, round_to

ROW_THREADS = 256
GROUP_ELEMENTS = 2048


def _total(value, shape):
    runs = cl.sum(cl.reshape(value, (ROW_THREADS, shape[0] // ROW_THREADS)), 1)
    return cl.item(
        cl.sum(runs, 0, scratch=cl.scratch((ROW_THREADS,), cl.f32)),
        scratch=cl.scratch((1,), cl.f32),
    )


@cache
def make_rms_rows(dtype=cl.bf16, epilogue=None):

    @cl.kernel
    def rms_rows(
        source: cl.buffer(dtype),
        weight: cl.buffer(dtype),
        output: cl.buffer(dtype),
        outs: operand_buffers(epilogue),
        epsilon: cl.f32,
        WIDTH: cl.constexpr,
        SCHEDULE: cl.constexpr,
    ):
        column = cl.arange(SCHEDULE.size, layout=SCHEDULE)
        active = column < WIDTH
        offset = cl.program_id() * WIDTH + column
        value = cl.cast(cl.load(source, offset, active, other=0.0), cl.f32)
        inverse = cl.rsqrt(_total(value * value, SCHEDULE.shape) / WIDTH + epsilon)
        scale = cl.cast(cl.load(weight, column, active, other=0.0), cl.f32)
        normed = value * inverse * scale
        result = (
            normed
            if epilogue is None
            else epilogue(normed, cl.program_id(), column, outs)
        )
        cl.store(output, offset, cl.cast(result, dtype), active)

    return rms_rows


@cache
def make_rms_groups(dtype, *, groups, width, epilogue):
    layout = Layout.blocked((width,), threads=32)

    @cl.kernel
    def rms_groups(
        source: cl.buffer(dtype),
        weight: cl.buffer(dtype),
        output: cl.buffer(dtype),
        outs: operand_buffers(epilogue),
        epsilon: cl.f32,
    ):
        part = cl.program_id() % groups
        row = cl.program_id() // groups
        index = cl.arange(width, layout=layout)
        value = cl.cast(
            cl.load(source, cl.program_id() * width + index, True, other=0.0),
            cl.f32,
        )
        scratch = cl.scratch((width,), cl.f32)
        inverse = cl.rsqrt(
            cl.item(cl.sum(value * value, 0, scratch=scratch), scratch=scratch) / width
            + epsilon,
        )
        normed = (
            value * inverse * cl.cast(cl.load(weight, index, True, other=0.0), cl.f32)
        )
        result = (
            normed
            if epilogue is None
            else epilogue(normed, row, part * width + index, outs)
        )
        cl.store(output, cl.program_id() * width + index, cl.cast(result, dtype), True)

    return rms_groups


def rms_norm_rows(
    device,
    source,
    weight,
    output,
    *,
    rows,
    columns,
    dtype=cl.bf16,
    epsilon=1e-5,
    groups=1,
    epilogue=None,
    operands=(),
):
    """Dispatches RMSNorm over `rows` contiguous rows of `columns` elements, one
    threadgroup per row."""
    if len(operands) != (0 if epilogue is None else len(epilogue.operands)):
        raise ValueError("the epilogue takes one buffer per operand type")
    if groups > 1:
        if columns % groups or columns // groups % 32:
            raise ValueError(
                "groups split each row into parts of a multiple of 32 columns",
            )
        artifact = make_rms_groups(
            dtype,
            groups=groups,
            width=columns // groups,
            epilogue=epilogue,
        ).compile(constants={}, threads=32, device=device)
        return [
            artifact.dispatch(
                device,
                [source, weight, output, *operands],
                grid=(rows * groups, 1, 1),
                scalars={"epsilon": float(epsilon)},
            ),
        ]
    schedule = Layout.blocked(
        (-(-columns // ROW_THREADS) * ROW_THREADS,),
        threads=ROW_THREADS,
    )
    artifact = make_rms_rows(dtype, epilogue).compile(
        constants={"WIDTH": columns, "SCHEDULE": schedule},
        threads=ROW_THREADS,
        device=device,
    )
    return [
        artifact.dispatch(
            device,
            [source, weight, output, *operands],
            grid=(rows, 1, 1),
            scalars={"epsilon": epsilon},
        ),
    ]


def _layered(value, weight, bias, column, active, width, epsilon, shape):
    centered = cl.select(active, value - _total(value, shape) / width, 0.0)
    spread = _total(centered * centered, shape)
    scale = cl.cast(cl.load(weight, column, active, other=0.0), cl.f32)
    shift = cl.cast(cl.load(bias, column, active, other=0.0), cl.f32)
    return centered * cl.rsqrt(spread / width + epsilon) * scale + shift


@cache
def make_layer_norm(dtype=cl.bf16):
    if dtype not in (cl.f16, cl.bf16, cl.f32):
        raise ValueError("normalization storage must be f16, bf16, or f32")

    @cl.kernel
    def layer_norm(
        source: cl.buffer(dtype),
        weight: cl.buffer(dtype),
        bias: cl.buffer(dtype),
        output: cl.buffer(dtype),
        epsilon: cl.f32,
        WIDTH: cl.constexpr,
        SCHEDULE: cl.constexpr,
    ):
        column = cl.arange(SCHEDULE.size, layout=SCHEDULE)
        active = column < WIDTH
        offset = cl.program_id() * WIDTH + column
        value = cl.cast(cl.load(source, offset, active, other=0.0), cl.f32)
        cl.store(
            output,
            offset,
            cl.cast(
                _layered(
                    value,
                    weight,
                    bias,
                    column,
                    active,
                    WIDTH,
                    epsilon,
                    SCHEDULE.shape,
                ),
                dtype,
            ),
            active,
        )

    return layer_norm


def layer_norm(
    device,
    source,
    weight,
    bias,
    output,
    *,
    rows,
    columns,
    epsilon,
    dtype=cl.bf16,
):
    """Dispatch LayerNorm with a weight and a bias over `rows` rows of `columns`, one
    row per threadgroup."""
    schedule = Layout.blocked(
        (-(-columns // ROW_THREADS) * ROW_THREADS,),
        threads=ROW_THREADS,
    )
    artifact = make_layer_norm(dtype).compile(
        constants={"WIDTH": columns, "SCHEDULE": schedule},
        threads=ROW_THREADS,
        device=device,
    )
    return [
        artifact.dispatch(
            device,
            [source, weight, bias, output],
            grid=(rows, 1, 1),
            scalars={"epsilon": float(epsilon)},
        ),
    ]


@cache
def make_residual_norm(
    dtype=cl.bf16,
    *,
    sandwich=False,
    epilogue=None,
    normed=True,
    layered=False,
):
    if dtype not in (cl.f16, cl.bf16, cl.f32):
        raise ValueError("normalization storage must be f16, bf16, or f32")

    def normalized(value, weight, column, active, width, epsilon, shape):
        squares = _total(value * value, shape)
        scale = cl.cast(cl.load(weight, column, active, other=0.0), cl.f32)
        return round_to(value * cl.rsqrt(squares / width + epsilon) * scale, dtype)

    @cl.kernel
    def residual_norm(
        state: cl.buffer(dtype),
        update: cl.buffer(dtype),
        update_weight: (cl.buffer(dtype),) if sandwich else (),
        outs: operand_buffers(epilogue),
        weight: (cl.buffer(dtype),) if normed else (),
        bias: (cl.buffer(dtype),) if layered else (),
        output: (cl.buffer(dtype),) if normed else (),
        epsilon: cl.f32,
        FIRST: cl.constexpr,
        WIDTH: cl.constexpr,
        SCHEDULE: cl.constexpr,
    ):
        column = cl.arange(SCHEDULE.size, layout=SCHEDULE)
        active = column < WIDTH
        offset = cl.program_id() * WIDTH + column
        change = cl.cast(cl.load(update, offset, active, other=0.0), cl.f32)
        if sandwich:
            change = normalized(
                change,
                update_weight[0],
                column,
                active,
                WIDTH,
                epsilon,
                SCHEDULE.shape,
            )
        total = round_to(
            cl.cast(cl.load(state, offset, active, other=0.0), cl.f32) + change,
            dtype,
        )
        if epilogue is not None:
            total = round_to(epilogue(total, cl.program_id(), column, outs), dtype)
        cl.store(state, offset, cl.cast(total, dtype), active)
        if normed and layered:
            value = _layered(
                total,
                weight[0],
                bias[0],
                column,
                active,
                WIDTH,
                epsilon,
                SCHEDULE.shape,
            )
        elif normed:
            value = normalized(
                total,
                weight[0],
                column,
                active,
                WIDTH,
                epsilon,
                SCHEDULE.shape,
            )
        if normed:
            cl.store(
                output[0],
                (cl.program_id() - FIRST) * WIDTH + column,
                cl.cast(value, dtype),
                active & (cl.program_id() >= FIRST),
            )

    return residual_norm


def _times_first(value, row, column, scalar):
    return value * cl.cast(cl.load(scalar, 0, True, other=0.0), cl.f32)


def residual_norm(
    device,
    state,
    update,
    *,
    rows,
    columns,
    epsilon,
    update_weight=None,
    scalar=None,
    weight=None,
    output=None,
    first=0,
    bias=None,
    epilogue=None,
    operands=(),
    dtype=cl.bf16,
):
    """Return the dispatch that adds `update` to `state` and normalizes the new state
    into `output`."""
    if (weight is None) != (output is None):
        raise ValueError("weight and output come together")
    if bias is not None and weight is None:
        raise ValueError("a LayerNorm bias needs its weight")
    if scalar is not None:
        if epilogue is not None:
            raise ValueError("pass a scalar or an epilogue, not both")
        epilogue, operands = Hook(_times_first, (dtype,)), (scalar,)
    if len(operands) != (0 if epilogue is None else len(epilogue.operands)):
        raise ValueError("an epilogue takes one buffer per operand type")
    schedule = Layout.blocked(
        (-(-columns // ROW_THREADS) * ROW_THREADS,),
        threads=ROW_THREADS,
    )
    artifact = make_residual_norm(
        dtype,
        sandwich=update_weight is not None,
        epilogue=epilogue,
        normed=weight is not None,
        layered=bias is not None,
    ).compile(
        constants={"FIRST": first, "WIDTH": columns, "SCHEDULE": schedule},
        threads=ROW_THREADS,
        device=device,
    )
    bound = [
        buffer
        for buffer in (update_weight, *operands, weight, bias, output)
        if buffer is not None
    ]
    return [
        artifact.dispatch(
            device,
            [state, update, *bound],
            grid=(rows, 1, 1),
            scalars={"epsilon": float(epsilon)},
        ),
    ]


@cache
def make_group_stats(dtype=cl.bf16):

    @cl.kernel
    def group_stats(
        source: cl.buffer(dtype),
        partials: cl.buffer(cl.f32),
        pixels: cl.u32,
        C: cl.constexpr,
        CG: cl.constexpr,
        PIX: cl.constexpr,
        SCHEDULE: cl.constexpr,
    ):
        groups = C // CG
        group, block = cl.program_id() % groups, cl.program_id() // groups
        element = cl.arange(SCHEDULE.size, layout=SCHEDULE)
        pixel = block * PIX + element // CG
        live = (element < PIX * CG) & (pixel < pixels)
        value = cl.cast(
            cl.load(source, pixel * C + group * CG + element % CG, live, other=0.0),
            cl.f32,
        )
        count = cl.cast(cl.minimum(pixels - block * PIX, PIX) * CG, cl.f32)
        total = cl.item(
            cl.sum(value, 0, scratch=cl.scratch(SCHEDULE.shape, cl.f32)),
            scratch=cl.scratch((1,), cl.f32),
        )
        centered = cl.select(live, value - total / count, 0.0)
        spread = cl.item(
            cl.sum(centered * centered, 0, scratch=cl.scratch(SCHEDULE.shape, cl.f32)),
            scratch=cl.scratch((1,), cl.f32),
        )
        slot = (block * groups + group) * 2 + element
        cl.store(
            partials,
            slot,
            cl.select(element == 0, total / count, spread),
            element < 2,
        )

    return group_stats


@cache
def make_group_combine():
    schedule = Layout.blocked((ROW_THREADS,), threads=ROW_THREADS)

    def sized(block, blocks, pixels, PIX, CG):
        last = cl.minimum(pixels - cl.minimum(block, blocks - 1) * PIX, PIX)
        return cl.select(block < blocks, cl.cast(last * CG, cl.f32), 0.0)

    @cl.kernel
    def group_combine(
        partials: cl.buffer(cl.f32),
        moments: cl.buffer(cl.f32),
        pixels: cl.u32,
        epsilon: cl.f32,
        C: cl.constexpr,
        CG: cl.constexpr,
        PIX: cl.constexpr,
    ):
        groups = C // CG
        group = cl.program_id()
        lane = cl.arange(ROW_THREADS, layout=schedule)
        blocks = (pixels + PIX - 1) // PIX
        weighted = cl.full((ROW_THREADS,), 0.0, cl.f32, layout=schedule)
        for start in range(0, blocks, ROW_THREADS):
            block = start + lane
            at = (block * groups + group) * 2
            weighted = weighted + sized(block, blocks, pixels, PIX, CG) * cl.load(
                partials,
                at,
                block < blocks,
                other=0.0,
            )
        mean = cl.item(
            cl.sum(weighted, 0, scratch=cl.scratch((ROW_THREADS,), cl.f32)),
            scratch=cl.scratch((1,), cl.f32),
        ) / cl.cast(pixels * CG, cl.f32)
        spread = cl.full((ROW_THREADS,), 0.0, cl.f32, layout=schedule)
        for start in range(0, blocks, ROW_THREADS):
            block = start + lane
            at = (block * groups + group) * 2
            shift = cl.load(partials, at, block < blocks, other=0.0) - mean
            own = cl.load(partials, at + 1, block < blocks, other=0.0)
            spread = (
                spread + own + sized(block, blocks, pixels, PIX, CG) * shift * shift
            )
        variance = cl.item(
            cl.sum(spread, 0, scratch=cl.scratch((ROW_THREADS,), cl.f32)),
            scratch=cl.scratch((1,), cl.f32),
        ) / cl.cast(pixels * CG, cl.f32)
        cl.store(
            moments,
            group * 2 + lane,
            cl.select(lane == 0, mean, cl.rsqrt(variance + epsilon)),
            lane < 2,
        )

    return group_combine


@cache
def make_group_apply(dtype=cl.bf16, *, epilogue=None):
    schedule = Layout.blocked((GROUP_ELEMENTS,), threads=ROW_THREADS)

    @cl.kernel
    def group_apply(
        source: cl.buffer(dtype),
        moments: cl.buffer(cl.f32),
        weight: cl.buffer(dtype),
        bias: cl.buffer(dtype),
        output: cl.buffer(dtype),
        outs: operand_buffers(epilogue),
        pixels: cl.u32,
        C: cl.constexpr,
        CG: cl.constexpr,
    ):
        element = cl.program_id() * GROUP_ELEMENTS + cl.arange(
            GROUP_ELEMENTS,
            layout=schedule,
        )
        live = element < pixels * C
        channel = element % C
        group = channel // CG
        value = cl.cast(cl.load(source, element, live, other=0.0), cl.f32)
        mean = cl.load(moments, group * 2, live, other=0.0)
        scale = cl.load(moments, group * 2 + 1, live, other=0.0)
        result = (
            (value - mean)
            * scale
            * cl.cast(cl.load(weight, channel, live, other=0.0), cl.f32)
        )
        result = result + cl.cast(cl.load(bias, channel, live, other=0.0), cl.f32)
        result = (
            result
            if epilogue is None
            else epilogue(result, element // C, channel, outs)
        )
        cl.store(output, element, cl.cast(result, dtype), live)

    return group_apply


def group_norm(
    device,
    source,
    weight,
    bias,
    output,
    *,
    pixels,
    channels,
    groups,
    epsilon,
    epilogue=None,
    operands=(),
    dtype=cl.bf16,
):
    """Return dispatches for GroupNorm over an NHWC image as (pixels, channels) rows,
    then `epilogue`."""
    if channels % groups or pixels < 1 or pixels * channels >= 2**32:
        raise ValueError(
            "group_norm needs groups dividing the channels and pixels, under 2^32 "
            "elements",
        )
    if len(operands) != (0 if epilogue is None else len(epilogue.operands)):
        raise ValueError("the epilogue takes one buffer per operand type")
    per = channels // groups
    block = max(1, GROUP_ELEMENTS // per)
    blocks = -(-pixels // block)
    schedule = Layout.blocked(
        (-(-block * per // ROW_THREADS) * ROW_THREADS,),
        threads=ROW_THREADS,
    )
    partials, moments = device.buffer(blocks * groups * 8), device.buffer(groups * 8)
    constants = {"C": channels, "CG": per, "PIX": block}
    stats = make_group_stats(dtype).compile(
        constants={**constants, "SCHEDULE": schedule},
        threads=ROW_THREADS,
        device=device,
        max_grid=(2**32 - 1) // max(1, block * per) * groups,
    )
    combine = make_group_combine().compile(
        constants=constants,
        threads=ROW_THREADS,
        device=device,
        max_grid=groups,
    )
    apply = make_group_apply(dtype, epilogue=epilogue).compile(
        constants={"C": channels, "CG": per},
        threads=ROW_THREADS,
        device=device,
        max_grid=(2**32 - 1) // GROUP_ELEMENTS,
    )
    return [
        stats.dispatch(
            device,
            [source, partials],
            grid=(blocks * groups, 1, 1),
            scalars={"pixels": pixels},
        ),
        combine.dispatch(
            device,
            [partials, moments],
            grid=(groups, 1, 1),
            scalars={"pixels": pixels, "epsilon": float(epsilon)},
        ),
        apply.dispatch(
            device,
            [source, moments, weight, bias, output, *operands],
            grid=(-(-pixels * channels // GROUP_ELEMENTS), 1, 1),
            scalars={"pixels": pixels},
        ),
    ]


@cache
def make_segment_mean(dtype=cl.bf16, *, normalized=True):
    if dtype not in (cl.f16, cl.bf16, cl.f32):
        raise ValueError("normalization storage must be f16, bf16, or f32")

    @cl.kernel
    def segment_mean(
        source: cl.buffer(dtype),
        segments: cl.buffer(cl.u32),
        output: cl.buffer(cl.f32),
        WIDTH: cl.constexpr,
        SCHEDULE: cl.constexpr,
    ):
        column = cl.arange(SCHEDULE.size, layout=SCHEDULE)
        active = column < WIDTH
        first = cl.load(segments, cl.program_id(), True, other=0)
        last = cl.load(segments, cl.program_id() + 1, True, other=0)
        total = cl.full(SCHEDULE.shape, 0.0, cl.f32, layout=SCHEDULE)
        for row in range(first, last):
            total = total + cl.cast(
                cl.load(source, row * WIDTH + column, active, other=0.0),
                cl.f32,
            )
        mean = total / cl.cast(cl.maximum(last - first, 1), cl.f32)
        if normalized:
            mean = mean * cl.rsqrt(
                cl.maximum(_total(mean * mean, SCHEDULE.shape), 1e-24),
            )
        cl.store(output, cl.program_id() * WIDTH + column, mean, active)

    return segment_mean


def segment_mean(
    device,
    source,
    segments,
    output,
    *,
    sequences,
    columns,
    normalized=True,
    dtype=cl.bf16,
):
    """f32 (sequences, columns) means of the row segments that u32 `offsets` cut; with
    `normalized`, unit L2 norm."""
    if sequences < 1 or columns < 1:
        raise ValueError("segment_mean needs a sequence and a column")
    schedule = Layout.blocked(
        (-(-columns // ROW_THREADS) * ROW_THREADS,),
        threads=ROW_THREADS,
    )
    artifact = make_segment_mean(dtype, normalized=normalized).compile(
        constants={"WIDTH": columns, "SCHEDULE": schedule},
        threads=ROW_THREADS,
        device=device,
    )
    return [
        artifact.dispatch(device, [source, segments, output], grid=(sequences, 1, 1)),
    ]
