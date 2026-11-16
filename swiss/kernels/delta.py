from functools import cache

from carbon import language as cl
from carbon.compiler import Atom, CopyAtom, Layout
from carbon.kernels import shared_workspace
from carbon.kernels.common import Hook, operand_buffers, place

CHUNK = 32
HALF = CHUNK // 2
SCAN_ROWS = 32
SIMDS = CHUNK // 8
STEP_ROWS = 8
PAD = 4
PARTS = 4


class Gates(Hook):
    """`function(row, head, live, *operands)`."""

    def __call__(self, row, head, live, operands):
        decay, beta = self.function(row, head, live, *operands)
        return cl.select(live, decay, 0.0), cl.select(live, beta, 0.0)


@cache
def make_delta_chunks(
    *,
    key_heads,
    value_heads,
    key_dim,
    value_dim,
    gates,
    dtype=cl.bf16,
):
    HK, HV, DK, DV = key_heads, value_heads, key_dim, value_dim
    atom = Atom(cl.f32)
    threads = 32 * SIMDS
    fragment = atom.layout(threads=threads)
    rows_by_dim = Layout.blocked((CHUNK, DK), threads=threads)
    norms = rows_by_dim.reduced(1)
    lines = Layout.striped((CHUNK,), threads=threads)

    def normalize(qkv, scales, first, source, base, rows, CH, KEYS, EPS, SCALE):
        index = cl.arange(rows_by_dim.size, layout=rows_by_dim)
        live = first + index // DK < rows
        start = (first + index // DK) * CH + source + index % DK
        k = cl.cast(cl.load(qkv, start + KEYS, live, other=0.0), cl.f32)
        q = cl.cast(cl.load(qkv, start, live, other=0.0), cl.f32)
        key_scale = cl.rsqrt(cl.sum(k * k, 1, layout=norms) + EPS)
        query_scale = cl.rsqrt(cl.sum(q * q, 1, layout=norms) + EPS) * SCALE
        line = cl.arange(CHUNK, layout=norms)
        cl.store(scales, 2 * base + line, key_scale, True)
        cl.store(scales, 2 * base + CHUNK + line, query_scale, True)
        return key_scale, query_scale

    def key_tile(qkv, rows_at, dims_at, first, source, rows, CH, KEYS):
        at = (first + rows_at) * CH + KEYS + source + dims_at
        return cl.cast(cl.load(qkv, at, first + rows_at < rows, other=0.0), cl.f32)

    def query_tile(qkv, rows_at, dims_at, first, source, rows, CH):
        at = (first + rows_at) * CH + source + dims_at
        return cl.cast(cl.load(qkv, at, first + rows_at < rows, other=0.0), cl.f32)

    def key_products(qkv, first, source, rows, CH, KEYS):
        simd, local, column = place(fragment)
        mine = 8 * simd + local
        keys = [atom.zeros() for _ in range(SIMDS)]
        queries = [atom.zeros() for _ in range(SIMDS)]
        for depth in range(DK // 8):
            own = key_tile(qkv, mine, 8 * depth + column, first, source, rows, CH, KEYS)
            keys = [
                atom.mma(
                    own,
                    key_tile(
                        qkv,
                        8 * block + column,
                        8 * depth + local,
                        first,
                        source,
                        rows,
                        CH,
                        KEYS,
                    ),
                    total,
                )
                for block, total in enumerate(keys)
            ]
            queries = [
                atom.mma(
                    own,
                    query_tile(
                        qkv,
                        8 * block + column,
                        8 * depth + local,
                        first,
                        source,
                        rows,
                        CH,
                    ),
                    total,
                )
                for block, total in enumerate(queries)
            ]
        return keys, queries

    def mixed(strip, memory, scales, queries, logs, base):
        simd, local, column = place(fragment)
        mine = 8 * simd + local
        scale, log = (
            cl.load(scales, mine, True, other=0.0),
            cl.load(logs, mine, True, other=0.0),
        )
        for block, value in enumerate(strip):
            other = 8 * block + column
            entry = (
                value
                * scale
                * cl.load(queries, other, True, other=0.0)
                * cl.exp(cl.load(logs, other, True, other=0.0) - log)
            )
            cl.store(
                memory,
                (base + mine) * CHUNK + other,
                cl.select(other >= mine, entry, 0.0),
                True,
            )

    def lower_part(products, scales, betas, logs):
        simd, local, column = place(fragment)
        mine = 8 * simd + local
        scale, beta, log = (
            cl.load(memory, mine, True, other=0.0) for memory in (scales, betas, logs)
        )
        strip = []
        for block, value in enumerate(products):
            other = 8 * block + column
            entry = (
                value
                * scale
                * cl.load(scales, other, True, other=0.0)
                * beta
                * cl.exp(log - cl.load(logs, other, True, other=0.0))
            )
            strip.append(cl.select(other < mine, entry, 0.0))
        return strip

    def publish(square, strip):
        simd, local, column = place(fragment)
        cl.barrier()
        for block, value in enumerate(strip):
            cl.store(square, (8 * simd + local, 8 * block + column), value, True)
        cl.barrier()

    def substituted(square, inverse):
        item = cl.arange(CHUNK, layout=lines)
        corner, own = item // HALF * HALF, item % HALF
        solution = []
        for i in range(HALF):
            value = cl.select(own == i, 1.0, 0.0)
            for k in range(i):
                value = (
                    value
                    - cl.load(square, (corner + i, corner + k), True, other=0.0)
                    * solution[k]
                )
            solution.append(value)
        for i, value in enumerate(solution):
            cl.store(inverse, (corner + i, corner + own), value, True)
            cl.store(
                inverse,
                (i, HALF + own),
                cl.full((CHUNK,), 0.0, cl.f32, layout=lines),
                corner == 0,
            )

    def below(lower, inverse, crossed):
        simd, local, column = place(fragment)
        mine = 8 * simd + local
        for block in range(HALF // 8):
            total = atom.zeros()
            for inner in range(HALF // 8):
                total = atom.mma(
                    lower[inner],
                    cl.load(
                        inverse,
                        (8 * inner + local, 8 * block + column),
                        True,
                        other=0.0,
                    ),
                    total,
                )
            cl.store(crossed, (mine, 8 * block + column), total, True)
        cl.barrier()
        for block in range(HALF // 8):
            total = atom.zeros()
            for inner in range(HALF // 8):
                total = atom.mma(
                    cl.load(
                        inverse,
                        (mine, HALF + 8 * inner + column),
                        True,
                        other=0.0,
                    ),
                    cl.load(
                        crossed,
                        (HALF + 8 * inner + local, 8 * block + column),
                        True,
                        other=0.0,
                    ),
                    total,
                )
            cl.store(inverse, (mine, 8 * block + column), -total, simd >= HALF // 8)

    def inverted(square, inverse, crossed, lower):
        simd, local, column = place(fragment)
        publish(square, lower)
        substituted(square, inverse)
        cl.barrier()
        below(lower, inverse, crossed)
        cl.barrier()
        return [
            cl.load(inverse, (8 * simd + local, 8 * block + column), True, other=0.0)
            for block in range(SIMDS)
        ]

    def solved(inverse, qkv, memory, width, offset, weights, first, base, rows, CH):
        simd, local, column = place(fragment)
        mine = 8 * simd + local
        for depth in range(width // 8):
            total = atom.zeros()
            for block in range(SIMDS):
                row = 8 * block + local
                at = (first + row) * CH + offset + 8 * depth + column
                value = cl.cast(cl.load(qkv, at, first + row < rows, other=0.0), cl.f32)
                total = atom.mma(inverse[block], value * weights[block], total)
            cl.store(memory, (base + mine) * width + 8 * depth + column, total, True)

    def row_weights(scales, betas, logs):
        simd, local, column = place(fragment)
        values, keys = [], []
        for block in range(SIMDS):
            row = 8 * block + local
            beta = cl.load(betas, row, True, other=0.0)
            values.append(beta)
            keys.append(
                beta
                * cl.exp(cl.load(logs, row, True, other=0.0))
                * cl.load(scales, row, True, other=0.0),
            )
        return values, keys

    @cl.kernel
    def delta_chunks(
        qkv: cl.buffer(dtype),
        gate_buffers: operand_buffers(gates),
        scales_out: cl.buffer(cl.f32),
        corrected: cl.buffer(cl.f32),
        updates: cl.buffer(cl.f32),
        decays: cl.buffer(cl.f32),
        intra: cl.buffer(cl.f32),
        rows: cl.u32,
        CH: cl.constexpr,
        KEYS: cl.constexpr,
        VALUES: cl.constexpr,
        EPS: cl.constexpr,
        SCALE: cl.constexpr,
    ):
        head = cl.program_id() % HV
        chunk = cl.program_id() // HV
        first = chunk * CHUNK
        source = head // (HV // HK) * DK
        base = (chunk * HV + head) * CHUNK
        key_scale, query_scale = normalize(
            qkv,
            scales_out,
            first,
            source,
            base,
            rows,
            CH,
            KEYS,
            EPS,
            SCALE,
        )
        line = cl.arange(CHUNK, layout=lines)
        decay, beta = gates(first + line, head, first + line < rows, gate_buffers)
        cumulative = cl.cumsum(decay, 0, scratch=cl.scratch((CHUNK,), cl.f32))
        cl.store(decays, base + line, cumulative, True)
        scales, betas, logs = (
            cl.scratch((CHUNK,), cl.f32),
            cl.scratch((CHUNK,), cl.f32),
            cl.scratch((CHUNK,), cl.f32),
        )
        queries = cl.scratch((CHUNK,), cl.f32)
        cl.barrier()
        cl.store(scales, cl.arange(CHUNK, layout=norms), key_scale, True)
        cl.store(queries, cl.arange(CHUNK, layout=norms), query_scale, True)
        cl.store(betas, line, beta, True)
        cl.store(logs, line, cumulative, True)
        cl.barrier()
        square, triangle = (
            cl.scratch((CHUNK, CHUNK), cl.f32),
            cl.scratch((CHUNK, CHUNK), cl.f32),
        )
        crossed = cl.scratch((CHUNK, HALF), cl.f32)
        keys, mixing = key_products(qkv, first, source, rows, CH, KEYS)
        inverse = inverted(
            square,
            triangle,
            crossed,
            lower_part(keys, scales, betas, logs),
        )
        values, keys_weights = row_weights(scales, betas, logs)
        solved(
            inverse,
            qkv,
            corrected,
            DK,
            KEYS + source,
            keys_weights,
            first,
            base,
            rows,
            CH,
        )
        solved(
            inverse,
            qkv,
            updates,
            DV,
            VALUES + head * DV,
            values,
            first,
            base,
            rows,
            CH,
        )
        mixed(mixing, intra, scales, queries, logs, base)

    return delta_chunks


@cache
def make_delta_scan(*, key_heads, value_heads, key_dim, value_dim, dtype=cl.bf16):
    HK, HV, DK, DV = key_heads, value_heads, key_dim, value_dim
    atom = Atom(cl.f32)
    simds = SCAN_ROWS // 8
    threads = 32 * simds
    fragment = atom.layout(threads=threads)
    blocks = DV // SCAN_ROWS
    span = DK // PARTS
    copy = CopyAtom(cl.f32, 4).layout((CHUNK * span,), threads=threads)

    def staged(stage, memory, base, part):
        cl.barrier()
        index = cl.arange(CHUNK * span, layout=copy)
        row, dim = index // span, index % span
        cl.store(
            stage,
            dim * (CHUNK + PAD) + row,
            cl.load(memory, (base + row) * DK + part * span + dim, True, other=0.0),
            True,
        )
        cl.barrier()
        return lambda rows_at, columns_at: cl.load(
            stage,
            rows_at * (CHUNK + PAD) + columns_at,
            True,
            other=0.0,
        )

    def normalized(
        stage,
        qkv,
        scales,
        base,
        first,
        source,
        part,
        query,
        rows,
        CH,
        KEYS,
    ):
        width = CHUNK if query else span
        cl.barrier()
        index = cl.arange(CHUNK * span, layout=copy)
        row, dim = index // span, index % span
        at = (first + row) * CH + (0 if query else KEYS) + source + part * span + dim
        value = cl.cast(cl.load(qkv, at, first + row < rows, other=0.0), cl.f32)
        value = value * cl.load(
            scales,
            2 * base + (CHUNK if query else 0) + row,
            True,
            other=0.0,
        )
        place = dim * (CHUNK + PAD) + row if query else row * (span + PAD) + dim
        cl.store(stage, place, value, True)
        cl.barrier()
        return lambda rows_at, columns_at: cl.load(
            stage,
            rows_at * (width + PAD) + columns_at,
            True,
            other=0.0,
        )

    def advance(
        memory,
        chunk,
        qkv,
        scales,
        corrected,
        updates,
        decays,
        intra,
        output,
        stage,
        head,
        value_row,
        rows,
        CH,
        KEYS,
    ):
        simd, local, column = place(fragment)
        base = (chunk * HV + head) * CHUNK
        first = chunk * CHUNK
        source = head // (HV // HK) * DK
        last = cl.load(decays, base + CHUNK - 1, True, other=0.0)
        steps = span // 8
        totals = [atom.zeros() for _ in range(SIMDS)]
        for part in range(PARTS):
            read = staged(stage, corrected, base, part)
            for depth in range(steps):
                totals = [
                    atom.mma(
                        memory[part * steps + depth],
                        read(8 * depth + local, 8 * block + column),
                        total,
                    )
                    for block, total in enumerate(totals)
                ]
        fresh = [
            cl.load(
                updates,
                (base + 8 * block + column) * DV + value_row,
                True,
                other=0.0,
            )
            - total
            for block, total in enumerate(totals)
        ]
        outs = [atom.zeros() for _ in range(SIMDS)]
        for part in range(PARTS):
            read = normalized(
                stage,
                qkv,
                scales,
                base,
                first,
                source,
                part,
                True,
                rows,
                CH,
                KEYS,
            )
            for depth in range(steps):
                outs = [
                    atom.mma(
                        memory[part * steps + depth],
                        read(8 * depth + local, 8 * block + column),
                        total,
                    )
                    for block, total in enumerate(outs)
                ]
        for block, total in enumerate(outs):
            total = total * cl.exp(
                cl.load(decays, base + 8 * block + column, True, other=0.0),
            )
            for inner in range(SIMDS):
                weight = cl.load(
                    intra,
                    (base + 8 * inner + local) * CHUNK + 8 * block + column,
                    True,
                    other=0.0,
                )
                total = atom.mma(fresh[inner], weight, total)
            position = chunk * CHUNK + 8 * block + column
            cl.store(
                output,
                (position * HV + head) * DV + value_row,
                cl.cast(total, dtype),
                position < rows,
            )
        weights = [
            cl.exp(last - cl.load(decays, base + 8 * inner + local, True, other=0.0))
            for inner in range(SIMDS)
        ]
        carried = cl.exp(last)
        updated = []
        for part in range(PARTS):
            read = normalized(
                stage,
                qkv,
                scales,
                base,
                first,
                source,
                part,
                False,
                rows,
                CH,
                KEYS,
            )
            for depth in range(steps):
                total = memory[part * steps + depth] * carried
                for inner in range(SIMDS):
                    total = atom.mma(
                        fresh[inner],
                        read(8 * inner + local, 8 * depth + column) * weights[inner],
                        total,
                    )
                updated.append(total)
        return tuple(updated)

    def loaded(state, slot, value_row):
        simd, local, column = place(fragment)
        return tuple(
            cl.load(
                state,
                (slot + value_row) * DK + 8 * depth + column,
                True,
                other=0.0,
            )
            for depth in range(DK // 8)
        )

    def saved(state, slot, value_row, memory):
        simd, local, column = place(fragment)
        for depth in range(DK // 8):
            cl.store(
                state,
                (slot + value_row) * DK + 8 * depth + column,
                memory[depth],
                True,
            )

    def value_rows(first_value):
        simd, local, column = place(fragment)
        return first_value + 8 * simd + local

    @cl.kernel
    def delta_scan(
        qkv: cl.buffer(dtype),
        scales: cl.buffer(cl.f32),
        corrected: cl.buffer(cl.f32),
        updates: cl.buffer(cl.f32),
        decays: cl.buffer(cl.f32),
        intra: cl.buffer(cl.f32),
        state: cl.buffer(cl.f32),
        slots: cl.buffer(cl.u32),
        output: cl.buffer(dtype),
        rows: cl.u32,
        CH: cl.constexpr,
        KEYS: cl.constexpr,
    ):
        head = cl.program_id() // blocks
        value_row = value_rows(cl.program_id() % blocks * SCAN_ROWS)
        slot = (cl.load(slots, 0, True, other=0) * HV + head) * DV
        memory = loaded(state, slot, value_row)
        stage = cl.scratch((max(span * (CHUNK + PAD), CHUNK * (span + PAD)),), cl.f32)
        for chunk in range((rows + CHUNK - 1) // CHUNK):
            memory = advance(
                memory,
                chunk,
                qkv,
                scales,
                corrected,
                updates,
                decays,
                intra,
                output,
                stage,
                head,
                value_row,
                rows,
                CH,
                KEYS,
            )
        saved(state, slot, value_row, memory)

    return delta_scan


@cache
def make_delta_steps(
    *,
    key_heads,
    value_heads,
    key_dim,
    value_dim,
    gates,
    batch,
    snapshot=False,
    dtype=cl.bf16,
):
    HK, HV, DK, DV = key_heads, value_heads, key_dim, value_dim
    layout = Layout.blocked((STEP_ROWS, DK), threads=32 * STEP_ROWS)
    values = layout.reduced(1)
    parts = DV // STEP_ROWS

    @cl.kernel
    def delta_steps(
        qkv: cl.buffer(dtype),
        gate_buffers: operand_buffers(gates),
        state: cl.buffer(cl.f32),
        slots: cl.buffer(cl.u32),
        output: cl.buffer(dtype),
        snapshots: (cl.buffer(cl.u32),) if snapshot else (),
        rows: cl.u32,
        CH: cl.constexpr,
        KEYS: cl.constexpr,
        VALUES: cl.constexpr,
        EPS: cl.constexpr,
        SCALE: cl.constexpr,
    ):
        local = cl.program_id() % (HV * parts)
        start = cl.program_id() // (HV * parts) if batch else 0
        head = local // parts
        first = local % parts * STEP_ROWS
        source = head // (HV // HK) * DK
        index = cl.arange(layout.size, layout=layout)
        dimension = index % DK
        outputs = cl.arange(STEP_ROWS, layout=values)
        base = ((cl.load(slots, start, True, other=0) * HV + head) * DV + first) * DK
        memory = cl.load(state, base + index, True, other=0.0)
        for step in range(1 if batch else rows):
            row = start + step
            query = cl.cast(
                cl.load(qkv, row * CH + source + dimension, True, other=0.0),
                cl.f32,
            )
            key = cl.cast(
                cl.load(qkv, row * CH + KEYS + source + dimension, True, other=0.0),
                cl.f32,
            )
            value = cl.cast(
                cl.load(
                    qkv,
                    row * CH + VALUES + head * DV + first + outputs,
                    True,
                    other=0.0,
                ),
                cl.f32,
            )
            log_decay, beta = gates(row, head, True, gate_buffers)
            key_scale = cl.rsqrt(cl.sum(key * key, 1, layout=values) + EPS)
            query_scale = (
                cl.rsqrt(cl.sum(query * query, 1, layout=values) + EPS) * SCALE
            )
            memory = memory * cl.exp(log_decay)
            recalled = cl.sum(memory * key, 1, layout=values) * key_scale
            delta = (value - recalled) * beta * key_scale
            memory = memory + key * cl.broadcast_to(
                delta,
                (STEP_ROWS, DK),
                axes=(0,),
                layout=layout,
            )
            result = cl.sum(memory * query, 1, layout=values) * query_scale
            cl.store(
                output,
                (row * HV + head) * DV + first + outputs,
                cl.cast(result, dtype),
                True,
            )
            if snapshot:
                kept = (
                    (cl.load(snapshots[0], step, True, other=0) * HV + head) * DV
                    + first
                ) * DK
                cl.store(state, kept + index, memory, True)
        cl.store(state, base + index, memory, True)

    return delta_steps


def delta_workspace_bytes(rows, *, value_heads, key_dim, value_dim):
    """Bytes of the workspace `gated_delta` needs for a prefill of `rows` rows."""
    chunks = -(-rows // CHUNK)
    return 4 * chunks * value_heads * CHUNK * (key_dim + value_dim + CHUNK + 3)


def delta_rule(
    device,
    qkv,
    state,
    slots,
    output,
    *,
    rows,
    key_heads,
    value_heads,
    key_dim,
    value_dim,
    gates,
    gate_operands=(),
    row_stride=None,
    keys_at=None,
    values_at=None,
    scale=None,
    eps=1e-6,
    batch=False,
    workspace=None,
    snapshots=None,
    dtype=cl.bf16,
):
    """Return dispatches for the gated delta rule over `rows` tokens, updating an f32
    state slot in place."""
    if value_heads % key_heads or key_dim % 32 or value_dim % 32:
        raise ValueError(
            "value heads must share key heads evenly, and key_dim and value_dim divide "
            "by 32",
        )
    if dtype not in (cl.f16, cl.bf16):
        raise ValueError("the delta rule takes f16 or bf16 rows")
    if len(gate_operands) != len(gates.operands):
        raise ValueError("gates take one buffer per operand type")
    if batch and snapshots is not None:
        raise ValueError("snapshots keep the states after the rows of one sequence")
    keys_at = key_heads * key_dim if keys_at is None else keys_at
    values_at = keys_at + key_heads * key_dim if values_at is None else values_at
    row_stride = (
        values_at + value_heads * value_dim if row_stride is None else row_stride
    )
    constants = {
        "CH": row_stride,
        "KEYS": keys_at,
        "VALUES": values_at,
        "EPS": float(eps),
        "SCALE": float(key_dim**-0.5 if scale is None else scale),
    }
    shape = dict(
        key_heads=key_heads,
        value_heads=value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        dtype=dtype,
    )
    if batch or rows < 2 * CHUNK or snapshots is not None:
        steps = make_delta_steps(
            **shape,
            gates=gates,
            batch=batch,
            snapshot=snapshots is not None,
        ).compile(constants=constants, threads=32 * STEP_ROWS, device=device)
        grid = value_heads * value_dim // STEP_ROWS * (rows if batch else 1)
        kept = [] if snapshots is None else [snapshots]
        return [
            steps.dispatch(
                device,
                [qkv, *gate_operands, state, slots, output, *kept],
                grid=(grid, 1, 1),
                scalars={"rows": rows},
            ),
        ]
    needed = delta_workspace_bytes(
        rows,
        value_heads=value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
    )
    if workspace is None:
        workspace = shared_workspace(device, needed)
    elif workspace.size < needed:
        raise ValueError(
            "a chunked prefill needs a workspace of delta_workspace_bytes(rows, ...) "
            "bytes",
        )
    chunks = -(-rows // CHUNK)
    sizes = [
        4 * chunks * value_heads * CHUNK * width
        for width in (2, key_dim, value_dim, 1, CHUNK)
    ]
    parts, offset = [], 0
    for size in sizes:
        parts.append(workspace.view(offset, size))
        offset += size
    scales, corrected, updates, decays, intra = parts
    prepare = make_delta_chunks(**shape, gates=gates).compile(
        constants=constants,
        threads=32 * SIMDS,
        device=device,
    )
    scan = make_delta_scan(**shape).compile(
        constants={"CH": row_stride, "KEYS": keys_at},
        threads=32 * (SCAN_ROWS // 8),
        device=device,
    )
    return [
        prepare.dispatch(
            device,
            [qkv, *gate_operands, scales, corrected, updates, decays, intra],
            grid=(chunks * value_heads, 1, 1),
            scalars={"rows": rows},
        ),
        scan.dispatch(
            device,
            [qkv, scales, corrected, updates, decays, intra, state, slots, output],
            grid=(value_heads * (value_dim // SCAN_ROWS), 1, 1),
            scalars={"rows": rows},
        ),
    ]
