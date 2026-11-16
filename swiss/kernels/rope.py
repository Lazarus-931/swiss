"""The fused [queries | keys | values | gates] row of an attention projection: RoPE
into the KV pages, and the gate that Carbon attention applies to its output."""

from dataclasses import dataclass
from functools import cache

import numpy as np
from carbon import language as cl
from carbon.compiler import Layout
from carbon.kernels.attention import OutputMod


def layout(
    heads,
    kv_heads,
    head_dim,
    gated=False,
    head_gates=False,
    k_eq_v=False,
    queries_only=False,
):
    """Row width, query stride, and key, value and gate columns of a fused row; with
    `gated`, each query head is followed by its gates."""
    stride = head_dim * (2 if gated else 1)
    keys = heads * stride
    values = keys if k_eq_v else keys + kv_heads * head_dim
    gate = values + kv_heads * head_dim
    row = keys if queries_only else gate + (heads if head_gates else 0)
    return {"row": row, "stride": stride, "keys": keys, "values": values, "gate": gate}


@dataclass(frozen=True)
class SigmoidGate:
    """round(value) · round(sigmoid(gate)) in bf16, one gate per head or per element."""

    offset: int
    row_stride: int
    head_stride: int
    elements: bool

    def __call__(self, value, row, head, column, gate):
        at = (
            self.offset
            + row * self.row_stride
            + head * self.head_stride
            + (column if self.elements else 0)
        )
        weight = cl.cast(
            cl.cast(
                cl.sigmoid(cl.cast(cl.load(gate, at, True, other=0.0), cl.f32)),
                cl.bf16,
            ),
            cl.f32,
        )
        return cl.cast(cl.cast(value, cl.bf16), cl.f32) * weight


def gate(gates, *, heads, kv_heads, head_dim, head_gates=False, **_):
    """Options of Carbon `paged_attention` that gate its output by the sigmoid of the
    gates in fused rows `gates`: per element after each query head, or with
    `head_gates` one per head after the values."""
    shape = layout(heads, kv_heads, head_dim, not head_gates, head_gates)
    hook = (
        SigmoidGate(shape["gate"], shape["row"], 1, False)
        if head_gates
        else SigmoidGate(head_dim, shape["row"], 2 * head_dim, True)
    )
    return {"output_mod": OutputMod(hook, (cl.bf16,)), "output_operands": (gates,)}


def rope_tables(capacity, rotary, theta, angles=None):
    """cos and sin of position · theta^(-2i / rotary), (capacity, rotary / 2) f32
    computed in f64."""
    frequencies = theta ** (-np.arange(0, rotary, 2, dtype=np.float64) / rotary)
    if angles is not None:
        frequencies[angles:] = 0.0
    turns = np.arange(capacity, dtype=np.float64)[:, None] * frequencies[None, :]
    return np.cos(turns).astype(np.float32), np.sin(turns).astype(np.float32)


@cache
def make_rope_cache(
    dtype,
    *,
    head_dim,
    rotary,
    value_norm,
    positioned,
    normed,
    pairwise,
):
    D, half = head_dim, rotary // 2
    schedule = Layout.striped((D,), threads=32)

    def rotated(rows_buffer, weight, cos, sin, base, position, index, EPS, scratch):
        if not rotary:
            x = cl.cast(cl.load(rows_buffer, base + index, True, other=0.0), cl.f32)
            if normed:
                inverse = cl.rsqrt(
                    cl.item(cl.sum(x * x, 0, scratch=scratch), scratch=scratch) / D
                    + EPS,
                )
                x = (
                    x
                    * inverse
                    * cl.cast(cl.load(weight, index, True, other=0.0), cl.f32)
                )
            return x
        turning = index < rotary
        if pairwise:
            partner = cl.select(turning, index + 1 - 2 * (index % 2), index)
        else:
            partner = cl.select(turning, (index + half) % rotary, index)
        x = cl.cast(cl.load(rows_buffer, base + index, True, other=0.0), cl.f32)
        y = cl.cast(cl.load(rows_buffer, base + partner, True, other=0.0), cl.f32)
        if normed:
            inverse = cl.rsqrt(
                cl.item(cl.sum(x * x, 0, scratch=scratch), scratch=scratch) / D + EPS,
            )
            x = x * inverse * cl.cast(cl.load(weight, index, True, other=0.0), cl.f32)
            y = y * inverse * cl.cast(cl.load(weight, partner, True, other=0.0), cl.f32)
        angle = position * half + (index // 2 if pairwise else index % half)
        c = cl.load(cos, angle, turning, other=1.0)
        s = cl.load(sin, angle, turning, other=0.0)
        return cl.select(
            (index % 2 == 0) if pairwise else (index < half),
            x * c - y * s,
            x * c + y * s,
        )

    @cl.kernel
    def rope_cache(
        qkv: cl.buffer(dtype),
        q_weight: cl.buffer(dtype) if normed else (),
        k_weight: cl.buffer(dtype) if normed else (),
        cos: cl.buffer(cl.f32),
        sin: cl.buffer(cl.f32),
        table: cl.buffer(cl.u32),
        lengths: cl.buffer(cl.u32),
        queries: cl.buffer(dtype),
        keys: cl.buffer(dtype),
        values: cl.buffer(dtype),
        turns: (cl.buffer(cl.u32),) if positioned else (),
        rows: cl.u32,
        HQ: cl.constexpr,
        HK: cl.constexpr,
        ROW: cl.constexpr,
        QS: cl.constexpr,
        KEYS_AT: cl.constexpr,
        VALUES_AT: cl.constexpr,
        PAGE: cl.constexpr,
        MAXP: cl.constexpr,
        EPS: cl.constexpr,
    ):
        index = cl.arange(D, layout=schedule)
        scratch = cl.scratch((D,), cl.f32)
        if cl.program_id() < rows * HQ:
            head, row = cl.program_id() % HQ, cl.program_id() // HQ
            position = (
                cl.load(lengths, row, True, other=0)
                if MAXP
                else cl.load(lengths, 0, True, other=0) + row
            )
            turn = cl.load(turns[0], row, True, other=0) if positioned else position
            value = rotated(
                qkv,
                q_weight,
                cos,
                sin,
                row * ROW + head * QS,
                turn,
                index,
                EPS,
                scratch,
            )
            cl.store(queries, cl.program_id() * D + index, cl.cast(value, dtype), True)
        else:
            head, row = (
                (cl.program_id() - rows * HQ) % HK,
                (cl.program_id() - rows * HQ) // HK,
            )
            position = (
                cl.load(lengths, row, True, other=0)
                if MAXP
                else cl.load(lengths, 0, True, other=0) + row
            )
            page = cl.load(
                table,
                row * MAXP + position // PAGE if MAXP else position // PAGE,
                True,
                other=0,
            )
            target = ((page * HK + head) * PAGE + position % PAGE) * D + index
            turn = cl.load(turns[0], row, True, other=0) if positioned else position
            key = rotated(
                qkv,
                k_weight,
                cos,
                sin,
                row * ROW + KEYS_AT + head * D,
                turn,
                index,
                EPS,
                scratch,
            )
            cl.store(keys, target, cl.cast(key, dtype), True)
            value = cl.cast(
                cl.load(qkv, row * ROW + VALUES_AT + head * D + index, True, other=0.0),
                cl.f32,
            )
            if value_norm:
                second = cl.scratch((D,), cl.f32)
                value = value * cl.rsqrt(
                    cl.item(cl.sum(value * value, 0, scratch=second), scratch=second)
                    / D
                    + EPS,
                )
            cl.store(values, target, cl.cast(value, dtype), True)

    return rope_cache


def rope_cache(
    device,
    qkv,
    q_weight,
    k_weight,
    cos,
    sin,
    table,
    lengths,
    queries,
    keys,
    values,
    *,
    rows,
    heads,
    kv_heads,
    head_dim,
    page_size,
    eps,
    rotary=None,
    gated=False,
    head_gates=False,
    k_eq_v=False,
    queries_only=False,
    row_stride=None,
    keys_at=None,
    max_pages=0,
    v_norm=False,
    positions=None,
    pairwise=False,
    dtype=cl.bf16,
):
    """Dispatch: normed, rotated queries out, and keys and values into the pages.
    `row_stride` and `keys_at` place a row that is not one projection's."""
    shape = layout(heads, kv_heads, head_dim, gated, head_gates, k_eq_v, queries_only)
    row_stride = shape["row"] if row_stride is None else row_stride
    query_stride = shape["stride"]
    keys_at = shape["keys"] if keys_at is None else keys_at
    values_at = keys_at + shape["values"] - shape["keys"]
    write_cache = not queries_only
    rotary = head_dim if rotary is None else rotary
    if head_dim % 32 or rotary % 2 or not 0 <= rotary <= head_dim:
        raise ValueError(
            "rope_cache needs head_dim a multiple of 32 and an even rotary count up to "
            "head_dim",
        )
    if (q_weight is None) != (k_weight is None):
        raise ValueError("the query and key norm weights come together")
    normed = q_weight is not None
    grid = rows * (heads + kv_heads if write_cache else heads)
    artifact = make_rope_cache(
        dtype,
        head_dim=head_dim,
        rotary=rotary,
        value_norm=v_norm,
        positioned=positions is not None,
        normed=normed,
        pairwise=pairwise,
    ).compile(
        constants={
            "HQ": heads,
            "HK": kv_heads,
            "ROW": row_stride,
            "QS": query_stride,
            "KEYS_AT": keys_at,
            "VALUES_AT": values_at,
            "PAGE": page_size,
            "MAXP": max_pages,
            "EPS": eps,
        },
        threads=32,
        device=device,
        max_grid=(2**32 - 1) // (row_stride + heads * head_dim) * (heads + kv_heads),
    )
    turns = [] if positions is None else [positions]
    weights = [q_weight, k_weight] if normed else []
    return artifact.dispatch(
        device,
        [qkv, *weights, cos, sin, table, lengths, queries, keys, values, *turns],
        grid=(grid, 1, 1),
        scalars={"rows": rows},
    )
