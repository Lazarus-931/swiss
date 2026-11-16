"""EAGLE-3 drafters, in the SpecForge and speculators checkpoint layouts."""

from dataclasses import dataclass
from functools import cache

import numpy as np
from carbon import language as cl
from carbon.compiler import Layout
from carbon.kernels import attention
from carbon.kernels.ops import indexing, sampling

from ..core.model import PAGE_SIZE, Unsupported, take
from ..core.program import Program, allocate
from ..core.speculative import Attached, Chain, Pages
from ..core.weights import Uploads
from ..kernels import linear, normalization, rope


@dataclass(frozen=True)
class Eagle3Spec:
    width: int
    target_width: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    rotary: int
    theta: float
    eps: float
    draft_vocab: int
    taps: tuple | None
    norm_before_residual: bool
    norm_before_fc: bool


def parse(config):
    text = config.get("transformer_layer_config", config)
    if take(text, "num_hidden_layers", int) != 1:
        raise Unsupported("only one-layer EAGLE-3 drafters are implemented")
    width, heads = (
        take(text, "hidden_size", int),
        take(text, "num_attention_heads", int),
    )
    head_dim = text.get("head_dim") or width // heads
    rope = text.get("rope_parameters") or {}
    factor = rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 1.0))
    eagle = config.get("eagle_config") or {}
    taps = config.get(
        "eagle_aux_hidden_state_layer_ids",
        eagle.get("eagle_aux_hidden_state_layer_ids"),
    )
    return Eagle3Spec(
        width=width,
        target_width=config.get("target_hidden_size") or width,
        hidden=take(text, "intermediate_size", int),
        heads=heads,
        kv_heads=take(text, "num_key_value_heads", int),
        head_dim=head_dim,
        rotary=int(head_dim * factor),
        theta=float(rope.get("rope_theta", text.get("rope_theta", 10000.0))),
        eps=float(text.get("rms_norm_eps", 1e-6)),
        draft_vocab=int(
            config.get("draft_vocab_size") or take(text, "vocab_size", int),
        ),
        taps=None if taps is None else tuple(taps),
        norm_before_residual=bool(config.get("norm_before_residual", False)),
        norm_before_fc=bool(config.get("norm_before_fc", False)),
    )


@cache
def _remap_kernel():
    @cl.kernel
    def remap(
        token: cl.buffer(cl.u32),
        table: cl.buffer(cl.u32),
        SCHEDULE: cl.constexpr,
    ):
        lane = cl.arange(32, layout=SCHEDULE)
        first = lane == 0
        draft = cl.load(token, lane, first, other=0)
        cl.store(token, lane, cl.load(table, draft, first, other=0), first)

    return remap


def remap(device, token, table):
    """Dispatch: token[0] = table[token[0]], a draft-vocabulary ID as a target ID."""
    artifact = _remap_kernel().compile(
        constants={"SCHEDULE": Layout.blocked((32,), threads=32)},
        threads=32,
        device=device,
    )
    return artifact.dispatch(device, [token, table], grid=(1, 1, 1))


class Eagle3(Attached, Uploads):
    """One decoder layer over [norm(token embedding), norm(hidden)], fed the target's
    taps for every stored token and its own output while it drafts.

    Its KV entry i holds token i + 1 of the sequence with the target's taps of token i,
    so it lags the target by one token and its rotations run one position ahead.
    """

    lag = 1

    def __init__(self, spec, weights, target, bits=None):
        s = spec
        self.spec, self.target, self.device = spec, target, target.device
        self.weights, self.quantization, self.prefix = weights, None, ""
        self.requantize = (bits, 64) if bits else None
        layer = (
            "midlayer." if "midlayer.input_layernorm.weight" in weights else "layers.0."
        )
        a, m = layer + "self_attn.", layer + "mlp."
        self.block = {
            "input_norm": self.vector(layer + "input_layernorm.weight"),
            "hidden_norm": self.vector(layer + "hidden_norm.weight"),
            "post_norm": self.vector(layer + "post_attention_layernorm.weight"),
            "qkv": self.matrix(a + "q_proj", a + "k_proj", a + "v_proj"),
            "o_proj": self.matrix(a + "o_proj"),
            "gate_up": self.matrix(m + "gate_proj", m + "up_proj", paired=True),
            "down": self.matrix(m + "down_proj"),
        }
        self.fc = self.matrix("fc")
        self.fc_norm = self.vector("input_norm.weight") if s.norm_before_fc else None
        self.final_norm = self.vector("norm.weight")
        self.head = self.matrix("lm_head")
        self.embed = (
            self.matrix("embed_tokens")
            if "embed_tokens.weight" in weights
            else target.embed
        )
        offsets = weights.get("d2t")
        table = np.arange(s.draft_vocab, dtype=np.int64)
        if offsets is not None:
            table = table + np.asarray(offsets, np.int64)
        self.d2t = linear.raw(self.device, table.astype(np.uint32))
        del self.weights
        layers = target.spec.layers
        self.taps = s.taps or (1, layers // 2 - 1, layers - 4)
        if len(self.taps) * target.spec.width != self.fc.inputs:
            raise Unsupported("the drafter was made for another target width")
        self.scratch = linear.scratch(self.device, self.block.values())
        self.kv = Pages(
            self.device,
            target.sequences,
            target.capacity + 16,
            1,
            s.kv_heads,
            s.head_dim,
        )
        self.rope = [
            linear.raw(self.device, table[1:])
            for table in rope.rope_tables(self.kv.capacity + 1, s.rotary, s.theta)
        ]
        self.greedy = sampling.settings([0], [0.0]).tobytes()
        self.chain = Chain(
            self.device,
            self.sizes(1),
            lambda b: self.steps(1, b),
            limit=self.kv.capacity,
            lengths=self.kv.lengths,
        )
        self.chain.buffers["settings"].write(self.greedy)
        self.commits, self.seeds = {}, {}
        target.run.follow(self, self.taps, target.spec.width)

    @classmethod
    def attach(cls, spec, checkpoint, target, bits=None):
        """The drafter of `checkpoint` for `target`; `bits` quantizes its matrices."""
        return cls(spec, dict(checkpoint.tensors()), target, bits)

    def sizes(self, rows):
        s = self.spec
        return {
            "ids": rows * 4,
            "embedded": rows * s.width * 2,
            "embedded_normed": rows * s.width * 2,
            "hidden_normed": rows * s.width * 2,
            "joined": rows * 2 * s.width * 2,
            "x": rows * s.width * 2,
            "qkv": rows * (s.heads + 2 * s.kv_heads) * s.head_dim * 2,
            "q": rows * s.heads * s.head_dim * 2,
            "attended": rows * s.heads * s.head_dim * 2,
            "update": rows * s.width * 2,
            "normed": rows * s.width * 2,
            "hidden": rows * s.hidden * 2,
            "last": s.width * 2,
            "logits": s.draft_vocab * 2,
            "token": 4,
            "settings": 6 * 4,
        }

    def steps(self, rows, b):
        """The layer over `rows` tokens with their hidden states in b["x"], then the
        next token after the last row in b["token"]."""
        s, device, w, kv = self.spec, self.device, self.block, self.kv
        keys, values = kv.layers[0]
        norm = dict(rows=rows, columns=s.width, epsilon=s.eps)
        shape = dict(rows=rows, heads=s.heads, kv_heads=s.kv_heads, head_dim=s.head_dim)
        steps = [linear.embedding(device, b["ids"], self.embed, b["embedded"], rows)]
        steps += normalization.rms_norm_rows(
            device,
            b["embedded"],
            w["input_norm"],
            b["embedded_normed"],
            **norm,
        )
        steps += normalization.rms_norm_rows(
            device,
            b["x"],
            w["hidden_norm"],
            b["hidden_normed"],
            **norm,
        )
        steps += [
            linear.place(
                device,
                b["embedded_normed"],
                b["joined"],
                rows,
                s.width,
                2 * s.width,
                0,
            ),
            linear.place(
                device,
                b["hidden_normed"],
                b["joined"],
                rows,
                s.width,
                2 * s.width,
                s.width,
            ),
        ]
        if s.norm_before_residual:
            steps += [
                linear.place(
                    device,
                    b["hidden_normed"],
                    b["x"],
                    rows,
                    s.width,
                    s.width,
                    0,
                ),
            ]
        steps += linear.product(
            device,
            b["joined"],
            w["qkv"],
            b["qkv"],
            rows,
            self.scratch,
        )
        steps += [
            rope.rope_cache(
                device,
                b["qkv"],
                None,
                None,
                *self.rope,
                kv.table,
                kv.lengths,
                b["q"],
                keys,
                values,
                page_size=PAGE_SIZE,
                eps=s.eps,
                rotary=s.rotary,
                **shape,
            ),
        ]
        steps += attention.paged_attention(
            device,
            b["q"],
            keys,
            values,
            kv.table,
            kv.lengths,
            b["attended"],
            page_size=PAGE_SIZE,
            context=kv.capacity,
            **shape,
        )
        steps += linear.product(
            device,
            b["attended"],
            w["o_proj"],
            b["update"],
            rows,
            self.scratch,
        )
        steps += normalization.residual_norm(
            device,
            b["x"],
            b["update"],
            weight=w["post_norm"],
            output=b["normed"],
            **norm,
        )
        steps += linear.product(
            device,
            b["normed"],
            w["gate_up"],
            b["hidden"],
            rows,
            self.scratch,
            "silu",
        )
        steps += linear.product(
            device,
            b["hidden"],
            w["down"],
            b["update"],
            rows,
            self.scratch,
        )
        steps += normalization.residual_norm(
            device,
            b["x"],
            b["update"],
            weight=self.final_norm,
            output=b["last"],
            first=rows - 1,
            **norm,
        )
        steps += linear.product(
            device,
            b["last"],
            self.head,
            b["logits"],
            1,
            self.scratch,
        )
        return steps + [
            sampling.sample(
                device,
                b["logits"],
                b["settings"],
                kv.lengths,
                b["token"],
                rows=1,
                vocab=s.draft_vocab,
            ),
            remap(device, b["token"], self.d2t),
        ]

    def commit_program(self, taps, start, count):
        key = id(taps), start, count
        if key not in self.commits:
            s, device = self.spec, self.device
            columns = len(self.taps) * s.target_width
            b = allocate(
                device,
                {
                    **self.sizes(count),
                    "features": count * columns * 2,
                    "rows": count * 4,
                },
            )
            b["rows"].write(np.arange(start, start + count, dtype=np.uint32).tobytes())
            b["settings"].write(self.greedy)
            steps = indexing.gather_rows(
                device,
                taps,
                b["rows"],
                b["features"],
                rows=count,
                columns=columns,
            )
            features = b["features"]
            if self.fc_norm is not None:
                b["features_normed"] = device.buffer(count * columns * 2)
                steps += normalization.rms_norm_rows(
                    device,
                    features,
                    self.fc_norm,
                    b["features_normed"],
                    rows=count,
                    columns=columns,
                    epsilon=s.eps,
                )
                features = b["features_normed"]
            steps += linear.product(
                device,
                features,
                self.fc,
                b["x"],
                count,
                self.scratch,
            )
            program = Program(device)
            for step in steps + self.steps(count, b):
                program.kernel(step)
            program.finish()
            self.commits[key] = program, b
        return self.commits[key]

    def commit(self, sequence, buffers, start, count, following):
        """Store the pairs (token after each committed row, that row's taps), and keep
        the draft after `following` with the hidden state it came from."""
        run = self.target.run
        stored = run.tokens[sequence][-count:]
        ids = np.array([*stored[1:], following], np.uint32)
        program, b = self.commit_program(buffers["taps"], start, count)
        b["ids"].write(ids.tobytes() + bytes(b["ids"].size - ids.nbytes))
        self.kv.point(sequence, run.length(sequence) - count)
        program.run()
        size = self.spec.width * 2
        token = int(np.frombuffer(b["token"].read(), np.uint32)[0])
        self.seeds[sequence] = token, b["x"].read()[(count - 1) * size : count * size]

    def propose(self, sequence, token, count):
        """`count` greedy drafts after `token`, the next input of `sequence`: the one
        its last commit made, then a chain from that commit's hidden state."""
        if not count or sequence not in self.seeds:
            return []
        first, hidden = self.seeds[sequence]
        if count == 1:
            return [first]
        start = self.target.run.length(sequence)
        self.chain.buffers["x"].write(hidden)
        self.kv.point(sequence, start)
        return [first, *self.chain.run(first, count - 1, start)]
