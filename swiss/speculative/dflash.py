"""DFlash block drafters."""

from dataclasses import dataclass

import numpy as np
from carbon.kernels import attention
from carbon.kernels.ops import indexing, sampling

from ..core.model import PAGE_SIZE, Unsupported, reject, take
from ..core.program import Program, allocate
from ..core.speculative import Attached, Pages
from ..core.weights import Uploads
from ..kernels import linear, normalization, rope


@dataclass(frozen=True)
class DFlashSpec:
    width: int
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    theta: float
    eps: float
    block: int
    mask: int
    taps: tuple
    window: int | None
    causal: tuple


def parse(config):
    reject(config, ("attention_bias", "final_logit_softcapping", "rope_scaling"))
    flash = config.get("dflash_config") or {}
    if flash.get("causal") or flash.get("attention_sink_bias"):
        raise Unsupported(
            "dflash_config causal and sinks are not implemented",
        )
    if config.get("hidden_act", "silu") != "silu":
        raise Unsupported("only SiLU DFlash drafters are implemented")
    layers = take(config, "num_hidden_layers", int)
    kinds = config.get("layer_types") or ["full_attention"] * layers
    rope = config.get("rope_parameters") or {}
    causal = config.get("is_causal")
    return DFlashSpec(
        width=take(config, "hidden_size", int),
        layers=layers,
        hidden=take(config, "intermediate_size", int),
        heads=take(config, "num_attention_heads", int),
        kv_heads=take(config, "num_key_value_heads", int),
        head_dim=take(config, "head_dim", int),
        theta=float(rope.get("rope_theta", config.get("rope_theta", 10000.0))),
        eps=float(config.get("rms_norm_eps", 1e-6)),
        block=int(flash.get("block_size", config.get("block_size", 16))),
        mask=int(flash.get("mask_token_id", config.get("mask_token_id"))),
        taps=tuple(flash.get("target_layer_ids", config.get("target_layer_ids"))),
        window=(
            take(config, "sliding_window", int)
            if "sliding_attention" in kinds
            else None
        ),
        causal=tuple(
            kind == "sliding_attention" if causal is None else bool(causal)
            for kind in kinds
        ),
    )


class DFlash(Attached, Uploads):
    """Layers whose keys and values hold, for every stored token, a projection of the
    target's taps; a draft is one pass over [next token, mask, ..., mask] that sees all
    of them and the block, with the target's embedding and head. As in z-lab/dflash,
    sliding layers see the block causally unless the config sets `is_causal`.

    Sliding layers keep `sliding_window` - 1 stored tokens, so the target's context
    must fit in that.
    """

    def __init__(self, spec, weights, target, bits=None):
        s = spec
        if s.window is not None and target.capacity > s.window:
            raise Unsupported(
                f"a context of {target.capacity} exceeds the drafter's sliding window",
            )
        if len(s.taps) * target.spec.width != weights["fc.weight"].shape[1]:
            raise Unsupported("the drafter was made for another target width")
        self.spec, self.target, self.device = spec, target, target.device
        self.weights, self.quantization = weights, None
        self.prefix = "model." if "model.fc.weight" in weights else ""
        self.requantize = (bits, 64) if bits else None
        self.blocks = [self.block(layer) for layer in range(s.layers)]
        self.fc = self.matrix("fc")
        self.hidden_norm = self.vector("hidden_norm.weight")
        self.final_norm = self.vector("norm.weight")
        self.context = self.matrix(
            *(
                f"layers.{layer}.self_attn.{name}"
                for layer in range(s.layers)
                for name in ("k_proj", "v_proj")
            ),
        )
        del self.weights
        self.scratch = linear.scratch(
            self.device,
            [self.fc, self.context, *(w["qkv"] for w in self.blocks)],
        )
        self.kv = Pages(
            self.device,
            target.sequences,
            target.capacity + s.block,
            s.layers,
            s.kv_heads,
            s.head_dim,
        )
        self.rope = [
            linear.raw(self.device, table)
            for table in rope.rope_tables(self.kv.capacity, s.head_dim, s.theta)
        ]
        self.programs = {}
        target.run.follow(self, s.taps, target.spec.width)

    @classmethod
    def attach(cls, spec, checkpoint, target, bits=None):
        """The drafter of `checkpoint` for `target`; `bits` quantizes its matrices."""
        return cls(spec, dict(checkpoint.tensors()), target, bits)

    def block(self, layer):
        p, a, m = (
            f"layers.{layer}.",
            f"layers.{layer}.self_attn.",
            f"layers.{layer}.mlp.",
        )
        return {
            "input_norm": self.vector(p + "input_layernorm.weight"),
            "post_norm": self.vector(p + "post_attention_layernorm.weight"),
            "q_norm": self.vector(a + "q_norm.weight"),
            "k_norm": self.vector(a + "k_norm.weight"),
            "qkv": self.matrix(a + "q_proj", a + "k_proj", a + "v_proj"),
            "o_proj": self.matrix(a + "o_proj"),
            "gate_up": self.matrix(m + "gate_proj", m + "up_proj", paired=True),
            "down": self.matrix(m + "down_proj"),
        }

    @property
    def lane(self):
        """The columns of one layer's [keys | values] in a context row."""
        return 2 * self.spec.kv_heads * self.spec.head_dim

    def commit_program(self, taps, start, count):
        key = "commit", id(taps), start, count
        if key not in self.programs:
            s, device, kv = self.spec, self.device, self.kv
            columns = len(s.taps) * self.target.spec.width
            b = allocate(
                device,
                {
                    "rows": count * 4,
                    "features": count * columns * 2,
                    "projected": count * s.width * 2,
                    "normed": count * s.width * 2,
                    "context": count * s.layers * self.lane * 2,
                    "q": count * s.head_dim * 2,
                },
            )
            b["rows"].write(np.arange(start, start + count, dtype=np.uint32).tobytes())
            steps = indexing.gather_rows(
                device,
                taps,
                b["rows"],
                b["features"],
                rows=count,
                columns=columns,
            )
            steps += linear.product(
                device,
                b["features"],
                self.fc,
                b["projected"],
                count,
                self.scratch,
            )
            steps += normalization.rms_norm_rows(
                device,
                b["projected"],
                self.hidden_norm,
                b["normed"],
                rows=count,
                columns=s.width,
                epsilon=s.eps,
            )
            steps += linear.product(
                device,
                b["normed"],
                self.context,
                b["context"],
                count,
                self.scratch,
            )
            for layer, w in enumerate(self.blocks):
                keys, values = kv.layers[layer]
                steps += [
                    rope.rope_cache(
                        device,
                        b["context"],
                        w["k_norm"],
                        w["k_norm"],
                        *self.rope,
                        kv.table,
                        kv.lengths,
                        b["q"],
                        keys,
                        values,
                        rows=count,
                        heads=1,
                        kv_heads=s.kv_heads,
                        head_dim=s.head_dim,
                        page_size=PAGE_SIZE,
                        eps=s.eps,
                        row_stride=s.layers * self.lane,
                        keys_at=layer * self.lane,
                    ),
                ]
            program = Program(device)
            for step in steps:
                program.kernel(step)
            program.finish()
            self.programs[key] = program, b
        return self.programs[key]

    def commit(self, sequence, buffers, start, count, following):
        """Store the context keys and values of the committed rows' taps."""
        program, _ = self.commit_program(buffers["taps"], start, count)
        self.kv.point(sequence, self.target.run.length(sequence) - count)
        program.run()

    def draft_program(self, rows):
        key = "draft", rows
        if key not in self.programs:
            s, device, kv, target = self.spec, self.device, self.kv, self.target
            b = allocate(
                device,
                {
                    "ids": rows * 4,
                    "x": rows * s.width * 2,
                    "normed": rows * s.width * 2,
                    "qkv": rows * (s.heads + 2 * s.kv_heads) * s.head_dim * 2,
                    "q": rows * s.heads * s.head_dim * 2,
                    "attended": rows * s.heads * s.head_dim * 2,
                    "update": rows * s.width * 2,
                    "hidden": rows * s.hidden * 2,
                    "last": rows * s.width * 2,
                    "logits": rows * target.spec.vocab * 2,
                    "token": rows * 4,
                    "settings": rows * 6 * 4,
                    "positions": rows * 4,
                },
            )
            b["settings"].write(np.tile(sampling.settings([0], [0.0]), rows).tobytes())
            norm = dict(rows=rows, columns=s.width, epsilon=s.eps)
            shape = dict(
                rows=rows,
                heads=s.heads,
                kv_heads=s.kv_heads,
                head_dim=s.head_dim,
            )
            steps = [
                linear.embedding(
                    device,
                    b["ids"],
                    target.embed,
                    b["x"],
                    rows,
                    scale=getattr(target, "embed_scale", 1.0),
                ),
            ]
            steps += normalization.rms_norm_rows(
                device,
                b["x"],
                self.blocks[0]["input_norm"],
                b["normed"],
                **norm,
            )
            for layer, w in enumerate(self.blocks):
                keys, values = kv.layers[layer]
                steps += linear.product(
                    device,
                    b["normed"],
                    w["qkv"],
                    b["qkv"],
                    rows,
                    self.scratch,
                )
                steps += [
                    rope.rope_cache(
                        device,
                        b["qkv"],
                        w["q_norm"],
                        w["k_norm"],
                        *self.rope,
                        kv.table,
                        kv.lengths,
                        b["q"],
                        keys,
                        values,
                        page_size=PAGE_SIZE,
                        eps=s.eps,
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
                    causal=s.causal[layer],
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
                last = layer + 1 == s.layers
                steps += normalization.residual_norm(
                    device,
                    b["x"],
                    b["update"],
                    weight=self.final_norm
                    if last
                    else self.blocks[layer + 1]["input_norm"],
                    output=b["last"] if last else b["normed"],
                    **norm,
                )
            steps += linear.product(
                device,
                b["last"],
                target.head,
                b["logits"],
                rows,
                self.scratch,
            )
            steps += [
                sampling.sample(
                    device,
                    b["logits"],
                    b["settings"],
                    b["positions"],
                    b["token"],
                    rows=rows,
                    vocab=target.spec.vocab,
                ),
            ]
            program = Program(device)
            for step in steps:
                program.kernel(step)
            program.finish()
            self.programs[key] = program, b
        return self.programs[key]

    def propose(self, sequence, token, count):
        """`count` greedy drafts after `token`, the next input of `sequence`, from one
        block of `count` + 1 rows."""
        count = min(count, self.spec.block - 1)
        if not count:
            return []
        program, b = self.draft_program(count + 1)
        ids = np.array([token, *[self.spec.mask] * count], np.uint32)
        b["ids"].write(ids.tobytes() + bytes(b["ids"].size - ids.nbytes))
        self.kv.point(sequence, self.target.run.length(sequence))
        program.run()
        return np.frombuffer(b["token"].read(), np.uint32)[1 : count + 1].tolist()
