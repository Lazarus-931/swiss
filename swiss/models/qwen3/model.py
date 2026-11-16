"""Qwen3 dense models."""

from dataclasses import dataclass

from carbon.kernels import attention

from ...core.model import PAGE_SIZE, Model, Unsupported, reject, take
from ...core.program import open_device
from ...core.sequences import Sequences
from ...core.weights import Uploads, quantization
from ...kernels import linear, normalization, rope


@dataclass(frozen=True)
class Qwen3Spec:
    vocab: int
    width: int
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    eps: float
    theta: float
    tie_embeddings: bool


def parse(config):
    reject(config, ("rope_scaling", "attention_bias", "use_sliding_window"))
    quantization(config)
    rope = config.get("rope_parameters") or {}
    spec = Qwen3Spec(
        vocab=take(config, "vocab_size", int),
        width=take(config, "hidden_size", int),
        layers=take(config, "num_hidden_layers", int),
        hidden=take(config, "intermediate_size", int),
        heads=take(config, "num_attention_heads", int),
        kv_heads=take(config, "num_key_value_heads", int),
        head_dim=take(config, "head_dim", int),
        eps=float(take(config, "rms_norm_eps", (int, float))),
        theta=float(config.get("rope_theta") or take(rope, "rope_theta", (int, float))),
        tie_embeddings=take(config, "tie_word_embeddings", bool, False),
    )
    if spec.heads % spec.kv_heads:
        raise Unsupported("Qwen3 needs heads divisible by KV heads")
    return spec


class Qwen3(Uploads, Model):
    def __init__(self, spec, runtime, weights, quantization, device=None):
        super().__init__(spec, runtime)
        s = spec
        self.device, self.weights, self.quantization = (
            device or open_device(),
            weights,
            quantization,
        )
        self.prefix = (
            "language_model.model."
            if "language_model.model.embed_tokens.weight" in weights
            else "model."
        )
        self.embed = self.matrix("embed_tokens")
        self.head = (
            self.embed
            if s.tie_embeddings
            else self.matrix(self.prefix[:-6] + "lm_head", prefix="")
        )
        self.final_norm = self.vector("norm.weight")
        self.blocks = [self.block(layer) for layer in range(s.layers)]
        del self.weights
        self.run = Sequences(
            self.device,
            runtime,
            vocab=s.vocab,
            sizes=self.sizes,
            steps=self.steps,
            rewindable=True,
        )
        size = self.run.pages * s.kv_heads * PAGE_SIZE * s.head_dim * 2
        self.pages = [
            (self.device.buffer(size), self.device.buffer(size))
            for _ in range(s.layers)
        ]
        self.rope = [
            linear.raw(self.device, t)
            for t in rope.rope_tables(self.capacity, s.head_dim, s.theta)
        ]

    def block(self, layer):
        p = f"layers.{layer}."
        a = p + "self_attn."
        return {
            "input_norm": self.vector(p + "input_layernorm.weight"),
            "post_norm": self.vector(p + "post_attention_layernorm.weight"),
            "q_norm": self.vector(a + "q_norm.weight"),
            "k_norm": self.vector(a + "k_norm.weight"),
            "qkv": self.matrix(a + "q_proj", a + "k_proj", a + "v_proj"),
            "o_proj": self.matrix(a + "o_proj"),
            "gate_up": self.matrix(p + "mlp.gate_proj", p + "mlp.up_proj", paired=True),
            "down": self.matrix(p + "mlp.down_proj"),
        }

    def sizes(self, rows, batch):
        s = self.spec
        last = rows if batch else 1
        return {
            "ids": rows * 4,
            "x": rows * s.width * 2,
            "normed": rows * s.width * 2,
            "qkv": rows * (s.heads + 2 * s.kv_heads) * s.head_dim * 2,
            "q": rows * s.heads * s.head_dim * 2,
            "attended": rows * s.heads * s.head_dim * 2,
            "update": rows * s.width * 2,
            "hidden": rows * s.hidden * 2,
            "last": last * s.width * 2,
            "logits": last * s.vocab * 2,
            "token": last * 4,
        }

    def steps(self, rows, b):
        s, device, run = self.spec, self.device, self.run
        shape = dict(rows=rows, heads=s.heads, kv_heads=s.kv_heads, head_dim=s.head_dim)
        residual = dict(rows=rows, columns=s.width, epsilon=s.eps)
        steps = [linear.embedding(device, b["ids"], self.embed, b["x"], rows)]
        steps += normalization.rms_norm_rows(
            device,
            b["x"],
            self.blocks[0]["input_norm"],
            b["normed"],
            **residual,
        )
        for layer, w in enumerate(self.blocks):
            keys, values = self.pages[layer]
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
                    run.table,
                    run.lengths,
                    b["q"],
                    keys,
                    values,
                    page_size=PAGE_SIZE,
                    eps=s.eps,
                    max_pages=run.max_pages(b),
                    **shape,
                ),
            ]
            steps += attention.paged_attention(
                device,
                b["q"],
                keys,
                values,
                run.table,
                run.lengths,
                b["attended"],
                page_size=PAGE_SIZE,
                context=self.capacity,
                max_pages=run.max_pages(b),
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
                **residual,
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
            if layer + 1 < s.layers:
                steps += normalization.residual_norm(
                    device,
                    b["x"],
                    b["update"],
                    weight=self.blocks[layer + 1]["input_norm"],
                    output=b["normed"],
                    **residual,
                )
                steps += self.run.tap(layer, b, rows)
        last = self.run.last(b, rows)
        steps += normalization.residual_norm(
            device,
            b["x"],
            b["update"],
            weight=self.final_norm,
            output=b["last"],
            first=rows - last,
            **residual,
        )
        steps += self.run.tap(s.layers - 1, b, rows)
        steps += linear.product(device, b["last"], self.head, b["logits"], last)
        return steps + [run.choose(b, last)]
