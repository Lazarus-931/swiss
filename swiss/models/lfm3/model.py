"""LFM3 text models."""

from dataclasses import dataclass

from carbon import language as cl
from carbon.kernels import attention
from carbon.kernels.gemm.epilogue import Epilogue

from ...core.model import PAGE_SIZE, Model, Unsupported, reject, take
from ...core.program import open_device
from ...core.sequences import Sequences
from ...core.weights import Uploads, quantization
from ...kernels import conv1d, linear, normalization, rope

KINDS = ("conv", "full_attention", "sliding_attention")


@dataclass(frozen=True)
class Lfm3Spec:
    vocab: int
    width: int
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    eps: float
    theta: float
    kinds: tuple
    taps: int
    window: int
    sliding_theta: float
    angles: int


@dataclass(frozen=True)
class GatedInputs:
    """LFM's convolution input round(B · x), from projection rows [B | C | x], each
    `width` wide."""

    width: int

    def __call__(self, row, channel, live, proj):
        b = cl.cast(
            cl.load(proj, row * 3 * self.width + channel, live, other=0.0),
            cl.f32,
        )
        x = cl.cast(
            cl.load(
                proj,
                row * 3 * self.width + 2 * self.width + channel,
                live,
                other=0.0,
            ),
            cl.f32,
        )
        return cl.cast(cl.cast(b * x, cl.bf16), cl.f32)


@dataclass(frozen=True)
class GatedOutputs:
    """LFM's convolution output C · round(value), with C the middle part of the
    projection rows."""

    width: int

    def __call__(self, value, row, channel, proj):
        mixed = cl.cast(cl.cast(value, cl.bf16), cl.f32)
        return (
            cl.cast(
                cl.load(
                    proj,
                    row * 3 * self.width + self.width + channel,
                    True,
                    other=0.0,
                ),
                cl.f32,
            )
            * mixed
        )


def rotation(parameters, kind):
    settings = take(parameters, kind, dict)
    if settings.get("rope_type", "default") != "default":
        raise Unsupported("only default RoPE is implemented")
    return settings


def parse(config):
    """Lfm3Spec of any LFM3 config."""
    text = config.get("text_config") or config.get("lfm") or config
    reject(text, ("conv_bias",))
    if not text.get("use_sandwich_norm"):
        raise Unsupported("LFM3 without sandwich norms is not implemented")
    quantization(config)
    layers = take(text, "num_hidden_layers", int)
    kinds = tuple(take(text, "layer_types", list))
    if len(kinds) != layers or set(kinds) - set(KINDS):
        raise Unsupported(
            f"layer_types must list {layers} layers of {', '.join(KINDS)}",
        )
    parameters = take(text, "rope_parameters", dict)
    full, sliding = (
        rotation(parameters, "full_attention"),
        rotation(parameters, "sliding_attention"),
    )
    if sliding.get("partial_rotary_factor", 1.0) != 1.0:
        raise Unsupported("partial RoPE on sliding layers is not implemented")
    width, heads = (
        take(text, "hidden_size", int),
        take(text, "num_attention_heads", int),
    )
    head_dim = take(text, "head_dim", int, width // heads)
    spec = Lfm3Spec(
        vocab=take(text, "vocab_size", int),
        width=width,
        layers=layers,
        hidden=take(text, "intermediate_size", int),
        heads=heads,
        kv_heads=take(text, "num_key_value_heads", int),
        head_dim=head_dim,
        eps=float(take(text, "norm_eps", (int, float))),
        theta=float(take(full, "rope_theta", (int, float))),
        kinds=kinds,
        taps=take(text, "conv_L_cache", int),
        window=take(text, "sliding_window", int),
        sliding_theta=float(take(sliding, "rope_theta", (int, float))),
        angles=int(head_dim * float(full.get("partial_rotary_factor", 1.0))) // 2,
    )
    if heads % spec.kv_heads:
        raise Unsupported("LFM3 needs heads divisible by KV heads")
    if not take(text, "tie_word_embeddings", bool, True):
        raise Unsupported("only a head tied to the embedding is implemented")
    return spec


class Lfm3(Uploads, Model):
    def __init__(self, spec, runtime, weights, quantization, device=None):
        super().__init__(spec, runtime)
        s = spec
        self.device, self.weights, self.quantization = (
            device or open_device(),
            weights,
            quantization,
        )
        self.prefix = next(
            prefix
            for prefix in ("language_model.model.", "model.language_model.", "model.")
            if prefix + "embed_tokens.weight" in weights
        )
        self.embed = self.matrix("embed_tokens")
        self.final_norm = self.vector("embedding_norm.weight")
        self.blocks = [self.block(layer) for layer in range(s.layers)]
        del self.weights
        self.run = Sequences(
            self.device,
            runtime,
            vocab=s.vocab,
            sizes=self.sizes,
            steps=self.steps,
            rewindable=False,
            clear=self.clear,
        )
        size = self.run.pages * s.kv_heads * PAGE_SIZE * s.head_dim * 2
        self.pages = {
            layer: (self.device.buffer(size), self.device.buffer(size))
            for layer in range(s.layers)
            if s.kinds[layer] != "conv"
        }
        self.ropes = {
            "full_attention": rope.rope_tables(
                self.capacity,
                s.head_dim,
                s.theta,
                s.angles,
            ),
            "sliding_attention": rope.rope_tables(
                self.capacity,
                s.head_dim,
                s.sliding_theta,
            ),
        }
        self.ropes = {
            kind: [linear.raw(self.device, t) for t in tables]
            for kind, tables in self.ropes.items()
        }
        self.slot_bytes = (s.taps - 1) * s.width * 2
        self.states = {
            layer: self.run.state(self.slot_bytes)
            for layer in range(s.layers)
            if s.kinds[layer] == "conv"
        }

    def block(self, layer):
        s, p = self.spec, f"layers.{layer}."
        block = {
            "input_norm": self.vector(p + "operator_norm.weight"),
            "post_norm": self.vector(p + "ffn_norm.weight"),
            "post_operator": self.vector(p + "post_operator_norm.weight"),
            "post_ffn": self.vector(p + "post_ffn_norm.weight"),
            "gate_up": self.matrix(
                p + "feed_forward.w1",
                p + "feed_forward.w3",
                paired=True,
            ),
            "down": self.matrix(p + "feed_forward.w2"),
        }
        if block["down"].inputs != s.hidden:
            raise Unsupported(
                f"the MLP is {block['down'].inputs} wide, the config gives {s.hidden}",
            )
        if s.kinds[layer] == "conv":
            return {
                **block,
                "in_proj": self.matrix(p + "conv.in_proj"),
                "conv": self.vector(p + "conv.conv.weight"),
                "out_proj": self.matrix(p + "conv.out_proj"),
            }
        a = p + "self_attn."
        return {
            **block,
            "q_norm": self.vector(a + "q_norm.weight"),
            "k_norm": self.vector(a + "k_norm.weight"),
            "qkv": self.matrix(
                a + "q_proj",
                a + "k_proj",
                a + "v_proj",
                a + "gate_proj",
            ),
            "o_proj": self.matrix(a + "o_proj"),
        }

    def sizes(self, rows, batch):
        s = self.spec
        last = rows if batch else 1
        return {
            "ids": rows * 4,
            "x": rows * s.width * 2,
            "normed": rows * s.width * 2,
            "qkv": rows * ((s.heads + 2 * s.kv_heads) * s.head_dim + s.heads) * 2,
            "q": rows * s.heads * s.head_dim * 2,
            "attended": rows * s.heads * s.head_dim * 2,
            "update": rows * s.width * 2,
            "hidden": rows * s.hidden * 2,
            "proj": rows * 3 * s.width * 2,
            "conv": rows * s.width * 2,
            "last": last * s.width * 2,
            "logits": last * s.vocab * 2,
            "token": last * 4,
        }

    def attention_steps(self, layer, rows, b):
        s, device, run, w = self.spec, self.device, self.run, self.blocks[layer]
        keys, values = self.pages[layer]
        window = s.window if s.kinds[layer] == "sliding_attention" else 0
        shape = dict(rows=rows, heads=s.heads, kv_heads=s.kv_heads, head_dim=s.head_dim)
        steps = linear.product(
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
                *self.ropes[s.kinds[layer]],
                run.table,
                run.lengths,
                b["q"],
                keys,
                values,
                page_size=PAGE_SIZE,
                eps=s.eps,
                head_gates=True,
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
            **rope.gate(b["qkv"], head_gates=True, **shape),
            window=window,
            max_pages=run.max_pages(b),
            **shape,
        )
        return steps + linear.product(
            device,
            b["attended"],
            w["o_proj"],
            b["update"],
            rows,
            self.scratch,
        )

    def conv_steps(self, layer, rows, b):
        s, device, w = self.spec, self.device, self.blocks[layer]
        steps = linear.product(
            device,
            b["normed"],
            w["in_proj"],
            b["proj"],
            rows,
            self.scratch,
        )
        inputs = conv1d.Inputs(GatedInputs(s.width), (cl.bf16,))
        steps += conv1d.causal_conv(
            device,
            w["conv"],
            self.states[layer],
            self.run.slots,
            b["conv"],
            rows=rows,
            channels=s.width,
            taps=s.taps,
            inputs=inputs,
            input_operands=(b["proj"],),
            epilogue=Epilogue(GatedOutputs(s.width), (cl.bf16,)),
            operands=(b["proj"],),
            batch=b["batch"],
            snapshots=b.get("snapshots"),
        )
        return steps + linear.product(
            device,
            b["conv"],
            w["out_proj"],
            b["update"],
            rows,
            self.scratch,
        )

    def steps(self, rows, b):
        s, device = self.spec, self.device
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
            steps += (
                self.conv_steps if s.kinds[layer] == "conv" else self.attention_steps
            )(layer, rows, b)
            steps += normalization.residual_norm(
                device,
                b["x"],
                b["update"],
                weight=w["post_norm"],
                output=b["normed"],
                update_weight=w["post_operator"],
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
                    update_weight=w["post_ffn"],
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
            update_weight=self.blocks[-1]["post_ffn"],
            first=rows - last,
            **residual,
        )
        steps += self.run.tap(s.layers - 1, b, rows)
        steps += linear.product(device, b["last"], self.embed, b["logits"], last)
        return steps + [self.run.choose(b, last)]

    def clear(self, sequence):
        for state in self.states.values():
            state.write(bytes(self.slot_bytes), sequence * self.slot_bytes)
