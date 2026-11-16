"""Qwen3.5 dense text models."""

from dataclasses import dataclass

import numpy as np
from carbon import language as cl
from carbon.kernels import attention
from carbon.kernels.gemm import epilogue

from ...core.model import PAGE_SIZE, Model, Unsupported, reject, take
from ...core.program import open_device
from ...core.sequences import Sequences
from ...core.weights import Uploads, quantization
from ...kernels import conv1d, delta, linear, normalization, rope
from ...kernels.linear import from_bf16, to_bf16

CONV_KERNEL = 4


@dataclass(frozen=True)
class Qwen3_5Spec:
    vocab: int
    width: int
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    rotary: int
    eps: float
    theta: float
    tie_embeddings: bool
    interval: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int


def parse(config):
    text = config.get("text_config", config)
    reject(text, ("attention_bias", "num_experts", "rope_scaling"))
    if not text.get("attn_output_gate", True):
        raise Unsupported("Qwen3.5 attention without an output gate is not implemented")
    if take(text, "linear_conv_kernel_dim", int) != CONV_KERNEL:
        raise Unsupported(f"only a {CONV_KERNEL}-tap convolution is implemented")
    quantization(config)
    rope = text.get("rope_parameters") or {}
    head_dim, layers = take(text, "head_dim", int), take(text, "num_hidden_layers", int)
    interval = take(text, "full_attention_interval", int, 4)
    kinds = text.get("layer_types")
    if kinds and kinds != [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(layers)
    ]:
        raise Unsupported(
            "layer_types must repeat linear attention with full attention every "
            "interval",
        )
    spec = Qwen3_5Spec(
        vocab=take(text, "vocab_size", int),
        width=take(text, "hidden_size", int),
        layers=layers,
        hidden=take(text, "intermediate_size", int),
        heads=take(text, "num_attention_heads", int),
        kv_heads=take(text, "num_key_value_heads", int),
        head_dim=head_dim,
        rotary=int(
            head_dim
            * float(
                text.get("partial_rotary_factor")
                or rope.get("partial_rotary_factor", 0.25),
            ),
        ),
        eps=float(take(text, "rms_norm_eps", (int, float))),
        theta=float(text.get("rope_theta") or take(rope, "rope_theta", (int, float))),
        tie_embeddings=take(
            config,
            "tie_word_embeddings",
            bool,
            take(text, "tie_word_embeddings", bool, False),
        ),
        interval=interval,
        key_heads=take(text, "linear_num_key_heads", int),
        value_heads=take(text, "linear_num_value_heads", int),
        key_dim=take(text, "linear_key_head_dim", int),
        value_dim=take(text, "linear_value_head_dim", int),
    )
    if spec.heads % spec.kv_heads or spec.value_heads % spec.key_heads:
        raise Unsupported(
            "Qwen3.5 needs heads divisible by KV heads and value heads by key heads",
        )
    return spec


@dataclass(frozen=True)
class SiluGate:
    """silu(z) · round(value) for the gated norm, with z at column `z` of the projection
    rows, `width` apart."""

    width: int
    z: int

    def __call__(self, value, row, column, proj):
        normed = cl.cast(cl.cast(value, cl.bf16), cl.f32)
        z = cl.cast(
            cl.load(proj, row * self.width + self.z + column, True, other=0.0),
            cl.f32,
        )
        return z * cl.sigmoid(z) * normed


@dataclass(frozen=True)
class DecayGates:
    """Log decay and beta of one value head and row, from the projection rows."""

    width: int
    gates: int
    heads: int

    def __call__(self, row, head, live, proj, a_log, dt_bias):
        def rounded(value):
            return cl.cast(cl.cast(value, cl.bf16), cl.f32)

        gate = row * self.width + self.gates + head
        shifted = rounded(
            cl.cast(cl.load(proj, gate + self.heads, live, other=0.0), cl.f32)
            + cl.cast(cl.load(dt_bias, head, True, other=0.0), cl.f32),
        )
        softplus = rounded(
            cl.maximum(shifted, 0.0) + cl.log(1.0 + cl.exp(-cl.abs(shifted))),
        )
        decay = -cl.exp(cl.load(a_log, head, True, other=0.0)) * softplus
        return decay, rounded(
            cl.sigmoid(cl.cast(cl.load(proj, gate, live, other=0.0), cl.f32)),
        )


class Qwen3_5(Uploads, Model):
    def __init__(self, spec, runtime, weights, quantization, device=None):
        super().__init__(spec, runtime)
        s = spec
        self.device, self.weights, self.quantization = (
            device or open_device(),
            weights,
            quantization,
        )
        self.official = "model.language_model.norm.weight" in weights
        self.prefix = (
            "model.language_model." if self.official else "language_model.model."
        )
        self.channels = 2 * s.key_heads * s.key_dim + s.value_heads * s.value_dim
        self.gates = self.channels + s.value_heads * s.value_dim
        self.proj_width = self.gates + 2 * s.value_heads
        self.embed = self.matrix("embed_tokens")
        head = "lm_head" if self.official else "language_model.lm_head"
        self.head = self.embed if s.tie_embeddings else self.matrix(head, prefix="")
        self.final_norm = self.norm("norm.weight")
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
            if self.attention(layer)
        }
        self.rope = [
            linear.raw(self.device, t)
            for t in rope.rope_tables(self.capacity, s.rotary, s.theta)
        ]
        self.conv_bytes = (CONV_KERNEL - 1) * self.channels * 2
        self.state_bytes = s.value_heads * s.value_dim * s.key_dim * 4
        self.states = {
            layer: (self.run.state(self.conv_bytes), self.run.state(self.state_bytes))
            for layer in range(s.layers)
            if not self.attention(layer)
        }
        self.workspace = None

    def attention(self, layer):
        return (layer + 1) % self.spec.interval == 0

    def norm(self, name):
        """An RMSNorm weight; official checkpoints store w for a scale of 1 + w."""
        if not self.official:
            return self.vector(name)
        return linear.raw(
            self.device,
            to_bf16(from_bf16(self.tensor(name)) + np.float32(1.0)),
        )

    def block(self, layer):
        p = f"layers.{layer}."
        block = {
            "input_norm": self.norm(p + "input_layernorm.weight"),
            "post_norm": self.norm(p + "post_attention_layernorm.weight"),
            "gate_up": self.matrix(p + "mlp.gate_proj", p + "mlp.up_proj", paired=True),
            "down": self.matrix(p + "mlp.down_proj"),
        }
        if self.attention(layer):
            a = p + "self_attn."
            return {
                **block,
                "q_norm": self.norm(a + "q_norm.weight"),
                "k_norm": self.norm(a + "k_norm.weight"),
                "qkv": self.matrix(a + "q_proj", a + "k_proj", a + "v_proj"),
                "o_proj": self.matrix(a + "o_proj"),
            }
        a = p + "linear_attn."
        a_log, dt_bias = (
            np.asarray(self.tensor(a + "A_log")),
            np.asarray(self.tensor(a + "dt_bias")),
        )
        a_log = (
            from_bf16(a_log) if a_log.dtype == np.uint16 else a_log.astype(np.float32)
        )
        dt_bias = (
            dt_bias
            if dt_bias.dtype == np.uint16
            else to_bf16(dt_bias.astype(np.float32))
        )
        return {
            **block,
            "in_proj": self.matrix(
                *(a + f"in_proj_{name}" for name in ("qkv", "z", "b", "a")),
            ),
            "conv": self.vector(a + "conv1d.weight"),
            "norm": self.vector(a + "norm.weight"),
            "a_log": linear.raw(self.device, a_log),
            "dt_bias": linear.raw(self.device, dt_bias),
            "out_proj": self.matrix(a + "out_proj"),
        }

    def sizes(self, rows, batch):
        s = self.spec
        last = rows if batch else 1
        values = rows * s.value_heads * s.value_dim * 2
        return {
            "ids": rows * 4,
            "x": rows * s.width * 2,
            "normed": rows * s.width * 2,
            "qkv": rows * (2 * s.heads + 2 * s.kv_heads) * s.head_dim * 2,
            "q": rows * s.heads * s.head_dim * 2,
            "attended": rows * s.heads * s.head_dim * 2,
            "update": rows * s.width * 2,
            "hidden": rows * s.hidden * 2,
            "proj": rows * self.proj_width * 2,
            "conv": rows * self.channels * 2,
            "recurrent": values,
            "gated": values,
            "last": last * s.width * 2,
            "logits": last * s.vocab * 2,
            "token": last * 4,
        }

    def attention_steps(self, layer, rows, b):
        s, device, run, w = self.spec, self.device, self.run, self.blocks[layer]
        keys, values = self.pages[layer]
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
                *self.rope,
                run.table,
                run.lengths,
                b["q"],
                keys,
                values,
                page_size=PAGE_SIZE,
                eps=s.eps,
                rotary=s.rotary,
                gated=True,
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
            **rope.gate(b["qkv"], **shape),
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

    def delta_steps(self, layer, rows, b):
        s, device, run, w = self.spec, self.device, self.run, self.blocks[layer]
        conv_state, delta_state = self.states[layer]
        heads = dict(
            key_heads=s.key_heads,
            value_heads=s.value_heads,
            key_dim=s.key_dim,
            value_dim=s.value_dim,
        )
        steps = linear.product(device, b["normed"], w["in_proj"], b["proj"], rows)
        steps += conv1d.causal_conv(
            device,
            w["conv"],
            conv_state,
            run.slots,
            b["conv"],
            rows=rows,
            channels=self.channels,
            taps=CONV_KERNEL,
            inputs=conv1d.strided(self.proj_width),
            input_operands=(b["proj"],),
            epilogue=epilogue.silu,
            batch=b["batch"],
            snapshots=b.get("snapshots"),
        )
        workspace = (
            None
            if b["batch"]
            else self.delta_workspace(
                delta.delta_workspace_bytes(
                    rows,
                    value_heads=s.value_heads,
                    key_dim=s.key_dim,
                    value_dim=s.value_dim,
                ),
            )
        )
        gates = delta.Gates(
            DecayGates(self.proj_width, self.gates, s.value_heads),
            (cl.bf16, cl.f32, cl.bf16),
        )
        steps += delta.delta_rule(
            device,
            b["conv"],
            delta_state,
            run.slots,
            b["recurrent"],
            rows=rows,
            gates=gates,
            gate_operands=(b["proj"], w["a_log"], w["dt_bias"]),
            batch=b["batch"],
            workspace=workspace,
            snapshots=b.get("snapshots"),
            **heads,
        )
        gate = epilogue.Epilogue(SiluGate(self.proj_width, self.channels), (cl.bf16,))
        steps += normalization.rms_norm_rows(
            device,
            b["recurrent"],
            w["norm"],
            b["gated"],
            rows=rows,
            columns=s.value_heads * s.value_dim,
            dtype=cl.bf16,
            epsilon=s.eps,
            groups=s.value_heads,
            epilogue=gate,
            operands=(b["proj"],),
        )
        return steps + linear.product(
            device,
            b["gated"],
            w["out_proj"],
            b["update"],
            rows,
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
                self.attention_steps if self.attention(layer) else self.delta_steps
            )(layer, rows, b)
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
        return steps + [self.run.choose(b, last)]

    def delta_workspace(self, size):
        """One buffer of at least `size` bytes for the chunked delta rule of every
        prefill; it grows on demand."""
        if self.workspace is None or self.workspace.size < size:
            self.workspace = self.device.buffer(size)
        return self.workspace

    def clear(self, sequence):
        for conv_state, delta_state in self.states.values():
            conv_state.write(bytes(self.conv_bytes), sequence * self.conv_bytes)
            delta_state.write(bytes(self.state_bytes), sequence * self.state_bytes)
