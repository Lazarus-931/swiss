"""Gemma 4 text models."""

from dataclasses import dataclass

from carbon import language as cl
from carbon.kernels import attention
from carbon.kernels.gemm import epilogue

from ...core.model import PAGE_SIZE, Model, Unsupported, reject, take
from ...core.program import open_device
from ...core.sequences import Sequences
from ...core.weights import Uploads, quantization
from ...kernels import linear, normalization, rope


@dataclass(frozen=True)
class Gemma4Spec:
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
    window: int
    sliding_theta: float
    global_head_dim: int
    global_kv_heads: int
    angles: int
    k_eq_v: bool
    first_shared: int
    per_layer: int
    softcap: float


def parse(config):
    """Gemma4Spec of a dense Gemma 4 config."""
    text = take(config, "text_config", dict)
    reject(text, ("enable_moe_block", "attention_bias"))
    quantization(config)
    layers = take(text, "num_hidden_layers", int)
    kinds = tuple(take(text, "layer_types", list))
    if len(kinds) != layers or set(kinds) - {"sliding_attention", "full_attention"}:
        raise Unsupported(
            f"layer_types must list {layers} sliding_attention or full_attention "
            "layers",
        )
    if take(text, "hidden_activation", str) != "gelu_pytorch_tanh":
        raise Unsupported("only the tanh GELU activation is implemented")
    rope = take(text, "rope_parameters", dict)
    full, sliding = rope.get("full_attention", {}), rope.get("sliding_attention", {})
    if (
        full.get("rope_type", "default") not in ("default", "proportional")
        or full.get("factor", 1.0) != 1.0
    ):
        raise Unsupported(
            "only proportional RoPE without a factor is implemented on full layers",
        )
    if (
        sliding.get("rope_type", "default") != "default"
        or sliding.get("partial_rotary_factor", 1.0) != 1.0
    ):
        raise Unsupported("only full default RoPE is implemented on sliding layers")
    head_dim = take(text, "head_dim", int)
    global_head_dim = take(text, "global_head_dim", int, head_dim)
    kv_heads = take(text, "num_key_value_heads", int)
    k_eq_v = take(text, "attention_k_eq_v", bool, False)
    global_kv_heads = (
        take(text, "num_global_key_value_heads", int, kv_heads) if k_eq_v else kv_heads
    )
    per_layer = take(text, "hidden_size_per_layer_input", int, 0)
    if per_layer and take(text, "vocab_size_per_layer_input", int) != take(
        text,
        "vocab_size",
        int,
    ):
        raise Unsupported("per-layer inputs need the main vocabulary")
    shared = take(text, "num_kv_shared_layers", int, 0)
    spec = Gemma4Spec(
        vocab=take(text, "vocab_size", int),
        width=take(text, "hidden_size", int),
        layers=layers,
        hidden=take(text, "intermediate_size", int)
        * (2 if take(text, "use_double_wide_mlp", bool, False) else 1),
        heads=take(text, "num_attention_heads", int),
        kv_heads=kv_heads,
        head_dim=head_dim,
        eps=float(take(text, "rms_norm_eps", (int, float))),
        theta=float(take(full, "rope_theta", (int, float))),
        kinds=kinds,
        window=take(text, "sliding_window", int),
        sliding_theta=float(take(sliding, "rope_theta", (int, float))),
        global_head_dim=global_head_dim,
        global_kv_heads=global_kv_heads,
        angles=int(global_head_dim * float(full.get("partial_rotary_factor", 1.0)))
        // 2,
        k_eq_v=k_eq_v,
        first_shared=layers - shared if shared else layers,
        per_layer=per_layer,
        softcap=float(take(text, "final_logit_softcapping", (int, float), 0.0)),
    )
    if not take(text, "tie_word_embeddings", bool, True):
        raise Unsupported("only a head tied to the embedding is implemented")
    return spec


class Layers:
    """Gemma 4 decoder layers over `blocks`, `pages`, `ropes` and the `run` tables."""

    def sliding(self, layer):
        return self.spec.kinds[layer] == "sliding_attention"

    def shared(self, layer):
        return layer >= self.spec.first_shared

    def k_eq_v(self, layer):
        return self.spec.k_eq_v and not self.sliding(layer)

    def geometry(self, layer):
        """(KV heads, head dim) of a layer's pages."""
        s = self.spec
        return (
            (s.kv_heads, s.head_dim)
            if self.sliding(layer)
            else (s.global_kv_heads, s.global_head_dim)
        )

    def block(self, layer):
        p, a = f"layers.{layer}.", f"layers.{layer}.self_attn."
        projections = (
            ("q",)
            if self.shared(layer)
            else ("q", "k")
            if self.k_eq_v(layer)
            else ("q", "k", "v")
        )
        block = {
            "input_norm": self.vector(p + "input_layernorm.weight"),
            "post_operator": self.vector(p + "post_attention_layernorm.weight"),
            "post_norm": self.vector(p + "pre_feedforward_layernorm.weight"),
            "post_ffn": self.vector(p + "post_feedforward_layernorm.weight"),
            "scalar": self.vector(p + "layer_scalar"),
            "q_norm": self.vector(a + "q_norm.weight"),
            "k_norm": self.vector(
                a + ("q_norm" if self.shared(layer) else "k_norm") + ".weight",
            ),
            "qkv": self.matrix(*(a + f"{name}_proj" for name in projections)),
            "o_proj": self.matrix(a + "o_proj"),
            "gate_up": self.matrix(p + "mlp.gate_proj", p + "mlp.up_proj", paired=True),
            "down": self.matrix(p + "mlp.down_proj"),
        }
        if self.spec.per_layer:
            block.update(
                ple_gate=self.matrix(p + "per_layer_input_gate"),
                ple_proj=self.matrix(p + "per_layer_projection"),
                ple_norm=self.vector(p + "post_per_layer_input_norm.weight"),
            )
        return block

    def attention_steps(self, layer, rows, b):
        s, device, run, w = self.spec, self.device, self.run, self.blocks[layer]
        kv_heads, head_dim = self.geometry(layer)
        keys, values = self.pages[layer]
        shape = dict(rows=rows, heads=s.heads, kv_heads=kv_heads, head_dim=head_dim)
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
                max_pages=run.max_pages(b),
                v_norm=True,
                k_eq_v=self.k_eq_v(layer),
                queries_only=self.shared(layer),
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
            window=s.window if self.sliding(layer) else 0,
            max_pages=run.max_pages(b),
            context=self.capacity,
            scale=1.0,
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

    def per_layer_steps(self, layer, rows, b):
        s, device, w = self.spec, self.device, self.blocks[layer]
        steps = linear.product(device, b["x"], w["ple_gate"], b["ple_gate"], rows)
        steps += [
            linear.ple_gate(
                device,
                b["ple_gate"],
                b["ple"],
                b["ple_mixed"],
                rows,
                s.per_layer,
                s.layers * s.per_layer,
                layer * s.per_layer,
            ),
        ]
        return steps + linear.product(
            device,
            b["ple_mixed"],
            w["ple_proj"],
            b["update"],
            rows,
        )

    def layer_end(self, layer, rows, b, update_weight):
        """Residual add, layer scalar and the next norm (the final norm after the last
        layer)."""
        s, w = self.spec, self.blocks[layer]
        end = dict(
            rows=rows,
            columns=s.width,
            epsilon=s.eps,
            update_weight=update_weight,
            scalar=w["scalar"],
        )
        if layer + 1 < s.layers:
            return normalization.residual_norm(
                self.device,
                b["x"],
                b["update"],
                weight=self.blocks[layer + 1]["input_norm"],
                output=b["normed"],
                **end,
            )
        last = self.run.last(b, rows)
        return normalization.residual_norm(
            self.device,
            b["x"],
            b["update"],
            weight=self.final_norm,
            output=b["last"],
            first=rows - last,
            **end,
        )

    def layer_steps(self, layer, rows, b):
        s, device, w = self.spec, self.device, self.blocks[layer]
        residual = dict(rows=rows, columns=s.width, epsilon=s.eps)
        steps = self.attention_steps(layer, rows, b)
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
            "gelu_tanh",
        )
        steps += linear.product(
            device,
            b["hidden"],
            w["down"],
            b["update"],
            rows,
            self.scratch,
        )
        if s.per_layer:
            steps += normalization.residual_norm(
                device,
                b["x"],
                b["update"],
                update_weight=w["post_ffn"],
                **residual,
            )
            steps += self.per_layer_steps(layer, rows, b)
            steps += self.layer_end(layer, rows, b, w["ple_norm"])
        else:
            steps += self.layer_end(layer, rows, b, w["post_ffn"])
        return steps


class Gemma4(Layers, Uploads, Model):
    def __init__(self, spec, runtime, weights, quantization, device=None):
        super().__init__(spec, runtime)
        s = spec
        self.device, self.weights, self.quantization = (
            device or open_device(),
            weights,
            quantization,
        )
        official = "model.language_model.norm.weight" in weights
        self.prefix = "model.language_model." if official else "language_model.model."
        self.embed = self.matrix("embed_tokens")
        self.final_norm = self.vector("norm.weight")
        self.blocks = [self.block(layer) for layer in range(s.layers)]
        if s.per_layer:
            self.per_layer = (
                self.matrix("embed_tokens_per_layer"),
                self.matrix("per_layer_model_projection"),
                self.vector("per_layer_projection_norm.weight"),
            )
        del self.weights
        self.run = Sequences(
            self.device,
            runtime,
            vocab=s.vocab,
            sizes=self.sizes,
            steps=self.steps,
            rewindable=True,
        )
        latest, self.pages = {}, {}
        for layer, kind in enumerate(s.kinds):
            if layer < s.first_shared:
                kv_heads, head_dim = self.geometry(layer)
                size = self.run.pages * kv_heads * PAGE_SIZE * head_dim * 2
                self.pages[layer] = latest[kind] = (
                    self.device.buffer(size),
                    self.device.buffer(size),
                )
            else:
                self.pages[layer] = latest[kind]
        tables = {
            "full_attention": rope.rope_tables(
                self.capacity,
                s.global_head_dim,
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
            kind: [linear.raw(self.device, t) for t in pair]
            for kind, pair in tables.items()
        }

    def sizes(self, rows, batch):
        s = self.spec
        last = rows if batch else 1
        head_dim = max(s.head_dim, s.global_head_dim)
        sizes = {
            "ids": rows * 4,
            "x": rows * s.width * 2,
            "normed": rows * s.width * 2,
            "qkv": rows * max(w["qkv"].outputs for w in self.blocks) * 2,
            "q": rows * s.heads * head_dim * 2,
            "attended": rows * s.heads * head_dim * 2,
            "update": rows * s.width * 2,
            "hidden": rows * s.hidden * 2,
            "last": last * s.width * 2,
            "logits": last * s.vocab * 2,
            "token": last * 4,
        }
        if s.per_layer:
            stacked = rows * s.layers * s.per_layer * 2
            sizes.update(
                ple_raw=stacked,
                ple_proj=stacked,
                ple_normed=stacked,
                ple=stacked,
                ple_gate=rows * s.per_layer * 2,
                ple_mixed=rows * s.per_layer * 2,
            )
        return sizes

    def embed_steps(self, rows, b):
        s, device = self.spec, self.device
        steps = [
            linear.embedding(
                device,
                b["ids"],
                self.embed,
                b["x"],
                rows,
                scale=s.width**0.5,
            ),
        ]
        if s.per_layer:
            table, projection, weight = self.per_layer
            steps += [
                linear.embedding(
                    device,
                    b["ids"],
                    table,
                    b["ple_raw"],
                    rows,
                    scale=s.per_layer**0.5,
                ),
            ]
            steps += linear.product(
                device,
                b["x"],
                projection,
                b["ple_proj"],
                rows,
                self.scratch,
            )
            steps += normalization.rms_norm_rows(
                device,
                b["ple_proj"],
                weight,
                b["ple_normed"],
                rows=rows * s.layers,
                columns=s.per_layer,
                epsilon=s.eps,
            )
            steps += [
                linear.ple_combine(
                    device,
                    b["ple_normed"],
                    b["ple_raw"],
                    b["ple"],
                    rows * s.layers * s.per_layer,
                    2.0**-0.5,
                ),
            ]
        return steps + normalization.rms_norm_rows(
            device,
            b["x"],
            self.blocks[0]["input_norm"],
            b["normed"],
            rows=rows,
            columns=s.width,
            epsilon=s.eps,
        )

    def steps(self, rows, b):
        s, device = self.spec, self.device
        steps = self.embed_steps(rows, b)
        for layer in range(s.layers):
            steps += self.layer_steps(layer, rows, b)
            steps += self.run.tap(layer, b, rows)
        last = self.run.last(b, rows)
        cap = (
            epilogue.chain(epilogue.rounded(cl.bf16), epilogue.softcap(s.softcap))
            if s.softcap
            else None
        )
        steps += linear.product(
            device,
            b["last"],
            self.embed,
            b["logits"],
            last,
            epilogue=cap,
        )
        return steps + [self.run.choose(b, last)]
