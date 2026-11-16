"""EmbeddingGemma 2 text embeddings."""

import json
from dataclasses import dataclass

import numpy as np
from carbon import language as cl
from carbon.kernels import attention
from carbon.kernels.gemm import epilogue
from carbon.kernels.ops.indexing import gather_rows

from ...core.model import PAGE_SIZE, Model, Unsupported, reject, take
from ...core.program import Program, allocate, open_device
from ...core.weights import Uploads
from ...kernels import linear, normalization, rope


@dataclass(frozen=True)
class EmbeddingGemma2Spec:
    vocab: int
    width: int
    layers: int
    hidden: int
    heads: int
    kinds: tuple
    geometry: tuple
    window: int
    thetas: tuple
    eps: float
    per_layer: int
    dim: int

    def theta(self, kind):
        return dict(self.thetas)[kind]


def parse(config):
    """Parse an EmbeddingGemma 2 checkpoint into EmbeddingGemma2Spec."""
    text = take(config, "text_config", dict)
    reject(text, ("attention_bias",))
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
    for kind in set(kinds):
        if (
            rope[kind].get("rope_type", "default") != "default"
            or rope[kind].get("partial_rotary_factor", 1.0) != 1
        ):
            raise Unsupported("only full default RoPE is implemented")
    base = (take(text, "num_key_value_heads", int), take(text, "head_dim", int))
    overrides = take(text, "per_layer_config", dict, {})
    geometry = []
    for layer in range(layers):
        entry = overrides.get(f"{layer:02d}", {})
        geometry.append(
            (entry.get("num_key_value_heads", base[0]), entry.get("head_dim", base[1])),
        )
    for kind in set(kinds):
        if (
            len({geometry[layer] for layer in range(layers) if kinds[layer] == kind})
            > 1
        ):
            raise Unsupported(f"{kind} layers must share one attention geometry")
    return EmbeddingGemma2Spec(
        vocab=take(text, "vocab_size", int),
        width=take(text, "hidden_size", int),
        layers=layers,
        hidden=take(text, "intermediate_size", int),
        heads=take(text, "num_attention_heads", int),
        kinds=kinds,
        geometry=tuple(geometry),
        window=take(text, "sliding_window", int),
        thetas=tuple(
            (kind, float(rope[kind]["rope_theta"])) for kind in sorted(set(kinds))
        ),
        eps=float(take(text, "rms_norm_eps", (int, float))),
        per_layer=take(text, "hidden_size_per_layer_input", int),
        dim=take(text, "embedding_dim", int),
    )


class EmbeddingGemma2(Uploads, Model):
    """`embed(texts)` returns one L2-normalized embedding per text; a batch holds up to
    `capacity` tokens."""

    ACTIVATION = "gelu_tanh"
    prefix = "language_model."

    def __init__(self, spec, runtime, weights, device=None):
        super().__init__(spec, runtime)
        self.device = device or open_device()
        self.weights, s = weights, spec
        table = linear.from_bf16(self.tensor("embed_tokens.weight"))
        scale = float(linear.from_bf16(linear.to_bf16(np.float32([s.width**0.5])))[0])
        self.embedding = linear.raw(
            self.device,
            linear.to_bf16(table * np.float32(scale)),
        )
        del table
        self.ple_projection = self.matrix("ple.per_layer_model_projection")
        self.ple_norm = self.vector("ple.per_layer_projection_norm.weight")
        self.final_norm = self.vector("norm.weight")
        self.projection = self.matrix("embedding_projection")
        self.blocks = [self.block(layer) for layer in range(s.layers)]
        pages = -(-self.capacity // PAGE_SIZE)
        self.pages, self.tables = {}, {}
        for kind in set(s.kinds):
            kv_heads, head_dim = self.geometry(kind)
            size = pages * kv_heads * PAGE_SIZE * head_dim * 2
            self.pages[kind] = tuple(
                linear.raw(self.device, np.zeros(size, np.uint8)) for _ in range(2)
            )
            self.tables[kind] = tuple(
                linear.raw(self.device, table)
                for table in rope.rope_tables(
                    self.capacity,
                    head_dim,
                    s.theta(kind),
                )
            )
        self.table = linear.raw(self.device, np.arange(pages, dtype=np.uint32))
        self.lengths = linear.raw(self.device, np.zeros(4, np.uint32))
        self.programs = {}
        del self.weights

    @classmethod
    def from_checkpoint(cls, spec, runtime, checkpoint, device=None):
        pooling = checkpoint.path / "1_Pooling" / "config.json"
        if (
            pooling.is_file()
            and json.loads(pooling.read_text()).get("pooling_mode", "mean") != "mean"
        ):
            raise Unsupported("only mean pooling is implemented")
        return cls(spec, runtime, checkpoint.tensors(), device)

    def geometry(self, kind):
        return self.spec.geometry[self.spec.kinds.index(kind)]

    def block(self, layer):
        p, a, e = (
            f"layers.{layer}.",
            f"layers.{layer}.self_attn.",
            f"layers.{layer}.ple_block.",
        )
        return {
            "input_norm": self.vector(p + "input_layernorm.weight"),
            "post_operator": self.vector(p + "post_attention_layernorm.weight"),
            "post_norm": self.vector(p + "pre_feedforward_layernorm.weight"),
            "post_ffn": self.vector(p + "post_feedforward_layernorm.weight"),
            "scalar": self.vector(p + "layer_scalar"),
            "q_norm": self.vector(a + "q_norm.weight"),
            "k_norm": self.vector(a + "k_norm.weight"),
            "qkv": self.matrix(a + "q_proj", a + "k_proj", a + "v_proj"),
            "o_proj": self.matrix(a + "o_proj"),
            "gate_up": self.matrix(p + "mlp.gate_proj", p + "mlp.up_proj", paired=True),
            "down": self.matrix(p + "mlp.down_proj"),
            "ple_gate": self.matrix(e + "per_layer_input_gate"),
            "ple_proj": self.matrix(e + "per_layer_projection"),
            "ple_norm": self.vector(e + "post_per_layer_input_norm.weight"),
        }

    def buffers(self, rows, sequences):
        s = self.spec
        head_dim = max(head_dim for _, head_dim in s.geometry)
        sizes = {
            "ids": rows * 4,
            "positions": rows * 4,
            "segments": (sequences + 1) * 4,
            "x": rows * s.width * 2,
            "normed": rows * s.width * 2,
            "qkv": rows * max(w["qkv"].outputs for w in self.blocks) * 2,
            "q": rows * s.heads * head_dim * 2,
            "attended": rows * s.heads * head_dim * 2,
            "update": rows * s.width * 2,
            "hidden": rows * s.hidden * 2,
            "ple_raw": rows * s.layers * s.per_layer * 2,
            "ple": rows * s.layers * s.per_layer * 2,
            "ple_mixed": rows * s.per_layer * 2,
            "projected": rows * s.dim * 2,
            "pooled": sequences * s.dim * 4,
        }
        return allocate(self.device, sizes)

    def residual(
        self,
        b,
        rows,
        weight=None,
        output=None,
        *,
        update_weight,
        scalar=None,
    ):
        return normalization.residual_norm(
            self.device,
            b["x"],
            b["update"],
            rows=rows,
            columns=self.spec.width,
            epsilon=self.spec.eps,
            weight=weight,
            output=output,
            update_weight=update_weight,
            scalar=scalar,
        )

    def layer_steps(self, layer, rows, sequences, longest, b):
        s, device, w = self.spec, self.device, self.blocks[layer]
        kind = s.kinds[layer]
        kv_heads, head_dim = self.geometry(kind)
        shape = dict(rows=rows, heads=s.heads, kv_heads=kv_heads, head_dim=head_dim)
        keys, values = self.pages[kind]
        cos, sin = self.tables[kind]
        stacked = s.layers * s.per_layer
        gate = epilogue.chain(
            epilogue.rounded(cl.bf16),
            epilogue.gelu_tanh,
            epilogue.rounded(cl.bf16),
            epilogue.multiply(cl.bf16, s.per_layer, stacked),
        )
        ple = b["ple"].view(
            layer * s.per_layer * 2,
            (rows * stacked - layer * s.per_layer) * 2,
        )
        steps = (
            normalization.rms_norm_rows(
                device,
                b["x"],
                w["input_norm"],
                b["normed"],
                rows=rows,
                columns=s.width,
                epsilon=s.eps,
            )
            if layer == 0
            else []
        )
        steps += linear.product(device, b["normed"], w["qkv"], b["qkv"], rows)
        steps += [
            rope.rope_cache(
                device,
                b["qkv"],
                w["q_norm"],
                w["k_norm"],
                cos,
                sin,
                self.table,
                self.lengths,
                b["q"],
                keys,
                values,
                page_size=PAGE_SIZE,
                eps=s.eps,
                v_norm=True,
                positions=b["positions"],
                **shape,
            ),
        ]
        steps += attention.paged_attention(
            device,
            b["q"],
            keys,
            values,
            self.table,
            self.lengths,
            b["attended"],
            page_size=PAGE_SIZE,
            context=self.capacity,
            scale=1.0,
            causal=False,
            window=s.window if kind == "sliding_attention" else 0,
            segments=b["segments"],
            sequences=sequences,
            longest=longest,
            **shape,
        )
        steps += linear.product(
            device,
            b["attended"],
            w["o_proj"],
            b["update"],
            rows,
        )
        steps += self.residual(
            b,
            rows,
            w["post_norm"],
            b["normed"],
            update_weight=w["post_operator"],
        )
        steps += linear.product(
            device,
            b["normed"],
            w["gate_up"],
            b["hidden"],
            rows,
            activation=self.ACTIVATION,
        )
        steps += linear.product(device, b["hidden"], w["down"], b["update"], rows)
        steps += self.residual(b, rows, update_weight=w["post_ffn"])
        steps += linear.product(
            device,
            b["x"],
            w["ple_gate"],
            b["ple_mixed"],
            rows,
            epilogue=gate,
            operands=(ple,),
        )
        steps += linear.product(
            device,
            b["ple_mixed"],
            w["ple_proj"],
            b["update"],
            rows,
        )
        following = (
            self.blocks[layer + 1]["input_norm"]
            if layer + 1 < s.layers
            else self.final_norm
        )
        return steps + self.residual(
            b,
            rows,
            following,
            b["normed"],
            update_weight=w["ple_norm"],
            scalar=w["scalar"],
        )

    def steps(self, rows, sequences, longest, b):
        """Every dispatch of one packed batch."""
        s, device = self.spec, self.device
        steps = gather_rows(
            device,
            self.embedding,
            b["ids"],
            b["x"],
            rows=rows,
            columns=s.width,
        )
        steps += linear.product(
            device,
            b["x"],
            self.ple_projection,
            b["ple_raw"],
            rows,
            epilogue=linear.scaled(s.width**-0.5),
        )
        steps += normalization.rms_norm_rows(
            device,
            b["ple_raw"],
            self.ple_norm,
            b["ple"],
            rows=rows * s.layers,
            columns=s.per_layer,
            epsilon=s.eps,
        )
        for layer in range(s.layers):
            steps += self.layer_steps(layer, rows, sequences, longest, b)
        steps += linear.product(
            device,
            b["normed"],
            self.projection,
            b["projected"],
            rows,
        )
        return steps + normalization.segment_mean(
            device,
            b["projected"],
            b["segments"],
            b["pooled"],
            sequences=sequences,
            columns=s.dim,
            dtype=cl.bf16,
        )

    def program(self, rows, sequences, longest):
        key = rows, sequences, longest
        if key not in self.programs:
            b = self.buffers(rows, sequences)
            program = Program(self.device)
            for step in self.steps(rows, sequences, longest, b):
                program.kernel(step)
            program.finish()
            self.programs[key] = program, b
        return self.programs[key]

    def tokenize(self, texts):
        return [self.tokenizer(text)["input_ids"] for text in texts]

    def embed_tokens(self, batches, gpu=None):
        """f32 embeddings of token lists, packed into one step."""
        lengths = [len(tokens) for tokens in batches]
        rows = sum(lengths)
        if not batches or min(lengths) < 1 or rows > self.capacity:
            raise ValueError(
                f"a batch needs texts of at least one token, {self.capacity} tokens "
                "in all at most",
            )
        program, b = self.program(rows, len(batches), max(lengths))
        ids = np.concatenate(batches).astype(np.uint32)
        if ids.max() >= self.spec.vocab:
            raise ValueError("tokens must be vocabulary IDs")
        b["ids"].write(ids.tobytes() + bytes(b["ids"].size - ids.nbytes))
        positions = np.concatenate(
            [np.arange(length, dtype=np.uint32) for length in lengths],
        )
        b["positions"].write(
            positions.tobytes() + bytes(b["positions"].size - positions.nbytes),
        )
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint32)
        b["segments"].write(
            offsets.tobytes() + bytes(b["segments"].size - offsets.nbytes),
        )
        completion = program.run()
        if gpu is not None:
            gpu.append(completion.gpu_ms)
        return np.frombuffer(b["pooled"].read(), np.float32)[
            : len(batches) * self.spec.dim
        ].reshape(len(batches), -1)

    def embed(self, texts):
        """f32 unit embeddings of `texts`."""
        return self.embed_tokens(self.tokenize(texts))
