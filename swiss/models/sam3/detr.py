"""SAM 3 detector: text encoder, DETR encoder and decoder, and scores."""

import math
from dataclasses import dataclass

import numpy as np
from carbon import language as cl
from carbon.kernels import attention
from carbon.kernels.attention import ScoreMod
from carbon.kernels.gemm import epilogue
from carbon.kernels.ops import indexing

from ...core.model import PAGE_SIZE
from ...core.program import Program
from ...kernels import linear, normalization, rope


@dataclass(frozen=True)
class _BoxBias:
    """Box position bias X[row, key % width, head] + Y[row, key // width, head]."""

    width: int
    height: int
    heads: int
    y_offset: int

    def __call__(self, score, row, head, key, table):
        x = cl.load(
            table,
            row * (self.width * self.heads) + head + key % self.width * self.heads,
            True,
            other=0.0,
        )
        y = cl.load(
            table,
            self.y_offset
            + row * (self.height * self.heads)
            + head
            + key // self.width * self.heads,
            True,
            other=0.0,
        )
        return score + cl.cast(x, cl.f32) + cl.cast(y, cl.f32)


DETECTOR = "detector_model."


def sine_grid(size, features, temperature=10000.0):
    """Normalized sine position embedding of a (size, size) grid, y part then x part."""
    steps = (np.arange(size, dtype=np.float32) + 1) / (size + 1e-6) * 2 * math.pi
    dim = temperature ** (2 * (np.arange(features) // 2) / features)
    angle = steps[:, None] / dim
    encoded = np.stack([np.sin(angle[:, 0::2]), np.cos(angle[:, 1::2])], -1).reshape(
        size,
        features,
    )
    y = np.repeat(encoded, size, 0)
    x = np.tile(encoded, (size, 1))
    return np.concatenate([y, x], 1).astype(np.float32)


def sine_boxes(boxes, features, temperature=10000.0):
    """Sine embedding of cxcywh boxes."""
    dim = temperature ** (2 * (np.arange(features) // 2) / features)
    parts = []
    for coordinate in (boxes[:, 1], boxes[:, 0], boxes[:, 2], boxes[:, 3]):
        angle = coordinate[:, None] * 2 * math.pi / dim
        parts.append(
            np.stack([np.sin(angle[:, 0::2]), np.cos(angle[:, 1::2])], -1).reshape(
                len(boxes),
                -1,
            ),
        )
    return np.concatenate(parts, 1)


def log_deltas(coordinates, bounds):
    """Log-scaled distance of each grid coordinate to a box's edges."""
    deltas = (coordinates[None, :, None] - bounds[:, None, :]) * 8
    return np.sign(deltas) * np.log2(np.abs(deltas) + 1.0) / 3.0


def layer_norm(x, weight, bias, eps):
    mean = x.mean(-1, keepdims=True)
    centred = x - mean
    return (
        centred / np.sqrt((centred * centred).mean(-1, keepdims=True) + eps) * weight
        + bias
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Detector:
    """The SAM 3 detector, mixed into the SAM 3 model."""

    def load_detector(self, config):
        s, device = self.spec, self.device
        self.d = self.weights[
            DETECTOR + "detr_encoder.layers.0.self_attn.q_proj.weight"
        ].shape[0]
        self.text_width = self.weights[
            DETECTOR + "text_encoder.text_model.final_layer_norm.weight"
        ].shape[0]
        self.queries = self.weights[DETECTOR + "detr_decoder.query_embed.weight"].shape[
            0
        ]
        detector = config["detector_config"]
        self.detr_heads = detector["detr_encoder_config"]["num_attention_heads"]
        self.text_heads = detector["text_config"]["num_attention_heads"]
        self.detr_eps = detector["detr_encoder_config"]["layer_norm_eps"]
        self.text_eps = detector["text_config"]["layer_norm_eps"]
        tokens = s.grid * s.grid
        t = DETECTOR + "text_encoder.text_model."
        self.token_table = self.vector(t + "embeddings.token_embedding.weight")
        self.text_positions = self.vector(t + "embeddings.position_embedding.weight")
        self.text_layers = []
        for layer in range(detector["text_config"]["num_hidden_layers"]):
            p, a = (
                f"{t}encoder.layers.{layer}.",
                f"{t}encoder.layers.{layer}.self_attn.",
            )
            self.text_layers.append(
                {
                    "norm1": self.norm(p + "layer_norm1"),
                    "norm2": self.norm(p + "layer_norm2"),
                    "qkv": self.affine(a + "q_proj", a + "k_proj", a + "v_proj"),
                    "o": self.affine(a + "out_proj"),
                    "fc1": self.affine(p + "mlp.fc1"),
                    "fc2": self.affine(p + "mlp.fc2"),
                },
            )
        self.text_final = self.norm(t + "final_layer_norm")
        self.text_projection = self.affine(DETECTOR + "text_projection")
        position = sine_grid(s.grid, self.d // 2)
        self.encoder_layers = []
        for layer in range(detector["detr_encoder_config"]["num_layers"]):
            p = f"{DETECTOR}detr_encoder.layers.{layer}."
            self.encoder_layers.append(
                {
                    "norm1": self.norm(p + "layer_norm1"),
                    "norm2": self.norm(p + "layer_norm2"),
                    "norm3": self.norm(p + "layer_norm3"),
                    "qkv": self.affine(
                        p + "self_attn.q_proj",
                        p + "self_attn.k_proj",
                        p + "self_attn.v_proj",
                        positioned=(position, 2),
                    ),
                    "o": self.affine(p + "self_attn.o_proj"),
                    "cross_q": self.affine(p + "cross_attn.q_proj"),
                    "cross_kv": self.affine(
                        p + "cross_attn.k_proj",
                        p + "cross_attn.v_proj",
                    ),
                    "cross_o": self.affine(p + "cross_attn.o_proj"),
                    "fc1": self.affine(p + "mlp.fc1"),
                    "fc2": self.affine(p + "mlp.fc2"),
                },
            )
        self.decoder_layers = []
        for layer in range(detector["detr_decoder_config"]["num_layers"]):
            p = f"{DETECTOR}detr_decoder.layers.{layer}."
            self.decoder_layers.append(
                {
                    "self": self.affine(
                        p + "self_attn.q_proj",
                        p + "self_attn.k_proj",
                        p + "self_attn.v_proj",
                    ),
                    "self_o": self.affine(p + "self_attn.o_proj"),
                    "self_norm": self.norm(p + "self_attn_layer_norm"),
                    "text_q": self.affine(p + "text_cross_attn.q_proj"),
                    "text_kv": self.affine(
                        p + "text_cross_attn.k_proj",
                        p + "text_cross_attn.v_proj",
                    ),
                    "text_o": self.affine(p + "text_cross_attn.o_proj"),
                    "text_norm": self.norm(p + "text_cross_attn_layer_norm"),
                    "vision_q": self.affine(p + "vision_cross_attn.q_proj"),
                    "vision_kv": self.affine(
                        p + "vision_cross_attn.k_proj",
                        p + "vision_cross_attn.v_proj",
                        positioned=(position, 1),
                    ),
                    "vision_o": self.affine(p + "vision_cross_attn.o_proj"),
                    "vision_norm": self.norm(p + "vision_cross_attn_layer_norm"),
                    "fc1": self.affine(p + "mlp.fc1"),
                    "fc2": self.affine(p + "mlp.fc2"),
                    "mlp_norm": self.norm(p + "mlp_layer_norm"),
                    "host": {
                        name: self.host(p + f"{name}.weight", p + f"{name}.bias")
                        for name in (
                            "self_attn.q_proj",
                            "self_attn.k_proj",
                            "text_cross_attn.q_proj",
                            "vision_cross_attn.q_proj",
                        )
                    },
                },
            )
        g = DETECTOR + "detr_decoder."
        self.head = {
            name: self.host(g + f"{name}.weight", g + f"{name}.bias")
            for name in (
                "box_head.layer1",
                "box_head.layer2",
                "box_head.layer3",
                "presence_head.layer1",
                "presence_head.layer2",
                "presence_head.layer3",
                "ref_point_head.layer1",
                "ref_point_head.layer2",
                "box_rpb_embed_x.layer1",
                "box_rpb_embed_x.layer2",
                "box_rpb_embed_y.layer1",
                "box_rpb_embed_y.layer2",
                "output_layer_norm",
                "presence_layer_norm",
            )
        }
        self.head["query_embed"] = linear.from_bf16(
            self.weights[g + "query_embed.weight"],
        )
        self.head["reference_points"] = linear.from_bf16(
            self.weights[g + "reference_points.weight"],
        )
        self.head["presence_token"] = linear.from_bf16(
            self.weights[g + "presence_token.weight"],
        )
        c = DETECTOR + "dot_product_scoring."
        self.scoring = {
            name: self.host(c + f"{name}.weight", c + f"{name}.bias")
            for name in (
                "query_proj",
                "text_proj",
                "text_mlp.layer1",
                "text_mlp.layer2",
                "text_mlp_out_norm",
            )
        }
        m = DETECTOR + "mask_decoder."
        self.mask_head = {
            "q": self.affine(m + "prompt_cross_attn.q_proj"),
            "kv": self.affine(
                m + "prompt_cross_attn.k_proj",
                m + "prompt_cross_attn.v_proj",
            ),
            "o": self.affine(m + "prompt_cross_attn.o_proj"),
            "norm": self.norm(m + "prompt_cross_attn_norm"),
            "convs": [
                (
                    (
                        self.winograd(m + f"pixel_decoder.conv_layers.{i}"),
                        self.vector(m + f"pixel_decoder.conv_layers.{i}.bias"),
                    ),
                    self.norm(m + f"pixel_decoder.norms.{i}"),
                )
                for i in range(2)
            ],
            "instance": self.affine(m + "instance_projection"),
        }
        self.head.update(
            {
                f"mask_embedder.layers.{i}": self.host(
                    m + f"mask_embedder.layers.{i}.weight",
                    m + f"mask_embedder.layers.{i}.bias",
                )
                for i in range(3)
            },
        )
        self.upsample = {
            size: linear.raw(
                device,
                (
                    (np.arange(4 * size * size) // (2 * size) // 2) * size
                    + np.arange(4 * size * size) % (2 * size) // 2
                ).astype(np.uint32),
            )
            for size in (s.grid, 2 * s.grid)
        }
        self.unused = linear.raw(device, np.zeros(4, np.float32))
        self.zero_length = linear.raw(device, np.zeros(4, np.uint32))
        self.identity = linear.raw(device, np.arange(tokens, dtype=np.uint32))
        rows = self.queries + 1
        self.vision_length = linear.raw(
            device,
            np.array([tokens - rows, 0, 0, 0], np.uint32),
        )
        self.text_starts = {count: self.device.buffer(16) for count in (tokens, rows)}
        self.prompts = {}

    def norm(self, name):
        return self.vector(name + ".weight"), self.vector(name + ".bias")

    def host(self, weight, bias):
        return linear.from_bf16(self.weights[weight]), linear.from_bf16(
            self.weights[bias],
        )

    def affine(self, *names, positioned=None):
        """Stacked weights and biases of the named layers; `positioned` folds position
        products into a table."""
        weight = self.matrix(*names)
        bias = np.concatenate(
            [linear.from_bf16(self.weights[f"{name}.bias"]) for name in names],
        )
        if positioned is None:
            return weight, self.vector(*(f"{name}.bias" for name in names)), None
        position, parts = positioned
        rows = [
            position @ linear.from_bf16(self.weights[f"{name}.weight"]).T
            for name in names[:parts]
        ]
        rows += [
            np.zeros(
                (len(position), self.weights[f"{name}.weight"].shape[0]),
                np.float32,
            )
            for name in names[parts:]
        ]
        return (
            weight,
            None,
            linear.raw(self.device, linear.to_bf16(np.concatenate(rows, 1) + bias)),
        )

    def product(self, x, layer, output, rows, activation=None, table=None):
        """x · Wᵀ plus a bias or position table, then `activation`."""
        weight, bias, positioned = layer
        if table is not None or positioned is not None:
            parts, operands = (
                [epilogue.residual(cl.bf16, weight.outputs)],
                (table or positioned,),
            )
        else:
            parts, operands = [epilogue.bias(cl.bf16)], (bias,)
        if activation is not None:
            parts += [epilogue.rounded(cl.bf16), activation]
        return linear.product(
            self.device,
            x,
            weight,
            output,
            rows,
            epilogue=epilogue.chain(*parts),
            operands=operands,
        )

    def split(self, fused, rows, heads, width, queries, at):
        """Fused rows of `width` columns, queries at 0 and keys and values from `at`, as
        queries and pages."""
        head_dim = (width - at) // 2 // heads
        keys, values = self.pages
        return [
            rope.rope_cache(
                self.device,
                fused,
                None,
                None,
                self.unused,
                self.unused,
                self.table,
                self.zero_length,
                queries,
                keys,
                values,
                rows=rows,
                heads=heads,
                kv_heads=heads,
                head_dim=head_dim,
                page_size=PAGE_SIZE,
                eps=1e-6,
                row_stride=width,
                keys_at=at,
                rotary=0,
            ),
        ]

    def attend(
        self,
        q,
        output,
        rows,
        heads,
        head_dim,
        *,
        keys=None,
        causal=False,
        text=False,
        **bias,
    ):
        """Attention of `rows` queries over `keys` positions, or with `text` over the
        text keys."""
        pages_k, pages_v = self.pages
        shape = dict(
            rows=rows,
            heads=heads,
            kv_heads=heads,
            head_dim=head_dim,
            page_size=PAGE_SIZE,
        )
        if text:
            lengths = self.text_starts[rows]
        else:
            lengths = self.vision_length if keys and keys > rows else self.zero_length
        return attention.paged_attention(
            self.device,
            q,
            pages_k,
            pages_v,
            self.table,
            lengths,
            output,
            causal=causal,
            context=keys or rows,
            **shape,
            **bias,
        )

    def prompt_buffers(self, count):
        s, d = self.spec, self.d
        tokens = s.grid * s.grid
        sizes = {
            "ids": count * 2,
            "text_x": count * self.text_width,
            "text_normed": count * self.text_width,
            "text_qkv": count * 3 * self.text_width,
            "text_q": count * self.text_width,
            "text_attended": count * self.text_width,
            "text_update": count * self.text_width,
            "text_hidden": count * 4 * self.text_width,
            "text": count * d,
            "kv": tokens * 2 * d,
            "x": tokens * d,
            "normed": tokens * d,
            "qkv": tokens * 3 * d,
            "q": tokens * d,
            "attended": tokens * d,
            "update": tokens * d,
            "hidden": tokens * 2048 // 256 * d,
            "h": (self.queries + 1) * d,
            "h2": (self.queries + 1) * d,
            "self_table": (self.queries + 1) * 3 * d,
            "text_table": (self.queries + 1) * d,
            "vision_table": (self.queries + 1) * d,
            "box_bias": (self.queries + 1) * s.grid * self.detr_heads * 2,
            "embeddings": self.queries * d,
            "masks": self.queries * (4 * s.grid) ** 2,
        }
        return {name: self.buffer(size) for name, size in sizes.items()}

    def text_steps(self, count, b):
        """CLIP over `count` tokens, then the projection to the DETR width into
        b["text"]."""
        device, width, heads = self.device, self.text_width, self.text_heads
        steps = indexing.gather_rows(
            device,
            self.token_table,
            b["ids"],
            b["text_x"],
            rows=count,
            columns=width,
        )
        steps += normalization.residual_norm(
            device,
            b["text_x"],
            self.text_positions,
            rows=count,
            columns=width,
            epsilon=self.text_eps,
            weight=self.text_layers[0]["norm1"][0],
            bias=self.text_layers[0]["norm1"][1],
            output=b["text_normed"],
        )
        for index, w in enumerate(self.text_layers):
            steps += self.product(b["text_normed"], w["qkv"], b["text_qkv"], count)
            steps += self.split(
                b["text_qkv"],
                count,
                heads,
                3 * width,
                b["text_q"],
                width,
            )
            steps += self.attend(
                b["text_q"],
                b["text_attended"],
                count,
                heads,
                width // heads,
                causal=True,
            )
            steps += self.product(b["text_attended"], w["o"], b["text_update"], count)
            steps += normalization.residual_norm(
                device,
                b["text_x"],
                b["text_update"],
                rows=count,
                columns=width,
                epsilon=self.text_eps,
                weight=w["norm2"][0],
                bias=w["norm2"][1],
                output=b["text_normed"],
            )
            steps += self.product(
                b["text_normed"],
                w["fc1"],
                b["text_hidden"],
                count,
                epilogue.gelu,
            )
            steps += self.product(b["text_hidden"], w["fc2"], b["text_update"], count)
            following = (
                self.text_layers[index + 1]["norm1"]
                if index + 1 < len(self.text_layers)
                else self.text_final
            )
            steps += normalization.residual_norm(
                device,
                b["text_x"],
                b["text_update"],
                rows=count,
                columns=width,
                epsilon=self.text_eps,
                weight=following[0],
                bias=following[1],
                output=b["text_normed"],
            )
        return steps + self.product(
            b["text_normed"],
            self.text_projection,
            b["text"],
            count,
        )

    def cross_text(self, normed, layer_q, layer_kv, rows, count, b):
        """Cross-attention of `normed` rows to the text."""
        d, heads = self.d, self.detr_heads
        steps = self.product(b["text"], layer_kv, b["kv"], count)
        steps += self.split(b["kv"], count, heads, 2 * d, b["attended"], 0)
        steps += self.product(normed, layer_q, b["q"], rows, table=None)
        return steps + self.attend(
            b["q"],
            b["attended"],
            rows,
            heads,
            d // heads,
            keys=count,
            text=True,
        )

    def encoder_steps(self, count, b, levels):
        """The DETR encoder over the 72 x 72 level (raster rows) and the text, into
        b["x"]."""
        s, device, d, heads = self.spec, self.device, self.d, self.detr_heads
        tokens = s.grid * s.grid
        norm = dict(rows=tokens, columns=d, epsilon=self.detr_eps)
        steps = indexing.gather_rows(
            device,
            levels[s.scales.index(1.0)],
            self.identity,
            b["x"],
            rows=tokens,
            columns=d,
        )
        steps += normalization.layer_norm(
            device,
            b["x"],
            *self.encoder_layers[0]["norm1"],
            b["normed"],
            **norm,
        )
        for index, w in enumerate(self.encoder_layers):
            steps += self.product(b["normed"], w["qkv"], b["qkv"], tokens)
            steps += self.split(b["qkv"], tokens, heads, 3 * d, b["q"], d)
            steps += self.attend(b["q"], b["attended"], tokens, heads, d // heads)
            steps += self.product(b["attended"], w["o"], b["update"], tokens)
            steps += normalization.residual_norm(
                device,
                b["x"],
                b["update"],
                weight=w["norm2"][0],
                bias=w["norm2"][1],
                output=b["normed"],
                **norm,
            )
            steps += self.cross_text(
                b["normed"],
                w["cross_q"],
                w["cross_kv"],
                tokens,
                count,
                b,
            )
            steps += self.product(b["attended"], w["cross_o"], b["update"], tokens)
            steps += normalization.residual_norm(
                device,
                b["x"],
                b["update"],
                weight=w["norm3"][0],
                bias=w["norm3"][1],
                output=b["normed"],
                **norm,
            )
            steps += self.product(
                b["normed"],
                w["fc1"],
                b["hidden"],
                tokens,
                epilogue.relu,
            )
            steps += self.product(b["hidden"], w["fc2"], b["update"], tokens)
            if index + 1 < len(self.encoder_layers):
                following = self.encoder_layers[index + 1]["norm1"]
                steps += normalization.residual_norm(
                    device,
                    b["x"],
                    b["update"],
                    weight=following[0],
                    bias=following[1],
                    output=b["normed"],
                    **norm,
                )
            else:
                steps += normalization.residual_norm(
                    device,
                    b["x"],
                    b["update"],
                    **norm,
                )
        return steps

    def decoder_steps(self, layer, count, b):
        """One DETR decoder layer."""
        s, device, d, heads, w = (
            self.spec,
            self.device,
            self.d,
            self.detr_heads,
            self.decoder_layers[layer],
        )
        rows, tokens = self.queries + 1, s.grid * s.grid
        norm = dict(rows=rows, columns=d, epsilon=self.detr_eps)
        steps = self.product(b["h"], w["self"], b["qkv"], rows, table=b["self_table"])
        steps += self.split(b["qkv"], rows, heads, 3 * d, b["q"], d)
        steps += self.attend(b["q"], b["attended"], rows, heads, d // heads)
        steps += self.product(b["attended"], w["self_o"], b["update"], rows)
        steps += normalization.residual_norm(
            device,
            b["h"],
            b["update"],
            weight=w["self_norm"][0],
            bias=w["self_norm"][1],
            output=b["h2"],
            **norm,
        )
        steps += self.product(b["text"], w["text_kv"], b["kv"], count)
        steps += self.split(b["kv"], count, heads, 2 * d, b["attended"], 0)
        steps += self.product(b["h2"], w["text_q"], b["q"], rows, table=b["text_table"])
        steps += self.attend(
            b["q"],
            b["attended"],
            rows,
            heads,
            d // heads,
            keys=count,
            text=True,
        )
        steps += self.product(b["attended"], w["text_o"], b["update"], rows)
        steps += normalization.residual_norm(
            device,
            b["h2"],
            b["update"],
            weight=w["text_norm"][0],
            bias=w["text_norm"][1],
            output=b["h"],
            **norm,
        )
        steps += self.product(b["x"], w["vision_kv"], b["kv"], tokens)
        steps += self.split(b["kv"], tokens, heads, 2 * d, b["attended"], 0)
        steps += self.product(
            b["h"],
            w["vision_q"],
            b["q"],
            rows,
            table=b["vision_table"],
        )
        box = ScoreMod(
            _BoxBias(s.grid, s.grid, heads, rows * s.grid * heads),
            (cl.bf16,),
        )
        steps += self.attend(
            b["q"],
            b["attended"],
            rows,
            heads,
            d // heads,
            keys=tokens,
            score_mod=box,
            score_operands=(b["box_bias"],),
        )
        steps += self.product(b["attended"], w["vision_o"], b["update"], rows)
        steps += normalization.residual_norm(
            device,
            b["h"],
            b["update"],
            weight=w["vision_norm"][0],
            bias=w["vision_norm"][1],
            output=b["h2"],
            **norm,
        )
        steps += self.product(b["h2"], w["fc1"], b["hidden"], rows, epilogue.relu)
        steps += self.product(b["hidden"], w["fc2"], b["update"], rows)
        return steps + normalization.residual_norm(
            device,
            b["h2"],
            b["update"],
            weight=w["mlp_norm"][0],
            bias=w["mlp_norm"][1],
            output=b["h"],
            **norm,
        )

    def mask_steps(self, count, b):
        """The mask head steps into b["masks"]."""
        s, device, d, w = self.spec, self.device, self.d, self.mask_head
        vision = self.program[1]
        tokens = s.grid * s.grid
        steps = normalization.layer_norm(
            device,
            b["x"],
            *w["norm"],
            b["normed"],
            rows=tokens,
            columns=d,
            epsilon=self.detr_eps,
        )
        steps += self.cross_text(b["normed"], w["q"], w["kv"], tokens, count, b)
        steps += self.product(b["attended"], w["o"], b["update"], tokens)
        steps += normalization.residual_norm(
            device,
            b["x"],
            b["update"],
            rows=tokens,
            columns=d,
            epsilon=self.detr_eps,
        )
        source, size = b["x"], s.grid
        for (conv, (weight, bias)), level in zip(w["convs"], (1, 0)):
            pixels = 4 * size * size
            steps += indexing.gather_rows(
                device,
                source,
                self.upsample[size],
                vision["wide"],
                rows=pixels,
                columns=d,
            )
            steps += normalization.residual_norm(
                device,
                vision["wide"],
                vision["levels"][level],
                rows=pixels,
                columns=d,
                epsilon=self.detr_eps,
            )
            steps += self.conv3(
                vision["wide"],
                conv[0],
                conv[1],
                vision["narrow"],
                2 * size,
            )
            steps += normalization.group_norm(
                device,
                vision["narrow"],
                weight,
                bias,
                vision["projected"],
                pixels=pixels,
                channels=d,
                groups=8,
                epsilon=1e-5,
                epilogue=epilogue.relu,
            )
            source, size = vision["projected"], 2 * size
        steps += self.product(source, w["instance"], vision["wide"], size * size)
        pixels = linear.DenseWeight(vision["wide"], size * size, d)
        return steps + linear.product(
            device,
            b["embeddings"],
            pixels,
            b["masks"],
            self.queries,
        )

    def prompt(self, count):
        """The programs for a prompt of `count` tokens."""
        if count not in self.prompts:
            b = self.prompt_buffers(count)
            levels = self.program[1]["levels"]
            programs = []
            for steps in (
                [self.text_steps(count, b) + self.encoder_steps(count, b, levels)]
                + [
                    self.decoder_steps(layer, count, b)
                    for layer in range(len(self.decoder_layers))
                ]
                + [self.mask_steps(count, b)]
            ):
                program = Program(self.device)
                for step in steps:
                    program.kernel(step)
                program.finish()
                programs.append(program)
            self.prompts[count] = programs, b
        return self.prompts[count]

    def mlp(self, x, *layers, final=False):
        x = np.asarray(x, np.float32)
        for index, name in enumerate(layers):
            weight, bias = self.head[name]
            x = x @ weight.T + bias
            if index + 1 < len(layers) or final:
                x = np.maximum(x, 0)
        return x

    def box_bias(self, boxes):
        """Axial box bias tables of the decoder's boxes."""
        grid = self.spec.grid
        coordinates = np.arange(grid, dtype=np.float32) / grid
        tables = []
        for axis, centre, size in (("x", 0, 2), ("y", 1, 3)):
            bounds = np.stack(
                [
                    boxes[:, centre] - boxes[:, size] / 2,
                    boxes[:, centre] + boxes[:, size] / 2,
                ],
                1,
            )
            part = self.mlp(
                log_deltas(coordinates, bounds).reshape(-1, 2),
                f"box_rpb_embed_{axis}.layer1",
                f"box_rpb_embed_{axis}.layer2",
            ).reshape(len(boxes), grid, -1)
            tables.append(
                np.concatenate([np.zeros((1, grid, part.shape[2]), np.float32), part]),
            )
        return np.concatenate([table.reshape(-1) for table in tables])

    def detect(self, tokens, gpu=None):
        """(logits, boxes, presence) of the queries for a token prompt on the last
        image."""
        tokens = list(tokens)[: self.text_positions.size // 2 // self.text_width]
        count = self.last = len(tokens)
        programs, b = self.prompt(count)
        write = lambda buffer, array: buffer.write(
            array.tobytes() + bytes(buffer.size - array.nbytes),
        )
        write(b["ids"], np.asarray(tokens, np.uint32))
        for rows, buffer in self.text_starts.items():
            write(buffer, np.array([(count - rows) % 2**32, 0, 0, 0], np.uint32))
        total = programs[0].run().gpu_ms
        boxes = sigmoid(self.head["reference_points"])
        rows = len(boxes) + 1
        d = self.d
        for layer, program in enumerate(programs[1:-1]):
            if layer == 0:
                hidden = np.concatenate(
                    [self.head["presence_token"], self.head["query_embed"]],
                )
                write(b["h"], linear.to_bf16(hidden))
            position = np.concatenate(
                [
                    np.zeros((1, d), np.float32),
                    self.mlp(
                        sine_boxes(boxes, d // 2),
                        "ref_point_head.layer1",
                        "ref_point_head.layer2",
                    ),
                ],
            )
            host = self.decoder_layers[layer]["host"]
            v_bias = linear.from_bf16(
                np.frombuffer(self.decoder_layers[layer]["self"][1].read(), np.uint16)[
                    2 * d : 3 * d
                ],
            )
            terms = [
                position @ host[name][0].T + host[name][1]
                for name in ("self_attn.q_proj", "self_attn.k_proj")
            ]
            write(
                b["self_table"],
                linear.to_bf16(np.concatenate(terms + [np.tile(v_bias, (rows, 1))], 1)),
            )
            for name, buffer in (
                ("text_cross_attn.q_proj", "text_table"),
                ("vision_cross_attn.q_proj", "vision_table"),
            ):
                write(
                    b[buffer],
                    linear.to_bf16(position @ host[name][0].T + host[name][1]),
                )
            write(b["box_bias"], linear.to_bf16(self.box_bias(boxes)))
            total += program.run().gpu_ms
            hidden = linear.from_bf16(
                np.frombuffer(b["h"].read(), np.uint16)[: rows * d],
            ).reshape(rows, d)
            normed = layer_norm(
                hidden[1:],
                *self.head["output_layer_norm"],
                self.detr_eps,
            )
            delta = self.mlp(
                normed,
                "box_head.layer1",
                "box_head.layer2",
                "box_head.layer3",
            )
            clipped = np.clip(boxes, 1e-5, 1 - 1e-5)
            boxes = sigmoid(np.log(clipped / (1 - clipped)) + delta)
        presence = layer_norm(
            hidden[:1],
            *self.head["presence_layer_norm"],
            self.detr_eps,
        )
        presence = float(
            np.clip(
                self.mlp(
                    presence,
                    "presence_head.layer1",
                    "presence_head.layer2",
                    "presence_head.layer3",
                ),
                -10,
                10,
            )[0, 0],
        )
        embedding = self.mlp(
            normed,
            "mask_embedder.layers.0",
            "mask_embedder.layers.1",
            "mask_embedder.layers.2",
        )
        write(b["embeddings"], linear.to_bf16(embedding))
        total += programs[-1].run().gpu_ms
        if gpu is not None:
            gpu.append(total)
        corners = np.concatenate(
            [boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2],
            1,
        )
        return self.scores(normed, count, b), corners, presence

    def masks(self, queries):
        """Mask logits of the given queries from the last `detect`."""
        b = self.prompts[self.last][1]
        size = 4 * self.spec.grid
        logits = linear.from_bf16(
            np.frombuffer(b["masks"].read(), np.uint16)[: self.queries * size * size],
        )
        return logits.reshape(self.queries, size, size)[list(queries)]

    def scores(self, queries, count, b):
        text = linear.from_bf16(
            np.frombuffer(b["text"].read(), np.uint16)[: count * self.d],
        ).reshape(count, self.d)
        w = self.scoring
        mixed = np.maximum(
            text @ w["text_mlp.layer1"][0].T + w["text_mlp.layer1"][1],
            0,
        )
        mixed = mixed @ w["text_mlp.layer2"][0].T + w["text_mlp.layer2"][1] + text
        pooled = layer_norm(mixed, *w["text_mlp_out_norm"], 1e-5).mean(0)
        pooled = pooled @ w["text_proj"][0].T + w["text_proj"][1]
        projected = queries @ w["query_proj"][0].T + w["query_proj"][1]
        return np.clip(projected @ pooled / math.sqrt(self.d), -12, 12)
