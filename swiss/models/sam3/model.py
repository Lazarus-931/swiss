"""SAM 3 vision encoder: ViT backbone and FPN neck."""

from dataclasses import dataclass

import numpy as np
from carbon import language as cl
from carbon.kernels import attention
from carbon.kernels.gemm import epilogue
from carbon.kernels.ops import indexing

from ...core.model import PAGE_SIZE, Model, Unsupported, take
from ...core.program import Program, allocate, open_device
from ...core.weights import Uploads
from ...kernels import convolution, linear, normalization, rope, spatial
from .detr import Detector

VISION = "detector_model.vision_encoder."


@dataclass(frozen=True)
class Sam3Spec:
    image: int
    patch: int
    pretrain: int
    width: int
    layers: int
    heads: int
    hidden: int
    window: int
    global_layers: tuple
    theta: float
    eps: float
    fpn: int
    scales: tuple

    @property
    def grid(self):
        return self.image // self.patch


def parse(config):
    """Map the vision encoder of a SAM 3 detector into Sam3Spec."""
    vision = take(take(config, "detector_config", dict), "vision_config", dict)
    vit = take(vision, "backbone_config", dict)
    if (
        take(vit, "hidden_act", str) != "gelu"
        or take(vision, "hidden_act", str, "gelu") != "gelu"
    ):
        raise Unsupported("only the erf GELU is implemented")
    if take(vit, "layer_scale_init_value", (int, float), None) is not None:
        raise Unsupported("layer scale is not implemented")
    if (take(vision, "fpn_kernel_size", int), take(vision, "fpn_stride", int)) != (
        2,
        2,
    ):
        raise Unsupported(
            "only 2 x 2 transposed convolutions of stride 2 are implemented",
        )
    spec = Sam3Spec(
        image=take(vit, "image_size", int),
        patch=take(vit, "patch_size", int),
        pretrain=take(vit, "pretrain_image_size", int),
        width=take(vit, "hidden_size", int),
        layers=take(vit, "num_hidden_layers", int),
        heads=take(vit, "num_attention_heads", int),
        hidden=take(vit, "intermediate_size", int),
        window=take(vit, "window_size", int),
        global_layers=tuple(take(vit, "global_attn_indexes", list)),
        theta=float(take(vit, "rope_theta", (int, float))),
        eps=float(take(vit, "layer_norm_eps", (int, float))),
        fpn=take(vision, "fpn_hidden_size", int),
        scales=tuple(float(scale) for scale in take(vision, "scale_factors", list)),
    )
    if (
        spec.grid % spec.window
        or spec.grid % (spec.pretrain // spec.patch)
        or set(spec.scales) - {4.0, 2.0, 1.0, 0.5}
    ):
        raise Unsupported(
            "the patch grid must hold whole windows and position tiles, at scales 4, "
            "2, 1 and 0.5",
        )
    return spec


def window_order(grid, window):
    """Raster index of each token in window order: windows in raster order, each
    window's tokens in raster order."""
    token = np.arange(grid * grid)
    per, cut = window * window, grid // window
    which, inside = token // per, token % per
    return (
        (which // cut * window + inside // window) * grid
        + which % cut * window
        + inside % window
    )


def axial_angles(grid, window, head_dim, theta):
    """(cos, sin) 2D axial RoPE tables in window order."""
    frequencies = 1.0 / theta ** (
        np.arange(0, head_dim, 4, dtype=np.float32) / head_dim
    )
    raster = window_order(grid, window)
    y, x = (raster // grid).astype(np.float32), (raster % grid).astype(np.float32)
    tables = {}
    for name, (ty, tx) in {
        "window": (y % window, x % window),
        "global": (y * (window / grid), x * (window / grid)),
    }.items():
        angles = np.concatenate(
            [tx[:, None] * frequencies, ty[:, None] * frequencies],
            1,
        ).astype(np.float32)
        tables[name] = (np.cos(angles), np.sin(angles))
    return tables


class Sam3(Detector, Uploads, Model):
    """`features(pixels)`: the four FPN levels of a preprocessed (image, image, 3)
    image; then `detect(tokens)`."""

    def __init__(self, spec, runtime, weights, config, device=None):
        super().__init__(spec, runtime)
        self.device = device or open_device()
        self.weights, s = weights, spec
        tokens, raster = s.grid * s.grid, window_order(s.grid, s.window)
        self.order = linear.raw(self.device, raster.astype(np.uint32))
        self.restore = linear.raw(self.device, np.argsort(raster).astype(np.uint32))
        b = VISION + "backbone."
        tile = s.pretrain // s.patch
        position = self.weights[b + "embeddings.position_embeddings"].reshape(
            tile,
            tile,
            s.width,
        )
        self.position = linear.raw(
            self.device,
            np.tile(position, (s.grid // tile, s.grid // tile, 1)),
        )

        patch = linear.from_bf16(
            self.weights[b + "embeddings.patch_embeddings.projection.weight"],
        )
        self.patch = convolution.conv_weights(
            self.device,
            patch,
            kernel=s.patch,
            stride=s.patch,
        )
        self.pre_norm = (
            self.vector(b + "layer_norm.weight"),
            self.vector(b + "layer_norm.bias"),
        )
        self.blocks = [self.block(f"{b}layers.{layer}.") for layer in range(s.layers)]
        self.angles = {
            name: tuple(linear.raw(self.device, table) for table in tables)
            for name, tables in axial_angles(
                s.grid,
                s.window,
                s.width // s.heads,
                s.theta,
            ).items()
        }
        self.windows = linear.raw(
            self.device,
            np.arange(0, tokens + 1, s.window * s.window, dtype=np.uint32),
        )
        pages = -(-tokens // PAGE_SIZE)
        self.pages = tuple(
            linear.raw(self.device, np.zeros(pages * s.width * PAGE_SIZE * 2, np.uint8))
            for _ in range(2)
        )
        self.table = linear.raw(self.device, np.arange(pages, dtype=np.uint32))
        self.lengths = linear.raw(self.device, np.zeros(4, np.uint32))
        self.levels = [self.level(index, scale) for index, scale in enumerate(s.scales)]
        self.program = None
        self.load_detector(config)
        del self.weights

    @classmethod
    def from_checkpoint(cls, spec, runtime, checkpoint, device=None):
        return cls(spec, runtime, checkpoint.tensors(), checkpoint.config, device)

    def block(self, p):
        a = p + "attention."
        return {
            "norm1": (
                self.vector(p + "layer_norm1.weight"),
                self.vector(p + "layer_norm1.bias"),
            ),
            "norm2": (
                self.vector(p + "layer_norm2.weight"),
                self.vector(p + "layer_norm2.bias"),
            ),
            "qkv": self.matrix(a + "q_proj", a + "k_proj", a + "v_proj"),
            "qkv_bias": self.vector(*(a + f"{name}_proj.bias" for name in "qkv")),
            "o_proj": self.matrix(a + "o_proj"),
            "o_bias": self.vector(a + "o_proj.bias"),
            "fc1": self.matrix(p + "mlp.fc1"),
            "fc1_bias": self.vector(p + "mlp.fc1.bias"),
            "fc2": self.matrix(p + "mlp.fc2"),
            "fc2_bias": self.vector(p + "mlp.fc2.bias"),
        }

    def transposed(self, name):
        """A transposed convolution's (in, out, kh, kw) weight as (out · kh · kw, in)
        rows."""
        weight = self.tensor(name + ".weight")
        rows = weight.transpose(1, 2, 0, 3).reshape(-1, weight.shape[-1])
        return linear.DenseWeight.upload(self.device, rows)

    def level(self, index, scale):
        p = f"{VISION}neck.fpn_layers.{index}."
        level = {
            "scale": scale,
            "proj1": self.matrix(p + "proj1"),
            "proj1_bias": self.vector(p + "proj1.bias"),
            "proj2": self.winograd(p + "proj2"),
            "proj2_bias": self.vector(p + "proj2.bias"),
        }
        level["up"] = [
            (
                self.transposed(p + f"scale_layers.{i}"),
                self.vector(p + f"scale_layers.{i}.bias"),
            )
            for i in ((0, 2) if scale == 4.0 else (0,) if scale == 2.0 else ())
        ]
        return level

    def buffer(self, elements):
        """A device buffer of `elements` bf16 values."""
        return allocate(self.device, {"buffer": elements * 2})["buffer"]

    def buffers(self):
        s = self.spec
        tokens = s.grid * s.grid
        biggest = (s.grid * 4) ** 2
        b = {
            "pixels": self.buffer(s.image * s.image * 3),
            "x": self.buffer(tokens * s.width),
            "normed": self.buffer(tokens * s.width),
            "qkv": self.buffer(tokens * 3 * s.width),
            "q": self.buffer(tokens * s.width),
            "attended": self.buffer(tokens * s.width),
            "update": self.buffer(tokens * s.width),
            "hidden": self.buffer(tokens * s.hidden),
            "features": self.buffer(tokens * s.width),
            "wide": self.buffer(biggest * s.width // 2),
            "narrow": self.buffer(biggest * s.width // 2),
            "projected": self.buffer(biggest * s.fpn),
        }
        b["levels"] = [
            self.buffer(int(s.grid * scale) ** 2 * s.fpn) for scale in s.scales
        ]
        return b

    def bias(self):
        return epilogue.bias(cl.bf16)

    def block_steps(self, layer, b):
        s, device, w = self.spec, self.device, self.blocks[layer]
        tokens, head_dim = s.grid * s.grid, s.width // s.heads
        shape = dict(rows=tokens, heads=s.heads, kv_heads=s.heads, head_dim=head_dim)
        windowed = layer not in s.global_layers
        cos, sin = self.angles["window" if windowed else "global"]
        keys, values = self.pages
        segments = (
            dict(
                segments=self.windows,
                sequences=(s.grid // s.window) ** 2,
                longest=s.window * s.window,
            )
            if windowed
            else {}
        )
        steps = linear.product(
            device,
            b["normed"],
            w["qkv"],
            b["qkv"],
            tokens,
            epilogue=self.bias(),
            operands=(w["qkv_bias"],),
        )
        steps += [
            rope.rope_cache(
                device,
                b["qkv"],
                None,
                None,
                cos,
                sin,
                self.table,
                self.lengths,
                b["q"],
                keys,
                values,
                page_size=PAGE_SIZE,
                eps=s.eps,
                pairwise=True,
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
            context=tokens,
            causal=False,
            **segments,
            **shape,
        )
        steps += linear.product(
            device,
            b["attended"],
            w["o_proj"],
            b["update"],
            tokens,
            epilogue=self.bias(),
            operands=(w["o_bias"],),
        )
        steps += normalization.residual_norm(
            device,
            b["x"],
            b["update"],
            rows=tokens,
            columns=s.width,
            epsilon=s.eps,
            weight=w["norm2"][0],
            bias=w["norm2"][1],
            output=b["normed"],
        )
        steps += linear.product(
            device,
            b["normed"],
            w["fc1"],
            b["hidden"],
            tokens,
            epilogue=epilogue.chain(
                self.bias(),
                epilogue.rounded(cl.bf16),
                epilogue.gelu,
            ),
            operands=(w["fc1_bias"],),
        )
        steps += linear.product(
            device,
            b["hidden"],
            w["fc2"],
            b["update"],
            tokens,
            epilogue=self.bias(),
            operands=(w["fc2_bias"],),
        )
        following = (
            self.blocks[layer + 1]["norm1"] if layer + 1 < s.layers else (None, None)
        )
        return steps + normalization.residual_norm(
            device,
            b["x"],
            b["update"],
            rows=tokens,
            columns=s.width,
            epsilon=s.eps,
            weight=following[0],
            bias=following[1],
            output=b["normed"] if following[0] is not None else None,
        )

    def backbone_steps(self, b):
        s, device = self.spec, self.device
        tokens = s.grid * s.grid
        steps = convolution.conv2d(
            device,
            b["pixels"],
            self.patch,
            b["update"],
            height=s.image,
            width=s.image,
            epilogue=epilogue.chain(
                epilogue.rounded(cl.bf16),
                epilogue.residual(cl.bf16, s.width),
            ),
            operands=(self.position,),
        )
        steps += indexing.gather_rows(
            device,
            b["update"],
            self.order,
            b["normed"],
            rows=tokens,
            columns=s.width,
        )
        steps += normalization.layer_norm(
            device,
            b["normed"],
            *self.pre_norm,
            b["x"],
            rows=tokens,
            columns=s.width,
            epsilon=s.eps,
        )
        steps += normalization.layer_norm(
            device,
            b["x"],
            *self.blocks[0]["norm1"],
            b["normed"],
            rows=tokens,
            columns=s.width,
            epsilon=s.eps,
        )
        for layer in range(s.layers):
            steps += self.block_steps(layer, b)
        return steps + indexing.gather_rows(
            device,
            b["x"],
            self.restore,
            b["features"],
            rows=tokens,
            columns=s.width,
        )

    def winograd(self, name):
        """A 3 x 3 convolution's (filters, 3, 3, channels) weight on the device, as
        conv2d reads it."""
        return convolution.conv_weights(
            self.device,
            linear.from_bf16(self.weights[f"{name}.weight"]),
            kernel=3,
            stride=1,
            padding=1,
        )

    def conv3(self, source, weight, bias, output, size):
        """A 3 x 3 convolution with padding 1 and a bias over a (size, size, fpn)
        image."""
        return convolution.conv2d(
            self.device,
            source,
            weight,
            output,
            height=size,
            width=size,
            bias=bias,
        )

    def neck_steps(self, b):
        s, device, steps = self.spec, self.device, []
        for level, output in zip(self.levels, b["levels"]):
            source, size, channels = b["features"], s.grid, s.width
            for index, (weight, bias) in enumerate(level["up"]):
                target = b["narrow"] if index + 1 < len(level["up"]) else b["wide"]
                after = (
                    {"epilogue": epilogue.gelu} if index + 1 < len(level["up"]) else {}
                )
                steps += convolution.conv_transpose(
                    device,
                    source,
                    weight.buffer,
                    target,
                    height=size,
                    width=size,
                    channels=channels,
                    filters=channels // 2,
                    bias=bias,
                    **after,
                )
                source, size, channels = target, size * 2, channels // 2
            if level["scale"] == 0.5:
                steps += spatial.max_pool(
                    device,
                    source,
                    b["wide"],
                    height=size,
                    width=size,
                    channels=channels,
                )
                source, size = b["wide"], size // 2
            steps += linear.product(
                device,
                source,
                level["proj1"],
                b["projected"],
                size * size,
                epilogue=self.bias(),
                operands=(level["proj1_bias"],),
            )
            steps += self.conv3(
                b["projected"],
                level["proj2"],
                level["proj2_bias"],
                output,
                size,
            )
        return steps

    def features(self, pixels, gpu=None):
        """The FPN levels of a preprocessed image; `gpu`, a list, gets the GPU ms."""
        s = self.spec
        if self.program is None:
            b = self.buffers()
            program = Program(self.device)
            for step in self.backbone_steps(b) + self.neck_steps(b):
                program.kernel(step)
            program.finish()
            self.program = program, b
        program, b = self.program
        data = linear.to_bf16(np.ascontiguousarray(pixels, np.float32).reshape(-1))
        b["pixels"].write(data.tobytes() + bytes(b["pixels"].size - data.nbytes))
        completion = program.run()
        if gpu is not None:
            gpu.append(completion.gpu_ms)
        out = []
        for scale, level in zip(s.scales, b["levels"]):
            size = int(s.grid * scale)
            out.append(
                linear.from_bf16(
                    np.frombuffer(level.read(), np.uint16)[: size * size * s.fpn],
                ).reshape(size, size, s.fpn),
            )
        return out

    def backbone(self):
        """The last backbone output, (grid, grid, width) f32, in raster order."""
        s = self.spec
        raw = np.frombuffer(self.program[1]["features"].read(), np.uint16)[
            : s.grid * s.grid * s.width
        ]
        return linear.from_bf16(raw).reshape(s.grid, s.grid, s.width)

    def preprocess(self, image):
        """An image as the (image, image, 3) f32 input of the encoder."""
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image, np.uint8))
        size = self.spec.image
        resized = (
            np.asarray(
                image.convert("RGB").resize((size, size), Image.BILINEAR),
                np.float32,
            )
            / 255.0
        )
        return (resized - 0.5) / 0.5

    def predict(self, image, text, threshold=0.3):
        """(boxes, scores, masks) of the objects `text` names in `image`, best first."""
        from PIL import Image

        width, height = (
            image.size
            if isinstance(image, Image.Image)
            else np.asarray(image).shape[1::-1]
        )
        self.features(self.preprocess(image))
        logits, boxes, presence = self.detect(self.tokenizer(text)["input_ids"])
        scores = 1 / (1 + np.exp(-logits)) / (1 + np.exp(-presence))
        keep = np.argsort(-scores)[: int((scores > threshold).sum())]
        pixels = boxes[keep] * np.array([width, height, width, height], np.float32)
        masks = [
            np.asarray(Image.fromarray(mask).resize((width, height), Image.BILINEAR))
            > 0
            for mask in self.masks(keep)
        ]
        masks = np.stack(masks) if masks else np.zeros((0, height, width), bool)
        return np.clip(pixels, 0, max(width, height)), scores[keep], masks
