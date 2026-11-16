"""Checkpoint tensors: safetensors reading, quantization settings and weight uploads."""

import json
from functools import cached_property

import numpy as np

from ..kernels import linear
from .model import Unsupported, take

SAFETENSOR_TYPES = {
    "U32": np.uint32,
    "BF16": np.uint16,
    "F16": np.float16,
    "F32": np.float32,
    "I64": np.int64,
    "BOOL": np.bool_,
}


def read_safetensors(path):
    """Arrays by name, memory-mapped; BF16 comes back as its u16 bit patterns."""
    with open(path, "rb") as file:
        start = 8 + int.from_bytes(file.read(8), "little")
        header = json.loads(file.read(start - 8))
    header.pop("__metadata__", None)
    data = np.memmap(path, np.uint8, "r")
    tensors = {}
    for name, entry in header.items():
        if entry["dtype"] not in SAFETENSOR_TYPES:
            raise Unsupported(
                f"{name}: safetensors dtype {entry['dtype']} is not supported",
            )
        begin, end = entry["data_offsets"]
        tensors[name] = (
            data[start + begin : start + end]
            .view(SAFETENSOR_TYPES[entry["dtype"]])
            .reshape(entry["shape"])
        )
    return tensors


def quantization(config):
    """(bits, group, overrides) of an affine-quantized checkpoint, or None for bf16."""
    if config.get("quantization") is None:
        return None
    settings = take(config, "quantization", dict)
    layers = {name: v for name, v in settings.items() if isinstance(v, dict)}
    if any(v.get("mode", "affine") != "affine" for v in (settings, *layers.values())):
        raise Unsupported("only affine quantization is implemented")
    overrides = {
        name: (take(value, "bits", int), take(value, "group_size", int))
        for name, value in layers.items()
    }
    bits, group = take(settings, "bits", int), take(settings, "group_size", int)
    if {bits, *(b for b, _ in overrides.values())} - set(linear.WIDENS):
        raise Unsupported(
            f"only {', '.join(map(str, linear.WIDENS))}-bit codes are implemented",
        )
    return bits, group, overrides


class Uploads:
    """Upload `weights` under `prefix` to `device`.
    `requantize=(bits, group)` quantizes bf16 matrices."""

    prefix = ""
    requantize = None

    @classmethod
    def from_checkpoint(cls, spec, runtime, checkpoint, device=None):
        return cls(
            spec,
            runtime,
            checkpoint.tensors(),
            quantization(checkpoint.config),
            device,
        )

    @cached_property
    def scratch(self):
        return linear.scratch(
            self.device,
            [w for block in self.blocks for w in block.values()],
        )

    def tensor(self, name):
        return self.weights[self.prefix + name]

    def vector(self, *names):
        """The named tensors, concatenated, as one raw device buffer."""
        return linear.raw(
            self.device,
            np.concatenate([self.tensor(name).reshape(-1) for name in names]),
        )

    def matrix(self, *names, paired=False, prefix=None):
        """The named linear layers as one matrix, quantized or bf16 as the checkpoint
        stores them; bf16 `paired` rows interleave [gate | up], and quantized codes of
        mixed widths widen to 8 bits. With `requantize`, bf16 layers load quantized."""
        names = [(self.prefix if prefix is None else prefix) + name for name in names]
        if f"{names[0]}.scales" not in self.weights:
            matrix = np.concatenate([self.weights[f"{name}.weight"] for name in names])
            matrix = matrix.reshape(matrix.shape[0], -1)
            if self.requantize is None:
                return linear.DenseWeight.upload(self.device, matrix, paired)
            bits, group = self.requantize
            return linear.QuantizedWeight.upload(
                self.device,
                *linear.quantize(matrix, bits, group),
                bits=bits,
                group=group,
            )
        if self.quantization is None:
            raise Unsupported(
                f"{names[0]} is quantized, but the config has no quantization",
            )
        bits, group, overrides = self.quantization
        formats = [overrides.get(name, (bits, group)) for name in names]
        if len({group for _, group in formats}) > 1:
            raise Unsupported(f"{', '.join(names)} mix quantization group sizes")
        target = (
            formats[0][0]
            if len({b for b, _ in formats}) == 1 and formats[0][0] in (4, 8)
            else 8
        )
        packed = [
            linear.widen(self.weights[f"{name}.weight"], width, target)
            for name, (width, _) in zip(names, formats)
        ]
        scales, biases = (
            [self.weights[f"{name}.{kind}"] for name in names]
            for kind in ("scales", "biases")
        )
        return linear.QuantizedWeight.upload(
            self.device,
            np.concatenate(packed),
            np.concatenate(scales),
            np.concatenate(biases),
            bits=target,
            group=formats[0][1],
        )
