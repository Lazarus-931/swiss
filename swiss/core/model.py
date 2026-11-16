"""Model interface, checkpoints, config parsing, runtime settings and registry."""

import importlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

MISSING = object()


class Unsupported(ValueError):
    """A checkpoint asks for something this model path does not implement."""


def take(config, key, kind=None, default=MISSING):
    """Read a config key, rejecting missing or malformed values at load time."""
    if key not in config or config[key] is None:
        if default is MISSING:
            raise Unsupported(f"config is missing required key '{key}'")
        return default
    value = config[key]
    if (
        kind is not None
        and not isinstance(value, kind)
        or isinstance(value, bool)
        and kind in (int, float)
    ):
        raise Unsupported(
            f"config key '{key}' must be "
            f"{kind.__name__ if isinstance(kind, type) else kind}",
        )
    return value


def reject(config, keys, reason="is not implemented"):
    """Reject checkpoint features the model path does not implement."""
    for key in keys:
        if config.get(key) not in (None, False, 0, [], {}):
            raise Unsupported(f"config key '{key}'={config[key]!r} {reason}")


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        if not (self.path / "config.json").is_file():
            raise FileNotFoundError(f"{self.path} has no config.json")

    @cached_property
    def config(self):
        return json.loads((self.path / "config.json").read_text())

    @cached_property
    def generation_config(self):
        path = self.path / "generation_config.json"
        return json.loads(path.read_text()) if path.is_file() else {}

    @property
    def model_type(self):
        return self.config.get("model_type")

    @property
    def name(self):
        return self.path.resolve().name

    @property
    def weight_files(self):
        return sorted(self.path.glob("*.safetensors"))

    def tensors(self):
        """Memory-mapped safetensors arrays by name."""
        from .weights import read_safetensors

        tensors = {}
        for path in self.weight_files:
            tensors.update(read_safetensors(path))
        return tensors

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.path,
            local_files_only=True,
            trust_remote_code=False,
        )

    def eos_token_ids(self):
        tokens = set()
        for source in (self.generation_config, self.config):
            value = source.get("eos_token_id")
            tokens.update(
                value if isinstance(value, list) else [] if value is None else [value],
            )
        if self.tokenizer.eos_token_id is not None:
            tokens.add(self.tokenizer.eos_token_id)
        return tokens


PAGE_SIZE = 16
PREFILL_CHUNKS = (2048, 512, 128, 32, 8, 2, 1)


@dataclass(frozen=True)
class RuntimeConfig:
    """Per-sequence context capacity, sequence count and maximum draft length."""

    capacity: int = 4096
    max_sequences: int = 1
    draft: int = 0

    def __post_init__(self):
        if type(self.capacity) is not int or self.capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if type(self.max_sequences) is not int or self.max_sequences < 1:
            raise ValueError("max_sequences must be a positive integer")
        if type(self.draft) is not int or not 0 <= self.draft <= 16:
            raise ValueError("draft must be an integer from 0 to 16")


@dataclass(frozen=True)
class Output:
    """Last row's greedy token, optional f32 logits and GPU time in milliseconds."""

    token: int
    logits: object = None
    gpu_ms: float | None = None


class Model:
    """Checkpoint-backed model; generating models keep their sequences in `run`."""

    checkpoint = None

    def __init__(self, spec, runtime):
        self.spec, self.runtime = spec, runtime

    @property
    def capacity(self):
        return self.runtime.capacity

    @property
    def sequences(self):
        return self.runtime.max_sequences

    @property
    def tokenizer(self):
        return self.checkpoint.tokenizer


@dataclass(frozen=True)
class Architecture:
    model_type: str
    parse: object
    model: type


_ARCHITECTURES = {}


def register(model_type, *, parse):
    """Register a model class and its config parser for `model_type`."""

    def decorate(model):
        if (
            model_type in _ARCHITECTURES
            and _ARCHITECTURES[model_type].model is not model
        ):
            raise ValueError(f"model_type '{model_type}' is already registered")
        _ARCHITECTURES[model_type] = Architecture(model_type, parse, model)
        return model

    return decorate


def architectures():
    importlib.import_module("swiss.models")
    importlib.import_module("swiss.speculative")
    return dict(_ARCHITECTURES)


def lookup(config):
    """Look up drafters by architecture name, then fall back to `model_type`."""
    available = architectures()
    for name in config.get("architectures") or ():
        if name in available and getattr(available[name].model, "drafts", False):
            return available[name]
    model_type = config.get("model_type")
    if model_type not in available:
        raise Unsupported(
            f"model_type '{model_type}' is not supported; "
            f"available: {', '.join(sorted(available)) or 'none'}",
        )
    return available[model_type]


def load(path, runtime=None):
    """Parse a checkpoint into its spec and build its model path."""
    checkpoint = Checkpoint(path)
    architecture = lookup(checkpoint.config)
    if getattr(architecture.model, "drafts", False):
        raise Unsupported(
            f"'{checkpoint.model_type}' is a drafter; attach it to a model",
        )
    spec = architecture.parse(checkpoint.config)
    model = architecture.model.from_checkpoint(
        spec,
        runtime or RuntimeConfig(),
        checkpoint,
    )
    model.checkpoint = checkpoint
    return model, checkpoint


def attach(model, path, bits=None):
    """Attach the drafter checkpoint at `path` to `model`; `bits` quantizes it."""
    checkpoint = Checkpoint(path)
    architecture = lookup(checkpoint.config)
    if not getattr(architecture.model, "drafts", False):
        raise Unsupported(f"'{checkpoint.model_type}' is not a supported drafter")
    spec = architecture.parse(checkpoint.config)
    return architecture.model.attach(spec, checkpoint, model, bits)
