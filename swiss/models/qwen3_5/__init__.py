"""Qwen3.5 hybrid models: gated delta rule layers and gated attention."""

from ...core.model import register
from .model import Qwen3_5, parse

register("qwen3_5", parse=parse)(Qwen3_5)
