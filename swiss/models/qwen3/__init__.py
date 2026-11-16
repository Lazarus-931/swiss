"""Qwen3 dense models."""

from ...core.model import register
from .model import Qwen3, parse

register("qwen3", parse=parse)(Qwen3)
