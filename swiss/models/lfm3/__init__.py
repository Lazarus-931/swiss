"""Liquid AI LFM3 language models, alone or inside the VL, Audio and Omni
checkpoints."""

from ...core.model import register
from .model import Lfm3, parse

for model_type in ("lfm3", "lfm3_vl", "lfm3_audio", "lfm3_omni"):
    register(model_type, parse=parse)(Lfm3)
