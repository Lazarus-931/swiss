"""Drafters that run only attached to a target: DFlash and EAGLE-3."""

from ..core.model import register
from . import dflash, eagle3

register("DFlashDraftModel", parse=dflash.parse)(dflash.DFlash)
for name in (
    "LlamaForCausalLMEagle3",
    "Eagle3Speculator",
    "Eagle3DraftModel",
    "eagle3",
):
    register(name, parse=eagle3.parse)(eagle3.Eagle3)

__all__ = ["dflash", "eagle3"]
