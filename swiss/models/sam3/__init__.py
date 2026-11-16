"""Meta SAM 3: image features, text-prompted detection and masks."""

from ...core.model import register
from .model import Sam3, parse

register("sam3_video", parse=parse)(Sam3)
