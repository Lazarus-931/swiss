"""Google Gemma 4 language models, dense and unified checkpoints, and their
assistants."""

from ...core.model import register
from .assistant import Gemma4Assistant
from .model import Gemma4, parse

for model_type in ("gemma4", "gemma4_unified"):
    register(model_type, parse=parse)(Gemma4)
for model_type in ("gemma4_assistant", "gemma4_unified_assistant"):
    register(model_type, parse=parse)(Gemma4Assistant)
