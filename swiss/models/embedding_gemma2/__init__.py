"""Google EmbeddingGemma 2, its text tower."""

from ...core.model import register
from .model import EmbeddingGemma2, parse

register("embedding_gemma2", parse=parse)(EmbeddingGemma2)
