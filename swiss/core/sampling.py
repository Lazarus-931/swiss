"""Sampling parameters of a request."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int | None = None
    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self):
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if not 0.0 <= self.temperature <= 100.0:
            raise ValueError("temperature must be between 0 and 100")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if not 0.0 <= self.min_p < 1.0:
            raise ValueError("min_p must be in [0, 1)")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if (
            any(not isinstance(text, str) or not text for text in self.stop)
            or len(self.stop) > 16
        ):
            raise ValueError("stop must be at most 16 nonempty strings")
        if any(type(token) is not int or token < 0 for token in self.stop_token_ids):
            raise ValueError("stop_token_ids must be nonnegative integers")

    @property
    def greedy(self):
        return self.temperature == 0.0 or self.top_k == 1
