"""Continuous-batching generation engine over one model."""

import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .speculative import Lookup, Policy, Session

CHUNK = 2048
CHAIN = 16
LOOKUP_CHAIN = 4


class Detokenizer:
    def __init__(self, tokenizer, stop=()):
        self.tokenizer, self.stop = tokenizer, stop
        self.tokens, self.text, self.emitted = [], "", 0

    def push(self, token):
        """Add a token; return the newly safe text and whether a stop string was
        reached."""
        self.tokens.append(token)
        text = self.tokenizer.decode(self.tokens, skip_special_tokens=True)
        if text.endswith("�"):
            return "", False
        self.text = text
        for stop in self.stop:
            index = self.text.find(stop, max(0, self.emitted - len(stop)))
            if index >= 0:
                delta = self.text[self.emitted : index]
                self.emitted = len(self.text)
                return delta, True
        hold = max(
            (
                size
                for stop in self.stop
                for size in range(len(stop) - 1, 0, -1)
                if self.text.endswith(stop[:size])
            ),
            default=0,
        )
        end = max(self.emitted, len(self.text) - hold)
        delta, self.emitted = self.text[self.emitted : end], end
        return delta, False

    def flush(self):
        delta, self.emitted = self.text[self.emitted :], len(self.text)
        return delta


@dataclass(frozen=True)
class Event:
    text: str = ""
    token: int | None = None
    finish_reason: str | None = None
    usage: dict | None = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0

    def as_dict(self):
        rate = (
            self.completion_tokens / (self.decode_ms / 1000)
            if self.decode_ms > 0
            else None
        )
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": self.cached_tokens},
            "timings": {
                "prefill_ms": round(self.prefill_ms, 3),
                "decode_ms": round(self.decode_ms, 3),
                "decode_tokens_per_second": round(rate, 2) if rate else None,
            },
        }


@dataclass(eq=False)
class Request:
    prompt: list
    params: object
    cancelled: object
    detokenizer: Detokenizer
    seed: int
    budget: int
    events: queue.Queue = field(default_factory=queue.Queue)
    usage: Usage = field(default_factory=Usage)
    sequence: int | None = None
    pending: list = field(default_factory=list)
    drafter: object = None
    next: int | None = None
    started: float = 0.0
    decoding: float = 0.0


class Engine:
    """Requests from any number of threads, run together on up to `model.sequences`
    sequence slots."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        eos_token_ids=(),
        name="swiss",
        chunk=CHUNK,
    ):
        self.model, self.tokenizer, self.name, self.chunk = (
            model,
            tokenizer,
            name,
            chunk,
        )
        self.eos_token_ids = frozenset(int(token) for token in eos_token_ids)
        self.draft, self.policy = model.runtime.draft, Policy()
        self.waiting, self.running = deque(), []
        self.free = list(range(model.sequences))
        self.wake, self.closed = threading.Condition(), False
        self.thread = threading.Thread(
            target=self.loop,
            name="swiss-engine",
            daemon=True,
        )
        self.thread.start()

    @property
    def capacity(self):
        return self.model.capacity

    @property
    def busy(self):
        return bool(self.waiting or self.running)

    def warm(self):
        """Compile the prefill chunks and the single-sequence decode step before
        serving."""
        for size in sorted(
            {
                size
                for size in (self.chunk, 512, 128, 32, 8, 2, 1)
                if size <= self.capacity
            },
        ):
            self.model.prepare(size)
        self.model.prepare(1, True)
        for rows in range(2, self.draft + 2):
            self.model.prepare(rows, every=True)

    def encode(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def encode_chat(self, messages):
        return list(
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            ),
        )

    def submit(self, prompt, params, cancelled=lambda: False):
        prompt = [int(token) for token in prompt]
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if len(prompt) >= self.capacity:
            raise ValueError(
                f"prompt has {len(prompt)} tokens; the context holds {self.capacity}",
            )
        request = Request(
            prompt,
            params,
            cancelled,
            Detokenizer(self.tokenizer, params.stop),
            int(np.random.default_rng(params.seed).integers(2**32)),
            min(params.max_tokens, self.capacity - len(prompt)),
        )
        request.usage.prompt_tokens = len(prompt)
        with self.wake:
            self.waiting.append(request)
            self.wake.notify()
        return request

    def generate(self, prompt, params, cancelled=lambda: False):
        """Yield text deltas, then one event carrying the finish reason and usage."""
        request = self.submit(prompt, params, cancelled)
        while True:
            event = request.events.get()
            if isinstance(event, Exception):
                raise event
            yield event
            if event.finish_reason:
                return

    def complete(self, prompt, params):
        """Return the whole generated text, finish reason and usage."""
        pieces, final = [], None
        for event in self.generate(prompt, params):
            pieces.append(event.text)
            final = event
        return "".join(pieces), final.finish_reason, final.usage

    def close(self):
        """Stop the scheduler once the running requests finish."""
        with self.wake:
            self.closed = True
            self.wake.notify()
        self.thread.join()

    def loop(self):
        while True:
            with self.wake:
                while not self.waiting and not self.running and not self.closed:
                    self.wake.wait()
                if self.closed and not self.waiting and not self.running:
                    return
                while self.waiting and self.free:
                    request = self.waiting.popleft()
                    request.sequence = max(
                        reversed(self.free),
                        key=lambda s: self.model.reusable(s, request.prompt),
                    )
                    self.free.remove(request.sequence)
                    self.running.append(request)
            try:
                self.advance()
            except Exception as error:
                for request in list(self.running):
                    self.finish(request, None, error)

    def advance(self):
        for request in [r for r in self.running if not r.started]:
            request.started = perf_counter()
            request.usage.cached_tokens = self.model.admit(
                request.sequence,
                request.prompt,
            )
            self.model.configure(request.sequence, request.params, request.seed)
            request.pending = request.prompt[request.usage.cached_tokens :]
            request.drafter = self.drafter(request)
        for request in [r for r in self.running if r.pending and r.cancelled()]:
            self.finish(request, "cancelled")
        reading = next((r for r in self.running if r.pending), None)
        if reading is not None:
            piece, reading.pending = (
                reading.pending[: self.chunk],
                reading.pending[self.chunk :],
            )
            output = self.model.prefill(
                reading.sequence,
                piece,
                following=reading.pending[0] if reading.pending else None,
            )
            if not reading.pending:
                reading.usage.prefill_ms = (perf_counter() - reading.started) * 1000
                reading.decoding = perf_counter()
                self.emit(reading, output.token)
        ready = [r for r in self.running if r.next is not None]
        if not ready:
            return
        chain = LOOKUP_CHAIN if any(r.drafter is not None for r in ready) else CHAIN
        steps = min(chain, *(r.budget - r.usage.completion_tokens for r in ready))
        idle = (
            reading is None
            and not any(r.pending for r in self.running)
            and not (self.waiting and self.free)
        )
        if (
            idle
            and len(ready) == 1
            and ready[0].drafter is not None
            and self.speculate(ready[0])
        ):
            return
        if idle and steps > 1:
            start = perf_counter()
            chosen = self.model.decode_chain(
                [r.sequence for r in ready],
                [r.next for r in ready],
                steps,
            )
            self.policy.time(1, (perf_counter() - start) * 1000 / steps)
            for tokens in chosen:
                for row, request in enumerate(ready):
                    if request.next is not None:
                        self.emit(request, int(tokens[row]))
            return
        tokens = self.model.decode([r.sequence for r in ready], [r.next for r in ready])
        for row, request in enumerate(ready):
            self.emit(request, int(tokens[row]))

    def drafter(self, request):
        """The model's drafter for `request` when it has one, else prompt lookup."""
        if not self.draft:
            return None
        if self.model.drafter is not None:
            return Session(self.model.drafter, request.sequence)
        return Lookup(request.prompt)

    def speculate(self, request):
        """One draft and verify step for `request` when the policy expects it to beat
        plain decode."""
        drafter = request.drafter
        room = min(
            self.draft,
            request.budget - request.usage.completion_tokens - 1,
            self.capacity - self.model.length(request.sequence) - 1,
        )
        drafted = self.policy.length(drafter.rate, max(room, 0))
        start = perf_counter()
        draft = drafter.propose(request.next, drafted)[:drafted]
        if not draft:
            drafter.forget()
            return False
        rows = [request.next, *draft]
        chosen = self.model.verify(request.sequence, rows)
        accepted = next(
            (index for index in range(len(draft)) if chosen[index] != rows[index + 1]),
            len(draft),
        )
        self.model.accept(request.sequence, rows[: accepted + 1])
        self.policy.time(len(rows), (perf_counter() - start) * 1000)
        drafter.observe(len(draft), accepted)
        for token in chosen[: accepted + 1]:
            self.emit(request, int(token))
            if request.next is None:
                break
        return True

    def emit(self, request, token):
        """Take the step's choice for `request`, stream its text, and finish it or queue
        the token for decode."""
        request.next = None
        if token in self.eos_token_ids or token in request.params.stop_token_ids:
            return self.finish(request, "stop")
        request.usage.completion_tokens += 1
        if request.drafter is not None:
            request.drafter.extend((token,))
        delta, stopped = request.detokenizer.push(token)
        if delta:
            request.events.put(Event(text=delta, token=token))
        if stopped:
            return self.finish(request, "stop")
        if request.usage.completion_tokens >= request.budget:
            return self.finish(request, "length")
        if request.cancelled():
            return self.finish(request, "cancelled")
        request.next = token

    def finish(self, request, reason, error=None):
        if request.decoding:
            request.usage.decode_ms = (perf_counter() - request.decoding) * 1000
        with self.wake:
            self.running.remove(request)
            self.free.append(request.sequence)
        if error is not None:
            request.events.put(error)
            return
        if reason != "stop":
            tail = request.detokenizer.flush()
            if tail:
                request.events.put(Event(text=tail))
        request.events.put(Event(finish_reason=reason, usage=request.usage.as_dict()))


class Embedder:
    """The engine of an encoder: `embed` packs texts into steps of up to `capacity`
    tokens, one step at a time."""

    def __init__(self, model, tokenizer, *, name="swiss"):
        self.model, self.tokenizer, self.name = model, tokenizer, name
        self.lock, self.running = threading.Lock(), 0

    @property
    def capacity(self):
        return self.model.capacity

    @property
    def busy(self):
        return bool(self.running)

    def warm(self):
        self.model.embed_tokens([[self.tokenizer.bos_token_id or 0]])

    def encode(self, text):
        return list(self.tokenizer(text)["input_ids"])

    def embed(self, inputs, dimensions=None):
        """(unit embeddings, token count) of the inputs, truncated to `dimensions` when
        given."""
        batches = [
            self.encode(item) if isinstance(item, str) else list(item)
            for item in inputs
        ]
        if any(len(tokens) > self.capacity for tokens in batches):
            raise ValueError(f"an input exceeds the {self.capacity} tokens of a step")
        vectors, start = [], 0
        self.running += 1
        try:
            while start < len(batches):
                end, total = start, 0
                while end < len(batches) and total + len(batches[end]) <= self.capacity:
                    total += len(batches[end])
                    end += 1
                with self.lock:
                    vectors.append(self.model.embed_tokens(batches[start:end]))
                start = end
        finally:
            self.running -= 1
        embeddings = np.concatenate(vectors)
        if dimensions is not None:
            embeddings = embeddings[:, :dimensions]
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings, sum(len(tokens) for tokens in batches)
