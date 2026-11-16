"""Speculative decoding: drafters propose tokens, one verify step checks them."""

import numpy as np

from ..kernels import linear
from .model import PAGE_SIZE
from .program import Program, allocate

NGRAMS = (4, 3, 2)
SMOOTHING = 0.2
FORGET = 0.97


class Drafter:
    """The acceptance rate of one request's drafts; subclasses propose the tokens."""

    hits = misses = 0.0

    def propose(self, token, count):
        """Up to `count` tokens to follow `token`, the next input of the sequence."""
        raise NotImplementedError

    def extend(self, tokens):
        """Tokens the sequence emitted."""

    @property
    def rate(self):
        """The chance that a drafted token is accepted after the ones before it were."""
        return (self.hits + 1.0) / (self.hits + self.misses + 2.0)

    def observe(self, drafted, accepted):
        self.hits += accepted
        self.misses += accepted < drafted

    def forget(self):
        self.hits *= FORGET
        self.misses *= FORGET


class Lookup(Drafter):
    """Prompt lookup: the tokens that followed the latest match of the longest recent
    n-gram."""

    def __init__(self, tokens=()):
        self.tokens, self.follows = [], {}
        self.extend(tokens)

    def extend(self, tokens):
        for token in tokens:
            end = len(self.tokens)
            self.tokens.append(int(token))
            for n in NGRAMS:
                if end >= n:
                    self.follows[tuple(self.tokens[end - n : end])] = end

    def propose(self, token, count):
        for n in NGRAMS:
            start = self.follows.get(tuple(self.tokens[-n:]))
            if start is not None:
                return self.tokens[start : start + count]
        return []


class Session(Drafter):
    """One request's drafts from the model drafter attached to its sequences."""

    def __init__(self, model, sequence):
        self.model, self.sequence = model, sequence

    def propose(self, token, count):
        return self.model.propose(self.sequence, token, count) if count else []


class Attached:
    """A model drafter attached to a target's sequences.

    `taps` are the target layers whose output it reads, and `commit` is told of every
    stored step. `lag` is how many reused prompt tokens it needs to see again.
    """

    drafts = True
    taps = ()
    lag = 0

    def commit(self, sequence, buffers, start, count, following):
        """Rows [start, start + count) of a target step stored the last `count` tokens
        of `sequence`, and `following` comes next."""


class Pages:
    """A drafter's own KV pages, `layers` of them, one page range per sequence slot of
    `capacity` positions."""

    def __init__(self, device, sequences, capacity, layers, kv_heads, head_dim):
        self.device, self.capacity = device, capacity
        self.context_pages = -(-capacity // PAGE_SIZE)
        size = sequences * self.context_pages * kv_heads * PAGE_SIZE * head_dim * 2
        self.layers = [
            (device.buffer(size), device.buffer(size)) for _ in range(layers)
        ]
        self.table = device.buffer(max(16, self.context_pages * 4))
        self.lengths = device.buffer(16)

    def point(self, sequence, length):
        """The next step's rows start at `length` in the pages of `sequence`."""
        table = sequence * self.context_pages + np.arange(
            self.context_pages,
            dtype=np.uint32,
        )
        self.table.write(table.tobytes() + bytes(self.table.size - table.nbytes))
        self.lengths.write(np.array([length, 0, 0, 0], np.uint32).tobytes())


class Chain:
    """A drafter's one-token step, run `count` times in one command buffer.

    `steps(buffers)` reads the input token from buffers["ids"] and writes the drafted
    token to buffers["token"]; the chain feeds it back and records it. With
    `lengths`, the drafter's KV length, each step advances it and its token is recorded
    at that position.
    """

    def __init__(self, device, sizes, steps, limit=16, lengths=None):
        self.buffers = allocate(
            device,
            {**sizes, "ids": 4, "token": 4, "count": 4, "drafted": limit * 4},
        )
        self.own = lengths is None
        self.lengths = self.buffers["count"] if self.own else lengths
        self.program = Program(device)
        for step in steps(self.buffers):
            self.program.kernel(step)
        self.program.kernel(
            linear.feed(
                device,
                self.buffers["token"],
                self.buffers["ids"],
                self.lengths,
                self.buffers["drafted"],
                1,
                limit,
            ),
        )
        self.program.finish()

    def run(self, token, count, start=0):
        """The `count` tokens drafted after `token`; the KV length is at `start`."""
        self.buffers["ids"].write(np.array([token, 0, 0, 0], np.uint32).tobytes())
        if self.own:
            self.buffers["count"].write(bytes(16))
        self.program.submit(count).wait()
        drafted = np.frombuffer(self.buffers["drafted"].read(), np.uint32)
        return drafted[start : start + count].tolist()


def expected(rate, drafted):
    """Expected tokens of a verify step of `drafted` drafts: the accepted ones and the
    one the target adds."""
    return sum(rate**index for index in range(drafted + 1))


class Policy:
    """Chooses the draft length with the most expected tokens per millisecond, from
    measured step times."""

    def __init__(self):
        self.steps, self.seen = {}, set()

    def time(self, rows, ms):
        """Record the milliseconds of a step of `rows` rows, drafting included; the
        first of each row count is skipped, as it can include compilation."""
        if rows not in self.seen:
            self.seen.add(rows)
            return
        old = self.steps.get(rows)
        self.steps[rows] = ms if old is None else old + SMOOTHING * (ms - old)

    def length(self, rate, available):
        """The number of drafts to verify, 0 when plain decode is expected to be
        faster."""
        plain = self.steps.get(1)
        if plain is None or not available:
            return 0
        best, size = 1.0 / plain, 0
        for drafted in range(1, available + 1):
            score = expected(rate, drafted) / self.steps.get(drafted + 1, plain)
            if score > best:
                best, size = score, drafted
        return size
