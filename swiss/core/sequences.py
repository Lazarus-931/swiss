"""The run loop of a generating model: KV pages, tables, one program per step shape.

Sequence slot s owns KV pages [s · P, (s + 1) · P), with P the pages of one context, so
no page is ever allocated. A slot keeps its tokens after its request ends, and a later
prompt that starts with them reuses their KV.
"""

from functools import cached_property

import numpy as np
from carbon.kernels.ops import indexing, sampling

from ..kernels import linear
from .model import PAGE_SIZE, PREFILL_CHUNKS, Output
from .program import Program, allocate

SETTINGS = 6


class Sequences:
    def __init__(self, device, runtime, *, vocab, sizes, steps, rewindable, clear=None):
        self.device, self.runtime, self.vocab = device, runtime, vocab
        self.sizes, self.steps, self.rewindable, self.clear = (
            sizes,
            steps,
            rewindable,
            clear,
        )
        self.context_pages = -(-self.capacity // PAGE_SIZE)
        self.pages = self.sequences * self.context_pages
        self.tokens = [[] for _ in range(self.sequences)]
        self.table = device.buffer(max(16, self.pages * 4))
        self.lengths = device.buffer(max(16, self.sequences * 4))
        self.slots = device.buffer(max(16, self.sequences * 4))
        self.settings = device.buffer(self.sequences * SETTINGS * 4)
        self.choices = np.zeros((self.sequences, SETTINGS), np.uint32)
        for sequence in range(self.sequences):
            self.configure(sequence)
        self.programs, self.states, self.verifying = {}, [], None
        self.drafter, self.outputs, self.taps, self.tap_width = None, {}, {}, 0
        self.spare = runtime.draft + 1 if runtime.draft and not rewindable else 0

    @property
    def capacity(self):
        return self.runtime.capacity

    @property
    def sequences(self):
        return self.runtime.max_sequences

    @property
    def slot_count(self):
        """One recurrent state slot per sequence, plus rollback slots per verify row."""
        return self.sequences + self.spare

    def state(self, size):
        """Zeroed `size`-byte states per slot, restored on rollback."""
        buffer = self.device.buffer(self.slot_count * size)
        buffer.write(bytes(buffer.size))
        self.states.append((buffer, size))
        return buffer

    def last(self, buffers, rows):
        """Sample every row of a batch or verify step, otherwise only the last row."""
        return rows if buffers["batch"] or buffers["every"] else 1

    def tap(self, layer, buffers, rows):
        """Copy residuals after `layer` into side-by-side drafter taps.
        Return no dispatches when the drafter does not read this layer."""
        if layer not in self.taps:
            return []
        width = self.tap_width
        return [
            linear.place(
                self.device,
                buffers["x"],
                buffers["taps"],
                rows,
                width,
                len(self.taps) * width,
                self.taps[layer] * width,
            ),
        ]

    def follow(self, drafter, taps=(), width=0):
        """Attach `drafter`; it reads the residual stream after each of `taps`, which
        are `width` wide, and every stored step is committed to it."""
        self.drafter, self.tap_width = drafter, width
        self.taps = {layer: index for index, layer in enumerate(taps)}
        self.programs.clear()

    def commit(self, sequence, buffers, start, count, following):
        """Tell the drafter that rows [start, start + count) of a step stored the last
        `count` tokens of `sequence`, and that `following` comes next."""
        if self.drafter is not None:
            self.drafter.commit(sequence, buffers, start, count, following)

    def max_pages(self, buffers):
        """Attention table width for a batch; 0 for one sequence's rows."""
        return self.context_pages if buffers["batch"] else 0

    def choose(self, buffers, rows):
        """Sample `rows` logits rows under their sequences' settings."""
        settings, positions = (
            (buffers["settings"], buffers["positions"])
            if buffers["every"]
            else (self.settings, self.lengths)
        )
        return sampling.sample(
            self.device,
            buffers["logits"],
            settings,
            positions,
            buffers["token"],
            rows=rows,
            vocab=self.vocab,
        )

    def configure(self, sequence, params=None, seed=0):
        """Sample `sequence` under `params`, keyed by (seed, position); greedy without
        `params`."""
        if params is None or params.greedy:
            self.choices[sequence] = sampling.settings([seed], [0.0])[0]
        else:
            self.choices[sequence] = sampling.settings(
                [seed],
                [params.temperature],
                [params.top_k],
                [params.top_p],
                [params.min_p],
            )[0]

    def program(self, rows, batch=False, chain=False, every=False):
        """The program of a step over `rows` tokens."""
        key = rows, batch, chain, every
        if key not in self.programs:
            buffers = {
                **allocate(self.device, self.sizes(rows, batch or every)),
                "batch": batch,
                "every": every,
            }
            if self.taps:
                buffers["taps"] = self.device.buffer(
                    max(16, rows * len(self.taps) * self.tap_width * 2),
                )
            if every:
                buffers.update(
                    allocate(
                        self.device,
                        {"settings": rows * SETTINGS * 4, "positions": rows * 4},
                    ),
                )
            if every and self.states:
                buffers["snapshots"] = self.device.buffer(max(16, rows * 4))
                slots = np.arange(
                    self.sequences,
                    self.sequences + rows,
                    dtype=np.uint32,
                )
                buffers["snapshots"].write(
                    slots.tobytes() + bytes(buffers["snapshots"].size - slots.nbytes),
                )
            program = Program(self.device)
            for step in self.steps(rows, buffers):
                program.kernel(step)
            if chain:
                program.kernel(
                    linear.feed(
                        self.device,
                        buffers["token"],
                        buffers["ids"],
                        self.lengths,
                        self.history,
                        rows,
                        self.capacity,
                    ),
                )
            program.finish()
            self.programs[key] = program, buffers
        return self.programs[key]

    def prepare(self, rows, batch=False, every=False):
        self.program(rows, batch, every=every)

    @cached_property
    def history(self):
        """Tokens chosen by chained decode steps, at [batch row, position]."""
        return self.device.buffer(self.sequences * self.capacity * 4)

    def write_tables(self, sequences):
        sequences = np.asarray(sequences, np.uint32)
        for buffer, values in zip(
            (self.table, self.lengths, self.slots, self.settings),
            (
                sequences[:, None] * self.context_pages
                + np.arange(self.context_pages, dtype=np.uint32),
                np.array([len(self.tokens[s]) for s in sequences], np.uint32),
                sequences,
                self.choices[sequences],
            ),
        ):
            buffer.write(values.tobytes() + bytes(buffer.size - values.nbytes))

    def ids(self, tokens):
        values = np.asarray(tokens, np.int64)
        if (
            values.ndim != 1
            or not len(values)
            or values.min() < 0
            or values.max() >= self.vocab
        ):
            raise ValueError("tokens must be a nonempty vector of vocabulary IDs")
        return values.astype(np.uint32)

    def step(self, sequences, tokens, batch):
        """Run one program: `tokens` of one sequence, or with `batch` one token for each
        of `sequences`."""
        values = self.ids(tokens)
        program, buffers = self.program(len(tokens), batch)
        for sequence in sequences:
            self.reserve(sequence, len(tokens) // len(sequences))
        self.write_tables(sequences)
        buffers["ids"].write(
            values.tobytes() + bytes(buffers["ids"].size - values.nbytes),
        )
        completion = program.run()
        for index, sequence in enumerate(sequences):
            self.tokens[sequence] += tokens[index : index + 1] if batch else tokens
            self.outputs[sequence] = buffers, index
        return completion, buffers

    def verify(self, sequence, tokens, logits=False):
        """Run `tokens` as rows of `sequence` in one step without storing them."""
        values = self.ids(tokens)
        rows, length = len(values), self.length(sequence)
        if self.states and rows > self.spare:
            raise ValueError(
                f"a verify step of {rows} rows needs runtime draft >= {rows - 1}",
            )
        program, buffers = self.program(rows, every=True)
        self.reserve(sequence, rows)
        self.write_tables([sequence])
        for name, value in (
            ("ids", values),
            ("settings", np.tile(self.choices[sequence], rows)),
            ("positions", np.arange(length, length + rows, dtype=np.uint32)),
        ):
            buffers[name].write(
                value.tobytes() + bytes(buffers[name].size - value.nbytes),
            )
        program.run()
        self.verifying = sequence, rows, buffers
        chosen = np.frombuffer(buffers["token"].read(), np.uint32)[:rows].copy()
        return (chosen, self.logits(buffers, rows)) if logits else chosen

    def accept(self, sequence, tokens):
        """Store the first len(`tokens`) rows of the last `verify` of `sequence`."""
        if (
            self.verifying is None
            or self.verifying[0] != sequence
            or not 0 < len(tokens) <= self.verifying[1]
        ):
            raise ValueError(
                "accept takes a nonempty prefix of the last verify step of the "
                "sequence",
            )
        self.tokens[sequence] += [int(token) for token in tokens]
        self.outputs[sequence] = self.verifying[2], len(tokens) - 1
        if self.states and len(tokens) < self.verifying[1]:
            self.restore(len(tokens) - 1, sequence).run()
        buffers, self.verifying = self.verifying[2], None
        chosen = np.frombuffer(buffers["token"].read(), np.uint32)
        self.commit(sequence, buffers, 0, len(tokens), int(chosen[len(tokens) - 1]))

    def restore(self, row, sequence):
        """Copy recurrent states after verify row `row` into `sequence`'s slot."""
        key = "restore", row, sequence
        if key not in self.programs:
            sources, targets = self.device.buffer(16), self.device.buffer(16)
            sources.write(
                np.array([self.sequences + row, 0, 0, 0], np.uint32).tobytes(),
            )
            targets.write(np.array([sequence, 0, 0, 0], np.uint32).tobytes())
            program = Program(self.device)
            for buffer, size in self.states:
                for step in indexing.copy_slots(
                    self.device,
                    buffer,
                    sources,
                    targets,
                    count=1,
                    slot_bytes=size,
                ):
                    program.kernel(step)
            program.finish()
            self.programs[key] = program
        return self.programs[key]

    def logits(self, buffers, rows):
        bits = np.frombuffer(buffers["logits"].read(), np.uint16)[: rows * self.vocab]
        return linear.from_bf16(bits).reshape(rows, -1)

    def reusable(self, sequence, prompt):
        """How many leading tokens of `prompt` the KV of `sequence` holds, never the
        last one; 0 when the model is not rewindable."""
        kept, prompt = self.tokens[sequence], list(prompt)[:-1]
        if not self.rewindable:
            return 0
        common = min(len(kept), len(prompt))
        reused = next((i for i in range(common) if kept[i] != prompt[i]), common)
        return max(reused - getattr(self.drafter, "lag", 0), 0)

    def admit(self, sequence, prompt=()):
        """Start `sequence`; returns how many prompt tokens are already stored."""
        if not 0 <= sequence < self.sequences:
            raise ValueError(f"sequence must be below max_sequences={self.sequences}")
        if len(prompt) > self.capacity:
            raise ValueError(
                f"prompt has {len(prompt)} tokens; the context holds {self.capacity}",
            )
        reused = self.reusable(sequence, prompt)
        self.tokens[sequence] = [int(token) for token in prompt[:reused]]
        if self.clear is not None:
            self.clear(sequence)
        return reused

    def reserve(self, sequence, count):
        if self.length(sequence) + count > self.capacity:
            raise ValueError(
                f"{self.length(sequence) + count} tokens exceed the context of "
                f"{self.capacity}",
            )

    def length(self, sequence):
        return len(self.tokens[sequence])

    def prefill(
        self,
        sequence,
        tokens,
        chunks=PREFILL_CHUNKS,
        logits=False,
        following=None,
    ):
        """Append `tokens` to `sequence` in steps of the largest chunk that fits; an
        Output for the last row. `following` is the token after `tokens` when they are
        not the end of the prompt."""
        tokens, start, gpu = list(map(int, tokens)), 0, 0.0
        self.reserve(sequence, len(tokens))
        while start < len(tokens):
            size = next(chunk for chunk in chunks if chunk <= len(tokens) - start)
            completion, buffers = self.step(
                [sequence],
                tokens[start : start + size],
                False,
            )
            gpu += completion.gpu_ms or 0.0
            start += size
            token = int(np.frombuffer(buffers["token"].read(), np.uint32)[0])
            if start < len(tokens):
                after = tokens[start]
            else:
                after = token if following is None else int(following)
            self.commit(sequence, buffers, 0, size, after)
        return Output(token, self.logits(buffers, 1)[0] if logits else None, gpu)

    def decode(self, sequences, tokens, logits=False):
        """One step per sequence, returning next tokens and optional logits."""
        _, buffers = self.step(list(sequences), list(map(int, tokens)), True)
        chosen = np.frombuffer(buffers["token"].read(), np.uint32)[: len(tokens)].copy()
        for row, sequence in enumerate(sequences):
            self.commit(sequence, buffers, row, 1, int(chosen[row]))
        return (chosen, self.logits(buffers, len(tokens))) if logits else chosen

    def decode_chain(self, sequences, tokens, steps):
        """`steps` decode steps in one command buffer; returns (steps, len(sequences))
        tokens. A drafter that reads taps needs every step, so then each step runs on
        its own."""
        sequences, tokens = list(sequences), [int(token) for token in tokens]
        if self.taps:
            chosen = []
            for _ in range(steps):
                tokens = self.decode(sequences, tokens)
                chosen.append(tokens)
            return np.stack(chosen)
        program, buffers = self.program(len(sequences), True, chain=True)
        starts = [self.length(sequence) for sequence in sequences]
        for sequence in sequences:
            self.reserve(sequence, steps)
        self.write_tables(sequences)
        buffers["ids"].write(
            np.asarray(tokens, np.uint32).tobytes()
            + bytes(buffers["ids"].size - 4 * len(tokens)),
        )
        program.submit(steps).wait()
        history = np.frombuffer(self.history.read(), np.uint32).reshape(
            self.sequences,
            self.capacity,
        )
        chosen = np.stack(
            [history[row, start : start + steps] for row, start in enumerate(starts)],
            1,
        )
        for row, sequence in enumerate(sequences):
            self.tokens[sequence] += [tokens[row], *chosen[:-1, row].tolist()]
            self.outputs[sequence] = buffers, row
        return chosen
