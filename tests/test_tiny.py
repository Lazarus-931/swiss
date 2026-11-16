"""Tiny random-weight models on the GPU against the reference model on the same weights,
and against Swiss itself."""

import numpy as np
import pytest
from swiss.core.model import Checkpoint, RuntimeConfig, attach, load
from swiss.kernels.linear import from_bf16

import tiny_model

pytestmark = pytest.mark.tiny

TOLERANCE = 6e-2
CASES = [pytest.param(case, id="-".join(case)) for case in tiny_model.cases()]


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory):
    made = {}

    def make(case):
        if case not in made:
            made[case] = tiny_model.save(
                *case,
                tmp_path_factory.mktemp("-".join(case).replace("/", "-")),
            )
        return made[case]

    return make


class Tensors(dict):
    """Tensors by name that remember which names were read."""

    def __getitem__(self, name):
        self.read.add(name)
        return super().__getitem__(name)


@pytest.mark.parametrize("case", CASES)
def test_reads_every_tensor(checkpoint, case, monkeypatch):
    loaded = Tensors(Checkpoint(checkpoint(case)).tensors())
    loaded.read = set()
    monkeypatch.setattr(Checkpoint, "tensors", lambda self: loaded)
    load(checkpoint(case), RuntimeConfig(capacity=64))
    assert (
        sorted(
            name for name in set(loaded) - loaded.read if not tiny_model.unused(name)
        )
        == []
    )


def tokens(count, seed=0):
    return [
        int(t) for t in np.random.default_rng(seed).integers(0, tiny_model.VOCAB, count)
    ]


def model(folder, sequences=1, draft=0):
    return load(
        folder,
        RuntimeConfig(capacity=128, max_sequences=sequences, draft=draft),
    )[0].run


def prefill(m, ids, sequence=0, **kwargs):
    m.admit(sequence)
    return m.prefill(sequence, ids, logits=True, **kwargs).logits


def decode(m, sequences, ids):
    return m.decode(sequences, ids, logits=True)[1]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prompt", [15, 16, 17, 33])
def test_prefill_then_decode(checkpoint, case, prompt):
    ids = tokens(prompt + 8)
    want = tiny_model.reference(checkpoint(case), ids)[prompt - 1 :]
    m = model(checkpoint(case))
    got = [prefill(m, ids[:prompt])] + [
        decode(m, [0], [token])[0] for token in ids[prompt:-1]
    ]
    errors = [round(tiny_model.relative(g, w), 4) for g, w in zip(got, want)]
    assert max(errors) < TOLERANCE, errors


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("chunks", [(41,), (16, 1), (8, 2, 1)])
def test_prefill_chunks(checkpoint, case, chunks):
    ids = tokens(41, seed=1)
    want = tiny_model.reference(checkpoint(case), ids)[-1]
    assert (
        tiny_model.relative(prefill(model(checkpoint(case)), ids, chunks=chunks), want)
        < TOLERANCE
    )


@pytest.mark.parametrize("case", CASES)
def test_batched_decode(checkpoint, case):
    prompts = [tokens(length, seed=length) for length in (5, 16, 30)]
    steps = [tokens(4, seed=100 + s) for s in range(3)]
    batch = model(checkpoint(case), sequences=3)
    for s, prompt in enumerate(prompts):
        prefill(batch, prompt, sequence=s)
    together = [
        decode(batch, [0, 1, 2], [steps[s][i] for s in range(3)]) for i in range(4)
    ]
    for s, prompt in enumerate(prompts):
        alone = model(checkpoint(case))
        prefill(alone, prompt)
        for i in range(4):
            assert (
                tiny_model.relative(
                    together[i][s],
                    decode(alone, [0], [steps[s][i]])[0],
                )
                < 3e-2
            )


@pytest.mark.parametrize("case", CASES)
def test_admit_starts_clean(checkpoint, case):
    used = model(checkpoint(case))
    prefill(used, tokens(37, seed=2))
    decode(used, [0], [5])
    second = tokens(21, seed=3)
    np.testing.assert_array_equal(
        prefill(used, second),
        prefill(model(checkpoint(case)), second),
    )


@pytest.mark.parametrize("case", CASES)
def test_prefix_reused(checkpoint, case):
    ids = tokens(40, seed=4)
    m = model(checkpoint(case), sequences=2)
    prefill(m, ids[:35])
    assert m.reusable(1, ids) == 0
    reused = m.admit(0, ids)
    assert reused == (35 if m.rewindable else 0)
    got = m.prefill(0, ids[reused:], logits=True).logits
    assert (
        tiny_model.relative(got, tiny_model.reference(checkpoint(case), ids)[-1])
        < TOLERANCE
    )


@pytest.mark.parametrize("case", CASES)
def test_decode_chain(checkpoint, case):
    prompt = tokens(20, seed=5)
    chained, stepped = model(checkpoint(case)), model(checkpoint(case))
    chained.admit(0)
    stepped.admit(0)
    token = chained.prefill(0, prompt).token
    chain = chained.decode_chain([0], [token], 6)[:, 0].tolist()
    stepped.prefill(0, prompt)
    chosen = []
    for _ in range(6):
        token = int(stepped.decode([0], [token])[0])
        chosen.append(token)
    assert chain == chosen


@pytest.mark.parametrize("case", CASES)
def test_verify_then_accept(checkpoint, case):
    prompt, ids = tokens(20, seed=6), tokens(6, seed=7)
    plain, speculative = model(checkpoint(case)), model(checkpoint(case), draft=4)
    prefill(plain, prompt)
    prefill(speculative, prompt)
    want = [decode(plain, [0], [token])[0] for token in ids]
    _, rows = speculative.verify(
        0,
        [*ids[:3], (ids[3] + 1) % tiny_model.VOCAB, ids[4]],
        logits=True,
    )
    speculative.accept(0, ids[:3])
    got = [*rows[:3], *(decode(speculative, [0], [token])[0] for token in ids[3:])]
    errors = [round(tiny_model.relative(g, w), 4) for g, w in zip(got, want)]
    assert max(errors) < 3e-2, errors


DRAFTED = [pytest.param(repo, id=repo) for repo in tiny_model.DRAFTERS]


@pytest.mark.parametrize("repo", DRAFTED)
@pytest.mark.parametrize("prompt", [17, 40])
def test_drafts_match_reference(checkpoint, tmp_path, repo, prompt):
    target = checkpoint((tiny_model.DRAFTERS[repo], "bf16"))
    drafter = tiny_model.save_drafter(repo, tmp_path / "drafter")
    ids = tokens(prompt, seed=8)
    token, want = tiny_model.drafter_reference(target, drafter, ids, 3)
    loaded = load(target, RuntimeConfig(capacity=128, draft=4))[0]
    attach(loaded, drafter)
    m = loaded.run
    m.admit(0)
    m.prefill(0, ids)
    got = []
    for steps in range(1, 4):
        m.drafter.propose(0, token, steps)
        bits = np.frombuffer(m.drafter.chain.buffers["logits"].read(), np.uint16)
        got.append(from_bf16(bits[: tiny_model.VOCAB]))
    errors = [round(tiny_model.relative(g, w), 4) for g, w in zip(got, want)]
    assert max(errors) < TOLERANCE, errors


TAPPED = [pytest.param(repo, id=repo) for repo in tiny_model.TAPPED]


@pytest.mark.parametrize("repo", TAPPED)
@pytest.mark.parametrize("prompt", [17, 40])
def test_tapped_drafts_match_reference(checkpoint, tmp_path, repo, prompt):
    target = checkpoint((tiny_model.TAPPED[repo], "bf16"))
    drafter = tiny_model.save_tapped(repo, tmp_path / "drafter")
    ids = tokens(prompt, seed=8)
    token, first, want = tiny_model.tapped_reference(target, drafter, ids, 3)
    loaded = load(target, RuntimeConfig(capacity=128, draft=4))[0]
    attach(loaded, drafter)
    m = loaded.run
    m.admit(0)
    assert m.prefill(0, ids).token == token
    small, errors = m.drafter, []
    if first is not None:
        hidden = np.frombuffer(small.seeds[0][1], np.uint16)
        errors.append(round(tiny_model.relative(from_bf16(hidden), first), 4))
        for steps in range(2, 4):
            small.propose(0, token, steps)
            bits = np.frombuffer(small.chain.buffers["logits"].read(), np.uint16)
            got = from_bf16(bits[: tiny_model.DRAFT_VOCAB])
            errors.append(round(tiny_model.relative(got, want[steps - 2]), 4))
    else:
        small.propose(0, token, 3)
        _, b = small.draft_program(4)
        bits = np.frombuffer(b["logits"].read(), np.uint16)
        got = from_bf16(bits[: 4 * tiny_model.VOCAB]).reshape(4, -1)[1:]
        errors += [round(tiny_model.relative(g, w), 4) for g, w in zip(got, want)]
    assert max(errors) < TOLERANCE, errors
