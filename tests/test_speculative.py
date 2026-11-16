"""Prompt lookup drafts and the draft-length policy, on the host."""

from swiss.core.speculative import Lookup, Policy, expected


def test_lookup_copies_what_followed_the_latest_match():
    lookup = Lookup([1, 2, 3, 4, 5, 9, 1, 2, 3, 7, 8, 1, 2, 3])
    assert lookup.propose(None, 3) == [7, 8, 1]


def test_lookup_prefers_the_longest_ngram():
    lookup = Lookup([1, 2, 3, 4, 0, 9, 2, 3, 6, 1, 2, 3])
    assert lookup.propose(None, 2) == [4, 0]


def test_lookup_needs_an_earlier_match():
    assert Lookup([1, 2, 3, 4]).propose(None, 4) == []
    lookup = Lookup([1, 2, 1])
    lookup.extend([2])
    assert lookup.propose(None, 4) == [1, 2]


def test_rate_follows_acceptance_and_forgets():
    lookup = Lookup()
    lookup.observe(4, 4)
    lookup.observe(4, 4)
    high = lookup.rate
    lookup.observe(4, 0)
    assert lookup.rate < high
    for _ in range(200):
        lookup.forget()
    assert abs(lookup.rate - 0.5) < 0.01


def test_expected_tokens():
    assert expected(0.0, 4) == 1
    assert expected(1.0, 4) == 5


def test_policy_waits_for_a_plain_measurement():
    policy = Policy()
    assert policy.length(0.9, 4) == 0
    policy.time(1, 10.0)
    assert policy.length(0.9, 4) == 0
    policy.time(1, 10.0)
    assert policy.length(0.9, 4) == 4


def test_policy_weighs_measured_verify_cost():
    policy = Policy()
    for rows, ms in ((1, 10.0), (2, 12.0), (3, 30.0), (4, 40.0), (5, 50.0)):
        policy.time(rows, ms)
        policy.time(rows, ms)
    assert policy.length(0.9, 4) == 1
    assert policy.length(0.1, 4) == 0
