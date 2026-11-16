"""One config per architecture in tests/model_config.json, each parsing into a
registered spec."""

import pytest
from swiss.core.model import lookup

from tiny_model import CONFIGS


@pytest.mark.parametrize("repo", CONFIGS)
def test_parses(repo):
    lookup(CONFIGS[repo]).parse(CONFIGS[repo])


def test_one_config_per_architecture():
    kinds = [lookup(config).model_type for config in CONFIGS.values()]
    assert len(set(kinds)) == len(kinds)
