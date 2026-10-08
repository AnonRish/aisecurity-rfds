import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic generator: every test is reproducible run to run."""
    return np.random.default_rng(12345)
