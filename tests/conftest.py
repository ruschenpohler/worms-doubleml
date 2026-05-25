import random

import pytest


@pytest.fixture
def rng():
    random.seed(42)
    return random
