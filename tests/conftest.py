from pathlib import Path

import pytest

from cascade.data import load

FIXTURE = Path(__file__).parent / "fixtures" / "mini_convfinqa.json"


@pytest.fixture
def convs():
    return load(FIXTURE)
