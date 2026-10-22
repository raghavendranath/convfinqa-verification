import random

import pytest

from cascade.inject import ERROR_TYPES, corrupt, fmt
from cascade.match import match


def test_each_type_gives_a_different_value(convs):
    c = convs[0]
    for kind in ERROR_TYPES:
        out = corrupt(206588.0, kind, c, random.Random(0))
        assert out is not None, kind
        assert not match(out, 206588.0)


def test_period_and_line_item_come_from_the_table(convs):
    c = convs[0]
    assert corrupt(206588.0, "period", c, random.Random(0)) in (181001.0, 170000.0)
    assert corrupt(206588.0, "line_item", c, random.Random(0)) == 50000.0


def test_not_applicable_returns_none(convs):
    c = convs[0]
    assert corrupt(25587.0, "period", c, random.Random(0)) is None  # computed value, not in the table
    assert corrupt(0.0, "sign", c, random.Random(0)) is None
    assert corrupt("yes", "sign", c, random.Random(0)) == "no"
    assert corrupt("yes", "scale", c, random.Random(0)) is None


def test_scale_and_percent():
    from cascade.data import Conversation

    c = Conversation("x", [], [], [], [])
    assert corrupt(0.14, "percent", c, random.Random(0)) == pytest.approx(14.0)
    assert corrupt(250.0, "scale", c, random.Random(0)) == 250000.0


def test_fmt():
    assert fmt(25587.0) == "25587"
    assert fmt(0.141364) == "0.141364"
    assert fmt("yes") == "yes"
