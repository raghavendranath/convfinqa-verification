import pytest

from cascade.match import extract_answer, match


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The change is 25,587.\nAnswer: 25587", 25587),
        ("Answer: $1,250.00", 1250),
        ("so roughly 14.1%\nAnswer: 0.1414", 0.1414),
        ("Answer: 14.1%", 0.141),
        ("Answer: (3,000)", -3000),
        ("Answer: -0.05", -0.05),
        ("Answer: yes", "yes"),
        ("no answer line, last number is 42", 42),
        ("nothing here", None),
        # Scientific notation is read whole (CHANGELOG R53; it used to drop the exponent).
        ("Answer: 6.4748e+06", 6474800.0),
        ("Answer: 2.25546e10", 22554600000.0),
        ("Answer: 590.9534164e6", 590953416.4),
        ("Answer: -1.5E-3", -0.0015),
        ("Answer: 12e", 12),
    ],
)
def test_extract(text, expected):
    got = extract_answer(text)
    if isinstance(expected, float):
        assert got == pytest.approx(expected)
    else:
        assert got == expected


def test_strict_vs_lenient():
    assert match(0.14136, 0.141364)
    assert not match(14.136, 0.14136)
    assert match(14.136, 0.14136, mode="lenient")
    assert not match(None, 1.0)
    assert match("Yes", "yes")
    assert not match(1.0, "yes")


def test_gold_rounded_to_five_decimals():
    # Cases from ConvFinQA train where the gold answer is the program result rounded to 5 decimals.
    assert match(-0.0005139133438914874, -0.00051)
    assert match(5.101539249568888e-06, 1e-05)
    assert match(0.00020394736842105262, 0.0002)
    assert not match(0.0006, 0.00051)
