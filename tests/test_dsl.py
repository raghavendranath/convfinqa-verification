import pytest

from cascade.dsl import ProgramError, execute, parse_number


def test_two_step_program_with_reference():
    ex = execute("subtract(206588, 181001), divide(#0, 181001)")
    assert ex.results[0] == 25587
    assert ex.answer == pytest.approx(0.141364, rel=1e-4)
    assert ex.literals == (206588.0, 181001.0, 181001.0)


def test_bare_literal_is_a_lookup():
    ex = execute("206588")
    assert ex.answer == 206588 and ex.literals == (206588.0,)


def test_constants_and_percent():
    assert parse_number("const_m1") == -1
    assert parse_number("const_1000") == 1000
    assert parse_number("5%") == pytest.approx(0.05)
    assert parse_number("(3000)") == -3000
    assert execute("multiply(100, const_100)").literals == (100.0,)


def test_greater_and_table_ops():
    table = [["", "2020", "2019"], ["shares", "10", "12"], ["price", "$ 5.5", "4.0"]]
    assert execute("greater(5.5, 4.0)").answer == "yes"
    assert execute("table_max(shares, none)", table).answer == 12
    assert execute("table_average(price, none)", table).answer == pytest.approx(4.75)


@pytest.mark.parametrize("bad", ["", "divide(1, 0)", "add(1)", "foo(1, 2)", "add(#3, 1)", "add(1, 2) subtract(3, 4)"])
def test_bad_programs_raise(bad):
    with pytest.raises(ProgramError):
        execute(bad)


def test_bad_constant_is_a_program_error():
    # Qwen3 32B wrote const_neg_3929 in the main run; it must count as an unusable program, not crash the stage.
    with pytest.raises(ProgramError):
        parse_number("const_neg_3929")
    with pytest.raises(ProgramError):
        execute("add(const_neg_3929, 5)")
    assert parse_number("const_m1") == -1.0 and parse_number("const_100") == 100.0
