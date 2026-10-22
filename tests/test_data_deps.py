from cascade.deps import build


def test_load_and_program_check(convs):
    assert [len(c.turns) for c in convs] == [4, 3, 3]
    assert convs[0].issues == [] and convs[1].issues == []
    assert convs[2].issues and "turn 2" in convs[2].issues[0]  # gold 999 does not match the program


def test_context_has_table(convs):
    ctx = convs[0].context()
    assert "net revenue | $ 206588" in ctx and "amounts in thousands" in ctx


def test_graph_edges(convs):
    g = build(convs[0])
    assert g.direct == {(0, 2), (1, 2), (0, 3), (1, 3), (2, 3)}
    assert g.parents(3) == [0, 1, 2]
    assert g.descendants(0) == {2, 3}
    assert g.path_count(0, 3) == 2  # 0->3 and 0->2->3
    assert g.depth(0, 3) == 1 and g.depth(2, 3) == 1 and g.depth(1, 0) is None


def test_independent_turns_have_no_edges(convs):
    assert build(convs[1]).direct == frozenset()
