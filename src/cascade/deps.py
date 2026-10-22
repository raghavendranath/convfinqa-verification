"""Turn-dependency graph for a conversation, recovered from the gold programs.

ConvFinQA writes each turn's program in full (no cross-turn references), so dependence is
recovered by value: turn k *uses* turn j < k when turn j's gold answer equals a literal in
turn k's program or an intermediate result of it. Lookup turns ("what was revenue in 2019?")
have the looked-up number as their answer, so later turns that compute with it link back.

This is a heuristic. Coincidental equal values can create false edges, so the graph is
audited by hand on a sample of 50 dev conversations (``python -m cascade audit-graph``
writes the sheet); the audited precision and recall are reported in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import Conversation
from .dsl import ProgramError, execute
from .match import match


@dataclass(frozen=True)
class Graph:
    n: int
    direct: frozenset[tuple[int, int]]  # (j, k): turn k directly uses turn j's answer

    def parents(self, k: int) -> list[int]:
        return sorted(j for j, kk in self.direct if kk == k)

    def descendants(self, j: int) -> set[int]:
        out: set[int] = set()
        frontier = [j]
        while frontier:
            x = frontier.pop()
            for a, b in self.direct:
                if a == x and b not in out:
                    out.add(b)
                    frontier.append(b)
        return out

    def path_count(self, j: int, k: int) -> int:
        """Number of distinct directed paths from j to k (0 if k does not depend on j)."""
        memo: dict[int, int] = {}

        def count(x: int) -> int:
            if x == k:
                return 1
            if x in memo:
                return memo[x]
            memo[x] = sum(count(b) for a, b in self.direct if a == x and b <= k)
            return memo[x]

        return count(j) if j < k else 0

    def depth(self, j: int, k: int) -> int | None:
        """Shortest number of hops from j to k, or None if k does not depend on j."""
        if j >= k:
            return None
        dist = {j: 0}
        frontier = [j]
        while frontier:
            nxt = []
            for x in frontier:
                for a, b in self.direct:
                    if a == x and b not in dist:
                        dist[b] = dist[x] + 1
                        nxt.append(b)
            frontier = nxt
        return dist.get(k)


def _values(program: str, table: list[list[str]]) -> list[float]:
    try:
        ex = execute(program, table)
    except ProgramError:
        return []
    vals = list(ex.literals)
    # Intermediate results, excluding the final answer (a turn does not "use" a repeat of itself).
    vals += [r for r in ex.results[:-1] if not isinstance(r, str)]
    return vals


def build(conv: Conversation) -> Graph:
    edges: set[tuple[int, int]] = set()
    for k, tk in enumerate(conv.turns):
        vals = _values(tk.program, conv.table)
        # A single-step lookup turn has its looked-up literal as the answer; it uses no earlier turn
        # unless that literal equals an earlier answer (a repeated lookup), which we keep.
        for j in range(k):
            aj = conv.turns[j].answer
            if isinstance(aj, str):
                continue
            if any(match(v, aj) for v in vals):
                edges.add((j, k))
    return Graph(len(conv.turns), frozenset(edges))
