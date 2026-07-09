"""Statechart engine (DESIGN.md SS5 conventions).

The YAML files in this directory are the single source of truth for every
explicit-trigger entity machine. Entities embed a ``Statechart`` and route all
state changes through :meth:`Statechart.fire`; any (state, trigger) pair not in
the YAML raises :class:`grid.errors.IllegalTransition`. SS9's statechart tests are
generated from the same YAML, never from documentation tables.

YAML schema::

    entity: DialectGrammar
    initial: DRAFT
    terminal: [FROZEN, INVALID]
    transitions:
      - {from: DRAFT,     trigger: parse_ok,    to: PARSED}
      - {from: DRAFT,     trigger: parse_error, to: INVALID}
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import yaml

from grid.errors import IllegalTransition

_CHART_DIR = pathlib.Path(__file__).parent


@dataclass(frozen=True)
class ChartSpec:
    entity: str
    initial: str
    terminal: frozenset[str]
    transitions: dict[tuple[str, str], str]  # (from, trigger) -> to

    @property
    def states(self) -> frozenset[str]:
        out = {self.initial} | set(self.terminal)
        for (frm, _), to in self.transitions.items():
            out.add(frm)
            out.add(to)
        return frozenset(out)


def load_chart(name: str) -> ChartSpec:
    """Load ``<name>.yaml`` from this directory into a validated ChartSpec."""
    raw = yaml.safe_load((_CHART_DIR / f"{name}.yaml").read_text())
    transitions: dict[tuple[str, str], str] = {}
    for row in raw["transitions"]:
        key = (row["from"], row["trigger"])
        if key in transitions:
            raise ValueError(f"{raw['entity']}: duplicate transition {key}")
        transitions[key] = row["to"]
    spec = ChartSpec(
        entity=raw["entity"],
        initial=raw["initial"],
        terminal=frozenset(raw.get("terminal", [])),
        transitions=transitions,
    )
    for term in spec.terminal:
        for (frm, trig), _to in spec.transitions.items():
            if frm == term:
                raise ValueError(f"{spec.entity}: terminal state {term} has outgoing transition {trig}")
    return spec


def all_chart_names() -> list[str]:
    return sorted(p.stem for p in _CHART_DIR.glob("*.yaml"))


@dataclass
class Statechart:
    """A live machine instance owned by an entity."""

    spec: ChartSpec
    state: str = field(default="")

    def __post_init__(self) -> None:
        if not self.state:
            self.state = self.spec.initial

    def fire(self, trigger: str) -> str:
        to = self.spec.transitions.get((self.state, trigger))
        if to is None:
            raise IllegalTransition(self.spec.entity, self.state, trigger)
        self.state = to
        return to

    def is_terminal(self) -> bool:
        return self.state in self.spec.terminal
