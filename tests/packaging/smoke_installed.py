"""Smoke test for an INSTALLED grid-guardrail (wheel or sdist), run outside the repo.

CI and the release workflow install the built package into a fresh virtualenv,
change to a directory without the source tree, and run this file. It fails if a
runtime data file is missing from the package (0.2.5-0.4.1 shipped without the
statechart YAML files, so every CFG grammar load raised FileNotFoundError) or if
the CFG generation path does not work end to end. Not collected by pytest
(no test_ prefix): it must not run against the source checkout.
"""

from __future__ import annotations

import sys
from importlib.resources import files
from pathlib import Path

import grid
from grid import generate
from grid._statecharts.engine import all_chart_names, load_chart
from grid.grammar import spec
from grid.jsonschema import compile_json_schema
from grid.models.mock import MockModel
from grid.models.tokenizer_adapter import MockTokenizer
from grid.samplers import greedy

GRAMMAR = """%start entry
%ignore WS
WS: /[ ]+/
NUM: /[1-9][0-9]*/
entry: "(" "pay" NUM ")"
"""


def main() -> None:
    here = Path(grid.__file__).resolve().parent
    if (here.parent / "pyproject.toml").exists():
        sys.exit(f"smoke test imported grid from a source tree ({here}); run it from an installed package")

    expected = {"compiled_grammar", "dialect_grammar", "logits_processor",
                "registry_slot", "role_projection", "schema_lexicon"}
    shipped = set(all_chart_names())
    missing = expected - shipped
    assert not missing, f"statechart YAML missing from the package: {sorted(missing)}"
    for name in sorted(shipped):
        load_chart(name)  # parses and validates every shipped chart
    assert files("grid.jsonschema").joinpath("SUPPORT.md").is_file(), "grid/jsonschema/SUPPORT.md missing"

    spec.load(GRAMMAR)  # raised FileNotFoundError in 0.4.1
    tok = MockTokenizer(extra_tokens=("(", ")", " ", "pay", "12"))
    result = generate.cfg(MockModel(tok, seed=0), GRAMMAR, sampler=greedy(), audit=True)("", max_tokens=16, seed=0)
    compact = "".join(result.text.split())  # the grammar ignores whitespace
    assert compact.startswith("(pay") and compact.endswith(")"), result.text
    assert result.audit.verify_chain(), "audit chain does not verify"

    compile_json_schema({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]})
    print(f"grid-guardrail {grid.__version__} installed package OK: {len(shipped)} statecharts, "
          f"CFG generation {result.text!r}, audit chain valid, JSON Schema compile OK")


if __name__ == "__main__":
    main()
