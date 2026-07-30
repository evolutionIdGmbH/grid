"""GrammarParts: the object manifest between the JSON-schema compiler and the
.grid text emitter / grammar loader (P2 direct emission).

One manifest, two consumers — structurally, so they cannot drift:

- :func:`render_text` renders the exact .grid text the compiler's legacy
  string emitter produced (byte-identical; gated over the full corpus), the
  permanent debug/audit path and the differential oracle;
- :meth:`grid.grammar.spec.DialectGrammar.from_parts` builds the grammar
  object directly, skipping the render + regex re-parse.

Fixed-header contract (this emitter always produces exactly this shape):

- ``%start start`` — the start symbol is the synthetic rule ``start``, whose
  single production body is ``start_target``;
- ``%ignore WS`` — ``terminal_defs[0]`` is always ``("WS", "[ \\t\\n\\r]+")``
  and WS is the only ignored terminal;
- named terminal definitions before any rule, in ``terminal_defs`` order
  (this order IS the decl_index numbering the loader assigns);
- rule alternatives are space-joined token tuples; the empty tuple is an
  epsilon alternative (rendered as an empty string, exactly like the legacy
  emitter's ``""``/``"|EPS|"`` forms). Alt tokens starting with ``"`` are
  quoted literal terminals (JSON punctuation and keywords), everything else
  is a named terminal or rule reference.

Leaf module: stdlib-only imports (the grid/perf_flags.py discipline). The
compiler's flag-off fast path must not pull spec.py's statechart machinery,
and spec.py must be importable without the jsonschema package.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GrammarParts:
    """Object form of an emitted .grid grammar (see fixed-header contract)."""

    # (name, regex source) in emission order; index 0 is always WS. The
    # pattern is the final text between the slashes (degradation aliasing
    # and key-literal escaping already applied by the compiler).
    terminal_defs: tuple[tuple[str, str], ...]
    # body token of the synthetic ``start:`` rule (the root rule's name)
    start_target: str
    # (rule name, alternatives) in emission order; each alternative is a
    # token tuple, () = epsilon
    rules: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...]


def render_text(parts: GrammarParts) -> str:
    """GrammarParts -> .grid source, byte-identical to the legacy emitter."""
    lines = ["%start start", "%ignore WS"]
    for name, pattern in parts.terminal_defs:
        lines.append(f"{name}: /{pattern}/")
    lines.append(f"start: {parts.start_target}")
    for name, alts in parts.rules:
        lines.append(f"{name}: " + " | ".join(" ".join(alt) for alt in alts))
    return "\n".join(lines) + "\n"
