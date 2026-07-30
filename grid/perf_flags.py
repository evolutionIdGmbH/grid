"""Call-time readers for the GRID_PERF_* environment flags.

Single source of truth for every GRID_PERF_* flag the library reads: one
named reader per flag, each preserving that flag's exact historical value
grammar. The grammars deliberately differ per flag ("" enables
ARTIFACT_STORE via ``!= "0"`` but leaves FACTORED_SCANNER off via
``== "1"``); "normalizing" them for consistency would be a semantic change
and is permanently out of scope for this module.

Flag table:

    flag                        reader                      grammar
    --------------------------  --------------------------  -----------------
    GRID_PERF_ARTIFACT_STORE    artifact_store_enabled()    != "0" (default
                                                            off; any other
                                                            value, incl. "",
                                                            enables)
    GRID_PERF_FACTORED_SCANNER  factored_scanner_enabled()  == "1" (default
                                                            on: unset = "1";
                                                            "0" = legacy
                                                            eager builder)
    GRID_PERF_FACTORED_BUDGET   factored_budget(default)    int(); ValueError
                                                            on garbage
    GRID_PERF_COMPONENT_BUDGET  component_budget(default)   int(); "0" = cap
                                                            disabled (None:
                                                            legacy eager
                                                            component builds);
                                                            ValueError on
                                                            garbage
    GRID_PERF_LALR_DP           lalr_algorithm()            "dp" if == "1"
                                                            (default on:
                                                            unset = "1") else
                                                            "lr1_merge"
    GRID_PERF_HASHCONS          hashcons_components()       ""/"0" = none,
                                                            "1"/"all" = all,
                                                            else comma list
                                                            (default on:
                                                            unset =
                                                            "norm,dedupe")
    GRID_PERF_HASHCONS_DEBUG    hashcons_debug_enabled()    == "1"
    GRID_PERF_DIRECT_EMIT       direct_emit_enabled()       == "1" (default
                                                            off; P2 bake-off
                                                            pending)
    GRID_PERF_DIRECT_EMIT_CHECK direct_emit_check_enabled() == "1"

Contract (enforced by tests/test_perf_flags.py):

- Leaf module: imports stdlib ``os`` only. Importing any grid submodule here
  would hand grid.jsonschema's flag-off fast path (the pre-check in
  grid/jsonschema/__init__.py exists to skip the ~3-7ms grid.serving import
  chain) the very import cost it avoids, or create an import cycle.
- Every reader hits os.environ at call time and is NEVER cached (no
  lru_cache, no module-level snapshot): tests monkeypatch these flags
  between calls, and long-lived serving processes may flip them.

Non-GRID_PERF flags (GRID_CACHE_DIR, GRID_ADMIT_WARM, GRID_DEFER, ...) are
out of scope: they are not performance-path selectors and keep their
existing read sites.
"""

import os


def artifact_store_enabled() -> bool:
    """GRID_PERF_ARTIFACT_STORE: versioned on-disk compile-artifact store
    (grid/serving/artifact_store.py). Default off; any value other than
    "0" — including "" — enables. grid/jsonschema/__init__.py uses the
    negation of this exact predicate to skip importing grid.serving on
    flag-off fast builds; both sides must stay this function."""
    return os.environ.get("GRID_PERF_ARTIFACT_STORE", "0") != "0"


def factored_scanner_enabled() -> bool:
    """GRID_PERF_FACTORED_SCANNER: per-terminal-DFA product scanner path
    (grid/lexer/factored.py) behind dfa.build_scanner. Default ON since the
    v0.3.0 full-corpus run (tmp/mb-grid-v030rc2); only "1" (now the unset
    default) enables — "0" or any other value is the kill switch restoring
    the eager union builder."""
    return os.environ.get("GRID_PERF_FACTORED_SCANNER", "1") == "1"


def factored_budget(default: int) -> int:
    """GRID_PERF_FACTORED_BUDGET -> product-state budget for the factored
    scanner's eager materialization. The default is injected by the call
    site (factored._DEFAULT_BUDGET stays in grid/lexer/factored.py); a
    non-integer value raises ValueError exactly like the historical inline
    int() read."""
    return int(os.environ.get("GRID_PERF_FACTORED_BUDGET", str(default)))


def component_budget(default: int) -> int | None:
    """GRID_PERF_COMPONENT_BUDGET -> per-terminal component state budget for
    the factored scanner (factored._build_component; sub-flag of
    GRID_PERF_FACTORED_SCANNER, never read on the eager path). Over-budget
    components come back as demand-driven LazyTerminalDFAs instead of eager
    subset constructions — the substring-union terminal family builds ~2^k
    eager states (BAKEOFF.md F1). "0" is the kill switch: returns None = cap
    disabled = the pre-cap eager component builds (and their hang on family
    schemas). Other values are int() with the default injected by the call
    site (factored._DEFAULT_COMPONENT_BUDGET), ValueError on garbage —
    the factored_budget grammar except for the "0" special case."""
    val = int(os.environ.get("GRID_PERF_COMPONENT_BUDGET", str(default)))
    return None if val == 0 else val


def lalr_algorithm() -> str:
    """GRID_PERF_LALR_DP -> "dp" (LR(0) + DeRemer-Pennello lookaheads, the
    default since the v0.3.0 full-corpus run) when "1" (now the unset
    default), else "lr1_merge" (canonical LR(1) merge — the kill-switch value
    "0", kept as the construction-independent oracle), for
    lalr.compile_tables."""
    return "dp" if os.environ.get("GRID_PERF_LALR_DP", "1") == "1" else "lr1_merge"


# 'rulefor' (structural rule_for memo) is planned but NOT implemented; the
# parser must not advertise components that silently no-op
HASHCONS_COMPONENTS = frozenset({"norm", "dedupe"})


def hashcons_components(value: str | None = None) -> frozenset[str]:
    """GRID_PERF_HASHCONS -> enabled component set ('' / '0' = none,
    '1' / 'all' = every component, else a comma list; unknown names
    ignored). Unset defaults to "norm,dedupe" — the exact configuration the
    v0.3.0 full-corpus run measured (pinned by name, NOT "all", so a future
    component added to HASHCONS_COMPONENTS needs its own default decision);
    "0" (or "") is the kill switch restoring legacy un-consed
    normalization. Moved verbatim from grid/jsonschema/normalize.py, which
    keeps back-compat aliases (_hashcons_components /
    _HASHCONS_COMPONENTS)."""
    if value is None:
        value = os.environ.get("GRID_PERF_HASHCONS", "norm,dedupe")
    value = value.strip()
    if value in ("", "0"):
        return frozenset()
    if value in ("1", "all"):
        return HASHCONS_COMPONENTS
    return frozenset(p.strip() for p in value.split(",")) & HASHCONS_COMPONENTS


def hashcons_debug_enabled() -> bool:
    """GRID_PERF_HASHCONS_DEBUG=1: re-digest every memoized node at the end
    of a normalize() run to catch in-place mutation of shared subtrees."""
    return os.environ.get("GRID_PERF_HASHCONS_DEBUG", "0") == "1"


def direct_emit_enabled() -> bool:
    """GRID_PERF_DIRECT_EMIT: build DialectGrammar objects straight from the
    compiler's GrammarParts manifest (spec.DialectGrammar.from_parts) in
    grid.jsonschema.compile_json_schema_grammar, skipping the .grid text
    render + regex re-parse on schema->grammar compiles. Default off (P2
    bake-off pending); only "1" enables. Text emission stays the permanent
    debug/audit path and the differential oracle; artifact-store schema_src
    HITS keep text -> spec.load regardless of this flag."""
    return os.environ.get("GRID_PERF_DIRECT_EMIT", "0") == "1"


def direct_emit_check_enabled() -> bool:
    """GRID_PERF_DIRECT_EMIT_CHECK=1: from_parts additionally renders the
    manifest to text, spec.load()s it, and asserts both paths agree — full
    grammar identity (start/ignored/terminals/productions/terminal_order/
    fingerprint) for valid manifests, GrammarInvalid message parity
    otherwise. The permanent CI oracle against renderer/object-builder
    drift."""
    return os.environ.get("GRID_PERF_DIRECT_EMIT_CHECK", "0") == "1"
