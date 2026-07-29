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
    GRID_PERF_FACTORED_SCANNER  factored_scanner_enabled()  == "1"
    GRID_PERF_NFA_LIVE          nfa_live_mode()             "0" | "verify" |
                                                            "nfa" (default;
                                                            all other values
                                                            mean "nfa")
    GRID_PERF_FACTORED_BUDGET   factored_budget(default)    int(); ValueError
                                                            on garbage
    GRID_PERF_LALR_DP           lalr_algorithm()            "dp" if == "1"
                                                            else "lr1_merge"
    GRID_PERF_HASHCONS          hashcons_components()       ""/"0" = none,
                                                            "1"/"all" = all,
                                                            else comma list
    GRID_PERF_HASHCONS_DEBUG    hashcons_debug_enabled()    == "1"

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
    """GRID_PERF_FACTORED_SCANNER=1: per-terminal-DFA product scanner path
    (grid/lexer/factored.py) behind dfa.build_scanner. Only "1" enables."""
    return os.environ.get("GRID_PERF_FACTORED_SCANNER", "0") == "1"


def nfa_live_mode() -> str:
    """GRID_PERF_NFA_LIVE -> live-set / co-accessibility computation:
    "0" = legacy DFA-graph pass (fixpoint in dfa.py, reverse BFS per
    component in factored.py), "verify" = both + cross-check, "nfa" = NFA
    terminal-reach (dfa._terminal_reach), the default. Every raw value
    other than "0"/"verify" normalizes to "nfa"; consumers must branch only
    on the ``!= "0"`` / ``== "verify"`` predicates, which are invariant
    under that normalization."""
    raw = os.environ.get("GRID_PERF_NFA_LIVE", "1")
    if raw == "0":
        return "0"
    return "verify" if raw == "verify" else "nfa"


def factored_budget(default: int) -> int:
    """GRID_PERF_FACTORED_BUDGET -> product-state budget for the factored
    scanner's eager materialization. The default is injected by the call
    site (factored._DEFAULT_BUDGET stays in grid/lexer/factored.py); a
    non-integer value raises ValueError exactly like the historical inline
    int() read."""
    return int(os.environ.get("GRID_PERF_FACTORED_BUDGET", str(default)))


def lalr_algorithm() -> str:
    """GRID_PERF_LALR_DP=1 -> "dp" (LR(0) + DeRemer-Pennello lookaheads),
    else "lr1_merge" (canonical LR(1) merge), for lalr.compile_tables."""
    return "dp" if os.environ.get("GRID_PERF_LALR_DP", "0") == "1" else "lr1_merge"


# 'rulefor' (structural rule_for memo) is planned but NOT implemented; the
# parser must not advertise components that silently no-op
HASHCONS_COMPONENTS = frozenset({"norm", "dedupe"})


def hashcons_components(value: str | None = None) -> frozenset[str]:
    """GRID_PERF_HASHCONS -> enabled component set ('' / '0' = none,
    '1' / 'all' = every component, else a comma list; unknown names
    ignored). Moved verbatim from grid/jsonschema/normalize.py, which keeps
    back-compat aliases (_hashcons_components / _HASHCONS_COMPONENTS)."""
    if value is None:
        value = os.environ.get("GRID_PERF_HASHCONS", "")
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
