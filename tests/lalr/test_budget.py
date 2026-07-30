"""P5: deterministic LALR construction budget -> declared LALRBudgetExceeded.

The budget is a size cap on the construction itself — items materialized
(sum of closure sizes at state creation) plus a derived state cap — so
grammars whose automaton diverges (helm-testsuite class) or whose conflict
detection sits behind tens of millions of items (o27148 class) terminate as
a declared decline instead of hanging. Counters are input-derived: the fire
point is identical run-to-run, warm or cold, on any machine.

Differential scoping (module docstring of grid/lalr/compile.py): table
equality is asserted for under-budget grammars only; over-budget grammars
assert that BOTH algorithms raise the declared class (the oracle may fire
at different counts — LR(1) materializes >= LR(0) items).
"""

import dataclasses

import pytest

from grid.errors import LALRBudgetExceeded, LALRConflictError
from grid.grammar import spec
from grid.grammar.projection import RoleProjection
from grid.lalr.compile import _DEFAULT_ITEM_BUDGET, _STATE_BUDGET_DIVISOR, compile_tables


def _proj(g):
    return RoleProjection.full(g).build()


def _fire(monkeypatch, g, budget: str, algorithm: str = "dp"):
    monkeypatch.setenv("GRID_LALR_BUDGET", budget)
    with pytest.raises(LALRBudgetExceeded) as ei:
        compile_tables(_proj(g), algorithm=algorithm)
    return ei.value


def test_fire_point_is_deterministic(monkeypatch, toy_grammar):
    first = _fire(monkeypatch, toy_grammar, "8")
    for _ in range(2):  # cold rebuilds: fresh projection every time
        again = _fire(monkeypatch, toy_grammar, "8")
        assert (again.states, again.items) == (first.states, first.items)
    # checked at state creation: the counts sit just past the cap, they are
    # the crossing counts, not some arbitrary later snapshot
    assert first.items > 8
    assert first.item_budget == 8
    assert first.state_budget == 1  # 8 // _STATE_BUDGET_DIVISOR
    assert "budget exceeded (size cap)" in str(first)


@pytest.mark.parametrize("algorithm", ["dp", "lr1_merge"])
def test_both_algorithms_raise_declared_class(monkeypatch, toy_grammar, algorithm):
    # over-budget scoping: class equality across constructions (fire counts
    # may differ — LR(1) materializes at least as many items as LR(0))
    _fire(monkeypatch, toy_grammar, "8", algorithm=algorithm)


def test_state_cap_is_a_real_backstop(monkeypatch, toy_grammar):
    # learn the true size, then pick an item budget that can't bind while
    # its derived state cap (budget // 8) can: fires with items under budget
    monkeypatch.setenv("GRID_LALR_BUDGET", "0")
    stats: dict = {}
    compile_tables(_proj(toy_grammar), stats=stats, algorithm="dp")  # lr0_* counters are dp-construction stats
    n_states, n_items = stats["lr0_states"], stats["lr0_items"]
    assert n_items < 4 * n_states  # toy grammar: ~2-3 items/state
    err = _fire(monkeypatch, toy_grammar, str(2 * n_items))
    assert err.items <= err.item_budget  # the STATE compare tripped
    assert err.states > err.state_budget == 2 * n_items // _STATE_BUDGET_DIVISOR


def test_zero_disables_both_caps(monkeypatch, toy_grammar):
    monkeypatch.setenv("GRID_LALR_BUDGET", "0")
    for algorithm in ("dp", "lr1_merge"):
        assert compile_tables(_proj(toy_grammar), algorithm=algorithm).action


def test_default_budget_is_invisible_under_budget(monkeypatch, toy_grammar, sql_grammar):
    # every completing corpus build must be unaffected; toy + sql stand in
    # here, the corpus-wide zero-fires sweep is the perfbench gate
    for g in (toy_grammar, sql_grammar):
        monkeypatch.delenv("GRID_LALR_BUDGET", raising=False)
        on = compile_tables(_proj(g))
        monkeypatch.setenv("GRID_LALR_BUDGET", "0")
        off = compile_tables(_proj(g))
        for f in dataclasses.fields(on):
            assert getattr(on, f.name) == getattr(off, f.name), f.name


def test_budget_precedes_conflict_detection(monkeypatch):
    # conflicts are a fill-stage outcome, AFTER construction: a budget that
    # fires mid-construction reports budget, not conflict (the o27148-class
    # semantics — detecting its conflicts needs the whole 48M-item build)
    g = spec.load("%start s\nA: /a/\ns: e\ne: A | e e\n")
    monkeypatch.setenv("GRID_LALR_BUDGET", "4")
    with pytest.raises(LALRBudgetExceeded):
        compile_tables(_proj(g))
    monkeypatch.setenv("GRID_LALR_BUDGET", "0")
    with pytest.raises(LALRConflictError):
        compile_tables(_proj(g))


def test_stats_untouched_on_fire(monkeypatch, toy_grammar):
    # completion counters never mix with fire counts: stats stays empty,
    # the crossing counts ride the exception
    monkeypatch.setenv("GRID_LALR_BUDGET", "8")
    stats: dict = {}
    with pytest.raises(LALRBudgetExceeded):
        compile_tables(_proj(toy_grammar), stats=stats, algorithm="dp")
    assert stats == {}


def test_decline_never_cached(monkeypatch, tmp_path, toy_grammar):
    # artifact store puts only on success: a budget decline recompiles and
    # re-fires at the identical point on every warm retry
    monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "1")
    monkeypatch.setenv("GRID_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GRID_LALR_BUDGET", "8")
    from grid.serving.artifact_store import load_or_compile_tables

    fires = []
    for _ in range(2):
        with pytest.raises(LALRBudgetExceeded) as ei:
            load_or_compile_tables(_proj(toy_grammar))
        fires.append((ei.value.states, ei.value.items))
    assert fires[0] == fires[1]
    lalr_entries = list(tmp_path.rglob("lalr/*"))
    assert lalr_entries == [], "budget decline must never be cached"


def test_garbage_budget_raises_at_compile(monkeypatch, toy_grammar):
    monkeypatch.setenv("GRID_LALR_BUDGET", "many")
    with pytest.raises(ValueError):
        compile_tables(_proj(toy_grammar))


def test_default_constants_document_the_calibration():
    # full-corpus calibration (P5 sweep, both maskbench legs per schema):
    # largest completing build is o21112 at 20,094,330 items / 926,457
    # states; smallest divergence target is o27148 at 48.17M items. 32M is
    # the log-midpoint of that gap (1.59x above / 1.51x below); the state
    # cap rides the same knob (32M // 8 = 4M, 4.3x the largest completer).
    # Changing either constant requires re-running the corpus sweep.
    assert 20_094_330 < _DEFAULT_ITEM_BUDGET < 48_170_000
    assert _DEFAULT_ITEM_BUDGET == 32_000_000
    assert 926_457 < _DEFAULT_ITEM_BUDGET // _STATE_BUDGET_DIVISOR == 4_000_000
