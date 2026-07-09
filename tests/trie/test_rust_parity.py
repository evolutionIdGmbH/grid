"""grid_core parity: the Rust walk must be behaviorally identical to the Python
walk (the executable specification) across grammars, lexicons, and random states.

The Rust kernel groups CD entries in-kernel and returns representatives only, so
parity is asserted at the semantically meaningful level:
- ci token sets are equal;
- the CD id-partition into groups is identical;
- per matched group, every verdict-relevant representative field is equal
  (candidate sequences always; segments/remainder when lexicons apply; the tail
  live set always — via the group key reconstruction).
"""

import random

import pytest

import grid.trie.walk as W
from grid.generate import build_guide
from grid.guide import COMPLETE
from grid.mask.cache import make_entry

pytestmark = pytest.mark.skipif(W._grid_core is None, reason="grid_core not installed")


def _entries_for(guide, result):
    """Normalize either walk output into {frozenset(group ids): verdict-relevant key}.

    Rust results arrive alias-expanded and grouped in-kernel; Python results are
    expanded/grouped here — mirroring MaskProducer.masks exactly."""
    live_of = lambda rem: guide.dfa.live[guide.dfa.scan_state(rem)]  # noqa: E731
    lex = guide.lexicons is not None
    if result.groups is not None:
        ci = result.ci_tokens
        expand = None
    else:
        ci = tuple(sorted(t for tid in result.ci_tokens for t in guide.trie.expand(tid)))
        expand = guide.trie.expand
    entry = make_entry(
        ("parity",), ci, result.cd_entries, guide.vocab_size,
        live_of=live_of, lexicon_sensitive=lex, expand=expand,
        precomputed_groups=result.groups,
        # verdict-equivalence grouping context, exactly as MaskProducer.masks
        # passes it — required for group-partition parity with the kernel key
        lexicons=guide.lexicons,
        ignored=guide.tables.ignored_terminal_ids,
        priority=guide.producer._priority,
    )
    out = {}
    for g in entry.cd_groups:
        rep = g.representative
        key = (
            tuple(ev.candidates for ev in rep.events),
            rep.segments if lex else None,
            rep.remainder if lex else None,
            live_of(rep.remainder),
        )
        out[frozenset(g.token_ids)] = key
    return entry.ci_tokens, out


def _compare(guide, state, ctx: str):
    A = guide.producer.allowed(state.stack)
    args = (
        guide.trie, guide.dfa, state.lexer.remainder, A,
        guide.tables.ignored_terminal_ids, guide.producer._priority, guide.lexicons,
    )
    rust_ci, rust_groups = _entries_for(guide, W.walk(*args))
    py_ci, py_groups = _entries_for(guide, W._walk_py(*args))
    assert sorted(rust_ci) == sorted(py_ci), f"{ctx}: ci mismatch"
    assert set(rust_groups) == set(py_groups), f"{ctx}: group partition mismatch"
    for ids, key in rust_groups.items():
        assert key == py_groups[ids], f"{ctx}: group key mismatch for {sorted(ids)[:4]}"


def _walk_states(guide, seed: int, steps: int, ctx: str):
    rng = random.Random(seed)
    state = guide.initial_state
    for step in range(steps):
        _compare(guide, state, f"{ctx} step {step}")
        ids, _ = guide._mask_ids(state)
        tok = rng.choice(sorted(set(ids) - {guide.eos_token_id}) or sorted(ids))
        state = guide.get_next_state(state, tok)
        if state.status == COMPLETE:
            break


def test_parity_toy(toy_source, toy_tokenizer):
    _walk_states(build_guide(toy_source, toy_tokenizer), seed=5, steps=12, ctx="toy")


def test_parity_wide_grammar(wide_source, wide_tokenizer):
    """>64 terminals: the [u64; W=2] kernel mask path vs the Python walk."""
    guide = build_guide(wide_source, wide_tokenizer)
    assert guide.tables.n_terminals > 64  # guards the fixture's purpose
    _walk_states(guide, seed=13, steps=12, ctx="wide")


def test_parity_sql_with_lexicons(sql_source, sql_tokenizer, sql_grammar):
    from grid.grammar.projection import RoleProjection
    from grid.lalr.compile import compile_tables
    from grid.policy.schema import SchemaSnapshot

    schema = SchemaSnapshot.from_dict({"users": ["id", "name"], "orders": ["id", "total"]})
    proj = RoleProjection.full(sql_grammar).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    guide = build_guide(sql_source, sql_tokenizer, projection=proj,
                        lexicons=schema.lexicons(tables), schema_fingerprint=schema.fingerprint)
    _walk_states(guide, seed=7, steps=10, ctx="sql")
