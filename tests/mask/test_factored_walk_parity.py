"""Factored-scanner mask parity (0.3.x #4): the lazy over-budget regime must
serve IDENTICAL masks through the pure-Python spec walk.

- _walk_py WalkResult equality eager-vs-lazy (ci ids, CDEntry events/segments/
  remainders — all numbering-free) is the direct mask-equivalence proof.
- Consumer gates: lazy DFAs never reach the Rust kernel (walk() falls back to
  the spec path, make_verdict_kernel returns None) and never take genN cache
  keys (demand-order state ids are instance-local; T2 is shared per-dialect
  across template instances — aliasing them is the forbidden wrong-mask class).
- End-to-end: a flag-off guide and a flag-on/budget-0 guide driven in lockstep
  produce identical per-step mask id sets and statuses.
"""

import random

import numpy as np
import pytest

from grid.generate import build_guide
from grid.guide import COMPLETE
from grid.lalr.stack import allowed_terminals, root_node, shift_terminal
from grid.lexer.factored import LazyProductDFA, build_factored_scanner
from grid.mask.producer import MaskProducer
from grid.models.tokenizer_adapter import MockTokenizer
from grid.trie.build import build_trie
from grid.trie.walk import _walk_py, make_verdict_kernel, walk


def _lazy_of(grammar) -> LazyProductDFA:
    lazy = build_factored_scanner(grammar.terminals, grammar.terminal_order, budget=0)
    assert isinstance(lazy, LazyProductDFA)
    return lazy


def _priority(tables) -> dict[int, tuple[int, int]]:
    return {
        tid: (0 if tid in tables.literal_terminal_ids else 1, tid)
        for tid in range(tables.n_terminals)
    }


def _a_sets(tables) -> list[frozenset[int]]:
    node = root_node(tables)
    out = [allowed_terminals(tables, node)]
    for _ in range(3):
        a = out[-1]
        nxt = None
        for t in sorted(a):
            nxt = shift_terminal(tables, node, t)
            if nxt is not None:
                break
        if nxt is None:
            break
        node = nxt
        out.append(allowed_terminals(tables, node))
    out.append(frozenset(range(tables.n_terminals - 1)) - tables.ignored_terminal_ids)
    return out


def _walk_parity(tables, eager, lazy, trie, remainders) -> None:
    prio = _priority(tables)
    ign = tables.ignored_terminal_ids
    for A in _a_sets(tables):
        for rem in remainders:
            if eager.scan_state(rem) < 0:
                continue
            r_e = _walk_py(trie, eager, rem, A, ign, prio)
            r_f = _walk_py(trie, lazy, rem, A, ign, prio)
            assert r_e == r_f, (rem, sorted(A))


def test_walk_parity_sql(sql_grammar, sql_tables, sql_dfa, sql_tokenizer):
    lazy = _lazy_of(sql_grammar)
    trie = build_trie(sql_tokenizer)
    _walk_parity(sql_tables, sql_dfa, lazy, trie,
                 [b"", b"sel", b"select", b" ", b"1", b"users", b"se"])


def test_walk_parity_toy(toy_grammar, toy_tables, toy_dfa, toy_tokenizer):
    lazy = _lazy_of(toy_grammar)
    trie = build_trie(toy_tokenizer)
    _walk_parity(toy_tables, toy_dfa, lazy, trie,
                 [b"", b"fo", b"12", b"1", b" "])


def test_lazy_never_reaches_kernel(sql_grammar, sql_tables, sql_tokenizer):
    lazy = _lazy_of(sql_grammar)
    assert make_verdict_kernel(sql_tables, lazy, None) is None
    trie = build_trie(sql_tokenizer)
    A = allowed_terminals(sql_tables, root_node(sql_tables))
    got = walk(trie, lazy, b"", A, sql_tables.ignored_terminal_ids, _priority(sql_tables))
    assert got.groups is None  # the Python spec path, kernel present or not
    assert isinstance(got.ci_tokens, tuple)


def test_lazy_takes_raw_schema_scoped_key(sql_grammar, sql_tables, sql_tokenizer):
    trie = build_trie(sql_tokenizer)
    vocab = max(sql_tokenizer.vocabulary.values()) + 1
    A = allowed_terminals(sql_tables, root_node(sql_tables))

    lazy_p = MaskProducer(tables=sql_tables, dfa=_lazy_of(sql_grammar), trie=trie,
                          vocab_size=vocab, schema_fingerprint="fp-lazy")
    lazy_p.set_genn_keys(True)
    key = lazy_p.cache_key(b"select", A)
    assert key == ("generic", b"select", tuple(sorted(A)), "fp-lazy")

    eager_p = MaskProducer(tables=sql_tables,
                           dfa=build_factored_scanner(sql_grammar.terminals,
                                                      sql_grammar.terminal_order,
                                                      budget=10**9),
                           trie=trie, vocab_size=vocab, schema_fingerprint="fp-mat")
    eager_p.set_genn_keys(True)
    assert eager_p.cache_key(b"select", A)[0] == "genN"  # materialized keeps genN


@pytest.mark.parametrize("source_fixture,tokens_seed", [("toy", 11), ("sql", 13)])
def test_guide_end_to_end_lockstep(request, monkeypatch, source_fixture, tokens_seed):
    source = request.getfixturevalue(f"{source_fixture}_source")
    tokenizer = request.getfixturevalue(f"{source_fixture}_tokenizer")

    monkeypatch.delenv("GRID_PERF_FACTORED_SCANNER", raising=False)
    g_off = build_guide(source, MockTokenizer(extra_tokens=tokenizer.extra_tokens))
    monkeypatch.setenv("GRID_PERF_FACTORED_SCANNER", "1")
    monkeypatch.setenv("GRID_PERF_FACTORED_BUDGET", "0")
    g_on = build_guide(source, MockTokenizer(extra_tokens=tokenizer.extra_tokens))
    assert isinstance(g_on.dfa, LazyProductDFA)

    rng = random.Random(tokens_seed)
    s_off, s_on = g_off.initial_state, g_on.initial_state
    for _step in range(30):
        ids_off, _ = g_off._mask_ids(s_off)
        ids_on, _ = g_on._mask_ids(s_on)
        set_off = {int(t) for t in np.asarray(ids_off)}
        set_on = {int(t) for t in np.asarray(ids_on)}
        assert set_off == set_on, _step
        assert s_off.status == s_on.status
        choices = sorted(set_off - {g_off.eos_token_id})
        if not choices or s_off.status == COMPLETE:
            break
        tok = rng.choice(choices)
        s_off = g_off.get_next_state(s_off, tok)
        s_on = g_on.get_next_state(s_on, tok)
