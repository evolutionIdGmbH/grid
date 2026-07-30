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


# -- v8 kernel-lazy leg (GRID_PERF_KERNEL_LAZY): the in-kernel lazy product ---
#
# GRID_PERF_COMPONENT_BUDGET=1 forces EVERY component over its budget (any
# non-degenerate terminal DFA has >= 2 subset states), so build_scanner
# returns an all-lazy LazyProductDFA — the walk must route it through the v8
# kernel and match _walk_py (the executable specification) exactly. State
# ids are instance-local demand-order on BOTH sides and excluded from
# comparison by construction: _entries_for compares ci sets, group
# partitions, and verdict-relevant representative fields only.

# _USE_RUST folds GRID_NO_RUST=1: these legs assert the kernel path is TAKEN,
# so a disabled kernel must skip them (walk() itself degrades to the spec)
_KERNEL_V8 = (W._USE_RUST
              and getattr(W._grid_core, "__kernel_version__", 0) >= 8)


def _force_lazy(monkeypatch):
    # pin the regime under test: the legacy CI leg disables the factored path
    # entirely, and these tests exercise the lazy product, not the leg env
    monkeypatch.setenv("GRID_PERF_FACTORED_SCANNER", "1")
    monkeypatch.setenv("GRID_PERF_COMPONENT_BUDGET", "1")
    monkeypatch.setenv("GRID_PERF_KERNEL_LAZY", "1")


needs_v8 = pytest.mark.skipif(
    not _KERNEL_V8, reason="grid_core v8 (lazy scanner) not installed/enabled")


@needs_v8
def test_parity_toy_lazy(toy_source, toy_tokenizer, monkeypatch):
    _force_lazy(monkeypatch)
    guide = build_guide(toy_source, toy_tokenizer)
    assert getattr(guide.dfa, "lazy", False), "forced-lazy fixture must build a lazy product"
    _walk_states(guide, seed=5, steps=12, ctx="toy-lazy")


@needs_v8
def test_parity_wide_grammar_lazy(wide_source, wide_tokenizer, monkeypatch):
    """>64 terminals: the lazy product under [u64; W=2] kernel masks."""
    _force_lazy(monkeypatch)
    guide = build_guide(wide_source, wide_tokenizer)
    assert guide.tables.n_terminals > 64
    assert getattr(guide.dfa, "lazy", False)
    _walk_states(guide, seed=13, steps=12, ctx="wide-lazy")


@needs_v8
def test_parity_sql_with_lexicons_lazy(sql_source, sql_tokenizer, sql_grammar, monkeypatch):
    """Lexicon walks consult lexeme_ok/prefix_ok against lazy live sets."""
    from grid.grammar.projection import RoleProjection
    from grid.lalr.compile import compile_tables
    from grid.policy.schema import SchemaSnapshot

    _force_lazy(monkeypatch)
    schema = SchemaSnapshot.from_dict({"users": ["id", "name"], "orders": ["id", "total"]})
    proj = RoleProjection.full(sql_grammar).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    guide = build_guide(sql_source, sql_tokenizer, projection=proj,
                        lexicons=schema.lexicons(tables), schema_fingerprint=schema.fingerprint)
    assert getattr(guide.dfa, "lazy", False)
    _walk_states(guide, seed=7, steps=10, ctx="sql-lazy")


def test_lazy_stays_off_kernel_with_kill_switch(toy_source, toy_tokenizer, monkeypatch):
    """GRID_PERF_KERNEL_LAZY=0 restores the wave-B regime regardless of the
    shipped default: lazy DFAs walk _walk_py (spec-path WalkResult,
    groups=None), never the kernel."""
    monkeypatch.setenv("GRID_PERF_FACTORED_SCANNER", "1")  # pin: legacy leg disables factored
    monkeypatch.setenv("GRID_PERF_COMPONENT_BUDGET", "1")
    monkeypatch.setenv("GRID_PERF_KERNEL_LAZY", "0")
    guide = build_guide(toy_source, toy_tokenizer)
    assert getattr(guide.dfa, "lazy", False)
    A = guide.producer.allowed(guide.initial_state.stack)
    result = W.walk(guide.trie, guide.dfa, b"", A,
                    guide.tables.ignored_terminal_ids, guide.producer._priority, None)
    assert result.groups is None, "flag-off lazy walk must be the Python spec path"


@needs_v8
def test_lazy_kernel_outputs_are_id_independent(toy_source, toy_tokenizer, monkeypatch):
    """Two kernel walkers over equal automata, driven through the SAME
    configurations in OPPOSITE orders (so their demand-order state interning
    diverges), must emit byte-identical outputs per configuration — ci id
    buffers and raw registration payloads (events/segments/remainder/ids).
    This pins the invariant that no kernel output embeds a scanner state id."""
    _force_lazy(monkeypatch)

    def fresh_outputs(reverse: bool):
        W._WALKERS.clear()
        guide = build_guide(toy_source, toy_tokenizer)
        assert getattr(guide.dfa, "lazy", False)
        state = guide.initial_state
        configs = []
        rng = random.Random(11)
        for _ in range(6):
            A = guide.producer.allowed(state.stack)
            configs.append((state.lexer.remainder, A))
            ids, _ = guide._mask_ids(state)
            tok = rng.choice(sorted(set(ids) - {guide.eos_token_id}) or sorted(ids))
            state = guide.get_next_state(state, tok)
            if state.status == COMPLETE:
                break
        # replay the recorded configurations on a FRESH walker in the given
        # order; the walker interning order then differs run-to-run
        W._WALKERS.clear()
        out = {}
        for rem, A in (reversed(configs) if reverse else configs):
            res = W.walk(guide.trie, guide.dfa, rem, A,
                         guide.tables.ignored_terminal_ids, guide.producer._priority, None)
            assert res.groups is not None, "lazy walk must take the kernel path"
            out[(bytes(rem), A)] = (
                [int(t) for t in res.ci_tokens],
                [payload for _rep, _ids, payload in res.groups],
            )
        return out

    fwd = fresh_outputs(reverse=False)
    rev = fresh_outputs(reverse=True)
    assert fwd.keys() == rev.keys()
    for key in fwd:
        assert fwd[key] == rev[key], f"kernel output depends on interning order at {key}"


@needs_v8
def test_lazy_kernel_parallel_walks_match_sequential(toy_source, toy_tokenizer, monkeypatch):
    """The rayon pool leg (plan step 7): walk_auto under GRID_WALK_THREADS
    must emit the same ci buffers and registration payloads as the sequential
    walk — lazy interning races (rayon workers hitting the build mutex in
    nondeterministic order) may permute internal state ids, never outputs."""
    _force_lazy(monkeypatch)

    def outputs(threads: str):
        monkeypatch.setenv("GRID_WALK_THREADS", threads)
        monkeypatch.setenv("GRID_WALK_PAR_MIN", "1")  # toy tries are tiny
        W._WALKERS.clear()
        guide = build_guide(toy_source, toy_tokenizer)
        assert getattr(guide.dfa, "lazy", False)
        state = guide.initial_state
        out = []
        rng = random.Random(23)
        for _ in range(8):
            A = guide.producer.allowed(state.stack)
            res = W.walk(guide.trie, guide.dfa, state.lexer.remainder, A,
                         guide.tables.ignored_terminal_ids, guide.producer._priority, None)
            assert res.groups is not None, "lazy walk must take the kernel path"
            out.append((
                [int(t) for t in res.ci_tokens],
                [payload for _rep, _ids, payload in res.groups],
            ))
            ids, _ = guide._mask_ids(state)
            tok = rng.choice(sorted(set(ids) - {guide.eos_token_id}) or sorted(ids))
            state = guide.get_next_state(state, tok)
            if state.status == COMPLETE:
                break
        return out

    seq = outputs("0")
    par = outputs("4")
    assert seq == par


@needs_v8
def test_lazy_kernel_valueerror_falls_back_to_spec(toy_source, toy_tokenizer, monkeypatch):
    """The intern-cap contract: a ValueError out of a LAZY kernel walk (cap
    breach / poisoned build mutex) degrades to the _walk_py specification —
    masks stay exact; a dense walk must re-raise (no sanctioned kernel
    failure mode)."""
    _force_lazy(monkeypatch)
    guide = build_guide(toy_source, toy_tokenizer)
    assert getattr(guide.dfa, "lazy", False)
    A = guide.producer.allowed(guide.initial_state.stack)
    args = (guide.trie, guide.dfa, b"", A,
            guide.tables.ignored_terminal_ids, guide.producer._priority, None)

    class _CapBreach:
        width = 1

        def walk(self, remainder, a_words):
            raise ValueError("grid_core lazy walk aborted (state intern cap exceeded)")

    key = (id(guide.trie), id(guide.dfa), id(None))
    monkeypatch.setitem(W._WALKERS, key, (_CapBreach(), None))
    got = W.walk(*args)
    assert got.groups is None, "cap-breach walk must be the Python spec result"
    spec = W._walk_py(*args)
    assert sorted(got.ci_tokens) == sorted(spec.ci_tokens)
    assert got.cd_entries == spec.cd_entries

    # dense: the same ValueError stays loud
    monkeypatch.delenv("GRID_PERF_COMPONENT_BUDGET", raising=False)
    dense_guide = build_guide(toy_source, toy_tokenizer)
    assert not getattr(dense_guide.dfa, "lazy", False)
    dense_args = (dense_guide.trie, dense_guide.dfa, b"",
                  dense_guide.producer.allowed(dense_guide.initial_state.stack),
                  dense_guide.tables.ignored_terminal_ids,
                  dense_guide.producer._priority, None)
    dense_key = (id(dense_guide.trie), id(dense_guide.dfa), id(None))
    monkeypatch.setitem(W._WALKERS, dense_key, (_CapBreach(), None))
    with pytest.raises(ValueError):
        W.walk(*dense_args)


@needs_v8
def test_lazy_payload_shape(toy_source, toy_tokenizer, monkeypatch):
    """kernel_lazy_payload structural invariants (the kernel re-validates at
    construction; this pins the Python-side contract)."""
    from grid.lexer.factored import LazyTerminalDFA, kernel_lazy_payload

    _force_lazy(monkeypatch)
    guide = build_guide(toy_source, toy_tokenizer)
    dfa = guide.dfa
    gclass, blobs, cmap = kernel_lazy_payload(dfa)
    assert len(gclass) == 256
    assert len(blobs) == len(dfa.comps) == len(cmap)
    n_g = max(gclass) + 1
    assert all(len(m) == n_g for m in cmap)
    for comp, blob, m in zip(dfa.comps, blobs, cmap, strict=True):
        kind = blob[0]
        assert kind == (1 if isinstance(comp, LazyTerminalDFA) else 0)
        n_classes = int.from_bytes(blob[1:3], "little")
        assert 0 < n_classes <= 256
        assert all(0 <= c < n_classes for c in m)
