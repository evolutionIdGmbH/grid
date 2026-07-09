"""Kernel v6 session differential: in-kernel accept/validate/rollback/fill
(GridGrammarSession fast path) against the v5 Python-state oracle sharing the
SAME producer — the red-team test plan for the session-in-kernel port.

The oracle is today's shipped v5 semantics AFTER the §0 post-COMPLETE fix
(guide._advance: COMPLETE consumes only (repeat-)eos). Every property listed
in the plan is covered: state/status/mask parity on random token streams
(legal + adversarial tokens, toy/sql-lexicon/wide grammars), pinned COMPLETE
semantics, COMPLETE fill bit-parity (bound and never-bound), rollback under
the spec-decode pattern (including eos-accept restoration), validate-no-commit,
vocab holes, the E6-normative token table, reset_interning with live sessions
(risk c), epoch-rollover binding invalidation (risk d), telemetry conservation
(risk e), the audit/E14 and GRID_NO_V6 gates, prefetcher flow, and a
registration-vs-session concurrency stress (GIL/borrow discipline).
"""

import os
import random
import threading

import numpy as np
import pytest

from grid.generate import build_guide
from grid.guide import ACCEPTING, COMPLETE, GRAMMAR_END
from grid.mask.producer import _token_table
from grid.models.vllm_structured import (
    _FLAG_COMPLETE,
    _FLAG_OK,
    _STATUS,
    GridGrammarSession,
)

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("GRID_NO_RUST") == "1",
        reason="v6 sessions require the grid_core kernel",
    ),
    pytest.mark.skipif(
        os.environ.get("GRID_NO_V6") == "1",
        reason="v6 sessions force-disabled (A/B env)",
    ),
]


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def toy_guide(toy_source, toy_tokenizer):
    return build_guide(toy_source, toy_tokenizer)


@pytest.fixture
def sql_lex_guide(sql_source, sql_tokenizer, sql_grammar):
    from grid.grammar.projection import RoleProjection
    from grid.lalr.compile import compile_tables
    from grid.policy.schema import SchemaSnapshot

    schema = SchemaSnapshot.from_dict({"users": ["id", "name"], "orders": ["id", "total"]})
    proj = RoleProjection.full(sql_grammar).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    return build_guide(sql_source, sql_tokenizer, projection=proj,
                       lexicons=schema.lexicons(tables), schema_fingerprint=schema.fingerprint)


@pytest.fixture
def wide_guide(wide_source, wide_tokenizer):
    return build_guide(wide_source, wide_tokenizer)


def _sessions(guide, prefetcher=None):
    """(v5 oracle, v6 kernel session) over ONE shared guide/producer."""
    if guide.producer._kernel is None:
        pytest.skip("grid_core kernel unavailable")
    s5 = GridGrammarSession(guide, _force_v5=True)
    s6 = GridGrammarSession(guide, prefetcher=prefetcher)
    assert s5._sid is None and s6._sid is not None
    return s5, s6


def _rows(guide):
    words = (guide.vocab_size + 31) // 32
    return np.zeros((1, words), dtype=np.int32), np.zeros((1, words), dtype=np.int32)


def _assert_state_parity(guide, s5, s6, ctx=""):
    st5 = s5.states[-1]
    kidx, rem, status, n_gen, prev = s6._kernel.session_state(s6._sid)
    assert _STATUS[status] == st5.status, f"{ctx}: status"
    assert rem == st5.lexer.remainder, f"{ctx}: remainder"
    assert n_gen == st5.n_generated, f"{ctx}: n_generated"
    assert (prev if prev >= 0 else None) == st5.prev_token, f"{ctx}: prev_token"
    assert s5.is_terminated() == s6.is_terminated(), f"{ctx}: is_terminated"


def _assert_fill_parity(guide, s5, s6, bm5, bm6, ctx=""):
    bm5.fill(-1)  # poisoned stale words must be fully overwritten
    bm6.fill(-1)
    s5.fill_bitmask(bm5, 0)
    s6.fill_bitmask(bm6, 0)
    assert (bm5 == bm6).all(), f"{ctx}: fill bit-parity"


def _drive(guide, seed, steps=90, p_bad=0.35, p_roll=0.12, p_eos=0.05):
    """The differential engine: identical token streams into both sessions;
    per-step state, fill, validate and accept parity; random rollbacks; runs
    THROUGH COMPLETE (post-complete probes included)."""
    rng = random.Random(seed)
    s5, s6 = _sessions(guide)
    bm5, bm6 = _rows(guide)
    eos = guide.eos_token_id
    accepted = rejected = 0
    for step in range(steps):
        ctx = f"seed {seed} step {step}"
        _assert_state_parity(guide, s5, s6, ctx)
        _assert_fill_parity(guide, s5, s6, bm5, bm6, ctx)

        st5 = s5.states[-1]
        ids, _ = guide._mask_ids(st5)
        # validate parity on a random probe (mask picks + noise), no commit
        probe = [int(rng.choice(ids)) if rng.random() < 0.7 else rng.randrange(guide.vocab_size)
                 for _ in range(3)]
        got6 = s6.validate_tokens(probe)
        got5 = s5.validate_tokens(probe)
        assert got5 == got6, f"{ctx}: validate {probe} -> {got5} vs {got6}"
        _assert_state_parity(guide, s5, s6, f"{ctx} (post-validate)")

        if s5.is_terminated():
            # pinned COMPLETE semantics: non-eos rejected, repeat-eos consumed
            bad = int(next(t for t in range(guide.vocab_size) if t != eos))
            assert s6.validate_tokens([eos, bad]) == s5.validate_tokens([eos, bad]) == [eos], ctx
            n5 = s5.rollback(1) or True  # step back to keep the walk going
            n6 = s6.rollback(1) or True
            assert n5 == n6
            continue

        if rng.random() < p_bad:
            tok = rng.randrange(guide.vocab_size)
        elif rng.random() < p_eos and eos in ids:
            tok = eos
        else:
            tok = int(rng.choice(ids))
        ok5 = s5.accept_tokens("r", [tok])
        ok6 = s6.accept_tokens("r", [tok])
        assert ok5 == ok6, f"{ctx}: accept({tok}) {ok5} vs {ok6}"
        accepted += ok5
        rejected += not ok5
        assert s5.num_processed_tokens == s6.num_processed_tokens, ctx

        if ok5 and rng.random() < p_roll:
            n = rng.randint(1, 3)
            s5.rollback(n)
            s6.rollback(n)
            _assert_state_parity(guide, s5, s6, f"{ctx} (post-rollback {n})")
    assert accepted > 10 and rejected > 3, "differential exercised too little"


# ------------------------------------------------------------- differentials


def test_differential_toy(toy_guide):
    for seed in (3, 11, 42):
        _drive(toy_guide, seed)


def test_differential_sql_lexicons(sql_lex_guide):
    """Identifier positions (L3 lexicons + schema fingerprint in the key)."""
    for seed in (5, 19):
        _drive(sql_lex_guide, seed, steps=70)


def test_differential_wide(wide_guide):
    """>64 terminals: the W=2 mask paths of scan/pick/status in-kernel."""
    assert wide_guide.producer._kernel.width == 2
    _drive(wide_guide, seed=21, steps=60)


def test_differential_multibatch_accepts(toy_guide):
    """accept_tokens with multi-token batches (spec-decode shape): False on
    the first non-viable token, prior tokens stay consumed — both impls."""
    rng = random.Random(23)
    s5, s6 = _sessions(toy_guide)
    for step in range(30):
        st5 = s5.states[-1]
        ids, _ = toy_guide._mask_ids(st5)
        batch = [int(rng.choice(ids))]
        cur = toy_guide._advance(st5, batch[0], audit=False)
        for _ in range(rng.randint(0, 3)):
            if cur is None:
                break
            if rng.random() < 0.3:
                batch.append(rng.randrange(toy_guide.vocab_size))
                cur = toy_guide._advance(cur, batch[-1], audit=False)
            else:
                nxt_ids, _ = toy_guide._mask_ids(cur)
                batch.append(int(rng.choice(nxt_ids)))
                cur = toy_guide._advance(cur, batch[-1], audit=False)
        ok5 = s5.accept_tokens("r", batch)
        ok6 = s6.accept_tokens("r", batch)
        assert ok5 == ok6, (step, batch)
        assert s5.num_processed_tokens == s6.num_processed_tokens, (step, batch)
        _assert_state_parity(toy_guide, s5, s6, f"step {step}")
        if s5.is_terminated():
            break


# --------------------------------------------------- pinned COMPLETE semantics


def _drive_to_accepting(guide, s5, s6, rng):
    while True:
        st5 = s5.states[-1]
        if st5.status in (ACCEPTING, GRAMMAR_END):
            return
        ids, _ = guide._mask_ids(st5)
        tok = int(rng.choice([t for t in ids if t != guide.eos_token_id]))
        assert s5.accept_tokens("r", [tok]) and s6.accept_tokens("r", [tok])


def test_post_complete_guide_advance(toy_guide):
    """§0 fix at the guide level: COMPLETE consumes only (repeat-)eos."""
    rng = random.Random(1)
    st = toy_guide.initial_state
    while st.status != ACCEPTING:
        ids, _ = toy_guide._mask_ids(st)
        st = toy_guide.get_next_state(
            st, int(rng.choice([t for t in ids if t != toy_guide.eos_token_id])))
    done = toy_guide._advance(st, toy_guide.eos_token_id, audit=False)
    assert done.status == COMPLETE
    assert done.n_generated == st.n_generated  # eos does not bump n_generated
    for tok in range(0, 40):
        if tok == toy_guide.eos_token_id:
            continue
        assert toy_guide._advance(done, tok, audit=False) is None, tok
    again = toy_guide._advance(done, toy_guide.eos_token_id, audit=False)
    assert again is not None and again.status == COMPLETE
    assert again.n_generated == done.n_generated


def test_post_complete_session_semantics(toy_guide):
    """One accept batch [.., eos, eos] terminates and consumes the repeat-eos;
    non-eos after eos rejects the batch — parity across v5/v6."""
    eos = toy_guide.eos_token_id
    rng = random.Random(2)
    s5, s6 = _sessions(toy_guide)
    _drive_to_accepting(toy_guide, s5, s6, rng)
    assert s5.accept_tokens("r", [eos, eos]) is True
    assert s6.accept_tokens("r", [eos, eos]) is True
    assert s5.is_terminated() and s6.is_terminated()
    _assert_state_parity(toy_guide, s5, s6, "double-eos")
    # terminated sessions refuse at entry (v5-pinned contract)
    assert s5.accept_tokens("r", [eos]) is False
    assert s6.accept_tokens("r", [eos]) is False

    s5b, s6b = _sessions(toy_guide)
    _drive_to_accepting(toy_guide, s5b, s6b, rng)
    bad = next(t for t in range(toy_guide.vocab_size) if t != eos)
    assert s5b.accept_tokens("r", [eos, bad]) is False
    assert s6b.accept_tokens("r", [eos, bad]) is False
    # the eos WAS consumed before the failure (prior tokens stay consumed)
    assert s5b.is_terminated() and s6b.is_terminated()
    assert s5b.num_processed_tokens == s6b.num_processed_tokens
    _assert_state_parity(toy_guide, s5b, s6b, "eos-then-bad")


def test_complete_fill_semantics(toy_guide):
    """Fill after COMPLETE: the entry is still consulted, the eos bit is
    excluded — bit-parity v5/v6, including the never-bound COMPLETE miss path
    (fresh v6 session that never filled before eos)."""
    eos = toy_guide.eos_token_id
    rng = random.Random(3)
    s5, s6 = _sessions(toy_guide)
    _drive_to_accepting(toy_guide, s5, s6, rng)
    bm5, bm6 = _rows(toy_guide)
    _assert_fill_parity(toy_guide, s5, s6, bm5, bm6, "pre-eos")
    pre = bm6.copy()
    assert (pre[0] >> (eos % 32))[eos // 32] & 1  # eos bit present pre-COMPLETE
    assert s5.accept_tokens("r", [eos]) and s6.accept_tokens("r", [eos])
    _assert_fill_parity(toy_guide, s5, s6, bm5, bm6, "post-eos")
    assert not (bm6[0].view(np.uint32)[eos // 32] >> (eos % 32)) & 1  # eos excluded
    nonzero = bm6[0].view(np.uint32).any()
    assert nonzero == bm5[0].view(np.uint32).any()

    # never-bound COMPLETE: a FRESH v6 session (fresh guide state, warm cache)
    # accepts straight to COMPLETE without a single fill, then fills — the
    # kernel misses, Python binds, the retry serves the same bits
    s5c, s6c = _sessions(toy_guide)
    _drive_to_accepting(toy_guide, s5c, s6c, rng)
    assert s5c.accept_tokens("r", [eos]) and s6c.accept_tokens("r", [eos])
    _assert_fill_parity(toy_guide, s5c, s6c, bm5, bm6, "never-bound COMPLETE")


# ------------------------------------------------------------------ rollback


def test_eos_rollback_restores_exactly(toy_guide):
    """The rollback log reproduces the eos-accept peculiarities: status
    restored to ACCEPTING/GRAMMAR_END, n_generated untouched (eos never
    bumped it), fill parity after the rewind."""
    eos = toy_guide.eos_token_id
    rng = random.Random(4)
    s5, s6 = _sessions(toy_guide)
    _drive_to_accepting(toy_guide, s5, s6, rng)
    n_before = s6._kernel.session_state(s6._sid)[3]
    assert s5.accept_tokens("r", [eos]) and s6.accept_tokens("r", [eos])
    assert s6._kernel.session_state(s6._sid)[3] == n_before  # not bumped
    s5.rollback(1)
    s6.rollback(1)
    assert not s5.is_terminated() and not s6.is_terminated()
    _assert_state_parity(toy_guide, s5, s6, "post-eos-rollback")
    bm5, bm6 = _rows(toy_guide)
    _assert_fill_parity(toy_guide, s5, s6, bm5, bm6, "post-eos-rollback")


def test_rollback_bounds(toy_guide):
    """rollback(<=0) is a no-op; rollback past the accepted count lands on the
    initial state (v5 truncation semantics)."""
    rng = random.Random(5)
    s5, s6 = _sessions(toy_guide)
    for s in (s5, s6):
        s.rollback(0)
        s.rollback(-3)
    _assert_state_parity(toy_guide, s5, s6, "no-op rollback")
    for _ in range(5):
        ids, _ = toy_guide._mask_ids(s5.states[-1])
        tok = int(rng.choice([t for t in ids if t != toy_guide.eos_token_id]))
        assert s5.accept_tokens("r", [tok]) and s6.accept_tokens("r", [tok])
    s5.rollback(999)
    s6.rollback(999)
    assert s5.num_processed_tokens == s6.num_processed_tokens == 0
    _assert_state_parity(toy_guide, s5, s6, "past-start rollback")
    st = s6._kernel.session_state(s6._sid)
    assert st[1] == b"" and _STATUS[st[2]] == toy_guide.initial_state.status


def test_reset_returns_to_initial(toy_guide):
    rng = random.Random(6)
    s5, s6 = _sessions(toy_guide)
    for _ in range(4):
        ids, _ = toy_guide._mask_ids(s5.states[-1])
        tok = int(rng.choice([t for t in ids if t != toy_guide.eos_token_id]))
        assert s5.accept_tokens("r", [tok]) and s6.accept_tokens("r", [tok])
    s5.reset()
    s6.reset()
    assert s6.num_processed_tokens == 0 and not s6.is_terminated()
    _assert_state_parity(toy_guide, s5, s6, "post-reset")
    bm5, bm6 = _rows(toy_guide)
    _assert_fill_parity(toy_guide, s5, s6, bm5, bm6, "post-reset")


# ------------------------------------------------- vocab holes / token table


def test_vocab_holes_and_out_of_range_rejected(toy_guide):
    """Pinned improvement: ids outside the table map to empty bytes and
    REJECT in-kernel (v5 raises KeyError — kept, documented); session state
    is untouched by the rejection."""
    s5, s6 = _sessions(toy_guide)
    before = s6._kernel.session_state(s6._sid)
    assert s6.accept_tokens("r", [toy_guide.vocab_size + 7]) is False
    assert s6.validate_tokens([toy_guide.vocab_size + 7]) == []
    assert s6._kernel.session_state(s6._sid) == before
    with pytest.raises(KeyError):
        s5.accept_tokens("r", [toy_guide.vocab_size + 7])


def test_token_table_is_e6_normative(toy_guide, toy_tokenizer):
    """blob[i] == adapter.token_bytes(i) for every id — the uploaded table is
    THE token->bytes definition (eos short-circuits before the lookup)."""
    blob, offs = _token_table(toy_tokenizer, toy_guide.vocab_size)
    assert len(offs) == toy_guide.vocab_size + 1
    for i in range(toy_guide.vocab_size):
        assert blob[offs[i]:offs[i + 1]] == toy_tokenizer.token_bytes(i), i
    # eos is accepted at ACCEPTING even though its table bytes are empty
    assert toy_tokenizer.token_bytes(toy_guide.eos_token_id) == b""


# -------------------------------------------- risks (c)/(d)/(e): invalidation


def test_reset_interning_with_live_sessions(toy_guide):
    """Risk (c): sessions own raw state chains; reset_interning mid-session
    must not dangle them — parity continues, bindings rebuilt on demand."""
    rng = random.Random(7)
    s5, s6 = _sessions(toy_guide)
    bm5, bm6 = _rows(toy_guide)
    for _ in range(4):
        ids, _ = toy_guide._mask_ids(s5.states[-1])
        tok = int(rng.choice([t for t in ids if t != toy_guide.eos_token_id]))
        assert s5.accept_tokens("r", [tok]) and s6.accept_tokens("r", [tok])
    _assert_fill_parity(toy_guide, s5, s6, bm5, bm6, "pre-reset")
    toy_guide.producer._reset_interning()
    _assert_state_parity(toy_guide, s5, s6, "post-reset_interning")
    _assert_fill_parity(toy_guide, s5, s6, bm5, bm6, "post-reset_interning")
    for step in range(6):
        ids, _ = toy_guide._mask_ids(s5.states[-1])
        tok = int(rng.choice(ids))
        assert s5.accept_tokens("r", [tok]) == s6.accept_tokens("r", [tok])
        _assert_fill_parity(toy_guide, s5, s6, bm5, bm6, f"post-reset step {step}")
        if s5.is_terminated():
            break
    # rollback across the reset boundary still restores exactly
    s5.rollback(2)
    s6.rollback(2)
    _assert_state_parity(toy_guide, s5, s6, "rollback across reset")
    _assert_fill_parity(toy_guide, s5, s6, bm5, bm6, "rollback across reset")


def test_epoch_rollover_clears_session_bindings(toy_guide):
    """Risk (d) regression: after invalidate_namespace a v6 fill must NOT be
    served from a stale binding — the first post-rollover fill recomputes
    (a counted T1 miss) and rebinds; bits stay identical."""
    prod = toy_guide.producer
    _s5, s6 = _sessions(toy_guide)
    bm6, _ = _rows(toy_guide)
    s6.fill_bitmask(bm6, 0)  # binds
    ref = bm6.copy()
    s6.fill_bitmask(bm6, 0)  # kernel-served
    assert (bm6 == ref).all()
    m0 = prod.cache.misses
    prod.cache.invalidate_namespace()
    bm6.fill(-1)
    s6.fill_bitmask(bm6, 0)
    assert prod.cache.misses > m0, "post-rollover fill must be a T1 miss"
    assert (bm6 == ref).all(), "content-identical rebuild"
    m1 = prod.cache.misses
    s6.fill_bitmask(bm6, 0)  # rebound: kernel-served again
    assert prod.cache.misses == m1


def test_telemetry_hit_conservation(toy_source, toy_tokenizer):
    """Risk (e): identical replay on v5 vs v6 yields equal hit totals once the
    kernel's fills_hit is folded into cache.hits (and equal misses)."""
    # one legal 30-token stream, precomputed on a scratch guide
    scratch = build_guide(toy_source, toy_tokenizer)
    if scratch.producer._kernel is None:
        pytest.skip("grid_core kernel unavailable")
    rng = random.Random(8)
    st = scratch.initial_state
    seq = []
    for _ in range(30):
        ids, _ = scratch._mask_ids(st)
        tok = int(rng.choice([t for t in ids if t != scratch.eos_token_id]))
        seq.append(tok)
        st = scratch.get_next_state(st, tok)

    def replay(force_v5: bool):
        guide = build_guide(toy_source, toy_tokenizer)
        s = GridGrammarSession(guide, _force_v5=force_v5)
        bm, _ = _rows(guide)
        for _rep in range(2):  # pass 1 mixed, pass 2 warm
            for t in seq:
                s.fill_bitmask(bm, 0)
                assert s.accept_tokens("r", [t])
            s.fill_bitmask(bm, 0)
            s.reset()
        guide.producer.fold_session_stats()
        return guide.producer.cache.hits, guide.producer.cache.misses

    h5, m5 = replay(True)
    h6, m6 = replay(False)
    assert (h5, m5) == (h6, m6)


# --------------------------------------------------------------------- gates


def test_audit_guide_stays_v5(toy_source, toy_tokenizer):
    """E14 gate: audit-enabled guides keep the v5 Python path (config_hash
    lives on Python StackNodes only) — and still serve correctly."""
    guide = build_guide(toy_source, toy_tokenizer, audit=True)
    s = GridGrammarSession(guide)
    assert s._sid is None
    ids, _ = guide._mask_ids(s.states[-1])
    assert s.accept_tokens("r", [int(ids[0])]) is True
    assert len(guide.audit.records) >= 1


def test_serving_registry_guide_selects_v6(sql_source, sql_tokenizer):
    """The serving path (mode 3): _GuideRegistry envelope builds audit=None
    guides, so GridGrammarSession takes the v6 kernel path — the same
    construction GridStructuredBackend.compile_grammar performs."""
    from grid.models.vllm_processor import _GuideRegistry

    reg = _GuideRegistry(sql_tokenizer)
    guide = reg.guide_for({
        "grammar": sql_source,
        "schema": {"users": ["id", "name"], "orders": ["id", "total"]},
    })
    assert guide.audit is None
    if guide.producer._kernel is None:
        pytest.skip("grid_core kernel unavailable")
    s = GridGrammarSession(guide)
    assert s._sid is not None  # v6 selected on the serving path
    ids, _ = guide._mask_ids(guide.initial_state)
    assert s.accept_tokens("r", [int(ids[0])]) is True


def test_grid_no_v6_env_forces_v5(toy_guide, monkeypatch):
    monkeypatch.setenv("GRID_NO_V6", "1")
    s = GridGrammarSession(toy_guide)
    assert s._sid is None
    monkeypatch.delenv("GRID_NO_V6")
    s2 = GridGrammarSession(toy_guide)
    assert s2._sid is not None


def test_initial_status_parity(toy_guide, sql_lex_guide):
    for guide in (toy_guide, sql_lex_guide):
        s = GridGrammarSession(guide)
        if s._sid is None:
            pytest.skip("grid_core kernel unavailable")
        st = s._kernel.session_state(s._sid)
        assert _STATUS[st[2]] == guide.initial_state.status


def test_end_of_statement_status_and_mask(sql_lex_guide):
    """End-of-statement parity: after a full statement the trailing ';' stays
    in the remainder (maximal munch — emission needs a dead byte), so the
    status is ACCEPTING and the mask still contains ignored-viable tokens
    (whitespace) + eos — parity, not intuition. (GRAMMAR_END proper requires
    an empty remainder AND an empty allowed set; the scan invariant leaves a
    nonempty remainder after every content token, so it only arises on
    degenerate grammars — the kernel derives it identically regardless.)"""
    guide = sql_lex_guide
    adapter = guide.adapter
    s5, s6 = _sessions(guide)
    text = b"select id from users;"
    toks = adapter.greedy_tokenize(text)
    assert s5.accept_tokens("r", [int(t) for t in toks])
    assert s6.accept_tokens("r", [int(t) for t in toks])
    assert s5.states[-1].status == ACCEPTING
    assert s5.states[-1].lexer.remainder == b";"
    _assert_state_parity(guide, s5, s6, "end-of-statement")
    bm5, bm6 = _rows(guide)
    _assert_fill_parity(guide, s5, s6, bm5, bm6, "end-of-statement")
    eos = guide.eos_token_id
    assert (bm6[0].view(np.uint32)[eos // 32] >> (eos % 32)) & 1
    assert bm6[0].view(np.uint32).sum() > 1  # ignored (whitespace) tokens too


# ---------------------------------------------------------- prefetcher / flags


def test_prefetch_flow_and_flags(toy_source, toy_tokenizer):
    """UNBOUND accepts schedule a pool build from captured (remainder, A);
    fill waits for it and serves in-kernel afterwards; rollback drops the
    scheduled target."""
    from grid.serving import MaskPrefetcher

    guide = build_guide(toy_source, toy_tokenizer)
    if guide.producer._kernel is None:
        pytest.skip("grid_core kernel unavailable")
    pf = MaskPrefetcher(max_workers=2)
    try:
        s6 = GridGrammarSession(guide, prefetcher=pf)
        assert s6._sid is not None
        rng = random.Random(9)
        st = guide.initial_state  # shadow for picking legal tokens
        bm, _ = _rows(guide)
        for step in range(12):
            ids, _ = guide._mask_ids(st)
            tok = int(rng.choice([t for t in ids if t != guide.eos_token_id]))
            flags = s6._kernel.session_validate(s6._sid, [tok])  # sanity: viable
            assert flags == 1
            assert s6.accept_tokens("r", [tok])
            st = guide.get_next_state(st, tok)
            s6.fill_bitmask(bm, 0)
            ref = np.zeros(bm.shape[1], dtype=np.uint32)
            guide.fill_bitmask(st, ref)
            assert (bm[0].view(np.uint32) == ref).all(), f"step {step}"
        assert pf.stats["scheduled"] >= 1
        s6.rollback(1)  # drop path exercised
    finally:
        pf.shutdown()


def test_accept_flags_encoding(toy_guide):
    """bit0 OK / bit1 COMPLETE surface as documented."""
    rng = random.Random(10)
    s5, s6 = _sessions(toy_guide)
    _drive_to_accepting(toy_guide, s5, s6, rng)
    flags = s6._kernel.session_accept(s6._sid, toy_guide.eos_token_id)
    assert flags & _FLAG_OK and flags & _FLAG_COMPLETE
    assert s6._kernel.session_accept(s6._sid, toy_guide.eos_token_id) & _FLAG_OK
    bad = next(t for t in range(toy_guide.vocab_size) if t != toy_guide.eos_token_id)
    assert s6._kernel.session_accept(s6._sid, bad) == 0


# ------------------------------------------------------------- concurrency


def test_registration_vs_session_concurrency(toy_guide):
    """GIL/borrow discipline: pool threads registering entries (&mut self via
    register_bytes) while the scheduler thread runs session accepts/fills must
    never raise BorrowMutError; results stay parity-correct."""
    guide = toy_guide
    prod = guide.producer
    s5, s6 = _sessions(guide)
    errors = []

    # collect distinct (remainder, A) build inputs by pre-walking states
    inputs = []
    st = guide.initial_state
    rng = random.Random(11)
    for _ in range(10):
        inputs.append((st.lexer.remainder, prod.allowed(st.stack)))
        ids, _ = guide._mask_ids(st)
        st = guide.get_next_state(
            st, int(rng.choice([t for t in ids if t != guide.eos_token_id])))

    def pound():
        try:
            for _ in range(50):
                for rem, a in inputs:
                    prod.prefetch_build(rem, a)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=pound) for _ in range(3)]
    for t in threads:
        t.start()
    bm5, bm6 = _rows(guide)
    for step in range(40):
        ids, _ = guide._mask_ids(s5.states[-1])
        tok = int(rng.choice(ids))
        assert s5.accept_tokens("r", [tok]) == s6.accept_tokens("r", [tok])
        _assert_fill_parity(guide, s5, s6, bm5, bm6, f"concurrent step {step}")
        if s5.is_terminated():
            s5.rollback(1)
            s6.rollback(1)
    for t in threads:
        t.join()
    assert not errors, errors


# ------------------------------------------------------- single-flight handle


def test_ensure_handle_single_flight(toy_guide):
    """Racing _ensure_handle on ONE unregistered entry yields exactly one
    kernel handle (the latent duplicate-registration inefficiency, now
    load-bearing for v6 binding keys)."""
    prod = toy_guide.producer
    st = toy_guide.initial_state
    entry = prod._entry_for(st.lexer.remainder, prod.allowed(st.stack))
    prod._kernel_handles.pop(entry.entry_id, None)  # force re-registration race
    got = []
    barrier = threading.Barrier(4)

    def grab():
        barrier.wait()
        got.append(prod._ensure_handle(entry))

    threads = [threading.Thread(target=grab) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(got)) == 1
