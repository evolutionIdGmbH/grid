"""Kernel v7 x v6 sessions (red-team plan §4.6): the v5 Python-state oracle
and the in-kernel v6 session must stay bit-identical when the producer builds
entries through the v7 path (walk_payload + register_blob), including the
NEEDS_BIND cold-fill flow; and the &self migration must survive a 4-thread
stress mixing detached register_blob calls with fill_bits / session_accept /
set_token_bytes — the exact overlap that would raise PyBorrowError if any of
those still took a mutable pyclass borrow (admission-warmup pool threads
inside a detached build racing ensure_session_tables)."""

import os
import random
import threading

import numpy as np
import pytest

from grid.generate import build_guide
from grid.mask.cache import MaskEntryV7
from grid.mask.producer import _token_table
from grid.models.vllm_structured import GridGrammarSession
from grid.trie.walk import _term_words

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("GRID_NO_RUST") == "1",
        reason="v7 requires the grid_core kernel",
    ),
    pytest.mark.skipif(
        os.environ.get("GRID_NO_V6") == "1",
        reason="v6 sessions force-disabled (A/B env)",
    ),
]


@pytest.fixture
def v7_guide(toy_source, toy_tokenizer, monkeypatch):
    monkeypatch.setenv("GRID_V7", "1")
    g = build_guide(toy_source, toy_tokenizer)
    assert g.producer._v7 and g.producer._kernel is not None
    return g


def _sessions(guide):
    s5 = GridGrammarSession(guide, _force_v5=True)
    s6 = GridGrammarSession(guide)
    assert s5._sid is None and s6._sid is not None
    return s5, s6


def _rows(guide):
    words = (guide.vocab_size + 31) // 32
    return np.zeros((1, words), dtype=np.int32), np.zeros((1, words), dtype=np.int32)


def _fill_parity(s5, s6, bm5, bm6, ctx=""):
    bm5.fill(-1)
    bm6.fill(-1)
    s5.fill_bitmask(bm5, 0)
    s6.fill_bitmask(bm6, 0)
    assert (bm5 == bm6).all(), f"{ctx}: fill bit-parity"


def test_v5_v6_session_differential_under_v7(v7_guide):
    """Random legal + adversarial token streams through both sessions on ONE
    v7 producer: accept verdicts and packed fill rows identical every step."""
    guide = v7_guide
    rng = random.Random(29)
    s5, s6 = _sessions(guide)
    bm5, bm6 = _rows(guide)
    for step in range(60):
        ctx = f"step {step}"
        _fill_parity(s5, s6, bm5, bm6, ctx)
        ids, _ = guide._mask_ids(s5.states[-1])
        pool = sorted(set(int(i) for i in ids))
        if rng.random() < 0.3:  # adversarial: off-mask token must reject on both
            bad = next(t for t in range(guide.vocab_size) if t not in set(pool))
            assert s5.accept_tokens("r", [bad]) == s6.accept_tokens("r", [bad]) is False, ctx
            # v5 treats False as terminal; re-create the pair to keep driving
            s5, s6 = _sessions(guide)
            continue
        tok = rng.choice(sorted(set(pool) - {guide.eos_token_id}) or pool)
        assert s5.accept_tokens("r", [tok]) == s6.accept_tokens("r", [tok]) is True, ctx
        assert s5.is_terminated() == s6.is_terminated(), ctx
        if s5.is_terminated():
            s5.rollback(2)
            s6.rollback(2)
    # the drive must actually have exercised v7 entries
    assert any(isinstance(e, MaskEntryV7) for e in guide.producer.cache._t1.values())


def test_needs_bind_cold_fill_row_identical(v7_guide):
    """The v6 NEEDS_BIND flow: a fill at an unbound configuration triggers
    session_bind_handle -> _entry_for (v7 build) -> session_bind -> retried
    in-kernel fill. The row must equal the v5 oracle's cold fill (which goes
    through masks() + fill retry) bit-for-bit, on a FRESH producer per side."""
    guide = v7_guide
    rng = random.Random(31)
    s5, s6 = _sessions(guide)
    bm5, bm6 = _rows(guide)
    # walk a few tokens WITHOUT filling, so successors stay unbound, then fill
    for step in range(8):
        ids, _ = guide._mask_ids(s5.states[-1])
        tok = rng.choice(sorted(set(int(i) for i in ids) - {guide.eos_token_id})
                         or [int(ids[0])])
        ok5 = s5.accept_tokens("r", [tok])
        ok6 = s6.accept_tokens("r", [tok])
        assert ok5 == ok6, f"step {step}"
        if not ok5:
            break
        _fill_parity(s5, s6, bm5, bm6, f"cold fill step {step}")
        if s5.is_terminated():
            break


def test_four_thread_stress_register_blob_vs_sessions(v7_guide):
    """4 threads for ~1s: (a)+(b) detached register_blob pounding (dedup
    path re-runs the full detached build each call), (c) set_token_bytes /
    set_dfa_accept re-uploads (identical tables — content-idempotent), while
    the main thread runs session_accept + fill_bitmask + validate. Pre-v7,
    (c) took `&mut self`: overlapping a detached `&self` call raised
    PyBorrowError. Assert: no exceptions anywhere, fill parity throughout."""
    guide = v7_guide
    prod = guide.producer
    kernel = prod._kernel
    st = guide.initial_state
    entry = prod._entry_for(st.lexer.remainder, prod.allowed(st.stack))
    assert isinstance(entry, MaskEntryV7)
    key_repr = repr(entry.key).encode()

    blob_v, offs = _token_table(guide.adapter, guide.vocab_size)
    w = kernel.width
    accept_bytes = np.asarray(guide.dfa.accept, dtype=np.int32).tobytes()
    accepts_all = [_term_words(s, w) for s in guide.dfa.accepts_all]

    errors: list = []
    stop = threading.Event()

    def pound_register():
        try:
            while not stop.is_set():
                h, eid, _t, _n = kernel.register_blob(
                    entry.blob, entry.ci_bytes, key_repr, prod.vocab_size)
                assert eid == entry.entry_id
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def pound_tables():
        try:
            while not stop.is_set():
                kernel.set_token_bytes(blob_v, offs.tobytes())
                kernel.set_dfa_accept(accept_bytes, accepts_all)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=pound_register),
               threading.Thread(target=pound_register),
               threading.Thread(target=pound_tables)]
    for t in threads:
        t.start()
    try:
        rng = random.Random(41)
        s5, s6 = _sessions(guide)
        bm5, bm6 = _rows(guide)
        for step in range(120):
            ids, _ = guide._mask_ids(s5.states[-1])
            pool = sorted(set(int(i) for i in ids) - {guide.eos_token_id}) \
                or sorted(int(i) for i in ids)
            tok = rng.choice(pool)
            assert s5.accept_tokens("r", [tok]) == s6.accept_tokens("r", [tok])
            assert s6._kernel.session_validate(s6._sid, [tok, tok, tok]) >= 0
            _fill_parity(s5, s6, bm5, bm6, f"stress step {step}")
            if s5.is_terminated():
                s5.rollback(3)
                s6.rollback(3)
    finally:
        stop.set()
        for t in threads:
            t.join()
    assert not errors, errors
