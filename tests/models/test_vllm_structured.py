"""GridGrammarSession (scheduler-side structured-output core) + kernel #4's
fill_bitmask — vllm-free. The bitmask must be bit-exact with _mask_ids at every
walked state; accept/validate/rollback follow vllm's XgrammarGrammar semantics.
"""

import random

import numpy as np
import pytest
import torch

from grid.guide import COMPLETE
from grid.models.vllm_structured import GridGrammarSession, _parse_spec


@pytest.fixture
def guide(toy_source, toy_tokenizer):
    from grid.generate import build_guide

    return build_guide(toy_source, toy_tokenizer)


def _bits_to_ids(words: np.ndarray, vocab: int) -> set[int]:
    out = set()
    for w, word in enumerate(words.tolist()):
        b = 0
        while word:
            if word & 1:
                out.add(w * 32 + b)
            word >>= 1
            b += 1
    return {t for t in out if t < vocab}


def test_fill_bitmask_matches_mask_ids_along_walk(guide):
    rng = random.Random(5)
    words = (guide.vocab_size + 31) // 32
    out = np.empty(words, dtype=np.uint32)
    state = guide.initial_state
    for _ in range(20):
        ids, _ = guide._mask_ids(state)
        guide.fill_bitmask(state, out)
        assert _bits_to_ids(out, guide.vocab_size) == {int(t) for t in ids}
        if state.status == "COMPLETE":
            break
        pick = guide.eos_token_id if guide.eos_token_id in ids else rng.choice(ids)
        state = guide.get_next_state(state, pick)


def test_session_accept_validate_rollback(guide):
    s = GridGrammarSession(guide)
    ids, _ = guide._mask_ids(s.states[-1])
    first = int(next(t for t in ids if t != guide.eos_token_id))
    bad = next(t for t in range(guide.vocab_size) if not bool((ids == t).any()))

    # validate does not advance
    assert s.validate_tokens([first, bad]) == [first]
    assert s.num_processed_tokens == 0 and len(s.states) == 1

    # accept advances; a bad token fails after consuming the good prefix
    assert s.accept_tokens("r0", [first]) is True
    assert s.num_processed_tokens == 1
    assert s.accept_tokens("r0", [bad]) is False

    # rollback truncates persistent states
    s.rollback(1)
    assert s.num_processed_tokens == 0 and len(s.states) == 1

    s.reset()
    assert len(s.states) == 1 and not s.is_terminated()


def test_session_terminates_and_refuses_after_eos(guide):
    rng = random.Random(9)
    s = GridGrammarSession(guide)
    # shadow the session with a GridState walked in lockstep: v6 kernel
    # sessions do not extend s.states, so picks come from the shadow's mask
    st = guide.initial_state
    for _ in range(64):
        ids, _ = guide._mask_ids(st)
        pick = guide.eos_token_id if guide.eos_token_id in ids else rng.choice(ids)
        assert s.accept_tokens("r1", [int(pick)])
        st = guide.get_next_state(st, int(pick))
        if s.is_terminated():
            break
    assert s.is_terminated()
    assert st.status == COMPLETE
    assert s.accept_tokens("r1", [0]) is False  # terminated sessions refuse


def test_fill_bitmask_through_torch_int32_row(guide):
    """The vllm bitmask is an int32 torch tensor; the uint32 view must write
    through (the exact fill path the backend uses)."""
    words = (guide.vocab_size + 31) // 32
    bm = torch.zeros((2, words), dtype=torch.int32)
    s = GridGrammarSession(guide)
    s.fill_bitmask(bm, 1)
    ids, _ = guide._mask_ids(s.states[-1])
    got = _bits_to_ids(bm[1].numpy().view(np.uint32), guide.vocab_size)
    assert got == {int(t) for t in ids}
    assert bm[0].abs().sum() == 0  # other rows untouched


def test_parse_spec_envelope_and_raw():
    assert _parse_spec('{"grammar": "x", "schema": {"t": ["c"]}}')["schema"] == {"t": ["c"]}
    assert _parse_spec("%start s\ns: \"a\"")["grammar"].startswith("%start")
    assert _parse_spec("{ not json")["grammar"].startswith("{ not json")


# ------------------------------------------- W6: mask_ready / defer chassis
#
# Soundness frame: mask_ready() is a pure timing hook — it never changes WHAT
# any fill computes, only WHEN the scheduler asks (a not-ready request is
# skipped for the round; cap expiry forces scheduling and the fill BLOCKS on
# the exact mask, never a substitute). GRID_DEFER=0 = always ready =
# byte-identical to today.


def _first_token(guide):
    ids, _ = guide._mask_ids(guide.initial_state)
    return int(next(t for t in ids if t != guide.eos_token_id))


def _gate_producer_build(monkeypatch, prod):
    """Block the v6 cold-walk entrypoint (prefetch_build) on a gate."""
    import threading

    gate, started = threading.Event(), threading.Event()
    orig = prod.prefetch_build

    def gated(w, A):
        started.set()
        assert gate.wait(10)
        return orig(w, A)

    monkeypatch.setattr(prod, "prefetch_build", gated)
    return gate, started


def _spin_until(pred, timeout=10.0):
    import time

    t0 = time.perf_counter()
    while not pred():
        if time.perf_counter() - t0 > timeout:
            raise AssertionError("condition never became true")
        time.sleep(0.001)


def _v6_session_or_skip(guide, pf, monkeypatch, defer="1"):
    # pin the lever explicitly (ambient GRID_DEFER must not skew the test)
    monkeypatch.setenv("GRID_DEFER", defer)
    monkeypatch.setenv("GRID_DEFER_MS", "60000")  # cap can't fire mid-test
    s = GridGrammarSession(guide, prefetcher=pf)
    if s._sid is None:
        pytest.skip("kernel v6 session unavailable (GRID_NO_V6/no kernel)")
    return s


def test_mask_ready_lifecycle_v6(guide, toy_source, toy_tokenizer, monkeypatch):
    """v6 matrix: ready with nothing scheduled; NOT ready while the cold
    successor build is in flight; ready when it completes; the fill after
    rejoin is the exact mask (bit-identical to a synchronous twin)."""
    from grid.generate import build_guide
    from grid.serving import MaskPrefetcher

    pf = MaskPrefetcher()
    s = _v6_session_or_skip(guide, pf, monkeypatch)
    assert s.mask_ready() is True, "no cold build scheduled yet"
    tok = _first_token(guide)
    gate, started = _gate_producer_build(monkeypatch, guide.producer)
    assert s.accept_tokens("r0", [tok]) is True
    assert pf.stats["scheduled"] == 1 and started.wait(10)
    assert s.mask_ready() is False, "in-flight cold build must defer"
    gate.set()
    _spin_until(s.mask_ready)
    words = (guide.vocab_size + 31) // 32
    bm = np.zeros((1, words), dtype=np.int32)
    s.fill_bitmask(bm, 0)
    twin = build_guide(toy_source, toy_tokenizer)
    st = twin.get_next_state(twin.initial_state, tok)
    ids, _ = twin._mask_ids(st)
    got = _bits_to_ids(bm[0].view(np.uint32), guide.vocab_size)
    assert got == {int(t) for t in ids}, "rejoin fill must be the exact mask"
    pf.shutdown()


def test_mask_ready_lifecycle_v5(guide, monkeypatch):
    """v5/spec-path variant: the schedule key is states[-1]; identical
    semantics ('a scheduled cold build is in flight'). This is also the
    GRID_NO_RUST spec-path logic (kernel-free sessions take this branch)."""
    import threading

    from grid.serving import MaskPrefetcher

    monkeypatch.setenv("GRID_DEFER", "1")  # pin the lever ON (ambient kill switch must not skew)
    monkeypatch.setenv("GRID_DEFER_MS", "60000")
    pf = MaskPrefetcher()
    s = GridGrammarSession(guide, prefetcher=pf, _force_v5=True)
    assert s._sid is None and s.mask_ready() is True
    tok = _first_token(guide)

    gate, started = threading.Event(), threading.Event()
    orig = guide._mask_ids

    def gated(state):
        started.set()
        assert gate.wait(10)
        return orig(state)

    monkeypatch.setattr(guide, "_mask_ids", gated)
    assert s.accept_tokens("r0", [tok]) is True
    assert pf.stats["scheduled"] == 1 and started.wait(10)
    assert s.mask_ready() is False, "in-flight cold build must defer (v5)"
    gate.set()
    _spin_until(s.mask_ready)
    words = (guide.vocab_size + 31) // 32
    bm = np.zeros((1, words), dtype=np.int32)
    s.fill_bitmask(bm, 0)
    assert pf.stats["waits"] == 1
    assert bm.any()
    pf.shutdown()


def test_mask_ready_cap_expiry_forces_blocking_exact_fill(
        guide, toy_source, toy_tokenizer, monkeypatch):
    """Clock injection: with the build still in flight, an expired
    GRID_DEFER_MS cap flips mask_ready True (starvation bound) — and the
    forced fill BLOCKS on the exact mask, never a substitute."""
    import threading
    import time

    from grid.generate import build_guide
    from grid.serving import MaskPrefetcher

    pf = MaskPrefetcher()
    s = _v6_session_or_skip(guide, pf, monkeypatch)
    tok = _first_token(guide)
    gate, started = _gate_producer_build(monkeypatch, guide.producer)
    assert s.accept_tokens("r0", [tok]) is True and started.wait(10)
    assert s.mask_ready() is False
    s._defer_t0 = time.monotonic() - 61.0  # inject: 60 s cap has expired
    assert s.mask_ready() is True, "cap expiry must force schedulability"
    # the forced fill blocks on the gated build and lands the EXACT mask
    threading.Timer(0.05, gate.set).start()
    words = (guide.vocab_size + 31) // 32
    bm = np.zeros((1, words), dtype=np.int32)
    t0 = time.perf_counter()
    s.fill_bitmask(bm, 0)
    blocked_s = time.perf_counter() - t0
    assert blocked_s >= 0.03, f"fill must block on the in-flight build ({blocked_s:.3f}s)"
    twin = build_guide(toy_source, toy_tokenizer)
    st = twin.get_next_state(twin.initial_state, tok)
    ids, _ = twin._mask_ids(st)
    assert _bits_to_ids(bm[0].view(np.uint32), guide.vocab_size) == {int(t) for t in ids}
    pf.shutdown()


def test_mask_ready_true_when_complete(guide, monkeypatch):
    """_complete short-circuits mask_ready regardless of any leftover
    prefetch target (a finished request never defers)."""
    import random as _random

    from grid.serving import MaskPrefetcher

    pf = MaskPrefetcher()
    s = _v6_session_or_skip(guide, pf, monkeypatch)
    rng = _random.Random(9)
    st = guide.initial_state
    for _ in range(64):
        ids, _ = guide._mask_ids(st)
        pick = guide.eos_token_id if guide.eos_token_id in ids else rng.choice(ids)
        assert s.accept_tokens("r1", [int(pick)])
        st = guide.get_next_state(st, int(pick))
        if s.is_terminated():
            break
    assert s.is_terminated() and s.mask_ready() is True
    pf.shutdown()


def test_mask_ready_cleared_on_rollback_and_reset(guide, monkeypatch):
    """rollback/reset drop the prefetch target: the session reads ready
    again (the dropped build still completes and warms the cache)."""
    from grid.serving import MaskPrefetcher

    pf = MaskPrefetcher()
    s = _v6_session_or_skip(guide, pf, monkeypatch)
    tok = _first_token(guide)
    gate, started = _gate_producer_build(monkeypatch, guide.producer)
    assert s.accept_tokens("r0", [tok]) is True and started.wait(10)
    assert s.mask_ready() is False
    s.rollback(1)
    assert s._pf_target is None and s.mask_ready() is True
    assert s.accept_tokens("r0", [tok]) is True
    assert s.mask_ready() is False, "re-scheduled cold build defers again"
    s.reset()
    assert s._pf_target is None and s.mask_ready() is True
    gate.set()  # release the parked builds before shutdown
    pf.shutdown()


def test_mask_ready_kill_switch_grid_defer_0(guide, monkeypatch):
    """GRID_DEFER=0: mask_ready is constantly True even mid-build — the
    scheduler guard becomes a no-op and behavior is byte-identical to
    today (the fill blocks exactly as it always has)."""
    from grid.serving import MaskPrefetcher

    pf = MaskPrefetcher()
    s = _v6_session_or_skip(guide, pf, monkeypatch, defer="0")
    tok = _first_token(guide)
    gate, started = _gate_producer_build(monkeypatch, guide.producer)
    assert s.accept_tokens("r0", [tok]) is True and started.wait(10)
    assert s.mask_ready() is True, "kill switch must force always-ready"
    gate.set()
    words = (guide.vocab_size + 31) // 32
    bm = np.zeros((1, words), dtype=np.int32)
    s.fill_bitmask(bm, 0)  # blocking fill, today's path
    assert bm.any()
    pf.shutdown()


def test_defer_ms_bad_env_degrades_to_default(guide, monkeypatch):
    """A garbage GRID_DEFER_MS must not raise at session construction (a
    compile_grammar exception is engine-fatal in vLLM 0.24)."""
    monkeypatch.setenv("GRID_DEFER_MS", "not-a-number")
    s = GridGrammarSession(guide)
    assert s._defer_ms == 100.0 and s.mask_ready() in (True, False)


def test_defer_on_off_masks_identical(toy_source, toy_tokenizer, monkeypatch):
    """Mini differential: emulate the scheduler defer (poll mask_ready,
    skip-spin until ready, then fill) vs defer-off immediate blocking fills
    over the same greedy token path — per-step masks are byte-identical.
    Defer is a pure timing transformation."""
    from grid.generate import build_guide
    from grid.serving import MaskPrefetcher

    monkeypatch.setenv("GRID_DEFER_MS", "60000")
    g_on = build_guide(toy_source, toy_tokenizer)
    g_off = build_guide(toy_source, toy_tokenizer)
    pf_on, pf_off = MaskPrefetcher(), MaskPrefetcher()
    monkeypatch.setenv("GRID_DEFER", "0")
    s_off = GridGrammarSession(g_off, prefetcher=pf_off)
    monkeypatch.setenv("GRID_DEFER", "1")
    s_on = GridGrammarSession(g_on, prefetcher=pf_on)
    words = (g_on.vocab_size + 31) // 32
    bm_on = np.zeros((1, words), dtype=np.int32)
    bm_off = np.zeros((1, words), dtype=np.int32)
    for step in range(12):
        bm_on[:] = 0
        bm_off[:] = 0
        _spin_until(s_on.mask_ready)  # the scheduler's skip-rounds, condensed
        s_on.fill_bitmask(bm_on, 0)
        s_off.fill_bitmask(bm_off, 0)  # defer-off: today's blocking fill
        on = _bits_to_ids(bm_on[0].view(np.uint32), g_on.vocab_size)
        off = _bits_to_ids(bm_off[0].view(np.uint32), g_off.vocab_size)
        assert on == off, f"defer changed mask content at step {step}"
        pick = sorted(t for t in on if t != g_on.eos_token_id)
        if not pick:
            break
        assert s_on.accept_tokens("a", [pick[0]])
        assert s_off.accept_tokens("b", [pick[0]])
        if s_on.is_terminated() or s_off.is_terminated():
            break
    pf_on.shutdown()
    pf_off.shutdown()
