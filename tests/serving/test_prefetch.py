"""MaskPrefetcher (§6 overlap contract): successor-state masks build on a
worker thread into the shared write-back cache; fill-time waits are bounded
residuals; duplicate schedules dedupe; results are never approximate — the
built entry is the same one the synchronous path would compute."""

import random

import numpy as np
import pytest

from grid.generate import build_guide
from grid.guide import COMPLETE
from grid.models.vllm_structured import GridGrammarSession
from grid.serving import MaskPrefetcher


@pytest.fixture
def guide(toy_source, toy_tokenizer):
    return build_guide(toy_source, toy_tokenizer)


def test_schedule_builds_into_shared_cache(guide):
    pf = MaskPrefetcher()
    state = guide.initial_state
    misses0 = guide.producer.cache.misses
    pf.schedule(guide, state)
    pf.wait(state, timeout=10)
    assert guide.producer.cache.misses > misses0, "prefetch walked the cold config"
    # the scheduler-thread fill is now a pure hit: no further misses
    m1 = guide.producer.cache.misses
    ids, _ = guide._mask_ids(state)
    assert guide.producer.cache.misses == m1
    assert ids.size > 0
    pf.shutdown()


def test_prefetched_mask_identical_to_synchronous(toy_source, toy_tokenizer):
    cold = build_guide(toy_source, toy_tokenizer)
    warm = build_guide(toy_source, toy_tokenizer)
    pf = MaskPrefetcher()
    rng = random.Random(3)
    s_cold, s_warm = cold.initial_state, warm.initial_state
    for _ in range(12):
        pf.schedule(warm, s_warm)
        pf.wait(s_warm, timeout=10)
        a, _ = cold._mask_ids(s_cold)
        b, _ = warm._mask_ids(s_warm)
        assert a.tolist() == b.tolist()
        pool = sorted(set(int(i) for i in a) - {cold.eos_token_id}) or [int(a[0])]
        tok = rng.choice(pool)
        s_cold = cold.get_next_state(s_cold, tok)
        s_warm = warm.get_next_state(s_warm, tok)
        if s_cold.status == COMPLETE:
            break
    pf.shutdown()


def test_dedupe_and_drop(guide):
    pf = MaskPrefetcher()
    state = guide.initial_state
    pf.schedule(guide, state)
    pf.schedule(guide, state)
    assert pf.stats["deduped"] == 1
    pf.wait(state, timeout=10)
    assert pf.wait(state) == 0.0, "second wait is a no-op (entry popped)"
    pf.schedule(guide, state)
    pf.drop(state)
    assert pf.wait(state) == 0.0
    pf.shutdown()


def test_session_prefetches_on_accept(guide):
    pf = MaskPrefetcher()
    session = GridGrammarSession(guide, prefetcher=pf)
    ids, _ = guide._mask_ids(guide.initial_state)
    tok = next(int(t) for t in ids if int(t) != guide.eos_token_id)
    assert session.accept_tokens("req-0", [tok])
    assert pf.stats["scheduled"] == 1
    # fill waits for the in-flight build, then fills from the warm entry
    words = (guide.vocab_size + 31) // 32
    bitmask = np.zeros((1, words), dtype=np.int32)
    session.fill_bitmask(bitmask, 0)
    assert pf.stats["waits"] == 1
    assert bitmask.any(), "mask must not be empty at a viable state"
    pf.shutdown()


def test_session_skips_schedule_for_warm_successor(guide):
    """Warm steady state never touches the pool (the G8 batched-TPOT fix):
    a successor whose configuration is already T1-warm is not scheduled."""
    pf = MaskPrefetcher()
    session = GridGrammarSession(guide, prefetcher=pf)
    ids, _ = guide._mask_ids(guide.initial_state)
    tok = next(int(t) for t in ids if int(t) != guide.eos_token_id)
    nxt = guide.get_next_state(guide.initial_state, tok)
    guide._mask_ids(nxt)  # warm the successor configuration synchronously
    assert guide.is_mask_warm(nxt)
    assert session.accept_tokens("req-0", [tok])
    assert pf.stats["scheduled"] == 0, "warm successor must not be scheduled"
    words = (guide.vocab_size + 31) // 32
    bitmask = np.zeros((1, words), dtype=np.int32)
    session.fill_bitmask(bitmask, 0)
    assert bitmask.any()
    pf.shutdown()


def test_walk_runs_from_worker_thread(guide):
    """The GIL-released walk is callable off the main thread (overlap smoke)."""
    import threading

    state = guide.initial_state
    err: list = []

    def worker():
        try:
            ids, _ = guide._mask_ids(state)
            assert ids.size > 0
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join(10)
    assert not err, err
