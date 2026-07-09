"""E17 single-flight: one build per fingerprint, waiters share the result or
the same exception object, FAILED negatively cached with TTL (DESIGN.md E17,
G8 concurrent-cold-start criterion)."""

import threading

import pytest

from grid.serving import SingleFlight


def test_concurrent_waiters_share_one_build():
    sf = SingleFlight()
    release = threading.Event()
    started = threading.Event()
    builds = []

    def builder():
        builds.append(1)
        started.set()
        release.wait(5)
        return object()

    results: list = [None] * 8

    def worker(i):
        results[i] = sf.get_or_build("k", builder)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    threads[0].start()
    assert started.wait(5), "builder never started"
    for t in threads[1:]:
        t.start()
    release.set()
    for t in threads:
        t.join(5)
    assert len(builds) == 1, "single-flight must build exactly once"
    assert all(r is results[0] for r in results), "all waiters share the READY object"
    assert sf.stats["builds"] == 1 and sf.stats["joined"] >= 1


def test_failure_fans_out_same_exception_and_is_negatively_cached():
    clock = [0.0]
    sf = SingleFlight(failed_ttl_s=10.0, clock=lambda: clock[0])
    boom = ValueError("bad grammar")
    release = threading.Event()

    def builder():
        release.wait(5)
        raise boom

    errors: list = [None] * 4

    def worker(i):
        try:
            sf.get_or_build("k", builder)
        except ValueError as e:
            errors[i] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    release.set()
    for t in threads:
        t.join(5)
    assert all(e is boom for e in errors), "every waiter gets the same exception object"

    # negative cache within TTL: no rebuild, same error, instantly
    calls = []
    with pytest.raises(ValueError):
        sf.get_or_build("k", lambda: calls.append(1))
    assert not calls and sf.stats["negative_hits"] >= 1

    # TTL expiry: the slot is removed and the next request rebuilds
    clock[0] = 11.0
    fresh = object()
    assert sf.get_or_build("k", lambda: fresh) is fresh


def test_ready_hits_and_evict():
    sf = SingleFlight()
    v1 = sf.get_or_build("k", lambda: object())
    assert sf.get_or_build("k", lambda: object()) is v1
    assert sf.stats["ready_hits"] == 1
    sf.evict("k")
    v2 = sf.get_or_build("k", lambda: object())
    assert v2 is not v1


def test_guide_registry_singleflight_and_failure(toy_source, toy_tokenizer):
    """The vLLM compile path: concurrent same-spec compiles build one template;
    an invalid grammar raises the same GrammarInvalid for every waiter."""
    from grid.errors import GrammarInvalid, LALRConflictError
    from grid.models.vllm_processor import _GuideRegistry

    reg = _GuideRegistry(toy_tokenizer)
    spec = {"grammar": toy_source}
    guides: list = [None] * 4

    def worker(i):
        guides[i] = reg.guide_for(spec)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert all(g is not None for g in guides)
    # copies of ONE template: they share the template's write-back mask cache
    assert len({id(g.producer.cache) for g in guides}) == 1
    assert reg.stats["builds"] == 1

    with pytest.raises((GrammarInvalid, LALRConflictError)) as first:
        reg.guide_for({"grammar": "start: BROKEN"})
    errs = reg.stats["failures"]
    with pytest.raises((GrammarInvalid, LALRConflictError)) as second:
        reg.guide_for({"grammar": "start: BROKEN"})  # negative-cached, no rebuild
    assert second.value is first.value, "negative cache serves the same exception object"
    assert reg.stats["failures"] == errs


def test_guide_registry_verb_rbac_envelope(sql_source, sql_tokenizer):
    """The vLLM grid envelope's optional `verbs` list projects the grammar to
    those statement kinds before compile — verb-RBAC over serving. Without it
    the full grammar admits every verb (the G6(b) prompt-suite finding)."""
    from grid.models.vllm_processor import _GuideRegistry

    reg = _GuideRegistry(sql_tokenizer)

    def can_start(guide, prefix: bytes) -> bool:
        st = guide.initial_state
        for t in sql_tokenizer.greedy_tokenize(prefix):
            nxt = guide._advance(st, int(t), audit=False)
            if nxt is None:
                return False
            st = nxt
        return True

    schema = {"users": ["id", "name"]}
    full = reg.guide_for({"grammar": sql_source, "schema": schema})
    sel = reg.guide_for({"grammar": sql_source, "schema": schema, "verbs": ["select"]})
    assert can_start(full, b"insert "), "full grammar should admit insert"
    assert can_start(sel, b"select "), "select must remain admitted under verbs=[select]"
    assert not can_start(sel, b"insert "), "verbs=[select] must block insert (verb-RBAC)"
    # distinct fingerprints -> distinct templates -> two builds
    assert reg.stats["builds"] == 2
