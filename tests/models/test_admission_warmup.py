"""W5 admission warmup (vllm-free core: admission_warmup): journal-driven
prewarm of a fresh schema's template inside the compile_grammar window.

Soundness frame: warmup only moves WHEN exact entries are built (walk/publish/
register through the existing paths), never WHAT they contain; any warmup
failure degrades to today's no-warmup behavior (a compile_grammar exception is
engine-fatal in vLLM 0.24); GRID_ADMIT_WARM=0 is byte-identical to today."""

import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import grid.mask.producer as P
from grid.models.vllm_processor import _GuideRegistry
from grid.models.vllm_structured import GridGrammarSession, admission_warmup

ROOT = pathlib.Path(__file__).parent.parent.parent

SCHEMA_1 = {"employees": ["id", "name"]}
SCHEMA_2 = {"orders": ["total", "qty"]}
SCHEMA_3 = {"library": ["shelf", "book"]}


@pytest.fixture(scope="module")
def spider_source() -> str:
    return (ROOT / "grammars" / "sql_spider.grid").read_text()


@pytest.fixture(autouse=True)
def _admit_warm_on(monkeypatch):
    """Pin the lever ON so an ambient GRID_ADMIT_WARM=0 cannot skew these
    feature tests; the kill-switch test overrides to "0" in its body."""
    monkeypatch.setenv("GRID_ADMIT_WARM", "1")


@pytest.fixture()
def pool():
    p = ThreadPoolExecutor(max_workers=8, thread_name_prefix="grid-warmup-test")
    yield p
    p.shutdown(wait=True)


def _drive_masks(guide, tok, text: bytes):
    """Advance along `text` computing the mask at EVERY state (the serving
    fill pattern): cold configs walk, journal, and publish to T1/T2."""
    st = guide.initial_state
    guide._mask_ids(st)
    for t in tok.greedy_tokenize(text):
        st = guide.get_next_state(st, int(t))
        guide._mask_ids(st)
    return st


def _warmed_registry(spider_source, sql_tokenizer):
    """One prior session over schema 1 = a journal-warm, T2-warm dialect."""
    reg = _GuideRegistry(sql_tokenizer)
    g1 = reg.guide_for({"grammar": spider_source, "schema": SCHEMA_1})
    _drive_masks(g1, sql_tokenizer, b"select name from employees where id=1")
    return reg


def _record_walks(monkeypatch, sink):
    orig = P.walk
    monkeypatch.setattr(
        P, "walk", lambda trie, dfa, rem, A, *a, **k:
        sink.append((rem, frozenset(A))) or orig(trie, dfa, rem, A, *a, **k))


# ------------------------------------------------------------- warmup warms


def test_warmup_warms_tier_ii_zero_boundary_walks_on_replay(
        spider_source, sql_tokenizer, pool, monkeypatch):
    """After warmup, the fresh schema's template has every tier-ii (word, A)
    config T1-warm; a replay of an ident-heavy segment does ZERO boundary or
    generic walks — the only permissible colds are the un-journaled cheap
    residual class (whitespace/mid-lexeme ident configs, ~0.03 ms walks the
    runtime prefetch pool owns)."""
    reg = _warmed_registry(spider_source, sql_tokenizer)
    g2 = reg.guide_for({"grammar": spider_source, "schema": SCHEMA_2})
    prod = g2.producer

    stats = admission_warmup(g2, pool)
    assert stats["enabled"] and stats["initial"] and stats["error"] is None
    assert stats["tier_i"] >= 2 and stats["tier_ii"] >= 5, stats
    assert stats["critical_done"] and stats["errors"] == 0, stats
    pool.shutdown(wait=True)  # tier-i is fire-and-forget: quiesce first

    # every planned tier-ii config is T1-warm under its exact runtime key
    tier_i, tier_ii = prod.journal.plan(g2.lexicons)
    assert stats["tier_i"] == len(tier_i) and stats["tier_ii"] == len(tier_ii)
    for w, ctx in tier_ii:
        key = prod.cache_key(w, ctx)
        assert key[0] == "ident" and key[3] == prod.schema_fingerprint
        assert prod.cache.peek(key) is not None, (w, ctx)

    # the literal acceptance: replaying the ident-heavy segment (the boundary
    # configs themselves) walks ZERO times
    walked: list = []
    _record_walks(monkeypatch, walked)
    for w, ctx in tier_ii:
        prod._entry_for(w, ctx)
    assert walked == [], "tier-ii configs must be T1 hits, never walks"

    # full replay: masks at every state; any cold walk must be in the cheap
    # un-journaled residual class (mid-lexeme ident configs) or a
    # schema-scoped genN first-touch: genN keys embed OUR schema_fp, so
    # another schema's tier-i journal entry can never soundly cover them
    # (their CD partitions are lexicon-filtered — the runtime prefetch/defer
    # owns these). Boundary words and schema-free generic configs stay
    # forbidden: tier-ii / tier-i must have covered those.
    lex_words = {w for ws in g2.lexicons.allowed.values() for w in ws}
    _drive_masks(g2, sql_tokenizer, b"select total from orders where total=1")
    for rem, ctx in walked:
        assert rem not in lex_words, f"boundary config {rem!r} walked cold"
        key = prod.cache_key(rem, ctx)
        if key[0] == "genN":
            assert key[5] == prod.schema_fingerprint and key not in tier_i, \
                f"tier-i-coverable genN config {rem!r} walked cold"
        else:
            assert key[0] == "ident", \
                f"generic config {rem!r} walked cold (tier-i must cover it)"
    assert len(walked) <= 8, f"residual class exploded: {walked}"


def test_warmup_without_journal_still_builds_initial(sql_source, sql_tokenizer, pool):
    """No journal wired (direct build_guide path): warmup degrades to the
    initial-position build alone."""
    from grid.generate import build_guide

    guide = build_guide(sql_source, sql_tokenizer)
    assert guide.producer.journal is None
    stats = admission_warmup(guide, pool)
    assert stats["enabled"] and stats["initial"] and stats["critical_done"]
    assert stats["tier_i"] == 0 and stats["tier_ii"] == 0
    st = guide.initial_state
    assert guide.is_mask_warm(st), "initial config must be T1-warm"


# ---------------------------------------------------------- failure guards


def test_poison_warmup_never_raises_sync_failure(
        spider_source, sql_tokenizer, pool, monkeypatch):
    """Injected failure in the SYNCHRONOUS initial build: admission_warmup
    returns (blanket guard) — compile_grammar would return the grammar and
    the engine stays alive."""
    reg = _warmed_registry(spider_source, sql_tokenizer)
    g2 = reg.guide_for({"grammar": spider_source, "schema": SCHEMA_2})

    def boom(_state):
        raise RuntimeError("poisoned walk")

    monkeypatch.setattr(g2, "_mask_ids", boom)
    stats = admission_warmup(g2, pool)  # must not raise
    assert stats["enabled"] and not stats["initial"]
    assert stats["error"] is not None and "poisoned walk" in stats["error"]


def test_poison_warmup_never_raises_worker_failure(
        spider_source, sql_tokenizer, pool, monkeypatch):
    """Injected failure in every tier-ii walk: errors stay inside their
    futures (counted), admission_warmup returns, and the guide still serves
    a working session afterwards."""
    reg = _warmed_registry(spider_source, sql_tokenizer)
    g3 = reg.guide_for({"grammar": spider_source, "schema": SCHEMA_3})

    def boom(_w, _a):
        raise RuntimeError("poisoned walk")

    monkeypatch.setattr(g3.producer, "prefetch_build", boom)
    stats = admission_warmup(g3, pool)  # must not raise
    assert stats["error"] is None and stats["initial"]
    assert stats["tier_ii"] > 0 and stats["errors"] == stats["tier_ii"], stats
    assert stats["critical_done"], "failed futures still complete the gate"

    # the grammar object built after warmup is fully functional
    session = GridGrammarSession(g3.copy())
    ids, _ = g3._mask_ids(g3.initial_state)
    first = int(next(t for t in ids if t != g3.eos_token_id))
    assert session.accept_tokens("r0", [first]) is True


# ------------------------------------------------------------------ deadline


def test_deadline_bounds_warmup_and_background_completes(
        spider_source, sql_tokenizer, pool, monkeypatch):
    """Slow tier-ii builds: admission_warmup returns at the deadline with the
    critical set incomplete; the builds finish in the background and land as
    exact entries (a late fill would block on them, never approximate)."""
    reg = _warmed_registry(spider_source, sql_tokenizer)
    g2 = reg.guide_for({"grammar": spider_source, "schema": SCHEMA_2})
    prod = g2.producer

    orig = prod.prefetch_build
    started = threading.Event()

    def slow_build(w, ctx):
        started.set()
        time.sleep(0.5)
        orig(w, ctx)

    monkeypatch.setattr(prod, "prefetch_build", slow_build)
    t0 = time.perf_counter()
    stats = admission_warmup(g2, pool, deadline_ms=50.0)
    elapsed_s = time.perf_counter() - t0
    assert started.is_set() and stats["tier_ii"] > 0
    assert stats["critical_done"] is False, "0.5 s builds cannot beat 50 ms"
    assert elapsed_s < 0.45, f"returned after the deadline: {elapsed_s:.3f}s"
    assert stats["deadline_ms"] == 50.0

    pool.shutdown(wait=True)  # background builds run to completion
    _, tier_ii = prod.journal.plan(g2.lexicons)
    for w, ctx in tier_ii:
        assert prod.cache.peek(prod.cache_key(w, ctx)) is not None, \
            "deadline released the request, not the builds"


# ---------------------------------------------------------------- kill switch


def test_kill_switch_is_byte_identical_no_op(spider_source, sql_tokenizer, monkeypatch):
    """GRID_ADMIT_WARM=0: no journal wired, admission_warmup touches NOTHING
    (no pool submit, no cache/T2/kernel/telemetry mutation) — the switch-off
    path is byte-identical to today."""
    monkeypatch.setenv("GRID_ADMIT_WARM", "0")
    reg = _GuideRegistry(sql_tokenizer)
    g1 = reg.guide_for({"grammar": spider_source, "schema": SCHEMA_1})
    _drive_masks(g1, sql_tokenizer, b"select name from employees")
    assert g1.producer.journal is None and reg._journals == {}

    g2 = reg.guide_for({"grammar": spider_source, "schema": SCHEMA_2})
    prod = g2.producer

    class BoomPool:
        def submit(self, *a, **k):
            raise AssertionError("pool must never be touched when off")

    before = (prod.cache.hits, prod.cache.misses, len(prod.cache._t1),
              prod.t2.hits, len(prod._kernel_handles))
    stats = admission_warmup(g2, BoomPool())
    assert stats == {"enabled": False, "initial": False, "tier_i": 0,
                     "tier_ii": 0, "critical_done": False, "errors": 0,
                     "deadline_ms": 0.0, "elapsed_ms": 0.0, "error": None}
    after = (prod.cache.hits, prod.cache.misses, len(prod.cache._t1),
             prod.t2.hits, len(prod._kernel_handles))
    assert after == before, "kill switch must leave every counter/cache untouched"
