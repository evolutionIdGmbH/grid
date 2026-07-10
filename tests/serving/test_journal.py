"""ContextJournal (W4): walk-miss recording, plan() shapes, bounds, thread
safety, and the registry/producer wiring. The journal stores KEY SHAPES only
(generic/genN keys verbatim; ident contexts word-abstracted) — it never
influences mask content, which the full suite + parity runs enforce."""

import pathlib
import threading

import pytest

from grid.models.vllm_processor import _GuideRegistry
from grid.serving import ContextJournal
from grid.trie.walk import Lexicons

ROOT = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture(scope="module")
def spider_source() -> str:
    return (ROOT / "grammars" / "sql_spider.grid").read_text()


@pytest.fixture(autouse=True)
def _admit_warm_on(monkeypatch):
    """Pin the lever ON so an ambient GRID_ADMIT_WARM=0 cannot skew the
    wiring tests; the kill-switch test overrides to "0" in its body."""
    monkeypatch.setenv("GRID_ADMIT_WARM", "1")


def _drive_masks(guide, tok, text: bytes):
    """Advance along `text` computing the mask at EVERY state (the serving
    fill pattern) so cold configurations actually walk and journal."""
    st = guide.initial_state
    guide._mask_ids(st)
    for t in tok.greedy_tokenize(text):
        st = guide.get_next_state(st, int(t))
        guide._mask_ids(st)
    return st


# ------------------------------------------------------------------- unit


def test_records_dedupe():
    j = ContextJournal()
    for _ in range(4):
        j.record_generic(("generic", b"x", (1, 2), None))
        j.record_ident_context(frozenset({1, 2}))
    assert j.stats == {"generic_keys": 1, "ident_contexts": 1, "cap": 4096}


def test_bounded_drops_overflow_keeps_first_seen():
    j = ContextJournal(cap=3)
    for i in range(10):
        j.record_generic(("generic", bytes([i]), (), None))
        j.record_ident_context(frozenset({i}))
    assert j.stats["generic_keys"] == 3 and j.stats["ident_contexts"] == 3
    tier_i, _ = j.plan(None)
    assert tier_i == [("generic", bytes([i]), (), None) for i in range(3)]


def test_env_cap(monkeypatch):
    monkeypatch.setenv("GRID_JOURNAL_MAX", "5")
    assert ContextJournal().stats["cap"] == 5
    assert ContextJournal(cap=2).stats["cap"] == 2  # explicit beats env


def test_plan_shapes_and_word_filtering():
    """tier_ii = journaled contexts x THIS schema's words, filtered to words
    whose terminal is IN the context (a word whose terminal the context
    disallows can never be its pending boundary lexeme); deterministic order
    (contexts first-seen, words sorted); no lexicons -> no tier-ii."""
    j = ContextJournal()
    k1 = ("genN", -1, 3, b"", (7, 9))
    k2 = ("generic", b"select", (4,), None)
    j.record_generic(k1)
    j.record_generic(k2)
    a1, a2 = frozenset({1, 7}), frozenset({1, 2})
    j.record_ident_context(a1)
    j.record_ident_context(a2)
    lex = Lexicons({1: {b"bb", b"aa"}, 2: {b"cc"}})
    tier_i, tier_ii = j.plan(lex)
    assert tier_i == [k1, k2]
    assert tier_ii == [(b"aa", a1), (b"bb", a1),
                       (b"aa", a2), (b"bb", a2), (b"cc", a2)]
    assert j.plan(None) == ([k1, k2], [])


def test_thread_safety_smoke():
    j = ContextJournal(cap=100_000)

    def rec(base: int) -> None:
        for i in range(500):
            j.record_generic(("generic", bytes([base]), (i,), None))
            j.record_ident_context(frozenset({base * 1000 + i}))

    threads = [threading.Thread(target=rec, args=(b,)) for b in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert j.stats["generic_keys"] == 8 * 500
    assert j.stats["ident_contexts"] == 8 * 500


# ---------------------------------------------------------------- wiring


def test_registry_scopes_one_journal_per_dialect(spider_source, sql_source, sql_tokenizer):
    reg = _GuideRegistry(sql_tokenizer)
    g1 = reg.guide_for({"grammar": spider_source, "schema": {"employees": ["id", "name"]}})
    g2 = reg.guide_for({"grammar": spider_source, "schema": {"orders": ["total", "qty"]}})
    assert isinstance(g1.producer.journal, ContextJournal)
    assert g1.producer.journal is g2.producer.journal, "one journal per dialect"
    assert g1.producer is not g2.producer
    g3 = reg.guide_for({"grammar": sql_source})
    assert g3.producer.journal is not g1.producer.journal, "dialects don't share"
    # copies share the template's producer, hence its journal
    assert g1.copy().producer.journal is g1.producer.journal


def test_kill_switch_wires_no_journal(spider_source, sql_tokenizer, monkeypatch):
    monkeypatch.setenv("GRID_ADMIT_WARM", "0")
    reg = _GuideRegistry(sql_tokenizer)
    g = reg.guide_for({"grammar": spider_source, "schema": {"employees": ["id", "name"]}})
    assert g.producer.journal is None, "switch off: today's exact producer"
    _drive_masks(g, sql_tokenizer, b"select name from employees")
    assert reg._journals == {}


def test_producer_journals_boundary_idents_and_generic_keys(spider_source, sql_tokenizer):
    """Driving one masked session records (a) generic/genN keys verbatim
    (tier-i) and (b) ident BOUNDARY A-contexts word-abstracted (tier-ii);
    mid-lexeme ident prefixes and whitespace-pending configs are never
    journaled (the cheap runtime-prefetch residual class)."""
    reg = _GuideRegistry(sql_tokenizer)
    guide = reg.guide_for({"grammar": spider_source, "schema": {"employees": ["id", "name"]}})
    journal = guide.producer.journal
    _drive_masks(guide, sql_tokenizer, b"select name from employees where id=1")

    stats = journal.stats
    # column position, table position, where position: >= 3 distinct contexts
    assert stats["ident_contexts"] >= 3, stats
    assert stats["generic_keys"] >= 2, stats  # b"" initial + b"select" at least

    ident_ids = guide.tables.identifier_terminal_ids
    tier_i, tier_ii = journal.plan(guide.lexicons)
    assert all(k[0] in ("generic", "genN") for k in tier_i), "never ident keys"
    assert {w for w, _ in tier_ii} == {b"id", b"name", b"employees"}
    for _, ctx in tier_ii:
        assert ctx & ident_ids, "contexts are ident positions by construction"
    # word abstraction: a SECOND schema's plan enumerates ITS words over the
    # same recorded contexts
    lex2 = Lexicons({tid: {b"zz_w"} for tid in guide.producer._LEX})
    _, tier_ii_2 = journal.plan(lex2)
    assert {w for w, _ in tier_ii_2} == {b"zz_w"}
    assert {a for _, a in tier_ii_2} == {a for _, a in tier_ii}
