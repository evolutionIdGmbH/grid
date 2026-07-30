"""ContextJournal persistence (S3, CANDIDATES #20a): snapshot/restore
roundtrips, cap enforcement, store wiring through the registry, write-back
points, warmed-vs-demand entry-id parity, and cross-process genN (p,q)
stability. The journal namespace stores keys/contexts ONLY — parity gates
prove a restored journal can never influence mask content."""

import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from grid.models.vllm_processor import _GuideRegistry
from grid.models.vllm_structured import admission_warmup
from grid.serving import ContextJournal
from grid.serving import artifact_store as store

ROOT = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture(scope="module")
def spider_source() -> str:
    return (ROOT / "grammars" / "sql_spider.grid").read_text()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "1")
    monkeypatch.setenv("GRID_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GRID_ADMIT_WARM", "1")
    return tmp_path


def _journal_bins(root):
    return [p for p in root.rglob("*.bin") if p.parent.name == "journal"]


def _drive_masks(guide, tok, text: bytes):
    """The serving fill pattern: compute the mask at every state so cold
    configurations walk, journal, and publish."""
    st = guide.initial_state
    ids = [guide._mask_ids(st)[1]]
    for t in tok.greedy_tokenize(text):
        st = guide.get_next_state(st, int(t))
        ids.append(guide._mask_ids(st)[1])
    return ids


K1 = ("genN", -1, 3, b"", (7, 9), None)
K2 = ("generic", b"select", (4,), None)


# ------------------------------------------------------------------- unit


def test_snapshot_restore_roundtrip():
    j = ContextJournal()
    j.record_generic(K1)
    j.record_generic(K2)
    j.record_ident_context(frozenset({1, 7}))
    snap = j.snapshot()
    j2 = ContextJournal()
    j2.restore(snap)
    assert j2.snapshot() == snap
    assert j2.stats["generic_keys"] == 2 and j2.stats["ident_contexts"] == 1
    tier_i, _ = j2.plan(None)
    assert tier_i == [K1, K2]  # first-seen order preserved


def test_restore_enforces_cap_and_first_seen():
    big = {"generic": tuple(("generic", bytes([i]), (), None) for i in range(10)),
           "ident": tuple(frozenset({i}) for i in range(10))}
    j = ContextJournal(cap=3)
    j.restore(big)
    assert j.stats["generic_keys"] == 3 and j.stats["ident_contexts"] == 3
    tier_i, _ = j.plan(None)
    assert tier_i == list(big["generic"][:3])


def test_restore_is_shape_defensive():
    j = ContextJournal()
    for garbage in (None, 42, "x", [], {"generic": 7}, {"generic": ["notuple"],
                    "ident": ["notafrozenset"]}, {"other": ()}):
        j.restore(garbage)
    assert j.stats["generic_keys"] == 0 and j.stats["ident_contexts"] == 0


def test_restore_never_marks_dirty_and_unbound_never_flushes(cache, monkeypatch):
    monkeypatch.setenv("GRID_PERF_STORE_JOURNAL_EVERY", "1")
    j = ContextJournal()  # unbound: no store key
    j.restore({"generic": (K1,), "ident": ()})
    j.record_generic(K2)  # dirty, but unbound
    j.flush()
    assert _journal_bins(cache) == []


def test_self_flush_every_n_records(cache, monkeypatch):
    monkeypatch.setenv("GRID_PERF_STORE_JOURNAL_EVERY", "2")
    j = store.load_or_restore_journal("dialect-src")
    j.record_generic(K1)
    assert _journal_bins(cache) == []  # 1 < 2: not yet
    j.record_generic(K2)
    assert len(_journal_bins(cache)) == 1  # threshold crossed
    # a fresh "process": restore sees both records
    j2 = store.load_or_restore_journal("dialect-src")
    assert j2.stats["generic_keys"] == 2


def test_loader_passthrough_when_flags_off(cache, tmp_path, monkeypatch):
    for flag, val in (("GRID_PERF_ARTIFACT_STORE", "0"),
                      ("GRID_PERF_STORE_JOURNAL", "0")):
        monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "1")
        monkeypatch.setenv(flag, val)
        j = store.load_or_restore_journal("dialect-src")
        j.record_generic(K1)
        j.flush()
        assert _journal_bins(cache) == [], flag


def test_explicit_flush_and_reload(cache):
    j = store.load_or_restore_journal("d2")
    j.record_generic(K1)
    j.record_ident_context(frozenset({3}))
    j.flush()
    j2 = store.load_or_restore_journal("d2")
    assert j2.snapshot() == j.snapshot()
    # dialect separation: another grammar source restores nothing
    assert store.load_or_restore_journal("d3").stats["generic_keys"] == 0


# ------------------------------------------------------------------ wiring


def test_registry_restores_journal_for_fresh_dialect(cache, spider_source,
                                                     sql_tokenizer):
    schema = {"employees": ["id", "name"]}
    reg_a = _GuideRegistry(sql_tokenizer)
    guide_a = reg_a.guide_for({"grammar": spider_source, "schema": schema})
    _drive_masks(guide_a, sql_tokenizer, b"select name from employees")
    stats_a = guide_a.producer.journal.stats
    assert stats_a["generic_keys"] > 0 and stats_a["ident_contexts"] > 0
    reg_a.flush_journals()
    assert len(_journal_bins(cache)) == 1

    # simulated redeploy: a fresh registry's journal is warm BEFORE any drive
    reg_b = _GuideRegistry(sql_tokenizer)
    guide_b = reg_b.guide_for({"grammar": spider_source, "schema": schema})
    stats_b = guide_b.producer.journal.stats
    assert stats_b["generic_keys"] == stats_a["generic_keys"]
    assert stats_b["ident_contexts"] == stats_a["ident_contexts"]


def test_admission_warmup_flushes_bound_journal(cache, spider_source,
                                                sql_tokenizer):
    reg = _GuideRegistry(sql_tokenizer)
    guide = reg.guide_for({"grammar": spider_source,
                           "schema": {"employees": ["id", "name"]}})
    _drive_masks(guide, sql_tokenizer, b"select id from employees")
    assert _journal_bins(cache) == []  # below the self-flush threshold
    with ThreadPoolExecutor(2) as pool:
        stats = admission_warmup(guide, pool)
    assert stats["enabled"] and stats["error"] is None
    assert len(_journal_bins(cache)) == 1  # warmup completion wrote back


def test_warmed_entries_are_entry_id_identical_to_demand(cache, spider_source,
                                                         sql_tokenizer,
                                                         monkeypatch):
    """The replay gate: a redeployed registry whose journal was RESTORED (not
    recorded) runs admission warmup, then serves the same drives — every
    per-step entry id must equal a no-journal producer's demand-computed ids
    (and OBL-KEY1 publish asserts fire during the drive on any divergence)."""
    schema = {"employees": ["id", "name"], "orders": ["total", "qty"]}
    texts = [b"select name from employees",
             b"select total from orders where qty=1",
             b"select id from employees where id=2"]

    reg_a = _GuideRegistry(sql_tokenizer)
    guide_a = reg_a.guide_for({"grammar": spider_source, "schema": schema})
    for t in texts:
        _drive_masks(guide_a, sql_tokenizer, t)
    reg_a.flush_journals()

    # redeploy: fresh registry, restored journal, warmup precomputes walks
    reg_b = _GuideRegistry(sql_tokenizer)
    guide_b = reg_b.guide_for({"grammar": spider_source, "schema": schema})
    assert guide_b.producer.journal.stats["ident_contexts"] > 0
    with ThreadPoolExecutor(2) as pool:
        stats = admission_warmup(guide_b, pool)
    assert stats["tier_ii"] > 0 and stats["errors"] == 0

    # oracle: same artifacts, journal machinery off entirely
    monkeypatch.setenv("GRID_ADMIT_WARM", "0")
    reg_c = _GuideRegistry(sql_tokenizer)
    guide_c = reg_c.guide_for({"grammar": spider_source, "schema": schema})
    for t in texts:
        ids_b = _drive_masks(guide_b, sql_tokenizer, t)
        ids_c = _drive_masks(guide_c, sql_tokenizer, t)
        assert ids_b == ids_c, t


# ------------------------------------------------- cross-process stability


def test_genn_pq_numbering_is_cross_process_stable(spider_source, tmp_path):
    """genN tier-i keys embed eager-scanner state ids (p, q): the coherence
    argument for restoring them across processes is that subset numbering is
    deterministic per (code epoch, grammar). Two fresh interpreters must
    produce identical (q, l, p) triples and transition-table hashes."""
    src_file = tmp_path / "g.grid"
    src_file.write_text(spider_source)
    probe = (
        "import hashlib, sys\n"
        "from grid.grammar import spec\n"
        "from grid.lexer.dfa import build_scanner\n"
        "g = spec.load(open(sys.argv[1]).read())\n"
        "dfa = build_scanner(g.terminals, g.terminal_order)\n"
        "if getattr(dfa, 'lazy', False):\n"
        "    dfa = dfa.materialize(10**9)\n"
        "rems = [b'', b'select', b'select ', b'select n', b'1', b'1e', b'1E',\n"
        "        b'where x', b'from']\n"
        "out = [repr(dfa.scan_with_last_accept(r)) for r in rems]\n"
        "h = hashlib.blake2b(repr(dfa.trans).encode(), digest_size=16)\n"
        "print('|'.join(out), h.hexdigest())\n"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", probe, str(src_file)],
            capture_output=True, text=True, check=True,
            cwd=str(ROOT), env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
        ).stdout
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    assert "|" in runs[0]  # the probe actually produced triples
