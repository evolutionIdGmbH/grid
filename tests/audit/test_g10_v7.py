"""G10 replay key-format compat UNDER KERNEL v7 (red-team plan §4.4): the
dual-key replay must stay green when the producer materializes entries
in-kernel (walk_payload + register_blob). dual_key_check's
``make_entry(k, list(e.ci_tokens), e.cd_entries, V).entry_id == e.entry_id``
becomes a free Python-vs-Rust entry_id differential on every consulted
config (e.entry_id was hashed by the kernel, the recomputation by hashlib),
and the lazy MaskEntryV7 decode feeds cd_entries. Mirrors
tests/audit/test_g10_key_format.py — the assertions are the same, only the
build regime differs (this file must never weaken them)."""

import pathlib
import sys

import pytest

from grid.generate import build_guide
from grid.mask.cache import MaskEntryV7
from grid.models.mock import MockModel
from grid.models.tokenizer_adapter import MockTokenizer
from grid.samplers import multinomial

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "bench"))

import g10_replay as g10  # noqa: E402

pytestmark = pytest.mark.skipif(
    __import__("grid.trie.walk", fromlist=["_USE_RUST"])._USE_RUST is False,
    reason="kernel v7 requires grid_core (disabled via GRID_NO_RUST)",
)


@pytest.fixture(scope="module")
def template():
    mp = pytest.MonkeyPatch()
    mp.setenv("GRID_V7", "1")
    source = (ROOT / "grammars" / "sql_subset.grid").read_text()
    t = build_guide(source, MockTokenizer(g10.SQL_TOKENS), audit=True)
    t.producer.set_genn_keys(True)
    assert t.producer._v7, "suite must run the v7 build regime"
    yield t
    mp.undo()


@pytest.fixture(scope="module")
def tok(template):
    return template.adapter


def _gen(template, tok, seed: int):
    return g10.generate_one(template, MockModel(tok, seed=seed),
                            multinomial(1.0), seed=seed)[1]


def test_v1_log_replays_bit_identical_via_dual_key_under_v7(template, tok):
    prod = template.producer
    prod.set_genn_keys(False)  # record v1 logs (legacy raw keys)
    try:
        v1_logs = [_gen(template, tok, seed) for seed in (101, 102, 103)]
    finally:
        prod.set_genn_keys(True)
    assert any(isinstance(e, MaskEntryV7) for e in prod.cache._t1.values()), \
        "v7 build regime not exercised (vacuous)"
    for log in v1_logs:
        rebuilt = g10.replay_records(template, log.records,
                                     header={"key_format": g10.KEY_FORMAT_RAW})
        assert rebuilt == [r.record_hash for r in log.records], \
            "v1 log must replay bit-identical on the v7 genN producer"
    assert g10.producer_key_format(prod) == g10.KEY_FORMAT_GENN


def test_v2_log_round_trips_natively_under_v7(template, tok):
    log = _gen(template, tok, 202)
    header = g10.replay_header(template)
    assert header == {"key_format": g10.KEY_FORMAT_GENN}
    rebuilt = g10.replay_records(template, log.records, header=header)
    assert rebuilt == [r.record_hash for r in log.records]
    assert g10.replay_records(template, log.records) == rebuilt


def test_dual_key_check_covers_generic_configs_under_v7(template, tok):
    log = _gen(template, tok, 303)
    prod = template.producer
    configs: set = set()
    orig = type(prod).cache_key
    prod.cache_key = lambda r, A: configs.add((bytes(r), A)) or orig(prod, r, A)
    try:
        g10._replay_chain(template, log.records)
    finally:
        del prod.cache_key
    assert configs, "replay consulted no configs (harness broken)"
    checked = g10.dual_key_check(prod, configs)
    assert checked >= 1, "dual-key check compared no generic configs (vacuous)"


def test_unknown_key_format_is_hard_error_under_v7(template, tok):
    log = _gen(template, tok, 404)
    with pytest.raises(ValueError, match="key_format"):
        g10.replay_records(template, log.records, header={"key_format": "v3"})
    with pytest.raises(ValueError, match="key_format"):
        g10.replay_records(template, log.records, header={})
