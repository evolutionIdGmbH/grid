"""G10 replay key-format compat (W3): logs recorded under the legacy v1 raw
keys replay bit-identically through a genN (v2) producer via the dual-key
path — every consulted config's entry is byte-compared under BOTH key forms
(proving the genN merge is a pure re-keying) — v2 logs round-trip natively,
and an unknown header version is a hard error."""

import pathlib
import sys

import pytest

from grid.generate import build_guide
from grid.models.mock import MockModel
from grid.models.tokenizer_adapter import MockTokenizer
from grid.samplers import multinomial

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "bench"))

import g10_replay as g10  # noqa: E402


@pytest.fixture(scope="module")
def template():
    source = (ROOT / "grammars" / "sql_subset.grid").read_text()
    t = build_guide(source, MockTokenizer(g10.SQL_TOKENS), audit=True)
    t.producer.set_genn_keys(True)  # the lever under test, independent of env
    return t


@pytest.fixture(scope="module")
def tok(template):
    return template.adapter


def _gen(template, tok, seed: int):
    return g10.generate_one(template, MockModel(tok, seed=seed),
                            multinomial(1.0), seed=seed)[1]


def test_v1_log_replays_bit_identical_via_dual_key(template, tok):
    prod = template.producer
    assert g10.producer_key_format(prod) == g10.KEY_FORMAT_GENN, \
        "suite baseline runs with genN keys on"
    prod.set_genn_keys(False)  # record v1 logs (legacy raw keys)
    try:
        v1_logs = [_gen(template, tok, seed) for seed in (101, 102, 103)]
    finally:
        prod.set_genn_keys(True)
    for log in v1_logs:
        rebuilt = g10.replay_records(template, log.records,
                                     header={"key_format": g10.KEY_FORMAT_RAW})
        assert rebuilt == [r.record_hash for r in log.records], \
            "v1 log must replay bit-identical on the genN producer"
    assert g10.producer_key_format(prod) == g10.KEY_FORMAT_GENN, \
        "replay must restore the native key format"


def test_v2_log_round_trips_natively(template, tok):
    log = _gen(template, tok, 202)
    header = g10.replay_header(template)
    assert header == {"key_format": g10.KEY_FORMAT_GENN}
    rebuilt = g10.replay_records(template, log.records, header=header)
    assert rebuilt == [r.record_hash for r in log.records]
    # header=None means "same-process native log": identical result
    assert g10.replay_records(template, log.records) == rebuilt


def test_dual_key_check_covers_generic_configs(template, tok):
    """The dual-key recompute must actually compare configs (not be vacuous)."""
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
    assert g10.producer_key_format(prod) == g10.KEY_FORMAT_GENN


def test_unknown_key_format_is_hard_error(template, tok):
    log = _gen(template, tok, 404)
    with pytest.raises(ValueError, match="key_format"):
        g10.replay_records(template, log.records, header={"key_format": "v3"})
    with pytest.raises(ValueError, match="key_format"):
        g10.replay_records(template, log.records, header={})
