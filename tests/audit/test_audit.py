"""G10a: audit chain integrity, tamper detection, replay smoke (DESIGN.md E14)."""

import dataclasses
import random

from grid import generate
from grid.audit.log import AuditLog
from grid.samplers import multinomial


def _generate_with_audit(model, source, seed):
    g = generate.cfg(model, source, sampler=multinomial(1.0), audit=True)
    return g("", max_tokens=25, seed=seed)


def test_chain_verifies_and_seals(toy_source, toy_model):
    r = _generate_with_audit(toy_model, toy_source, 1)
    log = r.audit
    assert log.sealed
    assert log.verify_chain()
    assert log.seal_info["stop_reason"] == r.stop_reason
    assert "grammar" in log.seal_info["artifacts"]
    # every step audited: one record per emitted token
    assert len(log.records) == len(r.token_ids)
    # E14: entry id iff GENERATE
    for rec in log.records:
        assert (rec.mask_entry_id is None) == (rec.instruction_kind in ("WRITE", "EOS"))


def test_eos_record_is_chain_tail(toy_model, toy_source):
    for seed in range(6):
        r = _generate_with_audit(toy_model, toy_source, seed)
        if r.stop_reason == "EOS_ACCEPT":
            assert r.audit.records[-1].instruction_kind == "EOS"
            return
    raise AssertionError("no EOS_ACCEPT stop in 6 seeds")


def test_tamper_detection_property(toy_model, toy_source):
    """Property: mutating any record field breaks chain verification (>=200 trials)."""
    r = _generate_with_audit(toy_model, toy_source, 2)
    base = r.audit
    rng = random.Random(0)
    fields = ["step", "config_hash", "chosen_token", "blocked_count", "instruction_kind"]
    detected = 0
    trials = 200
    for _ in range(trials):
        log = AuditLog(records=list(base.records), sealed=base.sealed,
                       seal_info=dict(base.seal_info))
        i = rng.randrange(len(log.records))
        field = rng.choice(fields)
        rec = log.records[i]
        cur = getattr(rec, field)
        new = cur + 1 if isinstance(cur, int) else ("WRITE" if cur != "WRITE" else "EOS")
        log.records[i] = dataclasses.replace(rec, **{field: new})
        if not log.verify_chain():
            detected += 1
    assert detected == trials, f"tampering missed: {trials - detected}/{trials}"


def test_replay_smoke_same_seed_same_chain(toy_model, toy_source):
    """G10a: identical seeds + artifacts reproduce the identical record chain."""
    r1 = _generate_with_audit(toy_model, toy_source, 3)
    r2 = _generate_with_audit(toy_model, toy_source, 3)
    h1 = [rec.record_hash for rec in r1.audit.records]
    h2 = [rec.record_hash for rec in r2.audit.records]
    assert h1 == h2
