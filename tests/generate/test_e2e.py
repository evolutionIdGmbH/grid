"""Mini-G5 + G6-lite: end-to-end soundness, termination, budget, RBAC (DESIGN.md SS10)."""

import pytest

from grid import generate
from grid._reference.guide import ReferenceGuide
from grid.policy.bundle import PolicyBundle
from grid.policy.schema import SchemaSnapshot
from grid.samplers import greedy, multinomial

N_SEEDS = 25


def test_toy_e2e_soundness_and_termination(toy_source, toy_model, toy_tokenizer):
    g = generate.cfg(toy_model, toy_source, sampler=multinomial(1.0), audit=True)
    guide = g.logits_processor.guide
    ref = ReferenceGuide(guide.tables, guide.dfa, toy_tokenizer)
    jump_completes = 0
    for seed in range(N_SEEDS):
        r = g("", max_tokens=30, seed=seed)
        # INV-OUT1 (binding oracle: own grammar via the reference guide)
        data = r.text.encode("latin-1")
        assert ref.eos_legal(data), f"seed {seed}: output does not parse: {r.text!r}"
        # G5: budget respected, stop reasons legal
        assert len(r.token_ids) <= 30
        assert r.stop_reason in ("EOS_ACCEPT", "MAX_TOKENS_WITH_JUMP_COMPLETE")
        jump_completes += r.stop_reason == "MAX_TOKENS_WITH_JUMP_COMPLETE"
        # audit chain valid and sealed with the EOS tail
        assert r.audit.verify_chain()
        assert r.audit.records[-1].instruction_kind in ("EOS", "WRITE")
    assert jump_completes >= 1, "coverage quota: reserve path unexercised"


def test_sql_e2e_rbac_and_schema(sql_source, sql_model, sql_tokenizer):
    schema = SchemaSnapshot.from_dict(
        {"users": ["id", "name", "email"], "orders": ["id", "user_id", "total"]}
    )
    store = {
        "analyst": {"verbs": ["select"]},
        "admin": {"verbs": ["select", "insert", "update", "delete"]},
    }
    forbidden_words = ("insert", "update", "delete", "salaries")
    for role, banned in (("analyst", forbidden_words), ("admin", ("salaries",))):
        pol = PolicyBundle.from_store(store, role)
        g = generate.sql(sql_model, sql_source, policy=pol, schema=schema,
                         sampler=multinomial(1.0))
        for seed in range(12):
            r = g("", max_tokens=60, seed=seed)
            text = r.text
            for word in banned:
                assert word not in text, f"{role} seed {seed}: RBAC violation {word!r} in {text!r}"
            assert r.stop_reason in ("EOS_ACCEPT", "MAX_TOKENS_WITH_JUMP_COMPLETE")
            assert len(r.token_ids) <= 60


def test_forbidden_identifier_multi_token_unreachable(sql_source, sql_model, sql_tokenizer):
    """G6(a)-lite: 'salaries' cannot be spelled even via sub-token pieces
    ('sala'+'ries', 's'+...); the lexicon prefix rule blocks every path."""
    schema = SchemaSnapshot.from_dict({"users": ["id", "name"]})
    g = generate.sql(sql_model, sql_source, schema=schema, sampler=multinomial(1.0))
    guide = g.logits_processor.guide

    # drive the guide manually into the table position: 'select * from '
    prefix = b"select * from "
    state = guide.initial_state
    for tid in sql_tokenizer.greedy_tokenize(prefix):
        state = guide.get_next_state(state, tid)
    ids, _ = guide._mask_ids(state)
    sala = sql_tokenizer.vocabulary["sala"]
    s_byte = sql_tokenizer.vocabulary["<0x73>"]  # 's'
    assert sala not in ids, "prefix of forbidden identifier admitted at table position"
    # 's' IS a viable prefix of nothing in {users} -> blocked too
    assert s_byte not in ids
    u_byte = sql_tokenizer.vocabulary["<0x75>"]  # 'u' -> prefix of 'users'
    assert u_byte in ids


def test_stop_at_rejected_in_sql_mode(sql_source, sql_model):
    g = generate.sql(sql_model, sql_source, schema=None, sampler=greedy())
    with pytest.raises(ValueError, match="stop_at"):
        g("", max_tokens=10, stop_at=";")


def test_greedy_is_deterministic(toy_source, toy_model):
    g = generate.cfg(toy_model, toy_source, sampler=greedy())
    r1 = g("", max_tokens=20, seed=0)
    r2 = g("", max_tokens=20, seed=5)  # greedy ignores rng
    assert r1.text == r2.text
