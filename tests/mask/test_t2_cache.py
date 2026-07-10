"""T2 cross-template tier (DESIGN §E10; the G4 cross-role/cross-template
slice): templates of different schemas over one dialect share the
schema-independent (generic) entries — a fresh template's literal-interior
giants arrive warm instead of re-walking — while identifier-position entries
stay schema-scoped by key (E11), and a T2 handover is bit-identical to a
local walk (OBL-KEY1)."""

import grid.mask.producer as P
from grid.models.vllm_processor import _GuideRegistry


def test_generic_entries_cross_schemas_without_walk(sql_source, sql_tokenizer, monkeypatch):
    """LEGACY (v1 key regime) behavior: cross-schema T2 handover of generic
    entries. In the v2 regime (GRID_GENN_KEYS=1, the default) generic keys are
    schema-scoped — walk-time CD filtering embeds schema words, so unscoped
    cross-schema handovers can serve one schema's continuations to another
    (tests/mask/test_genn_keys.py::test_raw_fallback_is_schema_scoped_v2, the
    50-seed shared-registry fuzz counterexample). This test pins the v1
    kill-switch path byte-for-byte, and the v2 scoping at the end."""
    reg = _GuideRegistry(sql_tokenizer)
    ga = reg.guide_for({"grammar": sql_source, "schema": {"users": ["id", "name"]}})
    ga.producer.set_genn_keys(False)  # v1 legacy regime: unscoped generic keys
    ids_a, _ = ga._mask_ids(ga.initial_state)  # cold walk on template A

    calls = []
    orig = P.walk
    monkeypatch.setattr(P, "walk", lambda *a, **k: calls.append(1) or orig(*a, **k))

    gb = reg.guide_for({"grammar": sql_source, "schema": {"orders": ["total", "qty"]}})
    gb.producer.set_genn_keys(False)
    assert gb.producer is not ga.producer, "distinct schemas -> distinct templates"
    assert gb.producer.t2 is ga.producer.t2, "one T2 pool per dialect"
    ids_b, _ = gb._mask_ids(gb.initial_state)  # generic config: T2 handover
    assert not calls, "fresh template re-walked a shared generic configuration"
    assert gb.producer.t2.hits >= 1
    assert ids_b.tolist() == ids_a.tolist(), "T2 handover must be bit-identical (OBL-KEY1)"

    # v2 regime: the same configuration is schema-scoped -> the fresh
    # template WALKS instead of adopting a foreign-schema entry
    gc = reg.guide_for({"grammar": sql_source, "schema": {"crew": ["vin", "rank"]}})
    gc.producer.set_genn_keys(True)
    n0 = len(calls)
    ids_c, _ = gc._mask_ids(gc.initial_state)
    assert len(calls) == n0 + 1, "v2: scoped generic key must re-walk, not adopt"
    assert ids_c.tolist() == ids_a.tolist(), \
        "walk result equals the handover here (vocab has no cross-lexeme tokens)"


def test_ident_entries_stay_schema_scoped(sql_source, sql_tokenizer):
    reg = _GuideRegistry(sql_tokenizer)
    ga = reg.guide_for({"grammar": sql_source, "schema": {"users": ["id", "name"]}})
    gb = reg.guide_for({"grammar": sql_source, "schema": {"orders": ["total", "qty"]}})

    def at_table_pos(g):
        st = g.initial_state
        for t in sql_tokenizer.greedy_tokenize(b"select * from "):
            st = g.get_next_state(st, int(t))
        return st

    sa, sb = at_table_pos(ga), at_table_pos(gb)
    ka = ga.producer.cache_key(sa.lexer.remainder, ga.producer.allowed(sa.stack))
    kb = gb.producer.cache_key(sb.lexer.remainder, gb.producer.allowed(sb.stack))
    assert ka[0] == "ident" == kb[0]
    assert ka != kb, "identifier keys must carry the schema fingerprint (E11)"
    ids_a, _ = ga._mask_ids(sa)
    ids_b, _ = gb._mask_ids(sb)
    assert set(ids_a.tolist()) != set(ids_b.tolist()), "different lexicons, different masks"
