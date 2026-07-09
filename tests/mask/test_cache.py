import pytest

from grid.errors import IdentifierMaskBypassError
from grid.mask.cache import TAG_ACCEPT, TAG_BITSET, TAG_REJECT, MaskCache, adaptive_encode, make_entry


def test_adaptive_encode_tag_selection():
    v = 1000
    assert adaptive_encode((1, 2), v)[0] == TAG_ACCEPT
    assert adaptive_encode(tuple(range(998)), v)[0] == TAG_REJECT
    assert adaptive_encode(tuple(range(0, 1000, 2)), v)[0] == TAG_BITSET


def test_encode_deterministic_and_order_free():
    a = adaptive_encode((3, 1, 2), 100)
    b = adaptive_encode((2, 3, 1), 100)
    assert a == b


def test_entry_id_idempotent_publish():
    cache = MaskCache()
    e1 = make_entry(("k",), (1, 2, 3), (), 100)
    e2 = make_entry(("k",), (3, 2, 1), (), 100)
    assert e1.entry_id == e2.entry_id
    cache.publish(e1)
    assert cache.publish(e2) is e1  # racing writers converge


def test_obl_key1_violation_is_loud():
    cache = MaskCache()
    cache.publish(make_entry(("k",), (1,), (), 100))
    with pytest.raises(AssertionError, match="OBL-KEY1"):
        cache.publish(make_entry(("k",), (2,), (), 100))


def test_namespace_rollover_serves_no_stale_entries():
    cache = MaskCache()
    cache.publish(make_entry(("k",), (1,), (), 100))
    assert cache.get(("k",)) is not None
    cache.invalidate_namespace()
    assert cache.get(("k",)) is None


def test_identifier_bypass_guard_fires(sql_source, sql_tokenizer, sql_grammar):
    """G6(c): consulting a generic key at an identifier position raises in ALL builds."""
    from grid.generate import build_guide
    from grid.grammar.projection import RoleProjection
    from grid.lalr.compile import compile_tables
    from grid.policy.schema import SchemaSnapshot

    schema = SchemaSnapshot.from_dict({"users": ["id"]})
    proj = RoleProjection.full(sql_grammar).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    guide = build_guide(sql_source, sql_tokenizer, projection=proj,
                        lexicons=schema.lexicons(tables), schema_fingerprint=schema.fingerprint)
    producer = guide.producer
    # inject: a generic-typed key at an identifier position must be refused
    ident_A = frozenset(guide.tables.identifier_terminal_ids)
    bad_key = ("generic", b"", tuple(sorted(ident_A)), None)
    with pytest.raises(IdentifierMaskBypassError):
        producer._guard_key(bad_key, ident_A)
