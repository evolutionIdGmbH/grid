"""Kernel v7 on/off differential (red-team plan §4.2/4.3/4.5/4.7 + blob
round-trip): GRID_V7=1 (walk_payload -> register_blob -> thin MaskEntryV7,
one GIL-released kernel call each) against GRID_V7=0 (today's walk() /
make_entry / register_bytes path, byte-for-byte untouched). The two regimes
must be observationally identical: per-step entry ids, mask id buffers (order
included — ci sorted ++ cd group order ++ eos is a contract), packed fill
rows, and the T1 key -> entry_id maps. The blob is also the cross-producer
export payload: decode must reproduce the classic WalkResult glue exactly,
and a foreign kernel's register_blob must serve verdicts identical to the
spec check under ITS OWN lexicons.
"""

import random

import numpy as np
import pytest

import grid.trie.walk as W
from grid.generate import build_guide
from grid.guide import COMPLETE
from grid.mask.cache import MaskCacheT2, MaskEntryV7, _decode_blob_v1
from grid.mask.producer import _StepMemo

pytestmark = pytest.mark.skipif(
    not W._USE_RUST or not hasattr(W._grid_core, "RustVerdicts"),
    reason="kernel v7 requires grid_core (disabled via GRID_NO_RUST)",
)


def _sql_lex_parts(sql_grammar):
    from grid.grammar.projection import RoleProjection
    from grid.lalr.compile import compile_tables
    from grid.policy.schema import SchemaSnapshot

    schema = SchemaSnapshot.from_dict(
        {"users": ["id", "name"], "orders": ["id", "total"]})
    proj = RoleProjection.full(sql_grammar).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    return proj, schema.lexicons(tables), schema.fingerprint


def _pair(monkeypatch, source, tokenizer, **kw):
    """(v7-off guide, v7-on guide): independent producers/kernels/caches, the
    GRID_V7 env read once at producer construction."""
    monkeypatch.setenv("GRID_V7", "0")
    g_off = build_guide(source, tokenizer, **kw)
    monkeypatch.setenv("GRID_V7", "1")
    g_on = build_guide(source, tokenizer, **kw)
    assert not g_off.producer._v7 and g_on.producer._v7
    return g_off, g_on


def _drive_pair(g_off, g_on, seed: int, steps: int, ctx: str) -> None:
    """Identical token streams through both guides; per-step entry_id, id
    buffer (order-exact), and packed fill row equality; final T1 map equality."""
    rng = random.Random(seed)
    s_off, s_on = g_off.initial_state, g_on.initial_state
    words = (g_off.vocab_size + 31) // 32
    v7_seen = 0
    for step in range(steps):
        c = f"{ctx} seed {seed} step {step}"
        ids_off, eid_off = g_off._mask_ids(s_off)
        ids_on, eid_on = g_on._mask_ids(s_on)
        assert eid_on == eid_off, c
        assert ids_on.tolist() == ids_off.tolist(), c
        row_off = np.full(words, 0xFFFFFFFF, dtype=np.uint32)
        row_on = np.full(words, 0xFFFFFFFF, dtype=np.uint32)
        g_off.fill_bitmask(s_off, row_off)
        g_on.fill_bitmask(s_on, row_on)
        assert row_on.tolist() == row_off.tolist(), c
        tok = rng.choice(
            sorted(set(int(i) for i in ids_off) - {g_off.eos_token_id})
            or [int(ids_off[0])])
        s_off = g_off.get_next_state(s_off, tok)
        s_on = g_on.get_next_state(s_on, tok)
        if s_off.status == COMPLETE:
            break
    t1_off = {k: e.entry_id for k, e in g_off.producer.cache._t1.items()}
    t1_on = {k: e.entry_id for k, e in g_on.producer.cache._t1.items()}
    assert t1_on == t1_off, f"{ctx}: T1 key/entry-id maps diverge"
    v7_seen = sum(isinstance(e, MaskEntryV7)
                  for e in g_on.producer.cache._t1.values())
    assert v7_seen == len(t1_on) > 0, f"{ctx}: v7 path not exercised (vacuous)"


def test_differential_toy(toy_source, toy_tokenizer, monkeypatch):
    g_off, g_on = _pair(monkeypatch, toy_source, toy_tokenizer)
    for seed in (11, 23):
        _drive_pair(g_off, g_on, seed=seed, steps=16, ctx="toy")


def test_differential_sql_with_lexicons(sql_source, sql_tokenizer, sql_grammar, monkeypatch):
    proj, lex, fp = _sql_lex_parts(sql_grammar)
    g_off, g_on = _pair(monkeypatch, sql_source, sql_tokenizer,
                        projection=proj, lexicons=lex, schema_fingerprint=fp)
    for seed in (3, 19):
        _drive_pair(g_off, g_on, seed=seed, steps=12, ctx="sql-lex")


def test_differential_wide_w2(wide_source, wide_tokenizer, monkeypatch):
    g_off, g_on = _pair(monkeypatch, wide_source, wide_tokenizer)
    assert g_on.producer._kernel.width == 2
    _drive_pair(g_off, g_on, seed=21, steps=12, ctx="wide")


def test_fuzz_50_seeds(toy_source, toy_tokenizer, monkeypatch):
    """§4.3: 50-seed generation fuzz, v7 on vs off — mask + entry_id equality
    at every step (fresh trajectories over the SAME two producers, so warm
    hits, alias memos, and interning all participate like a real serving mix)."""
    g_off, g_on = _pair(monkeypatch, toy_source, toy_tokenizer)
    for seed in range(50):
        _drive_pair(g_off, g_on, seed=1000 + seed, steps=10, ctx="fuzz")


# ------------------------------------------------------------ blob round-trip


def test_blob_decode_equals_walk_glue(sql_source, sql_tokenizer, sql_grammar, monkeypatch):
    """MaskEntryV7's lazy cd_groups/cd_entries/ci_tokens must be byte-identical
    to the classic WalkResult glue for the same configuration (CDEntry and
    CDGroup are frozen dataclasses: == is component equality, EmissionEvent
    candidates/length, segment bytes, remainder, token ids, group ORDER)."""
    proj, lex, fp = _sql_lex_parts(sql_grammar)
    g_off, g_on = _pair(monkeypatch, sql_source, sql_tokenizer,
                        projection=proj, lexicons=lex, schema_fingerprint=fp)
    rng = random.Random(19)
    s_off, s_on = g_off.initial_state, g_on.initial_state
    compared = 0
    for _ in range(12):
        rem = s_off.lexer.remainder
        A = g_off.producer.allowed(s_off.stack)
        e_off = g_off.producer._entry_for(rem, A)
        e_on = g_on.producer._entry_for(rem, g_on.producer.allowed(s_on.stack))
        assert np.array_equal(np.asarray(e_on.ci_tokens), np.asarray(e_off.ci_tokens))
        assert e_on.cd_entries == e_off.cd_entries
        assert e_on.cd_groups == e_off.cd_groups  # order-exact, reps included
        compared += len(e_off.cd_groups)
        ids, _ = g_off._mask_ids(s_off)
        tok = rng.choice(sorted(set(int(i) for i in ids) - {g_off.eos_token_id})
                         or [int(ids[0])])
        s_off = g_off.get_next_state(s_off, tok)
        s_on = g_on.get_next_state(s_on, tok)
        if s_off.status == COMPLETE:
            break
    assert compared > 0, "walk never reached a CD group (vacuous round-trip)"


def test_blob_version_and_shape_hard_errors(toy_source, toy_tokenizer, monkeypatch):
    monkeypatch.setenv("GRID_V7", "1")
    g = build_guide(toy_source, toy_tokenizer)
    prod = g.producer
    st = g.initial_state
    entry = prod._entry_for(st.lexer.remainder, prod.allowed(st.stack))
    assert isinstance(entry, MaskEntryV7)
    with pytest.raises(ValueError, match="version"):
        _decode_blob_v1(b"\x02" + entry.blob[1:])
    with pytest.raises(ValueError, match="version"):
        prod._kernel.register_blob(b"\x02" + entry.blob[1:], entry.ci_bytes,
                                   repr(entry.key).encode(), prod.vocab_size)
    with pytest.raises(ValueError):
        prod._kernel.register_blob(entry.blob + b"\x00", entry.ci_bytes,
                                   repr(entry.key).encode(), prod.vocab_size)


def test_register_blob_equivalent_to_register_bytes(toy_source, toy_tokenizer, monkeypatch):
    """The SAME kernel serves identical cd_pass_at / hit_pass / fill_bits from
    a register_blob handle and a register_bytes handle fed the decoded blob
    (the wire formats are two encodings of one payload)."""
    monkeypatch.setenv("GRID_V7", "1")
    g = build_guide(toy_source, toy_tokenizer)
    prod = g.producer
    kernel = prod._kernel
    rng = random.Random(5)
    st = g.initial_state
    checked = 0
    for _ in range(10):
        rem = st.lexer.remainder
        entry = prod._entry_for(rem, prod.allowed(st.stack))
        h_blob = prod._ensure_handle(entry)
        h_bytes = kernel.register_bytes(_decode_blob_v1(entry.blob), entry.ci_bytes)
        kx = prod._kidx(st.stack)
        assert kernel.cd_pass_at(h_blob, kx) == kernel.cd_pass_at(h_bytes, kx)
        assert kernel.hit_pass(h_blob, kx, g.eos_token_id) == \
            kernel.hit_pass(h_bytes, kx, g.eos_token_id)
        words = (g.vocab_size + 31) // 32
        r1 = np.full(words, 0xFFFFFFFF, dtype=np.uint32)
        r2 = np.full(words, 0xFFFFFFFF, dtype=np.uint32)
        kernel.fill_bits(h_blob, kx, -1, r1)
        kernel.fill_bits(h_bytes, kx, -1, r2)
        assert r1.tolist() == r2.tolist()
        checked += 1
        ids, _ = g._mask_ids(st)
        tok = rng.choice(sorted(set(int(i) for i in ids) - {g.eos_token_id})
                         or [int(ids[0])])
        st = g.get_next_state(st, tok)
        if st.status == COMPLETE:
            break
    assert checked >= 3


# ------------------------------------------------- rollover / reset under v7


def test_namespace_rollover_with_v7_entries(toy_source, toy_tokenizer, monkeypatch):
    """E10 rollover with v7 entries: post-rollover access is a counted T1 miss
    that REBUILDS (via register_blob) to the identical entry_id and mask; the
    kernel by_id dedup returns the ORIGINAL handle for the recomputed entry."""
    monkeypatch.setenv("GRID_V7", "1")
    g = build_guide(toy_source, toy_tokenizer)
    prod = g.producer
    st = g.initial_state
    ids1, e1 = g._mask_ids(st)
    h1 = prod._kernel_handles[e1]
    m0 = prod.cache.misses
    prod.cache.invalidate_namespace()
    prod._kernel_handles.clear()  # force re-registration through register_blob
    ids3, e3 = g._mask_ids(st)
    assert prod.cache.misses > m0, "post-rollover access must be a T1 miss"
    assert ids3.tolist() == ids1.tolist() and e3 == e1
    assert prod._kernel_handles[e1] == h1, \
        "kernel by_id dedup must return the original handle after rollover"


def test_reset_interning_with_v7_entries(toy_source, toy_tokenizer, monkeypatch):
    monkeypatch.setenv("GRID_V7", "1")
    g = build_guide(toy_source, toy_tokenizer)
    prod = g.producer
    st = g.initial_state
    ref, _ = g._mask_ids(st)
    prod._kidx(st.stack)
    gen = prod._kgen
    prod._reset_interning()
    assert prod._kgen == gen + 1
    ids, _ = g._mask_ids(st)
    assert ids.tolist() == ref.tolist()


# ------------------------------------------- cross-producer T2 handover (§4.5)


def test_cross_producer_t2_handover(sql_source, sql_tokenizer, sql_grammar, monkeypatch):
    """Producers X and Y (same schema fingerprint) share one T2. Y adopts X's
    v7 entries — register_blob on Y's kernel, VEvents/tails recomputed under
    Y's OWN lexicons — and must serve masks identical to a Y-only cold build;
    adopting twice yields ONE handle (kernel by_id); the adopted entry's
    kernel verdicts equal Y's spec check_context_dependent per group."""
    monkeypatch.setenv("GRID_V7", "1")
    from grid.grammar import spec as gspec
    from grid.grammar.projection import RoleProjection
    from grid.guide import GridGuide
    from grid.lalr.compile import compile_tables
    from grid.lexer.dfa import build_scanner
    from grid.policy.schema import SchemaSnapshot
    from grid.trie.build import build_trie

    grammar = gspec.load(sql_source)
    proj = RoleProjection.full(grammar).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    dfa = build_scanner(grammar.terminals, grammar.terminal_order)
    trie = build_trie(sql_tokenizer)
    schema = SchemaSnapshot.from_dict({"users": ["id", "name"], "orders": ["id", "total"]})
    lex = schema.lexicons(tables)
    t2 = MaskCacheT2()

    def guide():
        return GridGuide(tables=tables, dfa=dfa, trie=trie, adapter=sql_tokenizer,
                         lexicons=lex, schema_fingerprint=schema.fingerprint,
                         mask_t2=t2)

    gx, gy, gz = guide(), guide(), guide()  # z: the no-T2 cold-build oracle
    gz.producer.t2 = None
    assert gx.producer is not gy.producer and gx.producer.t2 is gy.producer.t2

    # drive X cold; collect its touched keys and one CD-bearing config
    rng = random.Random(19)
    sx = gx.initial_state
    for _ in range(10):
        ids, _ = gx._mask_ids(sx)
        tok = rng.choice(sorted(set(int(i) for i in ids) - {gx.eos_token_id})
                         or [int(ids[0])])
        sx = gx.get_next_state(sx, tok)
        if sx.status == COMPLETE:
            break
    keys = list(gx.producer.cache._t1.keys())
    assert any(isinstance(e, MaskEntryV7) for e in gx.producer.cache._t1.values())

    # Y adopts via warm_from_t2 (T2 -> T1 + register_blob import, no walks);
    # walk-freedom is proven by Y's OWN miss counter (a walk requires a
    # counted T1 miss through cache.get — warm_from_t2 and warm hits never
    # touch it), which stays attributable to Y unlike a module-level walk spy
    # (the Z oracle below builds cold through the same module functions)
    n = gy.producer.warm_from_t2(keys)
    assert n == len(keys), "every X key must adopt (same schema fingerprint)"
    assert gy.producer.cache.misses == 0, "adoption must be walk-free"

    # Y's masks over the same trajectory == Z's cold-built masks
    rng = random.Random(19)
    sy, sz = gy.initial_state, gz.initial_state
    for step in range(10):
        ids_y, eid_y = gy._mask_ids(sy)
        ids_z, eid_z = gz._mask_ids(sz)
        assert eid_y == eid_z and ids_y.tolist() == ids_z.tolist(), f"step {step}"
        tok = rng.choice(sorted(set(int(i) for i in ids_y) - {gy.eos_token_id})
                         or [int(ids_y[0])])
        sy = gy.get_next_state(sy, tok)
        sz = gz.get_next_state(sz, tok)
        if sy.status == COMPLETE:
            break
    assert gy.producer.cache.misses == 0, "adopted trajectory must stay walk-free on Y"

    # adopt-twice -> ONE kernel handle (by_id dedup, not a Python memo effect)
    entry = next(e for e in gy.producer.cache._t1.values() if e.cd_groups)
    h1 = gy.producer._ensure_handle(entry)
    gy.producer._kernel_handles.pop(entry.entry_id)
    h2 = gy.producer._ensure_handle(entry)
    assert h1 == h2, "kernel by_id must deduplicate a re-imported entry"

    # adopted-entry kernel verdicts == Y's Python check_context_dependent
    # (lexicon-recompute correctness: Y's kernel filtered under Y's lexicons)
    node = gy.initial_state.stack
    kx = gy.producer._kidx(node)
    got = np.frombuffer(gy.producer._kernel.cd_pass_at(h1, kx), dtype=np.int32)
    memo = _StepMemo()
    expected = [t for grp in entry.cd_groups
                if gy.producer.check_context_dependent(grp.representative, node, memo)
                for t in grp.token_ids]
    assert got.tolist() == expected


# --------------------------------------------------------- no-kernel guard


def test_v7_flag_without_kernel_takes_classic_path(toy_source, toy_tokenizer, monkeypatch):
    """GRID_V7=1 with no kernel (the GRID_NO_RUST/oversized-grammar shape)
    must fall through to the classic build — the spec oracle is untouched."""
    monkeypatch.setenv("GRID_V7", "1")
    g = build_guide(toy_source, toy_tokenizer)
    prod = g.producer
    prod._kernel = None  # simulate the no-kernel producer shape
    st = g.initial_state
    entry = prod._entry_for(st.lexer.remainder, prod.allowed(st.stack))
    assert not isinstance(entry, MaskEntryV7)
    assert entry.cd_groups is not None
