"""grid_core verdict-kernel parity: RustVerdicts must be behaviorally identical
to the Python executable specification — check_context_dependent/_StepMemo for
the per-step CD batch, and stack.py's simulate-based allowed_terminals /
eos_ok_stack for the LALR surface — across grammars, lexicons, and random states.

The batch comparison is order-exact (same passing ids, same concatenation order),
not just set-equal: producer._check_cd_batch's output order is part of the mask
assembly contract.
"""

import random

import numpy as np
import pytest

import grid.trie.walk as W
from grid.generate import build_guide
from grid.guide import COMPLETE
from grid.lalr.stack import allowed_terminals, eos_ok_stack, shift_terminal
from grid.mask.producer import _chain, _StepMemo

pytestmark = pytest.mark.skipif(
    not W._USE_RUST or not hasattr(W._grid_core, "RustVerdicts"),
    reason="grid_core RustVerdicts not installed (or disabled via GRID_NO_RUST)",
)


def _python_batch(prod, entry, node) -> tuple[int, ...]:
    """The executable-spec batch: check_context_dependent per group, in order."""
    memo = _StepMemo()
    out: list[int] = []
    for g in entry.cd_groups:
        if prod.check_context_dependent(g.representative, node, memo):
            out.extend(g.token_ids)
    return tuple(out)


def _assert_state(guide, state, ctx: str) -> int:
    prod = guide.producer
    assert prod._kernel is not None, "kernel must be active for parity tests"
    node, rem = state.stack, state.lexer.remainder

    # LALR surface, legacy chain APIs (intern_chain path) vs the Python spec
    chain = _chain(node)
    a_words = prod._kernel.allowed_mask(chain)
    assert W._unmask(W._words_int(a_words)) == allowed_terminals(guide.tables, node), ctx
    assert bool(prod._kernel.eos_ok(chain)) == eos_ok_stack(guide.tables, node), ctx

    # LALR surface, kidx-addressed v4 APIs
    kx = prod._kidx(node)
    assert W._unmask(W._words_int(prod._kernel.allowed_mask_at(kx))) == \
        allowed_terminals(guide.tables, node), ctx
    assert bool(prod._kernel.eos_ok_at(kx)) == eos_ok_stack(guide.tables, node), ctx

    # per-step CD batch: kernel cd_pass_at vs the Python loop on the same entry
    _ci, cd_pass, _entry_id = prod.masks(node, rem)
    entry = prod.cache.get(prod.cache_key(rem, prod.allowed(node)))
    assert entry is not None, ctx
    assert tuple(cd_pass) == _python_batch(prod, entry, node), ctx

    # assembled warm hit (hit_pass): bit- and order-exact vs the reference parts
    include_eos = guide.can_terminate_state(state) and state.status != COMPLETE
    hit = prod.mask_hit(node, rem, guide.eos_token_id if include_eos else -1)
    assert hit is not None, f"{ctx}: entry warm but mask_hit missed"
    ids, eid = hit
    expected = list(entry.ci_tokens) + list(_python_batch(prod, entry, node))
    if include_eos:
        expected.append(guide.eos_token_id)
    assert ids.tolist() == expected, ctx
    assert eid == entry.entry_id, ctx

    # packed-row warm fill (fill_bits / fill_bits_hit, kernel v5): bit-set
    # identical to the id buffer, and every word of a poisoned row overwritten
    # (vLLM reuses one bitmask tensor across steps — stale bits must not leak)
    words = (guide.vocab_size + 31) // 32
    row = np.full(words, 0xFFFFFFFF, dtype=np.uint32)
    eid_fill = prod.fill_bits_hit(node, rem, guide.eos_token_id if include_eos else -1, row)
    assert eid_fill == entry.entry_id, ctx
    ref = np.zeros(words, dtype=np.uint32)
    idx = np.asarray(expected, dtype=np.int64)
    np.bitwise_or.at(ref, idx >> 5, (np.uint32(1) << (idx & 31)).astype(np.uint32))
    assert row.tolist() == ref.tolist(), ctx
    return len(entry.cd_groups)


def _walk_states(guide, seed: int, steps: int, ctx: str) -> None:
    rng = random.Random(seed)
    state = guide.initial_state
    groups_seen = 0
    for step in range(steps):
        groups_seen += _assert_state(guide, state, f"{ctx} step {step}")
        ids, _ = guide._mask_ids(state)
        tok = rng.choice(sorted(set(ids) - {guide.eos_token_id}) or sorted(ids))
        state = guide.get_next_state(state, tok)
        if state.status == COMPLETE:
            break
    assert groups_seen > 0, f"{ctx}: walk never reached a CD group (vacuous parity)"


def test_kernel_parity_toy(toy_source, toy_tokenizer):
    _walk_states(build_guide(toy_source, toy_tokenizer), seed=11, steps=14, ctx="toy")


def test_kernel_parity_sql_with_lexicons(sql_source, sql_tokenizer, sql_grammar):
    from grid.grammar.projection import RoleProjection
    from grid.lalr.compile import compile_tables
    from grid.policy.schema import SchemaSnapshot

    schema = SchemaSnapshot.from_dict({"users": ["id", "name"], "orders": ["id", "total"]})
    proj = RoleProjection.full(sql_grammar).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    guide = build_guide(sql_source, sql_tokenizer, projection=proj,
                        lexicons=schema.lexicons(tables), schema_fingerprint=schema.fingerprint)
    for seed in (3, 19):  # two trajectories: identifier-heavy + nesting-heavy corners
        _walk_states(guide, seed=seed, steps=12, ctx=f"sql seed {seed}")


def test_kernel_parity_wide_grammar(wide_source, wide_tokenizer):
    """>64 terminals: verdict batch + LALR surface on the W=2 mask path."""
    guide = build_guide(wide_source, wide_tokenizer)
    assert guide.tables.n_terminals > 64 and guide.producer._kernel is not None
    assert guide.producer._kernel.width == 2
    _walk_states(guide, seed=21, steps=12, ctx="wide")


def test_fill_bitmask_cold_and_warm_rows_identical(toy_source, toy_tokenizer):
    """GridGuide.fill_bitmask: the cold path (walk + numpy pack) and the warm
    path (kernel fill_bits, one FFI call) write the identical row — including
    full overwrite of poisoned stale words."""
    guide = build_guide(toy_source, toy_tokenizer)
    words = (guide.vocab_size + 31) // 32
    cold_row = np.full(words, 0xFFFFFFFF, dtype=np.uint32)
    guide.fill_bitmask(guide.initial_state, cold_row)  # cold: miss -> walk + pack
    warm_row = np.full(words, 0xFFFFFFFF, dtype=np.uint32)
    guide.fill_bitmask(guide.initial_state, warm_row)  # warm: kernel fill_bits
    assert cold_row.tolist() == warm_row.tolist()
    # decode set bits and compare against the id buffer
    ids, _ = guide._mask_ids(guide.initial_state)
    set_bits = {int(w * 32 + b) for w in range(words)
                for b in range(32) if warm_row[w] >> np.uint32(b) & np.uint32(1)}
    assert set_bits == {int(i) for i in ids}


def test_fill_bits_row_memo_eos_variants(toy_source, toy_tokenizer):
    """Kernel packed-row memo: repeated fill_bits at the SAME (handle, kidx)
    with DIFFERENT eos ids must each be bit-identical to the freshly computed
    reference — the memoized row excludes the eos bit, which is OR'd into the
    caller's buffer per call — and every word of a poisoned buffer must still
    be overwritten on the memo-hit path (stale bits must not leak)."""
    guide = build_guide(toy_source, toy_tokenizer)
    prod = guide.producer
    st = guide.initial_state
    node, rem = st.stack, st.lexer.remainder
    prod.masks(node, rem)  # publish + register so the warm path hits
    words = (guide.vocab_size + 31) // 32
    # alternate eos across calls: call 0 is the memo miss (row built fresh),
    # calls 1+ are memo hits that must re-derive the eos bit every time
    for i, eos in enumerate((-1, guide.eos_token_id, -1, guide.eos_token_id)):
        row = np.full(words, 0xFFFFFFFF, dtype=np.uint32)
        eid = prod.fill_bits_hit(node, rem, eos, row)
        assert eid is not None, f"call {i}: warm entry but fill_bits_hit missed"
        ids, _ = prod.mask_hit(node, rem, eos)  # id-buffer reference
        ref = np.zeros(words, dtype=np.uint32)
        idx = np.asarray(ids, dtype=np.int64)
        np.bitwise_or.at(ref, idx >> 5, (np.uint32(1) << (idx & 31)).astype(np.uint32))
        assert row.tolist() == ref.tolist(), f"call {i} eos={eos}"
    # eos bit itself must toggle between the two variants
    w, b = guide.eos_token_id >> 5, guide.eos_token_id & 31
    no_eos = np.full(words, 0xFFFFFFFF, dtype=np.uint32)
    prod.fill_bits_hit(node, rem, -1, no_eos)
    with_eos = np.full(words, 0xFFFFFFFF, dtype=np.uint32)
    prod.fill_bits_hit(node, rem, guide.eos_token_id, with_eos)
    assert (with_eos[w] >> np.uint32(b)) & np.uint32(1) == 1
    assert with_eos[w] | np.uint32(1) << np.uint32(b) == no_eos[w] | np.uint32(1) << np.uint32(b)
    others = [i for i in range(words) if i != w]
    assert with_eos[others].tolist() == no_eos[others].tolist()


def test_kernel_registration_is_per_entry(toy_source, toy_tokenizer):
    """register() fires once per entry_id; repeat steps reuse the handle."""
    guide = build_guide(toy_source, toy_tokenizer)
    prod = guide.producer
    state = guide.initial_state
    prod.masks(state.stack, state.lexer.remainder)
    n = len(prod._kernel_handles)
    prod.masks(state.stack, state.lexer.remainder)  # same configuration again
    assert len(prod._kernel_handles) == n


def test_advance_frames_mirrors_shift_terminal(sql_source, sql_tokenizer):
    """producer.shift (kernel lalr_advance + StackNode mirror) ≡ the Python
    shift_terminal: same state chain, syms, depth, and config_hash — and the
    same None verdict on non-viable terminals — along a random walk."""
    guide = build_guide(sql_source, sql_tokenizer)
    prod = guide.producer
    tables = guide.tables
    rng = random.Random(7)
    state = guide.initial_state
    compared = 0
    for step in range(40):
        node = state.stack
        viable = sorted(prod.allowed(node))
        rest = [t for t in range(tables.n_terminals) if t not in set(viable)]
        probe = viable + rng.sample(rest, min(4, len(rest))) + [tables.end_id]
        for t in probe:
            py = shift_terminal(tables, node, t)
            rs = prod.shift(node, t)
            ctx = f"step {step} t {t}"
            if py is None:
                assert rs is None, ctx
                continue
            assert rs is not None, ctx
            assert _chain(rs) == _chain(py), ctx
            assert rs.config_hash == py.config_hash, ctx
            assert rs.depth == py.depth and rs.sym == py.sym, ctx
            compared += 1
        ids, _ = guide._mask_ids(state)
        pool = sorted(set(int(i) for i in ids) - {guide.eos_token_id}) or [int(ids[0])]
        state = guide.get_next_state(state, rng.choice(pool))
        if state.status == COMPLETE:
            break
    assert compared > 40, "walk exercised too few shifts to be meaningful"


def test_namespace_rollover_drops_hit_aliases(toy_source, toy_tokenizer):
    """E10 rollover: the (kidx, remainder) lookaside must not serve across
    invalidate_namespace — the first post-rollover access recomputes (a counted
    T1 miss) and republishes the identical entry."""
    guide = build_guide(toy_source, toy_tokenizer)
    prod = guide.producer
    st = guide.initial_state
    ids1, e1 = guide._mask_ids(st)
    ids2, e2 = guide._mask_ids(st)  # warm: served by mask_hit
    assert ids2.tolist() == ids1.tolist() and e2 == e1
    m0 = prod.cache.misses
    prod.cache.invalidate_namespace()
    ids3, e3 = guide._mask_ids(st)
    assert prod.cache.misses > m0, "post-rollover access must be a T1 miss"
    assert ids3.tolist() == ids1.tolist() and e3 == e1  # content-identical rebuild
    ids4, _ = guide._mask_ids(st)  # alias re-taught
    assert ids4.tolist() == ids1.tolist()


def test_reset_interning_regenerates_kidx(toy_source, toy_tokenizer):
    """reset_interning invalidates every kidx; the kgen guard re-interns lazily
    and masks stay identical."""
    guide = build_guide(toy_source, toy_tokenizer)
    prod = guide.producer
    st = guide.initial_state
    ref, _ = guide._mask_ids(st)
    node = st.stack
    prod._kidx(node)
    gen_before = prod._kgen
    prod._reset_interning()
    assert prod._kgen == gen_before + 1
    assert node.kgen != prod._kgen
    ids, _ = guide._mask_ids(st)
    assert ids.tolist() == ref.tolist()
    assert node.kgen == prod._kgen  # re-interned on demand
