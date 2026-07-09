"""Mini-G3: fast-path mask (walk + cache + CD residue + EOS gate) is bit-identical
to the ReferenceGuide trial-parse oracle along grammar-guided random walks,
including mid-lexeme states and identifier positions (DESIGN.md SS10 G3)."""

import random

from grid._reference.guide import ReferenceGuide
from grid.generate import build_guide
from grid.guide import COMPLETE
from grid.policy.schema import SchemaSnapshot


def _guide_mask(guide, state) -> set[int]:
    ids, _ = guide._mask_ids(state)
    return set(ids)


def _walk_and_compare(guide, ref, seed: int, max_steps: int, quota_counters: dict) -> None:
    rng = random.Random(seed)
    state = guide.initial_state
    prefix: list[int] = []
    for _ in range(max_steps):
        fast = _guide_mask(state=state, guide=guide)
        oracle = ref.valid_next_tokens(prefix)
        assert fast == oracle, (
            f"mask mismatch at prefix {prefix} "
            f"(fast-only={sorted(fast - oracle)[:5]}, oracle-only={sorted(oracle - fast)[:5]})"
        )
        if state.lexer.remainder:
            quota_counters["mid_lexeme"] += 1
        if guide.tables.identifier_terminal_ids and any(
            t in guide.tables.identifier_terminal_ids
            for t in __import__("grid.lalr.stack", fromlist=["allowed_terminals"]).allowed_terminals(
                guide.tables, state.stack
            )
        ):
            quota_counters["identifier_position"] += 1
        choices = sorted(fast - {guide.eos_token_id}) or sorted(fast)
        tok = rng.choice(choices)
        state = guide.get_next_state(state, tok)
        prefix.append(tok)
        if state.status == COMPLETE:
            break
    quota_counters["walks"] += 1


def test_toy_differential(toy_source, toy_tokenizer):
    guide = build_guide(toy_source, toy_tokenizer)
    ref = ReferenceGuide(guide.tables, guide.dfa, toy_tokenizer)
    counters = {"mid_lexeme": 0, "identifier_position": 0, "walks": 0}
    for seed in range(8):
        _walk_and_compare(guide, ref, seed, max_steps=10, quota_counters=counters)
    assert counters["mid_lexeme"] >= 3, "coverage quota: mid-lexeme states unexercised"


def test_sql_differential_with_lexicons(sql_source, sql_tokenizer, sql_grammar):
    from grid.grammar.projection import RoleProjection
    from grid.lalr.compile import compile_tables

    schema = SchemaSnapshot.from_dict({"users": ["id", "name"], "orders": ["id", "total"]})
    proj = RoleProjection.full(sql_grammar).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    lexicons = schema.lexicons(tables)
    guide = build_guide(
        sql_source, sql_tokenizer, projection=proj, lexicons=lexicons,
        schema_fingerprint=schema.fingerprint,
    )
    ref = ReferenceGuide(guide.tables, guide.dfa, sql_tokenizer, lexicons=lexicons)
    counters = {"mid_lexeme": 0, "identifier_position": 0, "walks": 0}
    for seed in range(6):
        _walk_and_compare(guide, ref, seed, max_steps=8, quota_counters=counters)
    assert counters["identifier_position"] >= 3, "coverage quota: identifier positions unexercised"


def test_cache_on_equals_cache_off(toy_source, toy_tokenizer):
    """G4: cached masks identical to fresh masks along the same walk."""
    g1 = build_guide(toy_source, toy_tokenizer)
    g2 = build_guide(toy_source, toy_tokenizer)
    rng = random.Random(9)
    s1, s2 = g1.initial_state, g2.initial_state
    for _ in range(12):
        m1 = _guide_mask(g1, s1)
        m1b = _guide_mask(g1, s1)  # same configuration -> guaranteed T1 hit
        assert m1 == m1b
        g2.producer.cache.invalidate_namespace()  # cache-off arm: always recompute
        m2 = _guide_mask(g2, s2)
        assert m1 == m2
        tok = rng.choice(sorted(m1 - {g1.eos_token_id}) or sorted(m1))
        s1 = g1.get_next_state(s1, tok)
        s2 = g2.get_next_state(s2, tok)
        if s1.status == COMPLETE:
            break
    assert g1.producer.cache.hits > 0, "warm arm never hit the cache"


def test_forced_span_replays_exactly(toy_source, toy_tokenizer):
    """SS4.5: every Write span token is in its own step's mask by construction."""
    guide = build_guide(toy_source, toy_tokenizer)
    ref = ReferenceGuide(guide.tables, guide.dfa, toy_tokenizer)
    state = guide.initial_state
    prefix: list[int] = []
    rng = random.Random(4)
    for _ in range(30):
        from grid.protocols import Write

        instr = guide.get_next_instruction(state)
        if isinstance(instr, Write):
            for t in (int(x) for x in instr.tokens):
                assert t in ref.valid_next_tokens(prefix) or t == guide.eos_token_id
                state = guide.get_next_state(state, t)
                prefix.append(t)
                if state.status == COMPLETE:
                    return
        else:
            ids = [int(x) for x in instr.tokens]
            tok = rng.choice(ids)
            state = guide.get_next_state(state, tok)
            prefix.append(tok)
            if state.status == COMPLETE:
                return
