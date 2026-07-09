"""E4a: reserve completions are minimal, token-denominated, lexicon-aware, and parse."""

import random

from grid._reference.guide import ReferenceGuide
from grid.generate import build_guide
from grid.guide import COMPLETE
from grid.policy.schema import SchemaSnapshot


def test_completion_parses_from_random_states(toy_source, toy_tokenizer):
    guide = build_guide(toy_source, toy_tokenizer)
    ref = ReferenceGuide(guide.tables, guide.dfa, toy_tokenizer)
    rng = random.Random(11)
    for trial in range(10):
        state = guide.initial_state
        prefix: list[int] = []
        for _ in range(rng.randrange(0, 8)):
            ids, _ = guide._mask_ids(state)
            ids = [t for t in ids if t != guide.eos_token_id] or ids
            tok = rng.choice(sorted(ids))
            state = guide.get_next_state(state, tok)
            prefix.append(tok)
            if state.status == COMPLETE:
                break
        if state.status == COMPLETE:
            continue
        completion = guide._completion_tokens(state)
        assert completion is not None, f"trial {trial}: no completion from viable state"
        full = prefix + [t for t in completion if t != guide.eos_token_id]
        data = b"".join(toy_tokenizer.token_bytes(t) for t in full)
        assert ref.eos_legal(data), f"trial {trial}: completion does not parse: {data!r}"


def test_completion_respects_identifier_lexicon(sql_source, sql_tokenizer, sql_grammar):
    """Regression: completions must render identifiers from the ALLOWED set,
    never the BFS-shortest lexeme ('_')."""
    from grid.grammar.projection import RoleProjection
    from grid.lalr.compile import compile_tables

    schema = SchemaSnapshot.from_dict({"users": ["id"]})
    proj = RoleProjection.full(sql_grammar).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    lexicons = schema.lexicons(tables)
    guide = build_guide(sql_source, sql_tokenizer, projection=proj, lexicons=lexicons,
                        schema_fingerprint=schema.fingerprint)
    state = guide.initial_state
    for tid in sql_tokenizer.greedy_tokenize(b"select * from "):
        state = guide.get_next_state(state, tid)
    completion = guide._completion_tokens(state)
    assert completion is not None
    data = b"".join(sql_tokenizer.token_bytes(t) for t in completion if t != guide.eos_token_id)
    assert b"users" in data and b"_" not in data.replace(b"user", b"")
    ref = ReferenceGuide(guide.tables, guide.dfa, sql_tokenizer, lexicons=lexicons)
    assert ref.eos_legal(b"select * from " + data)


def test_reserve_costs_are_token_denominated(toy_source, toy_tokenizer):
    guide = build_guide(toy_source, toy_tokenizer)
    reserve = guide.reserve
    # every cost equals the greedy token count of ' ' + shortest lexeme
    for tid, lexeme in reserve.term_lexeme.items():
        assert reserve.term_cost[tid] == len(toy_tokenizer.greedy_tokenize(b" " + lexeme))
