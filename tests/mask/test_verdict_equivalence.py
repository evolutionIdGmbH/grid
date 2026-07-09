"""CD-group verdict-equivalence soundness (the STAGE-1 keying change).

Groups are keyed by exactly the finite predicates the per-step CD check
consumes — per event the lexeme_ok-filtered candidate set + the min-priority
ignored pick, and for the tail the (live, prefix_ok-filtered allow, ign_ok)
triple — so two CD entries in one group must be verdict-INDISTINGUISHABLE at
EVERY parser configuration. This test binds that theorem to the executable
spec: on the SQL grammar + schema lexicons it walks several states, groups the
pure-Python walk's CD entries with the new key, and asserts (a) all members of
a group share the per-entry check_context_dependent verdict across random
reachable stacks, and (b) the grouped pass-set (one verdict per group applied
to every member id) equals the per-entry pass-set — the mask-SET invariant.
"""

import random

import pytest

import grid.trie.walk as W
from grid.generate import build_guide
from grid.guide import COMPLETE
from grid.mask.cache import make_entry
from grid.mask.producer import _StepMemo


@pytest.fixture(scope="module")
def sql_lex_guide(sql_source, sql_tokenizer, sql_grammar):
    from grid.grammar.projection import RoleProjection
    from grid.lalr.compile import compile_tables
    from grid.policy.schema import SchemaSnapshot

    schema = SchemaSnapshot.from_dict({"users": ["id", "name"], "orders": ["id", "total"]})
    proj = RoleProjection.full(sql_grammar).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    return build_guide(sql_source, sql_tokenizer, projection=proj,
                       lexicons=schema.lexicons(tables), schema_fingerprint=schema.fingerprint)


def _collect_states_and_stacks(guide):
    """Random trajectories: the visited states + a dedup'd pool of reachable stacks."""
    states, stacks, seen = [], [], set()
    for seed in (3, 11, 29):
        rng = random.Random(seed)
        st = guide.initial_state
        for _ in range(14):
            states.append(st)
            if st.stack.config_hash not in seen:
                seen.add(st.stack.config_hash)
                stacks.append(st.stack)
            ids, _ = guide._mask_ids(st)
            pool = sorted(set(int(i) for i in ids) - {guide.eos_token_id}) \
                or [int(i) for i in ids]
            st = guide.get_next_state(st, rng.choice(pool))
            if st.status == COMPLETE:
                break
    return states, stacks


def test_group_members_verdict_indistinguishable(sql_lex_guide):
    guide = sql_lex_guide
    prod = guide.producer
    tables = guide.tables
    states, stacks = _collect_states_and_stacks(guide)
    assert len(stacks) >= 5, "trajectories reached too few distinct stacks"

    rng = random.Random(97)
    cd_states = 0
    multi_member_checks = 0
    for st in states:
        A = prod.allowed(st.stack)
        result = W._walk_py(
            guide.trie, guide.dfa, st.lexer.remainder, A,
            tables.ignored_terminal_ids, prod._priority, guide.lexicons,
        )
        if not result.cd_entries:
            continue
        cd_states += 1
        memo_live = _StepMemo()
        entry = make_entry(
            ("equiv", st.lexer.remainder), result.ci_tokens, result.cd_entries,
            guide.vocab_size,
            live_of=lambda rem, _m=memo_live: _m.live_of(guide.dfa, rem),
            lexicon_sensitive=True, expand=guide.trie.expand,
            lexicons=guide.lexicons, ignored=tables.ignored_terminal_ids,
            priority=prod._priority,
        )
        # map each group back to its member CDEntries (alias sets are disjoint,
        # so an entry's first expanded id picks its group unambiguously)
        first_to_group = {t: gi for gi, g in enumerate(entry.cd_groups) for t in g.token_ids}
        members: dict[int, list] = {gi: [] for gi in range(len(entry.cd_groups))}
        for e in result.cd_entries:
            members[first_to_group[guide.trie.expand(e.token_id)[0]]].append(e)
        assert all(ms for ms in members.values()), "group without members (mapping bug)"

        # largest groups first: they carry the collapsed-singleton soundness load
        gidx = sorted(members, key=lambda gi: -len(members[gi]))[:10]
        node_sample = stacks if len(stacks) <= 8 else rng.sample(stacks, 8)
        for node in node_sample:
            memo = _StepMemo()
            # (a) every member of a group shares one verdict (rep included:
            # the representative IS the group's first member)
            for gi in gidx:
                verdicts = {prod.check_context_dependent(m, node, memo) for m in members[gi]}
                assert len(verdicts) == 1, (
                    f"verdict split inside one group: remainder={st.lexer.remainder!r} "
                    f"group={gi} members={[m.token_id for m in members[gi]]}"
                )
                if len(members[gi]) >= 2:
                    multi_member_checks += 1
            # (b) mask-SET invariance: one-verdict-per-group == per-entry verdicts
            grouped = {t for g in entry.cd_groups
                       if prod.check_context_dependent(g.representative, node, memo)
                       for t in g.token_ids}
            individual = {t for e in result.cd_entries
                          if prod.check_context_dependent(e, node, memo)
                          for t in guide.trie.expand(e.token_id)}
            assert grouped == individual, (
                f"grouped pass-set diverges from per-entry pass-set: "
                f"remainder={st.lexer.remainder!r} "
                f"only_grouped={sorted(grouped - individual)[:5]} "
                f"only_individual={sorted(individual - grouped)[:5]}"
            )
    assert cd_states >= 3, "walk never reached CD-bearing configurations (vacuous)"
    assert multi_member_checks > 0, \
        "no multi-member verdict-equivalence group exercised (vacuous collapse test)"
