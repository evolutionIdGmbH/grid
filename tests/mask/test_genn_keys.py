"""genN cache-key normalization (W2): non-ident configurations whose walks are
provably indistinguishable share ONE T1/T2 entry via the normalized key
("genN", p, q, v, sorted(A), schema_fp) — guarded by lexicon visibility, with
the legacy raw key as fallback, the E11 ident path byte-for-byte untouched,
and the GRID_GENN_KEYS=0 kill switch restoring legacy keys exactly. The
schema_fp component (None without lexicons) scopes sharing to one schema:
walked bytes cross lexeme boundaries, where CD partitions are lexeme_ok/
prefix_ok-filtered under the walker's lexicons — cross-schema reuse of such
entries serves one schema's CD partition to another (wrong masks; see
test_genn_keys_are_schema_scoped below for the live counterexample).

Runs identically on the kernel and the GRID_NO_RUST=1 spec path (cache_key is
shared by both: parity by construction, proven by the parity suite run)."""

import pathlib

import numpy as np
import pytest

import grid.mask.producer as P
from grid.errors import IdentifierMaskBypassError
from grid.generate import build_guide
from grid.mask.cache import adaptive_encode, make_entry
from grid.models.tokenizer_adapter import MockTokenizer
from grid.models.vllm_processor import _GuideRegistry

ROOT = pathlib.Path(__file__).parent.parent.parent

# the verified '1e'/'1E' counterexample grammar: [eE]-exponent numbers make the
# post-accept suffix bytes v the ONLY separator of the two remainders
EXPONENT_GRAMMAR = """%start s
%ignore WS
WS: /[ \\t\\n]+/
NUMBER: /[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?/
NAME: /[a-z_][a-z0-9_]*/
s: item | s item
item: NUMBER | NAME | "+"
"""

EXP_TOKENS = (
    " ", "  ", "+", "e", "E", "5", "12", "e5", "E5", " x", "x", "abc",
    "1.5", ".5", "e+2", "+ ", " 1", "5 ", "9", "e9 x",
)


@pytest.fixture(scope="module")
def exp_guide():
    g = build_guide(EXPONENT_GRAMMAR, MockTokenizer(extra_tokens=EXP_TOKENS))
    g.producer.set_genn_keys(True)  # the lever under test, independent of env
    return g


@pytest.fixture(scope="module")
def spider_source() -> str:
    return (ROOT / "grammars" / "sql_spider.grid").read_text()


def _drive(guide, tok, text: bytes):
    st = guide.initial_state
    for t in tok.greedy_tokenize(text):
        st = guide.get_next_state(st, int(t))
    return st


def _count_walks(monkeypatch):
    # counts BOTH cold-build entrypoints: the classic walk() and the kernel-v7
    # _v7_build (walk_payload + register_blob) — under GRID_V7=1 the miss path
    # never calls walk(), so counting only it would blind every "must not
    # re-walk"/"must walk" assertion below (no assertion is weakened: one
    # cold build is one count on either path)
    calls = []
    orig = P.walk
    monkeypatch.setattr(P, "walk", lambda *a, **k: calls.append(1) or orig(*a, **k))
    orig_v7 = P.MaskProducer._v7_build
    monkeypatch.setattr(P.MaskProducer, "_v7_build",
                        lambda self, *a, **k: calls.append(1) or orig_v7(self, *a, **k))
    return calls


# ---------------------------------------------------------------- genN sharing


def test_genn_sharing_one_walk_bit_identical(sql_source, sql_tokenizer, monkeypatch):
    """Two byte-different open-string interiors with equal (p, q, v, A) share
    one entry: ONE walk, same entry object, bit-identical masks."""
    guide = build_guide(sql_source, sql_tokenizer)  # no lexicons: LEX empty
    prod = guide.producer
    prod.set_genn_keys(True)  # the lever under test, independent of env
    s1 = _drive(guide, sql_tokenizer, b"select * from users where name = 'ab")
    s2 = _drive(guide, sql_tokenizer, b"select * from users where email = 'zz")
    r1, r2 = s1.lexer.remainder, s2.lexer.remainder
    assert (r1, r2) == (b"'ab", b"'zz") and r1 != r2
    A = prod.allowed(s1.stack)
    assert prod.allowed(s2.stack) == A
    k1, k2 = prod.cache_key(r1, A), prod.cache_key(r2, A)
    assert k1[0] == "genN" and k1 == k2, f"{k1} vs {k2}"
    assert k1[1] == -1 and k1[3] == b""  # open literal: no accepting prefix

    calls = _count_walks(monkeypatch)
    e1 = prod._entry_for(r1, A)
    e2 = prod._entry_for(r2, A)
    assert len(calls) == 1, "second remainder must be a T1 hit, not a walk"
    assert e1 is e2 and e1.entry_id == e2.entry_id
    ids1, eid1 = guide._mask_ids(s1)
    ids2, eid2 = guide._mask_ids(s2)
    assert len(calls) == 1
    assert eid1 == eid2 and np.array_equal(ids1, ids2), "masks must be bit-identical"

    # entry_id determinism + publish idempotence under the shared key
    again = make_entry(k1, list(e1.ci_tokens), e1.cd_entries, prod.vocab_size)
    assert again.entry_id == e1.entry_id
    assert prod.cache.publish(again) is e1  # racing writers converge (OBL-KEY1)


def test_genn_alias_memo_and_warm_hit_inherit_sharing(sql_source, sql_tokenizer, monkeypatch):
    """peek_warm/_warm_handle flow through cache_key: a byte-fresh remainder in
    a warmed genN class is warm (no walk) — the (kidx, remainder) alias memo
    keeps raw-bytes granularity but resolves to the shared entry."""
    guide = build_guide(sql_source, sql_tokenizer)
    prod = guide.producer
    prod.set_genn_keys(True)
    s1 = _drive(guide, sql_tokenizer, b"select * from users where name = 'ab")
    s2 = _drive(guide, sql_tokenizer, b"select * from users where email = 'zz")
    ids1, eid1 = guide._mask_ids(s1)  # cold: walk + publish under the genN key
    calls = _count_walks(monkeypatch)
    assert prod.peek_warm(s2.stack, s2.lexer.remainder), "genN class must be warm"
    if prod._kernel is not None:
        got = prod.mask_hit(s2.stack, s2.lexer.remainder, -1)
        assert got is not None, "warm-hit path must serve the shared entry"
        _ids, eid2 = got
        assert eid2 == eid1
    assert not calls, "no path may re-walk a genN-warm configuration"


# ---------------------------------------------------------------- guard + E11


def test_guard_falls_back_to_legacy_key(spider_source, sql_tokenizer):
    """Lexicon-visible states (keyword prefixes / ident partials) must fall
    back to the legacy raw generic key; lexicon-clean states normalize."""
    reg = _GuideRegistry(sql_tokenizer)
    guide = reg.guide_for({"grammar": spider_source, "schema": {"employees": ["id", "name"]}})
    prod = guide.producer
    prod.set_genn_keys(True)
    tn = {n: i for i, n in enumerate(guide.tables.terminal_names)}
    assert prod._LEX == frozenset({tn["TABLE_NAME"], tn["COLUMN_NAME"]})
    A = frozenset({tn["STRING"]})  # non-ident A
    for rem in (b"se", b"lim", b"uni", b"z_c", b"sal", b""):
        # live[q] contains COLUMN_NAME/TABLE_NAME (NAME-superset states) or, for
        # b"", every terminal: guard trips, key is the raw fallback — schema-
        # scoped in the v2 regime (walk-time CD filtering embeds schema words)
        assert prod.cache_key(rem, A) == \
            ("generic", rem, tuple(sorted(A)), prod.schema_fingerprint), rem
        assert prod.schema_fingerprint is not None
    for rem in (b"'ab", b"'abc'", b"42", b"42.", b"' "):
        k = prod.cache_key(rem, A)
        assert k[0] == "genN", (rem, k)


def test_ident_path_untouched_and_tripwire_fires(spider_source, sql_tokenizer):
    """E11: ident-position keys are byte-for-byte the legacy schema_fp-keyed
    form, and a genN-shaped key consulted at an ident position raises."""
    reg = _GuideRegistry(sql_tokenizer)
    guide = reg.guide_for({"grammar": spider_source, "schema": {"employees": ["id", "name"]}})
    prod = guide.producer
    ident_A = frozenset(guide.tables.identifier_terminal_ids)
    key = prod.cache_key(b"na", ident_A)
    assert key == ("ident", b"na", tuple(sorted(ident_A)), prod.schema_fingerprint)
    assert key[3] is not None
    with pytest.raises(IdentifierMaskBypassError):
        prod._guard_key(("genN", -1, 36, b"", tuple(sorted(ident_A))), ident_A)
    with pytest.raises(IdentifierMaskBypassError):
        prod._guard_key(("generic", b"na", tuple(sorted(ident_A)), None), ident_A)


# ------------------------------------------------------- '1e'/'1E' regression


def test_v_component_is_load_bearing_1e_vs_1E(exp_guide):
    """The REFUTED v-less key's counterexample: b'1e' and b'1E' share (q,l,p)
    but produce DIFFERENT masks — the genN key must embed v = r[l:] verbatim.
    A v-less merge would have served one entry for both (OBL-KEY1 violation)."""
    prod = exp_guide.producer
    root = exp_guide.initial_state.stack
    A = prod.allowed(root)
    s1 = prod.dfa.scan_with_last_accept(b"1e")
    s2 = prod.dfa.scan_with_last_accept(b"1E")
    assert s1 == s2, "counterexample seed: identical (q, l, p)"
    k1, k2 = prod.cache_key(b"1e", A), prod.cache_key(b"1E", A)
    assert k1[0] == k2[0] == "genN"
    assert (k1[1], k1[2], k1[4]) == (k2[1], k2[2], k2[4]), "p, q, A all equal"
    assert k1[3] == b"e" and k2[3] == b"E" and k1 != k2, "ONLY v separates them"

    def outputs(rem):
        e = prod._entry_for(rem, A)
        return set(int(t) for t in np.asarray(e.ci_tokens)) | \
            set(int(t) for t in prod._check_cd_batch(e, root))

    assert outputs(b"1e") != outputs(b"1E"), \
        "masks differ: a v-less key would have merged them unsoundly"


def test_positive_pairs_merge_on_exponent_grammar(exp_guide, monkeypatch):
    """SURVIVED positive classes: same (p, q, v), different prefix content and
    length => one key, one walk, identical masks."""
    prod = exp_guide.producer
    root = exp_guide.initial_state.stack
    A = prod.allowed(root)
    for ra, rb in [(b"1e", b"23e"), (b"1e5", b"9873e2"), (b"1E5", b"9E2"),
                   (b"12", b"9"), (b"1.5", b"22.75")]:
        ka, kb = prod.cache_key(ra, A), prod.cache_key(rb, A)
        assert ka == kb and ka[0] == "genN", (ra, rb, ka, kb)
        calls = _count_walks(monkeypatch)
        ea = prod._entry_for(ra, A)
        eb = prod._entry_for(rb, A)
        assert eb is ea and len(calls) <= 1, (ra, rb)
        assert np.array_equal(np.asarray(ea.ci_tokens), np.asarray(eb.ci_tokens))


# ---------------------------------------------------------------- kill switch


def test_kill_switch_restores_legacy_keys(exp_guide, monkeypatch):
    """GRID_GENN_KEYS=0 (env, read at construction) and set_genn_keys(False)
    (runtime, replay/tests) both restore the legacy raw keys byte-for-byte."""
    prod = exp_guide.producer
    root = exp_guide.initial_state.stack
    A = prod.allowed(root)
    legacy = ("generic", b"1e", tuple(sorted(A)), None)
    assert prod.cache_key(b"1e", A)[0] == "genN"
    prod.set_genn_keys(False)
    try:
        assert prod.cache_key(b"1e", A) == legacy
        assert prod.cache_key(b"", A) == ("generic", b"", tuple(sorted(A)), None)
    finally:
        prod.set_genn_keys(True)
    assert prod.cache_key(b"1e", A)[0] == "genN"

    monkeypatch.setenv("GRID_GENN_KEYS", "0")
    fresh = build_guide(EXPONENT_GRAMMAR, MockTokenizer(extra_tokens=EXP_TOKENS)).producer
    assert fresh._genn_keys is False
    assert fresh.cache_key(b"1e", A) == legacy


# --------------------------------------------------- T2 cross-schema scoping


def test_genn_keys_are_schema_scoped(spider_source, sql_tokenizer, monkeypatch):
    """genN keys embed schema_fp: producers of DIFFERENT schemas never share a
    genN entry through the per-dialect T2 (the fresh schema WALKS), because
    walked bytes cross lexeme boundaries where CD partitions are filtered
    under the walker's lexicons. Counterexample making fp load-bearing: the
    same non-ident configuration (b"42", {NUMBER}) fresh-walks to DIFFERENT
    CD partitions under the two schemas — an fp-less key would have served
    one schema's partition to the other (the 50-seed fuzz failure)."""
    reg = _GuideRegistry(sql_tokenizer)
    ga = reg.guide_for({"grammar": spider_source, "schema": {"employees": ["id", "name"]}})
    ga.producer.set_genn_keys(True)
    sa = _drive(ga, sql_tokenizer, b"select * from employees where name like 'ab")
    assert sa.lexer.remainder == b"'ab"
    Aa = ga.producer.allowed(sa.stack)
    assert not (Aa & ga.tables.identifier_terminal_ids), "LIKE position is generic"
    ids_a, _ = ga._mask_ids(sa)  # cold walk, published to the dialect's T2

    gb = reg.guide_for({"grammar": spider_source, "schema": {"orders": ["total", "qty"]}})
    gb.producer.set_genn_keys(True)
    assert gb.producer is not ga.producer and gb.producer.t2 is ga.producer.t2
    sb = _drive(gb, sql_tokenizer, b"select * from orders where total like 'zz")
    assert sb.lexer.remainder == b"'zz" != sa.lexer.remainder
    ka = ga.producer.cache_key(sa.lexer.remainder, Aa)
    kb = gb.producer.cache_key(sb.lexer.remainder, gb.producer.allowed(sb.stack))
    assert ka[0] == kb[0] == "genN" and ka[:5] == kb[:5], "normalized alike"
    assert ka[5] != kb[5] and ka != kb, "schema_fp separates the schemas"
    calls = _count_walks(monkeypatch)
    gb._mask_ids(sb)
    assert len(calls) == 1, "fresh schema must WALK: no cross-schema genN hit"

    # the counterexample: same config, different schemas => different walks
    tn = {n: i for i, n in enumerate(ga.tables.terminal_names)}
    A_num = frozenset({tn["NUMBER"]})
    assert ga.producer.cache_key(b"42", A_num)[0] == "genN"
    ea = ga.producer._entry_for(b"42", A_num)
    eb = gb.producer._entry_for(b"42", A_num)

    def part(e):
        return sorted(tuple(sorted(int(t) for t in g.token_ids)) for g in e.cd_groups)

    assert np.array_equal(np.asarray(ea.ci_tokens), np.asarray(eb.ci_tokens)), \
        "CI is schema-independent here"
    assert part(ea) != part(eb), \
        "CD partitions differ across schemas: sharing them would be unsound"


def test_raw_fallback_is_schema_scoped_v2(spider_source, sql_tokenizer, monkeypatch):
    """The guard-tripping raw FALLBACK key is schema-scoped in the v2 regime:
    byte-identical remainders across schemas must not share through the
    per-dialect T2. Live counterexample (caught by the 50-seed shared-registry
    fuzz): a boundary-keyword config (b"select", non-ident A) fresh-walks to
    DIFFERENT CD partitions under two schemas — the walk's successor
    exploration lexeme_ok/prefix_ok-filters the pending identifier lexeme
    under the builder's words, so an unscoped key serves one schema's
    continuations to the other (the consumer's own column tokens go missing).
    GRID_GENN_KEYS=0 restores the unscoped legacy tuple byte-for-byte."""
    from tests.conftest import SQL_TOKENS

    # tokens crossing the SELECT boundary INTO the column-name lexeme make the
    # schema words reachable from the b"select" walk (the poison carrier)
    tok = MockTokenizer(extra_tokens=SQL_TOKENS + (" name", " total"))
    reg = _GuideRegistry(tok)
    ga = reg.guide_for({"grammar": spider_source, "schema": {"employees": ["id", "name"]}})
    gb = reg.guide_for({"grammar": spider_source, "schema": {"orders": ["total", "qty"]}})
    ga.producer.set_genn_keys(True)
    gb.producer.set_genn_keys(True)
    assert gb.producer.t2 is ga.producer.t2

    sa = _drive(ga, tok, b"select")
    assert sa.lexer.remainder == b"select"
    Aa = ga.producer.allowed(sa.stack)
    assert not (Aa & ga.tables.identifier_terminal_ids)
    ka = ga.producer.cache_key(b"select", Aa)
    kb = gb.producer.cache_key(b"select", Aa)
    assert ka[0] == kb[0] == "generic", "keyword boundary trips the guard"
    assert ka[:3] == kb[:3] and ka[3] != kb[3] and ka != kb, \
        "schema_fp separates the raw fallback keys"

    ea = ga.producer._entry_for(b"select", Aa)  # cold walk, published to T2
    calls = _count_walks(monkeypatch)
    eb = gb.producer._entry_for(b"select", Aa)
    assert len(calls) == 1, "fresh schema must WALK: no cross-schema raw-key hit"

    def outputs(prod, e, stack):
        return set(int(t) for t in np.asarray(e.ci_tokens)) | \
            set(int(t) for t in prod._check_cd_batch(e, stack))

    out_a = outputs(ga.producer, ea, sa.stack)
    out_b = outputs(gb.producer, eb, sa.stack)
    t_name, t_total = tok.vocabulary[" name"], tok.vocabulary[" total"]
    assert t_name in out_a and t_name not in out_b
    assert t_total in out_b and t_total not in out_a
    assert out_a != out_b, \
        "masks differ across schemas: unscoped sharing would serve wrong words"

    ga.producer.set_genn_keys(False)
    try:
        assert ga.producer.cache_key(b"select", Aa) == \
            ("generic", b"select", tuple(sorted(Aa)), None), \
            "kill switch restores the pre-stage unscoped fallback byte-for-byte"
    finally:
        ga.producer.set_genn_keys(True)


def test_t2_cross_schema_misses_without_genn(spider_source, sql_tokenizer, monkeypatch):
    """Negative control: with GRID_GENN_KEYS=0 the same scenario WALKS (the
    raw key embeds the literal bytes) — the test above measures genN, not a
    pre-existing behavior."""
    monkeypatch.setenv("GRID_GENN_KEYS", "0")
    reg = _GuideRegistry(sql_tokenizer)
    ga = reg.guide_for({"grammar": spider_source, "schema": {"employees": ["id", "name"]}})
    sa = _drive(ga, sql_tokenizer, b"select * from employees where name like 'ab")
    ga._mask_ids(sa)
    gb = reg.guide_for({"grammar": spider_source, "schema": {"orders": ["total", "qty"]}})
    sb = _drive(gb, sql_tokenizer, b"select * from orders where total like 'zz")
    calls = _count_walks(monkeypatch)
    ids_b, _ = gb._mask_ids(sb)
    assert len(calls) == 1, "legacy keys: byte-different literal content re-walks"


# ------------------------------------------------ pure re-keying (content eq)


def test_genn_rekeying_preserves_entry_bytes(sql_source, sql_tokenizer):
    """The genN merge is a pure re-keying: for one configuration, the entry
    built under the legacy key and under the genN key carry byte-identical
    canonical payloads (adaptive tag + payload) and cd token partitions."""
    guide = build_guide(sql_source, sql_tokenizer)
    prod = guide.producer
    prod.set_genn_keys(True)
    st = _drive(guide, sql_tokenizer, b"select * from users where name = 'ab")
    rem, A = st.lexer.remainder, prod.allowed(st.stack)
    prod.set_genn_keys(False)
    try:
        e1 = prod._entry_for(rem, A)
    finally:
        prod.set_genn_keys(True)
    e2 = prod._entry_for(rem, A)
    assert e1.key != e2.key and e1.key[0] == "generic" and e2.key[0] == "genN"
    assert adaptive_encode(e1.ci_tokens, prod.vocab_size) == \
        adaptive_encode(e2.ci_tokens, prod.vocab_size)

    def part(e):
        return sorted(tuple(sorted(int(t) for t in g.token_ids)) for g in e.cd_groups)

    assert part(e1) == part(e2)
