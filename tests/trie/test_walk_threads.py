"""W8 rayon intra-walk parallelism: GRID_WALK_THREADS >= 2 must be BIT-IDENTICAL
to the sequential walk (GRID_WALK_THREADS=0, the kill switch — the pre-W8
walk_raw byte-for-byte), across the parity corpus (toy / sql-with-lexicons /
wide) and a big synthetic trie covering giant-CI open-literal walks and
CD-heavy lexicon boundary walks.

Identity is asserted at the raw kernel FFI surface — the i32-le ci buffer plus
the full ordered group list (events, segments, remainder, tid order) — which is
strictly stronger than mask-level equality: group ORDER and representative
bytes feed cache.make_entry and the normative entry_id hash, so any merge
reordering or wrong-representative pick would corrupt entry_id determinism.

GRID_WALK_PAR_MIN (default 4096 nodes) is the inline threshold; tests pin it
to 1 to force the parallel path onto the small corpus tries, and also run the
big trie at the default to cover the production dispatch.
"""

import random
import time

import pytest

import grid.trie.walk as W
from grid.generate import build_guide
from grid.guide import COMPLETE
from grid.models.tokenizer_adapter import MockTokenizer

pytestmark = pytest.mark.skipif(not W._USE_RUST, reason="grid_core not installed")

THREADS = ("1", "2", "8", "14")  # 1 = sequential by cap; others = rayon pool


def _walker(guide):
    return W._rust_walker(guide.trie, guide.dfa, guide.tables.ignored_terminal_ids,
                          guide.producer._priority, guide.lexicons)


def _harvest_configs(guide, seed: int, steps: int):
    """(remainder, A-words) configs along a random in-grammar trajectory."""
    walker = _walker(guide)
    rng = random.Random(seed)
    state = guide.initial_state
    configs = []
    for _step in range(steps):
        A = guide.producer.allowed(state.stack)
        configs.append((bytes(state.lexer.remainder), W._term_words(A, walker.width)))
        ids, _ = guide._mask_ids(state)
        tok = rng.choice(sorted(set(int(i) for i in ids) - {guide.eos_token_id})
                         or [int(ids[0])])
        state = guide.get_next_state(state, tok)
        if state.status == COMPLETE:
            break
    return configs


def _assert_threads_identical(walker, configs, monkeypatch, ctx: str,
                              threads=THREADS) -> int:
    """Every config: threaded output == sequential output, bit-for-bit at the
    FFI surface (ci i32-le bytes; ordered groups incl. representatives and tid
    order). Returns total groups seen sequentially (vacuity guard)."""
    groups_seen = 0
    for n, (rem, aw) in enumerate(configs):
        monkeypatch.setenv("GRID_WALK_THREADS", "0")
        ci0, g0 = walker.walk(rem, aw)
        groups_seen += len(g0)
        for th in threads:
            monkeypatch.setenv("GRID_WALK_THREADS", th)
            ci1, g1 = walker.walk(rem, aw)
            assert ci1 == ci0, f"{ctx} config {n} threads {th}: ci buffer diverged"
            assert g1 == g0, f"{ctx} config {n} threads {th}: groups diverged"
    return groups_seen


# ---------------------------------------------------------------- parity corpus


def test_walk_threads_toy(toy_source, toy_tokenizer, monkeypatch):
    monkeypatch.setenv("GRID_WALK_PAR_MIN", "1")  # force par path on a tiny trie
    guide = build_guide(toy_source, toy_tokenizer)
    configs = _harvest_configs(guide, seed=5, steps=12)
    _assert_threads_identical(_walker(guide), configs, monkeypatch, "toy")


def test_walk_threads_sql_with_lexicons(sql_source, sql_tokenizer, sql_grammar,
                                        monkeypatch):
    """Lexicon-sensitive walks (the lex group-key branch + ident boundaries)."""
    from grid.grammar.projection import RoleProjection
    from grid.lalr.compile import compile_tables
    from grid.policy.schema import SchemaSnapshot

    monkeypatch.setenv("GRID_WALK_PAR_MIN", "1")
    schema = SchemaSnapshot.from_dict({"users": ["id", "name"], "orders": ["id", "total"]})
    proj = RoleProjection.full(sql_grammar).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    guide = build_guide(sql_source, sql_tokenizer, projection=proj,
                        lexicons=schema.lexicons(tables),
                        schema_fingerprint=schema.fingerprint)
    groups = 0
    for seed in (3, 7, 19):
        configs = _harvest_configs(guide, seed=seed, steps=12)
        groups += _assert_threads_identical(_walker(guide), configs, monkeypatch,
                                            f"sql seed {seed}")
    assert groups > 0, "sql corpus never produced CD groups (vacuous merge test)"


def test_walk_threads_wide_grammar(wide_source, wide_tokenizer, monkeypatch):
    """>64 terminals: the [u64; W=2] mask width on the parallel path."""
    monkeypatch.setenv("GRID_WALK_PAR_MIN", "1")
    guide = build_guide(wide_source, wide_tokenizer)
    assert guide.tables.n_terminals > 64
    configs = _harvest_configs(guide, seed=13, steps=12)
    _assert_threads_identical(_walker(guide), configs, monkeypatch, "wide")


def test_walk_threads_garbage_and_capped_values(toy_source, toy_tokenizer, monkeypatch):
    """Invalid GRID_WALK_THREADS degrades to sequential (never raises); a value
    far above ncpu is capped, still bit-identical."""
    monkeypatch.setenv("GRID_WALK_PAR_MIN", "1")
    guide = build_guide(toy_source, toy_tokenizer)
    configs = _harvest_configs(guide, seed=5, steps=6)
    _assert_threads_identical(_walker(guide), configs, monkeypatch, "env edge",
                              threads=("not-a-number", "-3", "9999"))


# ------------------------------------------------------------ big synthetic trie


def _big_guide(n_tokens: int, seed: int = 42):
    """sql_subset + schema lexicons over a big random vocab: trie >= the default
    GRID_WALK_PAR_MIN, giant-CI open-literal walks, and cross-subtree CD groups
    (multi-lexeme tokens sharing verdict-equivalence keys across first bytes)."""
    import pathlib

    from grid.grammar import spec
    from grid.grammar.projection import RoleProjection
    from grid.lalr.compile import compile_tables
    from grid.policy.schema import SchemaSnapshot

    root = pathlib.Path(__file__).parent.parent.parent
    sql_src = (root / "grammars" / "sql_subset.grid").read_text()
    rng = random.Random(seed)
    alpha = "abcdefgh0123_"
    toks: set[str] = set()
    while len(toks) < n_tokens:
        n = rng.randint(1, 8)
        toks.add("".join(rng.choice(alpha) for _ in range(n)))
    # multi-lexeme tokens: CD entries whose groups span top-level subtrees
    toks |= {"ab=", "cd=", "id=", "total=", "ab,", "cd,", "id,", "ab ", "cd ", "id "}
    tok = MockTokenizer(extra_tokens=tuple(sorted(toks)))
    g = spec.load(sql_src)
    schema = SchemaSnapshot.from_dict(
        {"users": ["id", "ab", "cd", "total"], "orders": ["id", "total"]})
    proj = RoleProjection.full(g).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    guide = build_guide(sql_src, tok, projection=proj, lexicons=schema.lexicons(tables),
                        schema_fingerprint=schema.fingerprint)
    return guide, tok


def _state_for(guide, tok, text: str):
    st = guide.initial_state
    for t in tok.greedy_tokenize(text.encode()):
        st = guide.get_next_state(st, t)
        assert st is not None, text
    return st


def test_walk_threads_big_trie_giant_ci_and_cd_boundaries(monkeypatch):
    guide, tok = _big_guide(6000)
    walker = _walker(guide)
    assert len(guide.trie.nodes) >= 4096  # default GRID_WALK_PAR_MIN engages

    texts = (
        "select ",                                   # ident boundary (CD-heavy)
        "select ab",                                 # mid-ident, grouped CD
        "select * from users where ",                # second ident context
        "select ab, ",                               # post-comma boundary
        "select id from users where ab = '",         # giant-CI open literal
        "select id from users where ab = 'ab_01",    # mid-literal extension
        "select id from users where ab = 42 ",       # keyword boundary
    )
    configs = []
    giant_seen = 0
    span_groups = 0
    monkeypatch.setenv("GRID_WALK_THREADS", "0")
    for text in texts:
        st = _state_for(guide, tok, text)
        A = guide.producer.allowed(st.stack)
        rem = bytes(st.lexer.remainder)
        aw = W._term_words(A, walker.width)
        configs.append((rem, aw))
        ci, gs = walker.walk(rem, aw)
        if len(ci) // 4 > 5000:
            giant_seen += 1
        span_groups += sum(
            1 for _evs, _segs, _rem, ids in gs
            if len({tok.token_bytes(int(i))[:1] for i in ids}) > 1)
    assert giant_seen >= 2, "no giant-CI open-literal walk in the corpus (vacuous)"
    assert span_groups >= 2, \
        "no CD group spans top-level subtrees (cross-chunk merge untested)"

    # fuzz: real (remainder, A) pairs from prefixes of a corpus text (the lexer
    # invariant — a single scannable partial lexeme — must hold for walk input),
    # with random extra terminals OR'd into A as viable-set noise
    rng = random.Random(99)
    full = "select id, ab from users where cd = 'x0' and total = 42 limit 7;"
    for _ in range(24):
        st = _state_for(guide, tok, full[: rng.randint(0, len(full) - 1)])
        A = set(guide.producer.allowed(st.stack))
        A |= set(rng.sample(range(guide.tables.n_terminals), 3))
        configs.append((bytes(st.lexer.remainder), W._term_words(frozenset(A), walker.width)))

    # default inline threshold (production dispatch) ...
    _assert_threads_identical(walker, configs, monkeypatch, "big trie (par-min default)")
    # ... and forced fine chunking (one chunk per top-level subtree at high
    # thread counts) to stress the ordered merge
    monkeypatch.setenv("GRID_WALK_PAR_MIN", "1")
    _assert_threads_identical(walker, configs, monkeypatch, "big trie (par-min 1)")


def test_walk_threads_timing_smoke(monkeypatch):
    """NON-GATING speedup report on a giant open-literal walk (the W8 target
    class). Asserts only output identity; the ratio is printed for the record
    (run with -s), never asserted — CI boxes throttle unpredictably."""
    guide, tok = _big_guide(60000)
    walker = _walker(guide)
    st = _state_for(guide, tok, "select id from users where ab = '")
    rem = bytes(st.lexer.remainder)
    aw = W._term_words(guide.producer.allowed(st.stack), walker.width)

    def best_ms(runs: int = 5) -> float:
        best = float("inf")
        for _ in range(runs):
            t0 = time.perf_counter()
            walker.walk(rem, aw)
            best = min(best, time.perf_counter() - t0)
        return best * 1e3

    monkeypatch.setenv("GRID_WALK_THREADS", "0")
    ci0, g0 = walker.walk(rem, aw)
    seq_ms = best_ms()
    monkeypatch.setenv("GRID_WALK_THREADS", "8")
    ci1, g1 = walker.walk(rem, aw)
    par_ms = best_ms()
    assert ci1 == ci0 and g1 == g0
    assert len(ci0) // 4 >= 50000, "timing walk is not the giant-CI class"
    print(f"\n[W8 timing smoke] trie={len(guide.trie.nodes)} nodes, "
          f"ci={len(ci0) // 4}: seq {seq_ms:.3f} ms, 8-thread {par_ms:.3f} ms, "
          f"speedup {seq_ms / par_ms:.2f}x (non-gating)")
