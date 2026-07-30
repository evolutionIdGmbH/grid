"""S1 jump-forward: GridGuide.forced_run + GridGrammarSession.jump_tokens.

MockTokenizer differentials, vllm-free:

- forced spans are byte-identical to step-by-step decoding (the S1 core
  gate): a decode loop that consumes Write spans and one that re-derives
  every step's mask produce identical token streams and identical bytes,
  because each span token is the singleton element of its own step's mask
  (DESIGN.md §4.5 — asserted directly);
- jump_tokens() == guide-side forced_run for the same configuration on
  BOTH session paths (v5 Python states, v6 kernel session);
- jump_tokens() is state-neutral (the manager advances/rolls back the
  grammar itself around per-position fills) and warm-only on v6 (cold
  successors END the chain, never walk);
- eos / COMPLETE / j_max / choice-point stop rules;
- GRID_JUMP default-off is a strict no-op (no guide walk, no kernel op).
"""

import random

import numpy as np
import pytest

from grid import perf_flags
from grid.generate import build_guide
from grid.guide import COMPLETE
from grid.models.tokenizer_adapter import MockTokenizer
from grid.models.vllm_structured import GridGrammarSession
from grid.protocols import Generate, Write

# A 16-byte literal over the byte-fallback-only tokenizer: every position is
# a forced singleton (no multi-byte token admits an alternative
# tokenization), so the grammar carries one maximal forced run of length 16.
RUN_LITERAL = b"abcdefghijklmnop"
RUN_SRC = '%start s\ns: "abcdefghijklmnop"\n'
# Two literals sharing the "ab" prefix: forced run of exactly 2, then a
# genuine choice point (mask cardinality 2).
FORK_SRC = '%start s\ns: "abc" | "abz"\n'


def _mini_guide(source: str, adapter):
    """GridGuide over a reserve-free literal grammar (the maskbench engine
    construction; build_guide requires an ignored-whitespace terminal for
    its ReserveTable, which these single-literal grammars lack)."""
    from grid.grammar import spec
    from grid.grammar.projection import RoleProjection
    from grid.guide import GridGuide
    from grid.lalr.compile import compile_tables
    from grid.lexer.dfa import build_scanner
    from grid.trie.build import build_trie

    grammar = spec.load(source)
    proj = RoleProjection.full(grammar).build()
    tables = compile_tables(proj)
    dfa = build_scanner(grammar.terminals, grammar.terminal_order)
    return GridGuide(tables=tables, dfa=dfa, trie=build_trie(adapter),
                     adapter=adapter)


@pytest.fixture(scope="module")
def byte_tokenizer():
    return MockTokenizer(extra_tokens=())


@pytest.fixture()
def run_guide(byte_tokenizer):
    return _mini_guide(RUN_SRC, byte_tokenizer)


@pytest.fixture()
def fork_guide(byte_tokenizer):
    return _mini_guide(FORK_SRC, byte_tokenizer)


@pytest.fixture()
def toy_guide(toy_source, toy_tokenizer):
    return build_guide(toy_source, toy_tokenizer)


def _bits_to_ids(words: np.ndarray, vocab: int) -> set[int]:
    out = set()
    for w, word in enumerate(words.tolist()):
        b = 0
        while word:
            if word & 1:
                out.add(w * 32 + b)
            word >>= 1
            b += 1
    return {t for t in out if t < vocab}


# ------------------------------------------------------- guide: forced_run


def test_forced_run_agrees_with_instruction(toy_guide):
    """Random walk: forced_run(state) == the Write span wherever
    get_next_instruction emits a singleton non-eos Write, [] wherever it
    emits Generate or an eos-flavored Write."""
    guide = toy_guide
    for seed in (3, 11, 42):
        rng = random.Random(seed)
        state = guide.initial_state
        for _ in range(40):
            got = guide.forced_run(state)
            instr = guide.get_next_instruction(state)
            if isinstance(instr, Write):
                span = [int(t) for t in instr.tokens]
                if span[0] == guide.eos_token_id:
                    assert got == []
                else:
                    assert got == span
            else:
                assert isinstance(instr, Generate)
                assert got == []
            if state.status == COMPLETE:
                break
            ids, _ = guide._mask_ids(state)
            pick = guide.eos_token_id if guide.eos_token_id in ids else rng.choice(ids)
            state = guide.get_next_state(state, int(pick))


def test_forced_run_long_literal_and_j_max(run_guide, byte_tokenizer):
    """The 16-token run: j_max bounds the chain; raising it exposes the
    full run; the span IS the literal's canonical byte tokenization."""
    guide = run_guide
    assert guide.j_max == 8
    span = guide.forced_run(guide.initial_state)
    assert span == byte_tokenizer.greedy_tokenize(RUN_LITERAL[:8])
    guide.j_max = 32
    full = guide.forced_run(guide.initial_state)
    assert full == byte_tokenizer.greedy_tokenize(RUN_LITERAL)
    assert len(full) == 16


def test_forced_run_stops_at_choice_point(fork_guide, byte_tokenizer):
    span = fork_guide.forced_run(fork_guide.initial_state)
    assert span == byte_tokenizer.greedy_tokenize(b"ab")


def test_forced_run_excludes_eos_and_complete(run_guide):
    """After the full literal the mask is exactly {eos}: not a forced run
    (a jump never proposes eos). After eos: COMPLETE -> []."""
    guide = run_guide
    state = guide.initial_state
    for t in guide.adapter.greedy_tokenize(RUN_LITERAL):
        state = guide.get_next_state(state, t)
    ids, _ = guide._mask_ids(state)
    assert set(ids.tolist()) == {guide.eos_token_id}
    assert guide.forced_run(state) == []
    done = guide._advance(state, guide.eos_token_id, audit=False)
    assert done.status == COMPLETE
    assert guide.forced_run(done) == []


def test_forced_run_is_pure(toy_source, toy_tokenizer):
    """Pure query: no audit records, _pending untouched, the state's next
    instruction unchanged."""
    guide = build_guide(toy_source, toy_tokenizer, audit=True)
    state = guide.initial_state
    instr_before = guide.get_next_instruction(state)  # populates _pending
    pending_before = dict(guide._pending)
    n_records = len(guide.audit.records)
    guide.forced_run(state)
    assert len(guide.audit.records) == n_records
    assert guide._pending == pending_before
    instr_after = guide.get_next_instruction(state)
    assert type(instr_after) is type(instr_before)
    assert (instr_after.tokens == instr_before.tokens).all()


def _stepwise_pick(guide, state, score):
    """One no-span decode step: the exact token a masked sampler must pick.
    At a singleton mask ANY sampler picks the one allowed token
    (probability 1); at free steps `score` is the deterministic model."""
    ids, _ = guide._mask_ids(state)
    if len(ids) == 1:
        return int(ids[0])
    return max((int(t) for t in ids), key=score)


def test_forced_spans_byte_identical_to_stepwise_decode(toy_guide, run_guide):
    """THE S1 differential: a loop consuming Write spans (mode-1 shape) and
    a loop re-deriving every step's mask produce identical token streams
    and identical bytes; every span token is the singleton element of its
    own step's mask (asserted at each span position)."""
    for guide, seed in ((toy_guide, 5), (toy_guide, 19), (run_guide, 7)):
        rng = random.Random(seed)
        score_tbl = [rng.random() for _ in range(guide.vocab_size)]

        def score(t, _tbl=score_tbl, _guide=guide):
            return (_tbl[t] + (0.5 if t == _guide.eos_token_id else 0.0), t)

        # arm A: mode-1 shape — consume Write spans without "forward passes"
        out_a: list[int] = []
        state = guide.initial_state
        for _ in range(64):
            instr = guide.get_next_instruction(state)
            if isinstance(instr, Write):
                for t in (int(x) for x in instr.tokens):
                    if t != guide.eos_token_id:
                        ids, _ = guide._mask_ids(state)
                        assert len(ids) == 1 and int(ids[0]) == t, \
                            "span token must be its step's singleton mask"
                    state = guide.get_next_state(state, t)
                    out_a.append(t)
                    if state.status == COMPLETE:
                        break
            else:
                t = max((int(x) for x in instr.tokens), key=score)
                state = guide.get_next_state(state, t)
                out_a.append(t)
            if state.status == COMPLETE:
                break
        assert state.status == COMPLETE

        # arm B: stepwise — every token from a fresh mask, no spans
        out_b: list[int] = []
        state = guide.initial_state
        for _ in range(256):
            t = _stepwise_pick(guide, state, score)
            state = guide.get_next_state(state, t)
            out_b.append(t)
            if state.status == COMPLETE:
                break
        assert state.status == COMPLETE

        assert out_a == out_b, f"token streams diverged (seed {seed})"
        bytes_a = b"".join(guide.adapter.token_bytes(t) for t in out_a)
        bytes_b = b"".join(guide.adapter.token_bytes(t) for t in out_b)
        assert bytes_a == bytes_b, f"bytes diverged (seed {seed})"


# -------------------------------------------------- session: jump_tokens


def _twin_sessions(guide, monkeypatch, jump="1"):
    """(v5 oracle, v6 kernel) sessions over ONE shared guide/producer, with
    the jump lever pinned; the v6 leg skips without the kernel."""
    monkeypatch.setenv("GRID_JUMP", jump)
    s5 = GridGrammarSession(guide, _force_v5=True)
    s6 = GridGrammarSession(guide)
    if s6._sid is None:
        pytest.skip("kernel v6 session unavailable (GRID_NO_V6/no kernel)")
    return s5, s6


def _k_state(s6):
    return s6._kernel.session_state(s6._sid)


def test_jump_disabled_by_default_is_strict_noop(run_guide, monkeypatch):
    """GRID_JUMP unset: [] without a single guide walk or kernel op."""
    monkeypatch.delenv("GRID_JUMP", raising=False)
    assert perf_flags.jump_enabled() is False
    s = GridGrammarSession(run_guide)
    calls = []
    monkeypatch.setattr(run_guide, "forced_run",
                        lambda st: calls.append(st) or [])
    if s._sid is not None:
        before = dict(s._kernel.session_stats())
    assert s.jump_tokens() == []
    assert calls == [], "flag-off must not touch the guide"
    if s._sid is not None:
        assert dict(s._kernel.session_stats()) == before, \
            "flag-off must not touch the kernel"


def test_jump_flag_read_at_construction(run_guide, monkeypatch):
    monkeypatch.delenv("GRID_JUMP", raising=False)
    s_off = GridGrammarSession(run_guide, _force_v5=True)
    monkeypatch.setenv("GRID_JUMP", "1")
    assert s_off.jump_tokens() == [], "lever is per-session, read at init"
    s_on = GridGrammarSession(run_guide, _force_v5=True)
    assert s_on.jump_tokens() != []


def test_jump_v5_matches_forced_run_and_is_state_neutral(run_guide, monkeypatch):
    monkeypatch.setenv("GRID_JUMP", "1")
    s = GridGrammarSession(run_guide, _force_v5=True)
    tok = run_guide.adapter.greedy_tokenize(RUN_LITERAL[:1])
    assert s.accept_tokens("r", tok)
    want = run_guide.forced_run(s.states[-1])
    assert want == run_guide.adapter.greedy_tokenize(RUN_LITERAL[1:9])
    n_states, n_proc = len(s.states), s.num_processed_tokens
    got = s.jump_tokens()
    assert got == want
    assert len(s.states) == n_states and s.num_processed_tokens == n_proc
    assert s.jump_tokens() == want, "idempotent (pure query)"


def test_jump_v6_parity_and_state_neutral(run_guide, monkeypatch):
    """Warm chain: v6 == v5 == forced_run; kernel session state (kidx,
    remainder, status, n_generated, prev_token) identical before/after."""
    s5, s6 = _twin_sessions(run_guide, monkeypatch)
    tok = run_guide.adapter.greedy_tokenize(RUN_LITERAL[:1])
    assert s5.accept_tokens("r", tok) and s6.accept_tokens("r", tok)
    span5 = s5.jump_tokens()  # v5 walk publishes every chain config to T1
    assert span5 == run_guide.adapter.greedy_tokenize(RUN_LITERAL[1:9])
    before = _k_state(s6)
    span6 = s6.jump_tokens()
    assert span6 == span5
    assert _k_state(s6) == before, "jump must be state-neutral"
    assert s6.num_processed_tokens == s5.num_processed_tokens == 1
    assert s6.jump_tokens() == span6, "idempotent (pure query)"


def test_jump_v6_warm_only_truncates_on_cold(byte_tokenizer, monkeypatch):
    """Cold configs END the v6 chain (never walked eagerly): with only the
    current config warm the jump takes one hop; each warmed successor
    extends it by one; the warm result equals the v5 oracle."""
    guide = _mini_guide(RUN_SRC, byte_tokenizer)  # fresh producer: cold T1
    monkeypatch.setenv("GRID_JUMP", "1")
    s6 = GridGrammarSession(guide)
    if s6._sid is None:
        pytest.skip("kernel v6 session unavailable (GRID_NO_V6/no kernel)")
    # nothing warm at all: even the current config is unbound and unpeekable
    assert s6.jump_tokens() == []
    # warm the current config only -> exactly one hop, then cold stop
    shadow = guide.initial_state
    guide._mask_ids(shadow)
    assert s6.jump_tokens() == guide.adapter.greedy_tokenize(RUN_LITERAL[:1])
    # each successor warmed extends the chain by one hop
    shadow = guide.get_next_state(shadow, guide.adapter.greedy_tokenize(RUN_LITERAL[:1])[0])
    guide._mask_ids(shadow)
    assert s6.jump_tokens() == guide.adapter.greedy_tokenize(RUN_LITERAL[:2])
    # v5 has no warm gate: it walks the full j_max chain regardless
    s5 = GridGrammarSession(guide, _force_v5=True)
    assert s5.jump_tokens() == guide.adapter.greedy_tokenize(RUN_LITERAL[:8])
    # and the v6 chain, once every config is warm, matches it
    assert s6.jump_tokens() == s5.jump_tokens()


def test_jump_v6_rollback_interplay(run_guide, monkeypatch):
    """jump after rollback re-derives the rolled-back position's run; the
    kernel log survives the jump's internal accept/rollback cycles."""
    s5, s6 = _twin_sessions(run_guide, monkeypatch)
    toks = run_guide.adapter.greedy_tokenize(RUN_LITERAL[:3])
    assert s5.accept_tokens("r", toks) and s6.accept_tokens("r", toks)
    s5.jump_tokens()  # warm the chain from position 3
    assert s6.jump_tokens() == s5.jump_tokens()
    s5.rollback(2)
    s6.rollback(2)
    span5 = s5.jump_tokens()  # warm the chain from position 1
    assert span5 == run_guide.forced_run(s5.states[-1])
    assert s6.jump_tokens() == span5
    before = _k_state(s6)
    s6.jump_tokens()
    assert _k_state(s6) == before
    # the rolled-back sessions still accept and stay in lockstep
    nxt = run_guide.adapter.greedy_tokenize(RUN_LITERAL[1:2])
    assert s5.accept_tokens("r", nxt) and s6.accept_tokens("r", nxt)
    assert s5.num_processed_tokens == s6.num_processed_tokens == 2


def test_jump_v6_eos_only_mask_and_complete(run_guide, monkeypatch):
    """At the {eos}-only position the jump is empty (never proposes eos);
    after eos (COMPLETE) it is empty and touches nothing."""
    s5, s6 = _twin_sessions(run_guide, monkeypatch)
    toks = run_guide.adapter.greedy_tokenize(RUN_LITERAL)
    assert s5.accept_tokens("r", toks) and s6.accept_tokens("r", toks)
    s5.fill_bitmask(np.zeros((1, (run_guide.vocab_size + 31) // 32),
                             dtype=np.int32), 0)  # warm/publish current config
    assert s5.jump_tokens() == []
    assert s6.jump_tokens() == []
    eos = run_guide.eos_token_id
    assert s5.accept_tokens("r", [eos]) and s6.accept_tokens("r", [eos])
    assert s5.is_terminated() and s6.is_terminated()
    assert s5.jump_tokens() == [] and s6.jump_tokens() == []


def test_jump_draft_verification_flow(run_guide, monkeypatch):
    """The manager dance, emulated: jump -> per-position fills along the
    span (each row is exactly {span[i]} — the acceptance-rate-1.0 property)
    -> rollback -> real accept of the whole span. State parity v5/v6 and
    fill parity at the landing position."""
    s5, s6 = _twin_sessions(run_guide, monkeypatch)
    tok = run_guide.adapter.greedy_tokenize(RUN_LITERAL[:1])
    assert s5.accept_tokens("r", tok) and s6.accept_tokens("r", tok)
    span = s5.jump_tokens()
    assert len(span) == 8
    assert s6.jump_tokens() == span
    words = (run_guide.vocab_size + 31) // 32
    bm = np.zeros((1, words), dtype=np.int32)
    for s in (s5, s6):
        # advance-fill-rollback: what vllm's grammar_bitmask does with drafts
        for i, t in enumerate(span[:-1]):
            assert s.accept_tokens("r", [t])
            bm.fill(0)
            s.fill_bitmask(bm, 0)
            got = _bits_to_ids(bm[0].view(np.uint32), run_guide.vocab_size)
            assert got == {span[i + 1]}, "forced draft positions are singletons"
        s.rollback(len(span) - 1)
        # verification accepted everything: the real accept consumes the span
        assert s.accept_tokens("r", span)
    assert s5.num_processed_tokens == s6.num_processed_tokens == 1 + len(span)
    bm5 = np.zeros((1, words), dtype=np.int32)
    bm6 = np.zeros((1, words), dtype=np.int32)
    s5.fill_bitmask(bm5, 0)
    s6.fill_bitmask(bm6, 0)
    assert (bm5 == bm6).all(), "landing-position fill parity"
