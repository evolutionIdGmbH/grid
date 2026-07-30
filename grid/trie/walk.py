"""Trie walk: exact token-mask computation with the CI/CD split (DESIGN.md E11, SS6 step 6).

INCREMENTAL implementation (the grid_core kernel algorithm): the DFS carries an
O(1)-updatable scan state per frame — (dfa_state, current segment, last-accept) —
instead of rescanning ``remainder + path`` from scratch at every node. Forced
emissions cascade through a small pending-byte queue that exactly replicates
``lexer.run.scan``'s maximal-munch restart semantics; the G3 differential binds
the two.

Classification (identical semantics to v1):
- REJECT-SUBTREE: unlexable byte, an emission event with no viable choice, or a
  zero-emission partial with no viable hypothesis. Monotone in extensions
  (live sets shrink; future emission candidates are subsets of current live), so
  the whole subtree is skipped.
- CI (context-independent, cacheable): zero non-ignored emissions with a viable
  partial; or exactly one non-ignored emission whose terminal is in A, followed
  by nothing, pure-ignored events, or an ignored-viable partial. Depends only on
  (remainder, A, lexicons) — the cache key.
- CD (context-dependent, never cached): one in-A emission followed by content
  whose viability needs the post-shift allowed set, or >= 2 non-ignored
  emissions. Checked per step against the live stack
  (mask/producer.check_context_dependent).

Identifier positions (E3/G6): when a chosen terminal is an L3 identifier
category, the emitted lexeme must be in the category's allow-list and a partial
identifier must be a prefix of some allowed identifier — the identifier
composition rule, enforced here, never by unioning generic masks.

S2 slicer (trie.slices present, i.e. GRID_PERF_SLICER=1 at build_trie time):
before the full-trie DFS, prove structurally that EVERY slice token (spelled
entirely inside the JSON-string-safe byte class) is CI in this configuration —
closure BFS from the seeded state over the class bytes with every transition
non-DEAD (so no emission event can fire inside a sliced token), every reachable
state partial-viable (live & (A|ignored) != 0) and lexicon-inert (live disjoint
from allow-listed terminals, the genN guard's condition — prefix_ok is then
vacuous), |closure| <= _SLICE_CLOSURE_CAP. Proof passes -> walk only the
rest-trie and contribute the precomputed slice ids; proof fails -> the full
walk, byte-for-byte. Lazy factored DFAs never slice (same gate as the kernel:
the closure BFS would force product-state materialization).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from grid.lexer.dfa import DEAD, ScannerDFA
from grid.lexer.run import EmissionEvent
from grid.trie.build import TokenTrie

try:  # grid_core is optional; the Python walk is the executable specification
    import grid_core as _grid_core
except ImportError:  # pragma: no cover
    _grid_core = None

# hasattr guard: setuptools' editable install globs the grid_core/ crate dir in as
# an empty namespace package (import succeeds, no RustWalker); require the real symbol.
_USE_RUST = (
    _grid_core is not None
    and hasattr(_grid_core, "RustWalker")
    and os.environ.get("GRID_NO_RUST") != "1"
)

_MASK_CACHE: dict[int, frozenset[int]] = {}
_WALKERS: dict[tuple[int, int, int], tuple] = {}

MAX_KERNEL_TERMINALS = 512  # [u64; 8] bitmask bound in grid_core


def _term_mask(s) -> int:
    m = 0
    for t in s:
        m |= 1 << t
    return m


def _width(n_terminals: int) -> int:
    """Kernel mask width in u64 words (1/2/4/8), matching grid_core::width_for."""
    words = -(-n_terminals // 64) or 1
    for w in (1, 2, 4, 8):
        if words <= w:
            return w
    raise ValueError(f"{n_terminals} terminals exceeds the kernel bound")


def _term_words(s, w: int) -> list[int]:
    """Terminal set -> little-endian u64 word list of width w."""
    out = [0] * w
    for t in s:
        out[t >> 6] |= 1 << (t & 63)
    return out


def _words_int(words) -> int:
    m = 0
    for i, wd in enumerate(words):
        m |= int(wd) << (64 * i)
    return m


def _unmask(mask: int) -> frozenset[int]:
    got = _MASK_CACHE.get(mask)
    if got is None:
        got = frozenset(i for i in range(mask.bit_length()) if (mask >> i) & 1)
        _MASK_CACHE[mask] = got
    return got


def _rust_walker(trie: TokenTrie, dfa: ScannerDFA, ignored: frozenset[int],
                 priority: dict[int, tuple[int, int]], lexicons: Lexicons | None):
    # lazy factored DFAs (GRID_PERF_FACTORED_SCANNER over-budget regime) have
    # no dense trans/accept arrays — every kernel entry must gate on this;
    # counting DFAs (GRID_PERF_COUNTING) stay on the Python spec walk until
    # the kernel v8 counter step lands
    assert not getattr(dfa, "lazy", False), "lazy factored DFA reached the kernel walker"
    assert not getattr(dfa, "counters", ()), "counting DFA reached the Rust walker"
    key = (id(trie), id(dfa), id(lexicons))
    hit = _WALKERS.get(key)
    if hit is not None:
        return hit[0]
    import numpy as np

    n_terminals = len(priority)
    w = _width(n_terminals)
    literal_words = _term_words((t for t, (kind, _i) in priority.items() if kind == 0), w)
    trans = np.array(dfa.trans, dtype=np.int32)
    accept = np.array(dfa.accept, dtype=np.int32)
    lex = None if lexicons is None else {
        int(tid): [bytes(word) for word in words] for tid, words in lexicons.allowed.items()
    }
    kwargs = {}
    # S2 slice tables: (class bitmap as 4 u64 words, alias-complete i32-le
    # slice ids, rest-trie u64-le nodes). Only when the trie carries slices
    # AND the installed kernel knows the tables (feature-probe on the stats
    # getter added in the same kernel change) — an older grid_core keeps the
    # exact pre-slicer construction.
    if trie.slices is not None and hasattr(_grid_core.RustWalker, "slice_stats"):
        sl = trie.slices
        kwargs["slices"] = (
            [int(x) for x in sl.class_words],
            sl.ids.astype("<i4", copy=False).tobytes(),
            sl.rest_nodes.astype("<u8", copy=False).tobytes(),
        )
    walker = _grid_core.RustWalker(
        trie.nodes.tobytes(), trans.tobytes(), accept.tobytes(), n_terminals,
        [_term_words(s, w) for s in dfa.accepts_all],
        [_term_words(s, w) for s in dfa.live],
        dfa.start, _term_words(ignored, w), literal_words, lexicon=lex,
        aliases={int(k): [int(x) for x in v] for k, v in (trie.aliases or {}).items()},
        **kwargs,
    )
    _WALKERS[key] = (walker, (trie, dfa, lexicons))  # hold refs: id-keyed cache
    return walker


def make_verdict_kernel(tables, dfa: ScannerDFA, lexicons: Lexicons | None):
    """grid_core.RustVerdicts for (tables, dfa, lexicons), or None when the kernel
    is unavailable/disabled, the grammar exceeds the 512-terminal bitmask bound
    (same gate as the walk kernel), or the DFA is a lazy factored facade (no
    dense arrays to upload). SS2 kernel #2 + the LALR simulate behind it."""
    if (not _USE_RUST or not hasattr(_grid_core, "RustVerdicts")
            or tables.n_terminals > MAX_KERNEL_TERMINALS
            or getattr(dfa, "lazy", False)
            or getattr(dfa, "counters", ())):  # counter step is kernel v8; Python verdicts until then
        return None
    import numpy as np

    n_states = len(tables.action)
    n_cols = tables.n_terminals  # END column included; also the nonterminal id base
    w = _width(n_cols)
    kind = np.zeros((n_states, n_cols), dtype=np.uint8)
    arg = np.zeros((n_states, n_cols), dtype=np.uint32)
    for s, row in enumerate(tables.action):
        for t, (k, a) in row.items():
            kind[s, t] = k + 1  # 0 stays "not in row"
            arg[s, t] = a
    n_nts = len(tables.nonterminal_names)
    gt = np.full((n_states, n_nts), -1, dtype=np.int32)
    for s, row in enumerate(tables.goto):
        for nt, dst in row.items():
            gt[s, nt - n_cols] = dst
    trans = np.array(dfa.trans, dtype=np.int32)
    lex = None if lexicons is None else {
        int(tid): [bytes(word) for word in words] for tid, words in lexicons.allowed.items()
    }
    return _grid_core.RustVerdicts(
        kind.tobytes(), arg.tobytes(), n_states, n_cols,
        gt.tobytes(), n_nts,
        [(int(lhs), len(rhs)) for lhs, rhs in tables.prods],
        tables.end_id,
        _term_words(tables.ignored_terminal_ids, w),
        _term_words(tables.literal_terminal_ids, w),
        trans.tobytes(), [_term_words(s, w) for s in dfa.live], dfa.start,
        lexicon=lex,
    )


@dataclass(frozen=True)
class CDEntry:
    """A context-dependent token: re-checked against the live stack every step."""

    token_id: int
    events: tuple[EmissionEvent, ...]
    segments: tuple[bytes, ...]      # lexeme bytes per event (identifier checks)
    remainder: bytes                 # partial after the last emission


@dataclass(frozen=True)
class WalkResult:
    # rust path: a READ-ONLY int32 np.ndarray (np.frombuffer over the kernel's
    # i32-le buffer — no int-object materialization); Python-spec path: tuple.
    # Both support len()/iteration/list()/np.asarray/indexing identically.
    ci_tokens: tuple[int, ...]  # or np.ndarray[int32] on the kernel path
    cd_entries: tuple[CDEntry, ...]
    # rust path: (representative CDEntry, raw token ids, raw kernel payload)
    # per group, precomputed in-kernel (E10 cd_groups); None on the Python path
    # (make_entry groups there). The payload is the kernel's own walk output,
    # kept verbatim so registration needs no frozenset->words reconversion —
    # this was ~30% of the cold-miss cost.
    groups: tuple[tuple[CDEntry, tuple[int, ...], tuple], ...] | None = None


class Lexicons:
    """L3 SchemaLexicon view used by the walk: per-terminal allow-lists + prefix sets."""

    def __init__(self, allowed: dict[int, set[bytes]]) -> None:
        self.allowed = allowed
        self.prefixes: dict[int, set[bytes]] = {}
        for tid, words in allowed.items():
            pref: set[bytes] = set()
            for w in words:
                for k in range(len(w) + 1):
                    pref.add(w[:k])
            self.prefixes[tid] = pref

    def lexeme_ok(self, tid: int, lexeme: bytes) -> bool:
        return tid not in self.allowed or lexeme in self.allowed[tid]

    def prefix_ok(self, tid: int, partial: bytes) -> bool:
        return tid not in self.allowed or partial in self.prefixes[tid]


REJECT_SUBTREE = 0
CI = 1
CD = 2


def pick_viable(
    ev: EmissionEvent,
    lexeme: bytes,
    viable: frozenset[int],
    ignored: frozenset[int],
    priority: dict[int, tuple[int, int]],
    lexicons: Lexicons | None,
) -> int | None:
    """Contextual emission choice with L3 lexeme checks: first priority-ordered
    viable real candidate whose lexeme passes its lexicon; else an ignored
    candidate; else None (identifier composition rule, E3/G6)."""
    for t in sorted((t for t in ev.candidates if t in viable), key=lambda t: priority[t]):
        if lexicons is None or lexicons.lexeme_ok(t, lexeme):
            return t
    ign = [t for t in ev.candidates if t in ignored]
    if ign:
        return min(ign, key=lambda t: priority[t])
    return None


class _Frame:
    """Per-trie-node incremental walk state (the grid_core kernel state)."""

    __slots__ = ("end", "dfa_state", "seg", "last_len", "last_state", "counts",
                 "events", "n_real", "cd_flag")

    def __init__(self, end: int, dfa_state: int, seg: bytes, last_len: int, last_state: int,
                 counts: tuple[int, ...] | None,
                 events: tuple, n_real: int, cd_flag: bool) -> None:
        self.end = end
        self.dfa_state = dfa_state
        self.seg = seg
        self.last_len = last_len
        self.last_state = last_state
        self.counts = counts          # counting DFAs only; None otherwise
        self.events = events          # tuple[(EmissionEvent, lexeme_bytes)]
        self.n_real = n_real
        self.cd_flag = cd_flag


def _seed(dfa: ScannerDFA, remainder: bytes) -> tuple[int, bytes, int, int, tuple[int, ...] | None]:
    """Scan the (invariantly single-partial-lexeme) remainder, tracking last accept."""
    if getattr(dfa, "counters", ()):
        cur, counts, last_len, last_state = dfa.start, dfa.zero_counts(), 0, -1
        for i, b in enumerate(remainder):
            cur, counts = dfa.step(cur, counts, b)
            assert cur != DEAD, "LexerRun invariant: remainder must be scannable"
            if dfa.accept[cur] != -1:
                last_len, last_state = i + 1, cur
        return cur, remainder, last_len, last_state, counts
    cur, last_len, last_state = dfa.start, 0, -1
    for i, b in enumerate(remainder):
        cur = dfa.trans[cur][b]
        assert cur != DEAD, "LexerRun invariant: remainder must be scannable"
        if dfa.accept[cur] != -1:
            last_len, last_state = i + 1, cur
    return cur, remainder, last_len, last_state, None


def walk(
    trie: TokenTrie,
    dfa: ScannerDFA,
    remainder: bytes,
    A: frozenset[int],
    ignored: frozenset[int],
    priority: dict[int, tuple[int, int]],
    lexicons: Lexicons | None = None,
) -> WalkResult:
    """SS2 kernel #1: (ci_mask, cd_token_list) for the current configuration.

    Dispatches to the grid_core Rust kernel when available (bit-identical by
    tests/trie/test_rust_parity.py); falls back to the Python implementation for
    grammars with more than 512 terminals, when GRID_NO_RUST=1, when the DFA
    is a lazy factored facade (which materializes product states only along the
    trie paths this walk actually takes), or for counting DFAs
    (GRID_PERF_COUNTING — kernel v8 counter step pending)."""
    if (_USE_RUST and len(priority) <= MAX_KERNEL_TERMINALS
            and not getattr(dfa, "lazy", False)
            and not getattr(dfa, "counters", ())):
        import numpy as np

        walker = _rust_walker(trie, dfa, ignored, priority, lexicons)
        ci_bytes, raw_groups = walker.walk(bytes(remainder), _term_words(A, walker.width))
        # ci ids arrive as ONE i32-le buffer (sorted, alias-expanded in-kernel);
        # np.frombuffer is zero-copy and the view is read-only (immutability
        # matches the tuple it replaces) — no per-id int objects materialize.
        ci = np.frombuffer(ci_bytes, dtype=np.int32)
        groups = []
        reps = []
        for evs, segs, rem, ids in raw_groups:
            rep = CDEntry(
                ids[0],
                tuple(EmissionEvent(_unmask(_words_int(c)), int(ln)) for c, ln in evs),
                tuple(segs),
                rem,
            )
            reps.append(rep)
            # the raw (evs, segs, rem, ids) tuple IS the kernel registration
            # payload — kept verbatim (no words->frozenset->words round trip)
            groups.append((rep, tuple(ids), (evs, segs, rem, ids)))
        # ci and group ids arrive alias-expanded and sorted from the kernel
        return WalkResult(ci_tokens=ci, cd_entries=tuple(reps), groups=tuple(groups))
    return _walk_py(trie, dfa, remainder, A, ignored, priority, lexicons)


_SLICE_CLOSURE_CAP = 64


def _slice_contained(
    slices,
    dfa: ScannerDFA,
    q: int,
    A: frozenset[int],
    ignored: frozenset[int],
    lexicons: Lexicons | None,
) -> bool:
    """S2 containment proof at seeded state ``q``: True means EVERY slice token
    is provably CI here, so the walk may skip the sliced subtrees.

    Closure BFS from q over the slice byte class, requiring for every reachable
    state s (q included): (a) trans[s][b] != DEAD for every class byte b — no
    emission event (emissions fire only on DEAD transitions) can occur while
    walking an all-class token, so n_real stays 0 and the walk state stays in
    the closure; (b) live[s] & (A | ignored) != {} — the zero-emission node
    verdict's partial_viable holds, no subtree prune; (c) live[s] disjoint from
    the lexicon-constrained terminal ids — prefix_ok is vacuously true on every
    viability witness (the genN key normalization's lexicon-inertness guard,
    producer.cache_key); (d) |closure| <= _SLICE_CLOSURE_CAP (bounded
    maxLength/{m,n} counting chains blow the cap or hit DEAD and fall back).
    Deliberately derives byte exclusions from the DFA instead of trusting the
    slice regex: proof-or-walk, never a wrong mask."""
    lex_ids = frozenset() if lexicons is None else frozenset(lexicons.allowed)
    a_or_ign = A | ignored
    trans = dfa.trans
    live = dfa.live
    seen = {q}
    queue = [q]
    i = 0
    while i < len(queue):
        s = queue[i]
        i += 1
        lv = live[s]
        if not (lv & a_or_ign) or (lv & lex_ids):
            return False
        row = trans[s]
        for b in slices.class_bytes:
            nx = row[b]
            if nx == DEAD:
                return False
            if nx not in seen:
                if len(seen) >= _SLICE_CLOSURE_CAP:
                    return False
                seen.add(nx)
                queue.append(nx)
    return True


def _walk_py(
    trie: TokenTrie,
    dfa: ScannerDFA,
    remainder: bytes,
    A: frozenset[int],
    ignored: frozenset[int],
    priority: dict[int, tuple[int, int]],
    lexicons: Lexicons | None = None,
) -> WalkResult:
    """The executable-specification Python walk (the algorithm grid_core ports).

    With slice tables (S2) and a passing containment proof, only the rest-trie
    is walked: ci = slices.min_ids ++ rest DFS ci (per-node smallest-alias ids
    either way; consumers alias-expand and SORT — producer._entry_for and the
    parity harness both do), cd = the rest walk's cd verbatim. Token verdicts
    are path-determined, so this equals the full walk's output as a set, and
    the CD sequence is byte-identical (all slice tokens are CI under the
    proof; unsafe tokens keep their relative DFS order in the rest-trie)."""
    # counting DFAs skip slicing too: the containment proof BFSes dense
    # trans rows, which are advisory-only on guarded (state, byte) cells
    if (trie.slices is not None and not getattr(dfa, "lazy", False)
            and not getattr(dfa, "counters", ())):
        q = _seed(dfa, remainder)[0]
        if _slice_contained(trie.slices, dfa, q, A, ignored, lexicons):
            rest = _walk_py_nodes(
                trie.slices.rest_nodes, dfa, remainder, A, ignored, priority, lexicons)
            return WalkResult(
                ci_tokens=trie.slices.min_ids + rest.ci_tokens,
                cd_entries=rest.cd_entries,
            )
    return _walk_py_nodes(trie.nodes, dfa, remainder, A, ignored, priority, lexicons)


def _walk_py_nodes(
    nodes,
    dfa: ScannerDFA,
    remainder: bytes,
    A: frozenset[int],
    ignored: frozenset[int],
    priority: dict[int, tuple[int, int]],
    lexicons: Lexicons | None = None,
) -> WalkResult:
    """The spec DFS body over an explicit node array (full trie or rest-trie)."""
    n = len(nodes)
    ci: list[int] = []
    cd: list[CDEntry] = []
    trans = dfa.trans
    accept = dfa.accept
    accepts_all = dfa.accepts_all
    live = dfa.live
    a_or_ign = A | ignored

    def partial_viable(seg: bytes, dfa_state: int, candidates: frozenset[int]) -> bool:
        for t in live[dfa_state] & candidates:
            if lexicons is None or t in ignored or lexicons.prefix_ok(t, seg):
                return True
        return False

    counting = bool(getattr(dfa, "counters", ()))
    zero = dfa.zero_counts() if counting else None
    root = _Frame(0, *_seed(dfa, remainder), events=(), n_real=0, cd_flag=False)
    stack: list[_Frame] = [root]

    i = 0
    while i < n:
        while len(stack) > 1 and i >= stack[-1].end:
            stack.pop()
        parent = stack[-1]
        word = int(nodes[i])
        byte = word & 0xFF
        tid = ((word >> 8) & 0xFFFFFF) - 1
        size = word >> 32

        # ---- incremental byte step with emission cascade -------------------
        cur, seg = parent.dfa_state, parent.seg
        last_len, last_state = parent.last_len, parent.last_state
        counts = parent.counts
        events = parent.events
        n_real = parent.n_real
        cd_flag = parent.cd_flag
        reject = False

        pending = [byte]
        idx = 0
        while idx < len(pending):
            b = pending[idx]
            idx += 1
            if counting:
                nx, counts2 = dfa.step(cur, counts, b)
            else:
                nx = trans[cur][b]
            if nx != DEAD:
                seg = seg + bytes([b])
                cur = nx
                if counting:
                    counts = counts2
                if accept[nx] != -1:
                    last_len, last_state = len(seg), nx
                continue
            # forced emission (maximal munch)
            if last_state == -1:
                reject = True
                break
            ev = EmissionEvent(accepts_all[last_state], last_len)
            lexeme = seg[:last_len]
            # classification per event (identical to v1 classify loop)
            if n_real == 0:
                pick = pick_viable(ev, lexeme, A, ignored, priority, lexicons)
                if pick is None:
                    reject = True
                    break
                if pick not in ignored:
                    n_real = 1
            else:
                pure_ignored = ev.candidates & ignored and not (ev.candidates - ignored)
                if not pure_ignored:
                    cd_flag = True
            events = events + ((ev, lexeme),)
            rest = seg[last_len:]
            pending[idx:idx] = list(rest)
            pending.insert(idx + len(rest), b)
            cur, seg, last_len, last_state = dfa.start, b"", 0, -1
            counts = zero

        if reject:
            i += size
            continue

        frame = _Frame(i + size, cur, seg, last_len, last_state, counts,
                       events, n_real, cd_flag)

        # ---- node verdict (identical semantics to v1 classify tail); the
        # n_real == 0 non-viable case also prunes the subtree (monotone) ------
        if n_real == 0:
            if seg and not partial_viable(seg, cur, a_or_ign):
                i += size  # no extension can recover
                continue
            verdict = CI
        elif frame.cd_flag or n_real >= 2:
            verdict = CD
        elif not seg:
            verdict = CI
        elif partial_viable(seg, cur, ignored):
            verdict = CI
        else:
            verdict = CD
        if tid >= 0:
            if verdict == CI:
                ci.append(tid)
            else:
                cd.append(CDEntry(
                    tid,
                    tuple(ev for ev, _lx in events),
                    tuple(lx for _ev, lx in events),
                    seg,
                ))

        stack.append(frame)
        i += 1

    return WalkResult(ci_tokens=tuple(ci), cd_entries=tuple(cd))
