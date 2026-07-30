"""Counting-set component construction for {m,n} windows (P4 phase 1).

Behind GRID_PERF_COUNTING (perf_flags.counting_enabled, default off), eligible
counted loops in ONE terminal's regex keep a counted-loop NFA node instead of
the O(n) parse-time _expand_repeat unrolling, and determinize to a counting
automaton: O(1) control states per window plus a bounded counter. The result
is a CountingTerminalDFA — a ScannerComponent (factored.py Protocol) sibling
of TerminalDFA whose count-dependent transitions live in per-(state, class)
variant tables; the lazy product composes them with a global counter table
(per-component cid offsets) and the runtime scan state becomes
(state, counts) via ScannerDFA.step/scan_full.

This is the held COUNTING_WINDOWS runtime surface (worktree wf_12480d7a-e5d-1,
commits d69246a+044dad1: eligibility lowering, guarded-eps loop wiring,
annotated-closure determinization with per-seed-set transition memoization)
ported PER TERMINAL onto the component seam. The held combined construction
(_build_scanner_counting over the union NFA) is deliberately discarded — the
BAKEOFF verdict subsumed it under FACTORED — and its hardest machinery
collapses here: the cross-terminal _CountingFallback/rebuild loop reduces to
"this terminal's component falls back to plain expansion" (build_counting_
component returns None), because counter ids are owned per terminal and
per-terminal sub-NFAs are disjoint — the same disjointness that makes
LazyProductDFA state-for-state equal to build_scanner.

Eligibility (verbatim from the held _lower_reps): the rep is not nested under
a quantifier (one-shot per scan), its body is rep-free, non-epsilon and a
PREFIX CODE (unique iteration count: any input decomposes uniquely into
complete iterations plus a partial word), span >= _COUNTING_MIN_SPAN, and a
non-nullable continuation follows the loop somewhere (the loop exit cannot
eps-reach the terminal accept, keeping accept/accepts_all/live exact —
count-independent — per control state). Ineligible reps expand exactly as the
legacy parse-time expansion would — never wrong, only slower.

Two deltas against the held eager build, both load-bearing:

- co_acc comes from a geps-AWARE reverse reachability (_counting_reach), not
  the variant-graph live fixpoint: nfa._terminal_reach is blind to guarded
  eps edges, so a window component's live bits would under-report (all-zombie
  loop states) and forced emissions would fire a byte early. Guard optimism
  is exact here, not an approximation: eligibility makes accept reachability
  count-independent (from any reachable count the loop can iterate to m and
  exit — saturation keeps counts <= cap and m <= cap).
- guard variant tables INCLUDE the count regions where the merged control set
  dies as explicit (conds, DEAD, ()) boxes (the held build skipped them: in a
  single automaton an uncovered region just falls through to DEAD). The
  product needs the full box partition, because a member component dying
  under some counts only DROPS it from the product tuple — the product state
  lives on and the member's counters reset to canonical 0.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from grid.lexer.nfa import _NFABuilder
from grid.lexer.rx import _expand_repeat, _Node, _parse_regex
from grid.lexer.subset import DEAD, byte_classes, edges_by_class

# Windows below this span expand as before (expansion is cheap there); larger
# eligible windows keep a counted-loop NFA node.
_COUNTING_MIN_SPAN = 8
_MAX_COUNTERS = 8       # kernel v8 frame budget; excess loops expand (largest span wins)
_BODY_DFA_CAP = 512     # loop bodies are tiny; anything larger expands


class _CountingFallback(Exception):
    """Build signal: a counted loop cannot keep exact per-control-state
    knowledge — the terminal falls back to plain expansion (the caller
    returns None; no cross-terminal rebuild loop exists on the component
    path)."""


def _nullable(node: _Node) -> bool:
    k = node.kind
    if k == "eps":
        return True
    if k in ("char", "class"):
        return False
    if k in ("star", "opt"):
        return True
    if k == "plus":
        return _nullable(node.kids[0])
    if k == "cat":
        return all(_nullable(kid) for kid in node.kids)
    if k == "alt":
        return any(_nullable(kid) for kid in node.kids)
    if k == "rep":
        assert node.bounds is not None
        return node.bounds[0] == 0 or _nullable(node.kids[0])
    raise AssertionError(k)  # pragma: no cover


def _rep_body_ok(node: _Node) -> bool:
    """Counted-loop body admissibility: non-empty language that is a PREFIX
    CODE — every accepting state of the body's standalone DFA has zero
    out-transitions. Then any input decomposes uniquely into complete
    iterations plus a partial word, so one counter value per configuration
    is exact (mid-word and word-boundary occupancies never coexist)."""
    b = _NFABuilder()
    s, a = b.build(node)

    def closure(states) -> frozenset[int]:
        seen = set(states)
        stack = list(states)
        while stack:
            st = stack.pop()
            for nx in b.eps.get(st, ()):
                if nx not in seen:
                    seen.add(nx)
                    stack.append(nx)
        return frozenset(seen)

    start = closure({s})
    ids = {start}
    work = [start]
    found_accept = False
    while work:
        cur = work.pop()
        moves: dict[int, set[int]] = {}
        for st in cur:
            for chars, dst in b.edges.get(st, ()):
                for c in chars:
                    moves.setdefault(c, set()).add(dst)
        if a in cur:
            found_accept = True
            if moves:
                return False
        for _c, dsts in moves.items():
            nxt = closure(dsts)
            if nxt not in ids:
                if len(ids) > _BODY_DFA_CAP:
                    return False
                ids.add(nxt)
                work.append(nxt)
    return found_accept


def _lower_reps(node: _Node, under_quant: bool, follow_nonnull: bool,
                kept: list[_Node]) -> _Node:
    """Expand every rep that cannot be a counting-set loop; keep the rest.

    Eligible: not nested under a quantifier (one-shot per scan), body rep-free
    (nested reps expand first), body non-epsilon and a prefix code (unique
    iteration count), span >= _COUNTING_MIN_SPAN, and a non-nullable
    continuation somewhere after the loop (the loop exit cannot eps-reach the
    terminal accept, keeping accept/accepts_all/live counter-independent).
    Ineligible reps expand exactly as the legacy parse-time expansion would —
    never wrong, only slower."""
    k = node.kind
    if k in ("char", "class", "eps"):
        return node
    if k in ("star", "plus", "opt"):
        return _Node(k, kids=(_lower_reps(node.kids[0], True, follow_nonnull, kept),))
    if k == "alt":
        return _Node("alt", kids=tuple(
            _lower_reps(kid, under_quant, follow_nonnull, kept) for kid in node.kids))
    if k == "cat":
        kids = node.kids
        out = []
        for i, kid in enumerate(kids):
            fn = follow_nonnull or any(not _nullable(kids[j]) for j in range(i + 1, len(kids)))
            out.append(_lower_reps(kid, under_quant, fn, kept))
        return _Node("cat", kids=tuple(out))
    if k == "rep":
        assert node.bounds is not None
        m, n = node.bounds
        body = _lower_reps(node.kids[0], True, follow_nonnull, kept)
        span = n if n is not None else m
        if (under_quant or not follow_nonnull or span < _COUNTING_MIN_SPAN
                or _nullable(body) or not _rep_body_ok(body)):
            return _expand_repeat(body, m, n)
        rep = _Node("rep", kids=(body,), bounds=(m, n))
        kept.append(rep)
        return rep
    raise AssertionError(k)  # pragma: no cover


class _CountingNFABuilder(_NFABuilder):
    """_NFABuilder plus guard/op-annotated eps edges for kept counted loops.

    ``eps`` stays plain (the inherited traversals are reused untouched);
    guarded edges live in ``geps``: a -> [(b, guard, inc)] with guard =
    (cid, "iter"|"exit") and inc = cid or None. Loop wiring for counter cid:
    entry -eps-> L, L -CAN_ITER-> body.start, body.accept -INC-> L,
    L -CAN_EXIT-> exit; counts start at 0 per scan."""

    def __init__(self) -> None:
        super().__init__()
        self.geps: dict[int, list[tuple[int, tuple[int, str] | None, int | None]]] = {}
        self.counters: list[tuple[int, int | None]] = []
        self.boundary: list[int] = []       # per counter: its L state
        self.regions: list[set[int]] = []   # per counter: body states + L

    def add_geps(self, a: int, dst: int, guard: tuple[int, str] | None, inc: int | None) -> None:
        self.geps.setdefault(a, []).append((dst, guard, inc))

    def build(self, node: _Node) -> tuple[int, int]:
        if node.kind == "rep":
            assert node.bounds is not None
            m, n = node.bounds
            cid = len(self.counters)
            self.counters.append((m, n))
            s, a = self.new(), self.new()
            boundary = self.new()
            self.add_eps(s, boundary)
            lo_mark = self.n
            ks, ka = self.build(node.kids[0])
            region = set(range(lo_mark, self.n))
            region.add(boundary)
            self.add_geps(boundary, ks, (cid, "iter"), None)
            self.add_geps(ka, boundary, None, cid)
            self.add_geps(boundary, a, (cid, "exit"), None)
            self.boundary.append(boundary)
            self.regions.append(region)
            return s, a
        return super().build(node)


def _counting_reach(b: _CountingNFABuilder, accept: int) -> list[bool]:
    """reach[q] = the terminal accept is reachable from NFA state q via
    eps/geps/non-empty byte edges (>= 0 bytes) — nfa._terminal_reach with
    guarded eps edges treated as reachable. Guard optimism is EXACT per
    control state under the eligibility rules (module docstring): iterate
    guards fail only above n-1 and exit guards only below m, and from any
    reachable count the loop can iterate (saturating) to m and exit — so
    structural reachability equals reachability for every count value.
    Blindness to geps here is the known outcome-changing bug class: live
    would under-report and forced emissions would fire a byte early."""
    rev: list[list[int]] = [[] for _ in range(b.n)]
    for src, dsts in b.eps.items():
        for dst in dsts:
            rev[dst].append(src)
    for src, gedges in b.geps.items():
        for dst, _guard, _inc in gedges:
            rev[dst].append(src)
    for src, edges in b.edges.items():
        for chars, dst in edges:
            if not chars:
                continue
            rev[dst].append(src)
    reach = [False] * b.n
    stack = [accept]
    while stack:
        q = stack.pop()
        if reach[q]:
            continue
        reach[q] = True
        stack.extend(rev[q])
    return reach


@dataclass(frozen=True)
class CountingTerminalDFA:
    """One terminal's counting byte DFA (ScannerComponent sibling of
    factored.TerminalDFA; start state 0).

    - ``trans[s][class_of[byte]]``: successor CONTROL state or DEAD. On
      classes with a guard_rows entry the value is advisory only (the
      largest-control-set variant, the held dense-row convention) — count-
      aware callers must go through ``step_counts``; ``step`` asserts.
    - ``accepting``/``co_acc``/``matches_empty``: as TerminalDFA, exact per
      control state (eligibility makes them count-independent; co_acc is
      geps-aware — _counting_reach).
    - ``counters[k] = (m, n)`` with n=-1 for open {m,} — component-LOCAL
      counter ids; the product offsets them into its global table.
    - ``guard_rows[(state, cls)]``: ordered variants ``(conds, dst, ops)``
      over the pre-step counts — conds ``((cid, lo, hi), ...)``, ops
      ``((cid, op), ...)`` with op 1 = saturating increment, 2 = reset. The
      variant boxes cover the whole count space (DEAD boxes explicit, see
      module docstring), so exactly one variant matches any counts vector.
    - ``occupied[s]``: local cids whose loop region is live (has byte edges)
      inside control state ``s`` — the counters whose value is meaningful at
      ``s``; the product resets exactly these when the member drops.
    """

    trans: tuple[tuple[int, ...], ...]
    class_of: tuple[int, ...]
    accepting: tuple[bool, ...]
    co_acc: tuple[bool, ...]
    matches_empty: bool
    counters: tuple[tuple[int, int], ...]
    guard_rows: dict = field(hash=False)          # in __eq__, out of __hash__
    occupied: tuple[frozenset[int], ...]

    def step(self, state: int, cls: int) -> int:
        # the ScannerComponent protocol step is count-blind; consumers must
        # dispatch counting components to step_counts (caching a single dst
        # for a guarded transition is the forbidden wrong-mask class)
        assert (state, cls) not in self.guard_rows, \
            "counting component: guarded transition needs step_counts()"
        return self.trans[state][cls]

    def zero_counts(self) -> tuple[int, ...]:
        return (0,) * len(self.counters)

    def step_counts(
        self, state: int, counts: tuple[int, ...], cls: int,
    ) -> tuple[int, tuple[int, ...]]:
        """One counter-aware class step: (state', counts'). Component-local
        cids; mirrors ScannerDFA.step byte-for-byte modulo class keying."""
        var = self.guard_rows.get((state, cls))
        if var is None:
            return self.trans[state][cls], counts
        for conds, dst, ops in var:
            ok = True
            for cid, lo, hi in conds:
                c = counts[cid]
                if c < lo or c > hi:
                    ok = False
                    break
            if not ok:
                continue
            if dst == DEAD:
                return DEAD, counts
            if ops:
                lst = list(counts)
                for cid, op in ops:
                    if op == 1:
                        m, n = self.counters[cid]
                        cap = n if n >= 0 else m
                        v = lst[cid] + 1
                        lst[cid] = cap if v > cap else v
                    else:
                        lst[cid] = 0
                counts = tuple(lst)
            return dst, counts
        return DEAD, counts  # unreachable: variant boxes cover the count space


def build_counting_component(pattern: str, budget: int | None) -> CountingTerminalDFA | None:
    """Counting subset construction for ONE terminal's regex; None when the
    terminal has no eligible loops, more of them than _MAX_COUNTERS, breaches
    ``budget`` control states, or trips a _CountingFallback — the caller then
    uses the plain component (factored._component), which is exactly the
    flag-off artifact."""
    kept: list[_Node] = []
    node = _lower_reps(_parse_regex(pattern, keep_reps=True), False, False, kept)
    if not kept:
        return None
    if len(kept) > _MAX_COUNTERS:
        # one terminal alone over the kernel frame budget: expand wholesale
        # (the held build kept the largest spans WITHIN the terminal; at
        # component granularity partial keeps would need pattern-external
        # memo keys — deliberate simplification, recorded)
        return None
    try:
        return _build_counting(node, budget)
    except _CountingFallback:
        return None


def _build_counting(node: _Node, budget: int | None) -> CountingTerminalDFA:
    # ---- guarded single-terminal NFA ----------------------------------------
    b = _CountingNFABuilder()
    s0, accept = b.build(node)

    counters = b.counters
    nc = len(counters)
    caps = [n if n is not None else m for m, n in counters]
    regions = b.regions
    boundary = b.boundary
    has_byte = {st for st, edges in b.edges.items() if any(chars for chars, _d in edges)}

    def compose(ann: dict, guard: tuple[int, str] | None, inc: int | None) -> dict | None:
        """Fold one guarded eps edge into a derivation annotation: per counter
        (lo, hi, inc) constrains the PRE-STEP count; guards after an INC see
        c + 1 (saturation at the cap changes nothing inside the domain)."""
        ann2 = dict(ann)
        if inc is not None:
            lo, hi, ic = ann2.get(inc, (0, caps[inc], 0))
            if ic:  # two INCs in one byte step: body iterable via eps (not ours)
                raise _CountingFallback
            ann2[inc] = (lo, hi, 1)
        if guard is not None:
            cid, kind = guard
            lo, hi, ic = ann2.get(cid, (0, caps[cid], 0))
            m, n = counters[cid]
            if kind == "iter":
                if n is not None:
                    hi = min(hi, n - 1 - ic)
            else:
                lo = max(lo, m - ic)
            if lo < 0:
                lo = 0
            if lo > hi:
                return None  # infeasible derivation
            ann2[cid] = (lo, hi, ic)
        return ann2

    # per-NFA-state annotated eps closure, memoized: {reached: annotation};
    # None annotation table means every reached annotation is empty (fast path)
    closures: dict[int, tuple[frozenset[int], dict[int, dict] | None]] = {}

    def closure_of(v: int) -> tuple[frozenset[int], dict[int, dict] | None]:
        got = closures.get(v)
        if got is None:
            anns: dict[int, dict] = {}
            stack: list[tuple[int, dict]] = [(v, {})]
            while stack:
                st, ann = stack.pop()
                prev = anns.get(st)
                if prev is not None:
                    if prev != ann:  # two derivations disagree: not representable
                        raise _CountingFallback
                    continue
                anns[st] = ann
                for nx in b.eps.get(st, ()):
                    stack.append((nx, ann))
                for nx, guard, inc in b.geps.get(st, ()):
                    ann2 = compose(ann, guard, inc)
                    if ann2 is not None:
                        stack.append((nx, ann2))
            plain = not any(anns.values())
            got = (frozenset(anns), None if plain else anns)
            closures[v] = got
        return got

    def merge_closures(seeds) -> dict[int, dict]:
        merged: dict[int, dict] = {}
        for v in sorted(seeds):
            cs, canns = closure_of(v)
            if canns is None:
                for st in sorted(cs):
                    prev = merged.get(st)
                    if prev is None:
                        merged[st] = {}
                    elif prev:
                        raise _CountingFallback
            else:
                for st, ann in canns.items():
                    prev = merged.get(st)
                    if prev is None:
                        merged[st] = ann
                    elif prev != ann:
                        raise _CountingFallback
        return merged

    # alphabet compression + per-class edge index: the shared subset helpers
    blocks, class_of = byte_classes(b.edges)
    n_classes = len(blocks)
    edge_by_class = edges_by_class(b.edges, class_of, n_classes)

    def region_live(cid: int, states) -> bool:
        region = regions[cid]
        return any(st in region and st in has_byte for st in states)

    # ---- start control set: resolve guards at the zero count vector ---------
    merged0 = merge_closures([s0])
    sel0: list[int] = []
    for st, ann in merged0.items():
        ok = True
        for _cid, (lo, hi, ic) in ann.items():
            if ic:  # INC inside the start closure: body nullable (not ours)
                raise _CountingFallback
            if not lo <= 0 <= hi:
                ok = False
                break
        if ok:
            sel0.append(st)
    start_set = frozenset(sel0)

    # A transition's guard structure is a function of the SEED SET alone
    # (annotations, variant intervals, destination sets, region occupancy and
    # INC flags); control states reuse the same seed sets heavily, so the
    # expensive per-seeds work is memoized and only the cur-dependent finish
    # (alive counters, fresh-entry aliasing, ops assembly) runs per transition.
    # Cached shapes: ("plain", dst_set, occupied_cids) for guard-free
    # transitions, ("var", variants, fresh_cids) with variants =
    # ((conds, dst_set, occ_inc: {cid: inc_flag}, |sel|), ...) otherwise —
    # dst_set empty = DEAD under those counts (kept explicit for the product).
    trans_memo: dict[frozenset[int], tuple] = {}

    def transition_of(fseeds: frozenset[int]) -> tuple:
        got = trans_memo.get(fseeds)
        if got is not None:
            return got
        # plain fast path: no seed closure touches a guarded edge, so the
        # merged annotation table is all-empty by construction — take the
        # legacy-cost frozenset union instead of per-state dict merging
        plain = True
        union: frozenset[int] = frozenset()
        for v in fseeds:
            cs, canns = closure_of(v)
            if canns is not None:
                plain = False
                break
            union |= cs
        if plain:
            occ = frozenset(cid for cid in range(nc) if region_live(cid, union))
            got = ("plain", union, occ)
            trans_memo[fseeds] = got
            return got
        merged = merge_closures(fseeds)
        mentioned = sorted({cid for ann in merged.values() for cid in ann})
        if not mentioned:
            dst_set = frozenset(merged)
            occ = frozenset(cid for cid in range(nc) if region_live(cid, dst_set))
            got = ("plain", dst_set, occ)
            trans_memo[fseeds] = got
            return got
        # fresh loop entry (boundary L reached WITHOUT its own counter
        # constraint) is an aliasing hazard only when the same loop is already
        # running in cur — record the cids, checked per use site
        fresh = tuple(cid for cid in mentioned
                      if boundary[cid] in merged and cid not in merged[boundary[cid]])
        # count-dependent transition: one successor per distinct guard
        # valuation region (<= 4 regions per mentioned counter)
        domains: list[list[tuple[int, int]]] = []
        for cid in mentioned:
            pts = {0, caps[cid] + 1}
            for ann in merged.values():
                e = ann.get(cid)
                if e is not None:
                    pts.add(e[0])
                    pts.add(e[1] + 1)
            spts = sorted(p for p in pts if 0 <= p <= caps[cid] + 1)
            domains.append([(spts[j], spts[j + 1] - 1) for j in range(len(spts) - 1)])
        variants: list[tuple[tuple, frozenset[int], dict[int, int], int]] = []
        for combo in itertools.product(*domains):
            c0 = {cid: iv[0] for cid, iv in zip(mentioned, combo, strict=True)}
            sel = [st for st, ann in merged.items()
                   if all(lo <= c0[cid] <= hi for cid, (lo, hi, _ic) in ann.items())]
            conds = tuple((cid, iv[0], iv[1])
                          for cid, iv in zip(mentioned, combo, strict=True))
            if not sel:
                # dead under these counts — explicit for product composition
                variants.append((conds, frozenset(), {}, 0))
                continue
            occ_inc: dict[int, int] = {}
            for cid in range(nc):
                region = regions[cid]
                rstates = [st for st in sel if st in region and st in has_byte]
                if rstates:
                    incs = {merged[st].get(cid, (0, 0, 0))[2] for st in rstates}
                    if len(incs) > 1:  # mixed mid-word/boundary counts (not ours)
                        raise _CountingFallback
                    occ_inc[cid] = incs.pop()
            variants.append((conds, frozenset(sel), occ_inc, len(sel)))
        got = ("var", tuple(variants), fresh)
        trans_memo[fseeds] = got
        return got

    def intern(dst_set: frozenset[int]) -> int:
        dst_id = ids.get(dst_set)
        if dst_id is None:
            dst_id = ids[dst_set] = len(order)
            order.append(dst_set)
        return dst_id

    ids: dict[frozenset[int], int] = {start_set: 0}
    order = [start_set]
    trans: list[list[int]] = []
    accepting: list[bool] = []
    occupied: list[frozenset[int]] = []
    guard_rows: dict[tuple[int, int], tuple] = {}
    i = 0
    while i < len(order):
        if budget is not None and len(order) > budget:
            raise _CountingFallback  # control-state budget: plain path decides
        cur = order[i]
        sid = i
        i += 1
        alive = [cid for cid in range(nc) if region_live(cid, cur)]
        by_class: dict[int, set[int]] = {}
        for st in cur:
            per = edge_by_class.get(st)
            if per is None:
                continue
            for cl, dsts in enumerate(per):
                if dsts is not None:
                    by_class.setdefault(cl, set()).update(dsts)
        row = [DEAD] * n_classes
        for cl, seeds in sorted(by_class.items()):
            outcome = transition_of(frozenset(seeds))
            if outcome[0] == "plain":
                _tag, dst_set, occ = outcome
                dead_cids = [cid for cid in alive if cid not in occ]
                dst_id = intern(dst_set)
                row[cl] = dst_id
                if dead_cids:  # loop left the control set: counts reset to canonical 0
                    guard_rows[(sid, cl)] = (
                        (tuple(), dst_id, tuple((cid, 2) for cid in dead_cids)),)
                continue
            _tag, cvariants, fresh = outcome
            for cid in fresh:
                # fresh loop entry while the same loop is already running: the
                # single stored count cannot represent both hypotheses
                if region_live(cid, cur):
                    raise _CountingFallback
            variants_out: list[tuple[tuple, int, tuple]] = []
            best: tuple[int, int] | None = None  # (size, dst_id) for the dense row
            for conds, dst_set, occ_inc, n_sel in cvariants:
                if not dst_set:
                    variants_out.append((conds, DEAD, ()))
                    continue
                ops: list[tuple[int, int]] = []
                for cid in alive:
                    flag = occ_inc.get(cid)
                    if flag is None:
                        ops.append((cid, 2))
                    elif flag == 1:
                        ops.append((cid, 1))
                dst_id = intern(dst_set)
                variants_out.append((conds, dst_id, tuple(ops)))
                if best is None or n_sel > best[0]:
                    best = (n_sel, dst_id)
            if variants_out:
                # dense row: advisory only for guarded classes (counter-aware
                # consumers go through step_counts; step() asserts)
                assert best is not None, "merged annotations must be feasible somewhere"
                row[cl] = best[1]
                guard_rows[(sid, cl)] = tuple(variants_out)
        trans.append(row)
        accepting.append(accept in cur)
        occupied.append(frozenset(alive))

    # geps-aware co-accessibility (module docstring: the _terminal_reach
    # port would under-report through guarded edges)
    reach = _counting_reach(b, accept)
    co_acc = [any(reach[q] for q in subset) for subset in order]

    return CountingTerminalDFA(
        trans=tuple(tuple(r) for r in trans),
        class_of=tuple(class_of),
        accepting=tuple(accepting),
        co_acc=tuple(co_acc),
        matches_empty=accepting[0],
        counters=tuple((m, n if n is not None else -1) for m, n in counters),
        guard_rows=guard_rows,
        occupied=tuple(occupied),
    )
