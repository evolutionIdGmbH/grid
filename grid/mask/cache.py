"""E10/E11 MaskCache: T1 per-grammar cache with deterministic entry encoding.

Entry encoding (E10, cross-implementation determinism for G10 replay):
payload sizes compared as ``4*|accept|`` vs ``4*(V-|accept|)`` vs ``ceil(V/8)``;
ties broken accept-list < reject-list < bitset; token ids ascending;
``entry_id = BLAKE2b-128(canonical key bytes || tag || canonical payload)``.
Racing writers of one key produce the same entry_id -> publish is idempotent.

T2 (cross-grammar-family tier) is deferred to the M3+ serving work; the key
derivation here already separates the family-invariant components so T2 is a
second dict keyed by a suffix of the same tuple.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

from grid.trie.walk import CDEntry

TAG_ACCEPT, TAG_REJECT, TAG_BITSET = 0, 1, 2


def adaptive_encode(ci_tokens, vocab_size: int) -> tuple[int, bytes]:
    """Deterministic adaptive payload: (tag, canonical payload bytes).

    Vectorized (numpy) but BYTE-IDENTICAL to the original per-int loop — the
    payload is hashed into entry_id, so G10 audit replay depends on stability
    (tests/mask/test_adaptive_encode.py binds this against the loop as the
    reference implementation). ``ci_tokens`` may be any int sequence: tuple,
    list, or an int32/uint32 numpy array (the kernel-walk fast path).
    """
    ci = np.asarray(ci_tokens, dtype=np.uint32)
    n = ci.size
    size_accept = 4 * n
    size_reject = 4 * (vocab_size - n)
    size_bitset = (vocab_size + 7) // 8
    best = min((size_accept, TAG_ACCEPT), (size_reject, TAG_REJECT), (size_bitset, TAG_BITSET))
    tag = best[1]
    if tag == TAG_ACCEPT:
        # sorted() keeps duplicates; np.sort does too — identical bytes
        payload = np.sort(ci).astype("<u4", copy=False).tobytes()
    elif tag == TAG_REJECT:
        keep = np.zeros(vocab_size, dtype=bool)
        keep[ci] = True
        payload = np.nonzero(~keep)[0].astype("<u4").tobytes()
    else:
        bits = np.zeros(vocab_size, dtype=bool)
        bits[ci] = True
        # bitorder="little": bit t -> byte t>>3, bit t&7 (the loop's layout);
        # packbits pads the tail byte with zeros -> exactly ceil(V/8) bytes
        payload = np.packbits(bits, bitorder="little").tobytes()
    return tag, payload


@dataclass(frozen=True)
class CDGroup:
    """CD entries sharing one verdict-equivalence key — per event the
    lexeme_ok-filtered candidate set + the min-priority ignored pick, plus the
    tail (live set, prefix_ok-filtered allow set, ignored-viability) — get ONE
    stack-dependent verdict per step: the per-step check consumes an entry only
    through those finite predicates, so members are verdict-indistinguishable
    at every parser configuration (tests/mask/test_verdict_equivalence.py).
    Grouped once at publish time; the representative keeps real segment and
    remainder bytes for registration/audit payloads."""

    representative: CDEntry
    token_ids: tuple[int, ...]           # alias-expanded


@dataclass(frozen=True)
class MaskCacheEntry:
    entry_id: str
    key: tuple
    # kernel-walk entries carry a READ-ONLY int32 np.ndarray (immutable like
    # the tuple it replaces; len/iter/list/np.asarray/indexing all identical);
    # spec-path/seeded entries keep a tuple
    ci_tokens: tuple[int, ...]  # or np.ndarray[int32] on the kernel path
    cd_entries: tuple[CDEntry, ...]
    cd_groups: tuple[CDGroup, ...]
    origin: str  # SEEDED | COMPUTED
    # rust-walk entries: the kernel registration payload, verbatim from the
    # walk (derived data — not part of the entry_id hash); None on the spec path
    kernel_groups: tuple | None = None


def make_entry(key: tuple, ci, cd: tuple[CDEntry, ...], vocab_size: int,
               origin: str = "COMPUTED", live_of=None, lexicon_sensitive: bool = False,
               expand=None, precomputed_groups=None, lexicons=None, ignored=None,
               priority=None) -> MaskCacheEntry:
    tag, payload = adaptive_encode(ci, vocab_size)
    h = hashlib.blake2b(digest_size=16)
    h.update(repr(key).encode())
    h.update(bytes([tag]))
    h.update(payload)
    expand = expand or (lambda t: (t,))
    if precomputed_groups is not None:  # rust kernel: grouped + alias-expanded in-kernel
        cd_groups = tuple(CDGroup(rep, tuple(ids)) for rep, ids, _payload in precomputed_groups)
        return MaskCacheEntry(entry_id=h.hexdigest(), key=key, ci_tokens=ci, cd_entries=cd,
                              cd_groups=cd_groups, origin=origin,
                              kernel_groups=tuple(p for _r, _i, p in precomputed_groups))
    # Verdict-equivalence keying (mirrors grid_core walk_raw): the per-step CD
    # check (producer.check_context_dependent / kernel cd_groups_compute)
    # consumes an entry only through, per event, the lexeme_ok-filtered
    # candidate set and the min-priority ignored pick, and, for the tail, the
    # (live set, prefix_ok-filtered allow set, ignored-viability) triple.
    # Entries equal on those are verdict-indistinguishable at every parser
    # configuration, so they share one group. Requires the lexicons/ignored/
    # priority context; legacy callers without it keep the raw-bytes key.
    verdict_key = (lexicon_sensitive and lexicons is not None and live_of is not None
                   and ignored is not None and priority is not None)

    def _gkey(e: CDEntry) -> tuple:
        if not verdict_key:
            return (
                tuple(ev.candidates for ev in e.events),
                e.segments if lexicon_sensitive else None,
                e.remainder if lexicon_sensitive else None,
                live_of(e.remainder) if live_of is not None else e.remainder,
            )
        evkey = []
        for ev, seg in zip(e.events, e.segments, strict=True):
            cand_pass = frozenset(t for t in ev.candidates if lexicons.lexeme_ok(t, seg))
            ign = [t for t in ev.candidates if t in ignored]
            ign_pick = min(ign, key=lambda t: priority[t]) if ign else None
            evkey.append((cand_pass, ign_pick))
        if e.remainder:
            live = live_of(e.remainder)
            allow = frozenset(t for t in live if lexicons.prefix_ok(t, e.remainder))
            tail = (live, allow, bool(live & ignored))
        else:
            tail = None  # empty tail: verdict is always true
        return (tuple(evkey), tail)

    groups: dict[tuple, list[int]] = {}
    reps: dict[tuple, CDEntry] = {}
    for e in cd:
        gkey = _gkey(e)
        groups.setdefault(gkey, []).extend(expand(e.token_id))
        reps.setdefault(gkey, e)
    cd_groups = tuple(CDGroup(reps[k], tuple(v)) for k, v in groups.items())
    return MaskCacheEntry(entry_id=h.hexdigest(), key=key, ci_tokens=ci, cd_entries=cd,
                          cd_groups=cd_groups, origin=origin)


def _decode_blob_v1(blob: bytes) -> list:
    """Kernel v7 blob v1 -> the raw walk groups [(evs, segs, rem, ids)]
    (grid_core blob_encode's exact inverse; see lib.rs for the layout).
    Order-preserving — group order is part of the order-exact parity
    contract. Unknown version / trailing bytes are hard errors."""
    if not blob or blob[0] != 1:
        raise ValueError(f"unknown v7 blob version: {blob[:1]!r}")
    w, n_groups = struct.unpack_from("<II", blob, 1)
    off = 9
    out = []
    for _ in range(n_groups):
        (n_events,) = struct.unpack_from("<I", blob, off)
        off += 4
        evs, segs = [], []
        for _ in range(n_events):
            words = list(struct.unpack_from(f"<{w}Q", blob, off))
            off += 8 * w
            (ln,) = struct.unpack_from("<I", blob, off)
            off += 4
            segs.append(blob[off:off + ln])
            off += ln
            evs.append((words, ln))
        (rl,) = struct.unpack_from("<I", blob, off)
        off += 4
        rem = blob[off:off + rl]
        off += rl
        (ni,) = struct.unpack_from("<I", blob, off)
        off += 4
        ids = list(struct.unpack_from(f"<{ni}i", blob, off))
        off += 4 * ni
        out.append((evs, segs, rem, ids))
    if off != len(blob):
        raise ValueError("v7 blob trailing bytes")
    return out


class MaskEntryV7:
    """Kernel-v7 thin cache entry (duck-typed to MaskCacheEntry): the walk,
    group build, ci packing, adaptive encoding, entry-id hash and kernel
    registration all happened inside ONE GIL-released register_blob call —
    Python holds only (entry_id, key, tag, ci bytes, blob). ``blob`` is the
    kernel's own export payload: a foreign producer registers this entry
    Rust-to-Rust via register_blob (producer._ensure_handle), replacing the
    kernel_groups tuple path (``kernel_groups`` reads None by design).

    ``ci_tokens`` / ``cd_groups`` / ``cd_entries`` are lazy views for the
    parity/audit/test consumers (byte-identical to the WalkResult glue the
    eager path built); the serving path never decodes the blob. Lazy decode
    is race-benign: concurrent first touches compute identical values and
    the assignment is idempotent."""

    __slots__ = ("entry_id", "key", "origin", "tag", "ci_bytes", "blob",
                 "_ci", "_groups", "_reps")

    kernel_groups = None  # class attr: v7 entries have no verbatim payload

    def __init__(self, entry_id: str, key: tuple, tag: int, ci_bytes: bytes,
                 blob: bytes, origin: str = "COMPUTED") -> None:
        self.entry_id = entry_id
        self.key = key
        self.tag = tag
        self.ci_bytes = ci_bytes
        self.blob = blob
        self.origin = origin
        self._ci = None
        self._groups = None
        self._reps = None

    @property
    def ci_tokens(self):
        ci = self._ci
        if ci is None:
            # read-only view (frombuffer over bytes), like the kernel-walk path
            ci = np.frombuffer(self.ci_bytes, dtype=np.int32)
            self._ci = ci
        return ci

    def _decode(self):
        """Blob -> (CDGroup tuple, representative CDEntry tuple), exactly the
        WalkResult glue in grid/trie/walk.py (byte-identical reps)."""
        from grid.lexer.run import EmissionEvent
        from grid.trie.walk import _unmask, _words_int

        reps, groups = [], []
        for evs, segs, rem, ids in _decode_blob_v1(self.blob):
            rep = CDEntry(
                ids[0],
                tuple(EmissionEvent(_unmask(_words_int(c)), int(ln)) for c, ln in evs),
                tuple(segs),
                rem,
            )
            reps.append(rep)
            groups.append(CDGroup(rep, tuple(ids)))
        self._reps = tuple(reps)
        self._groups = tuple(groups)

    @property
    def cd_groups(self) -> tuple[CDGroup, ...]:
        if self._groups is None:
            self._decode()
        return self._groups

    @property
    def cd_entries(self) -> tuple[CDEntry, ...]:
        if self._reps is None:
            self._decode()
        return self._reps


class MaskCacheT2:
    """T2: the cross-family tier (DESIGN §E10). One instance per (dialect
    fingerprint, tokenizer trie) — the registry scopes it, exactly as the
    grammar fingerprint scopes a T1 instance. Producers of DIFFERENT schemas
    (and role projections — terminals are numbered at L1 freeze; projections
    subset productions, never renumber) share it: generic keys carry no schema
    fingerprint by construction and identifier-position keys carry it (E11),
    so cross-template hits are exactly the schema-independent entries and
    OBL-KEY1 holds by the same key refinement T1 relies on. The load-bearing
    case (G8 adversarial): a fresh schema's template starts with a cold T1,
    but the literal-interior giants (10k+-id entries, 100-500 ms to rebuild)
    are generic — T2 hands them over and only the schema's own identifier
    entries (lexicon-bounded, small) build cold.

    Flow per the design: T1 miss → T2 get (hit is copied into T1 by the
    caller) → walk; publish T1 sync and T2 sync (a dict store — the async
    publish in the design targets out-of-process T2 backends). Bounded FIFO;
    supersession is expressed by the serving registry as NEW fingerprints
    (never in-place mutation), so the rollover pointer-swap tier is not
    needed in-process."""

    def __init__(self, cap: int = 100_000) -> None:
        self._map: dict[tuple, MaskCacheEntry] = {}
        self._cap = cap
        self.hits = 0

    def get(self, key: tuple) -> MaskCacheEntry | None:
        e = self._map.get(key)
        if e is not None:
            self.hits += 1
        return e

    def publish(self, entry: MaskCacheEntry) -> None:
        cur = self._map.get(entry.key)
        if cur is not None:
            assert cur.entry_id == entry.entry_id, "OBL-KEY1 violation: same key, different mask"
            return
        if len(self._map) >= self._cap:
            self._map.pop(next(iter(self._map)))
        self._map[entry.key] = entry


class MaskCache:
    """T1: per-CompiledGrammar dict; publish is idempotent by content hash."""

    def __init__(self) -> None:
        self._t1: dict[tuple, MaskCacheEntry] = {}
        self.hits = 0
        self.misses = 0
        self.epoch = 0  # namespace generation; consumers drop entry aliases on change

    def get(self, key: tuple) -> MaskCacheEntry | None:
        e = self._t1.get(key)
        if e is None:
            self.misses += 1
        else:
            self.hits += 1
        return e

    def peek(self, key: tuple) -> MaskCacheEntry | None:
        """Uncounted lookup for lookaside paths that account hits themselves."""
        return self._t1.get(key)

    def publish(self, entry: MaskCacheEntry) -> MaskCacheEntry:
        cur = self._t1.get(entry.key)
        if cur is not None:
            assert cur.entry_id == entry.entry_id, "OBL-KEY1 violation: same key, different mask"
            return cur
        self._t1[entry.key] = entry
        return entry

    def invalidate_namespace(self) -> None:
        """E10 rollover: entries are never deleted in place; the namespace pointer moves."""
        self._t1 = {}
        self.epoch += 1
