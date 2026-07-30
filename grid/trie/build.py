"""E5 TokenTrie in the final artifact format (DESIGN.md SS2).

One numpy uint64 array of DFS-contiguous nodes plus nothing else — exactly the
buffer grid_core consumes zero-copy at M4. Node packing (8 bytes)::

    bits  0..7   byte value on the edge into this node
    bits  8..31  token_id + 1 ending exactly at this node (0 = none)
    bits 32..63  subtree size in nodes (self included) -> DFS sibling skip

The root is virtual: the array is the concatenation of the top-level subtrees.
Special tokens (E6) are excluded — EOS enters masks only via SS6 step 7's union.

S2 slicer (GRID_PERF_SLICER=1): the vocabulary is additionally partitioned by
the JSON-string-safe byte class into ``TrieSlices`` — one precomputed sorted
id array for the tokens spelled entirely inside the class (96% of Llama-3.1)
plus a rest-trie over the remaining tokens, in the same DFS format. The walk
(grid/trie/walk.py, grid_core walk_auto) proves slice containment structurally
per configuration and, on success, skips the sliced tokens' subtrees entirely:
``ci = merge(slice ids, rest walk)`` with CD groups only from the rest-trie.
Alias groups land wholly on one side because assignment is by spelling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid import perf_flags
from grid.errors import TrieBuildError

# The slice byte class: bytes legal inside a JSON string body — everything
# except '"' (0x22), '\\' (0x5C) and the C0 controls (0x00-0x1F). Exactly the
# [^"\\\x00-\x1f] class of grid.jsonschema.compiler.STRING_RX, so plain
# STRING-interior scanner states loop on every class byte and the containment
# proof can fire; bounded/pattern-constrained string terminals go DEAD on some
# class byte and fall back to the full walk (proof-or-walk, never wrong).
JSON_STRING_SAFE = frozenset(
    b for b in range(256) if b >= 0x20 and b != 0x22 and b != 0x5C
)

# Slice construction is skipped (slices=None, logged) below this fraction of
# sliceable tokens: a vocab that mostly fails the spelling test would pay the
# proof + rest-walk plumbing for no skippable mass (SQL-like grammars still
# see slices=None per-walk fallback semantics either way).
MIN_COVERAGE = 0.5


@dataclass(frozen=True)
class TrieSlices:
    """S2 slicer tables, built once per tokenizer alongside the trie.

    - ``class_bytes``: the slice byte class (spelling test + closure alphabet).
    - ``class_words``: the same class as a 256-bit bitmap, 4 little-endian u64
      words (the kernel upload format).
    - ``min_ids``: sorted node token ids (smallest alias per spelling) of the
      sliced tokens — the Python-spec walk's ci contribution, mirroring the
      per-node ids ``_walk_py`` emits (consumers alias-expand downstream).
    - ``ids``: sorted alias-COMPLETE int32 array of every sliced token id —
      the kernel's ci contribution (kernel ci is alias-expanded in-kernel).
    - ``rest_nodes``: DFS uint64 node array over the non-sliced tokens only,
      exact same packing as TokenTrie.nodes; walked with the standard walk.
    - ``coverage``: sliced fraction of the vocabulary (build-time log line).
    """

    class_bytes: frozenset[int]
    class_words: tuple[int, int, int, int]
    min_ids: tuple[int, ...]
    ids: np.ndarray            # int32, sorted, alias-complete
    rest_nodes: np.ndarray     # uint64, DFS-contiguous
    coverage: float


@dataclass(frozen=True)
class TokenTrie:
    nodes: np.ndarray          # uint64, DFS-contiguous
    n_tokens: int
    tokenizer_fingerprint: str
    # tokens with byte-identical spellings: node carries the smallest id; the mask
    # must include every alias (completeness — a mask over ids, not spellings)
    aliases: dict[int, tuple[int, ...]] = None  # type: ignore[assignment]
    slices: TrieSlices | None = None  # S2 slicer tables (GRID_PERF_SLICER=1)

    @staticmethod
    def unpack(word: int) -> tuple[int, int, int]:
        """-> (edge_byte, token_id or -1, subtree_size)"""
        return int(word & 0xFF), int(((word >> 8) & 0xFFFFFF) - 1), int(word >> 32)

    def expand(self, token_id: int) -> tuple[int, ...]:
        return self.aliases.get(token_id, (token_id,))


def _dfs_words(entries: list[tuple[bytes, int]]) -> list[int]:
    """entries [(spelling, tid)] -> packed DFS node words (the format above).
    Byte-identical to the historical inline build: nested insert keeping the
    smallest tid per exact spelling, children emitted in ascending byte order."""
    root: dict[int, list] = {}
    for bs, tid in entries:
        cur = root
        for i, byte in enumerate(bs):
            node = cur.setdefault(byte, [-1, {}])
            if i == len(bs) - 1:
                if node[0] == -1 or tid < node[0]:
                    node[0] = tid
            cur = node[1]

    words: list[int] = []

    def emit(byte: int, node: list) -> int:
        """DFS-emit; returns subtree size."""
        my_index = len(words)
        words.append(0)  # placeholder
        size = 1
        for b in sorted(node[1]):
            size += emit(b, node[1][b])
        tid = node[0]
        words[my_index] = (size << 32) | (((tid + 1) & 0xFFFFFF) << 8) | byte
        return size

    for b in sorted(root):
        emit(b, root[b])
    return words


def _build_slices(entries: list[tuple[bytes, int]]) -> TrieSlices | None:
    """Partition entries by JSON_STRING_SAFE spelling -> TrieSlices, or None
    when coverage is too low to be worth the proof machinery (logged) or the
    partition is degenerate (no rest tokens would mean an empty rest-trie —
    kept anyway; no sliced tokens means nothing to skip)."""
    safe: list[tuple[bytes, int]] = []
    rest: list[tuple[bytes, int]] = []
    for bs, tid in entries:
        (safe if all(b in JSON_STRING_SAFE for b in bs) else rest).append((bs, tid))
    coverage = len(safe) / len(entries)
    if coverage < MIN_COVERAGE or not safe:
        import logging

        logging.getLogger("grid.trie").info(
            "slicer: coverage %.1f%% below %.0f%% - slices skipped",
            100 * coverage, 100 * MIN_COVERAGE)
        return None
    words = [0, 0, 0, 0]
    for b in JSON_STRING_SAFE:
        words[b >> 6] |= 1 << (b & 63)
    # min_ids: the trie-node ids (smallest alias per spelling) — what the
    # full walk's per-node tid field would emit; ids: every alias, the
    # kernel's post-expansion form. Both sorted (merge order contract).
    by_bytes: dict[bytes, int] = {}
    for bs, tid in safe:
        cur = by_bytes.get(bs)
        if cur is None or tid < cur:
            by_bytes[bs] = tid
    return TrieSlices(
        class_bytes=JSON_STRING_SAFE,
        class_words=tuple(words),
        min_ids=tuple(sorted(by_bytes.values())),
        ids=np.array(sorted(tid for _bs, tid in safe), dtype=np.int32),
        rest_nodes=np.array(_dfs_words(rest), dtype=np.uint64),
        coverage=coverage,
    )


def build_trie(adapter) -> TokenTrie:
    """Build from TokenizerAdapter.token_bytes exclusively (E5)."""
    special = getattr(adapter, "special_token_ids", frozenset())
    entries: list[tuple[bytes, int]] = []
    for tid in sorted(set(adapter.vocabulary.values())):
        if tid in special:
            continue
        bs = adapter.token_bytes(tid)
        if not bs:
            continue
        if len(bs) > 2**16:
            raise TrieBuildError(f"token {tid} unreasonably long ({len(bs)} bytes)")
        entries.append((bs, tid))
    if not entries:
        raise TrieBuildError("empty vocabulary after excluding special tokens")

    # group byte-identical spellings; the trie node carries the smallest id
    by_bytes: dict[bytes, list[int]] = {}
    for bs, tid in entries:
        by_bytes.setdefault(bs, []).append(tid)
    aliases = {min(ids): tuple(sorted(ids)) for ids in by_bytes.values() if len(ids) > 1}

    words = _dfs_words(entries)

    import hashlib

    h = hashlib.blake2b(digest_size=16)
    for bs, tid in entries:
        h.update(tid.to_bytes(4, "little"))
        h.update(len(bs).to_bytes(2, "little"))
        h.update(bs)
    return TokenTrie(
        nodes=np.array(words, dtype=np.uint64),
        n_tokens=len(entries),
        tokenizer_fingerprint=h.hexdigest(),
        aliases=aliases,
        slices=_build_slices(entries) if perf_flags.slicer_enabled() else None,
    )
