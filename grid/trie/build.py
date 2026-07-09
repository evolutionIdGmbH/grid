"""E5 TokenTrie in the final artifact format (DESIGN.md SS2).

One numpy uint64 array of DFS-contiguous nodes plus nothing else — exactly the
buffer grid_core consumes zero-copy at M4. Node packing (8 bytes)::

    bits  0..7   byte value on the edge into this node
    bits  8..31  token_id + 1 ending exactly at this node (0 = none)
    bits 32..63  subtree size in nodes (self included) -> DFS sibling skip

The root is virtual: the array is the concatenation of the top-level subtrees.
Special tokens (E6) are excluded — EOS enters masks only via SS6 step 7's union.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid.errors import TrieBuildError


@dataclass(frozen=True)
class TokenTrie:
    nodes: np.ndarray          # uint64, DFS-contiguous
    n_tokens: int
    tokenizer_fingerprint: str
    # tokens with byte-identical spellings: node carries the smallest id; the mask
    # must include every alias (completeness — a mask over ids, not spellings)
    aliases: dict[int, tuple[int, ...]] = None  # type: ignore[assignment]

    @staticmethod
    def unpack(word: int) -> tuple[int, int, int]:
        """-> (edge_byte, token_id or -1, subtree_size)"""
        return int(word & 0xFF), int(((word >> 8) & 0xFFFFFF) - 1), int(word >> 32)

    def expand(self, token_id: int) -> tuple[int, ...]:
        return self.aliases.get(token_id, (token_id,))


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

    # nested dict trie: byte -> [token_id, children]
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
    )
