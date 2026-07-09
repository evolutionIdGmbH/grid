"""E6 TokenizerAdapter: pinned Tokenizer protocol + the canonical token->bytes function.

``token_bytes(token_id) -> bytes`` is THE single token-to-bytes definition, used
identically by the trie build, the fast path, and the reference guide (DESIGN.md
SS4.2 divergence note 2). Normative rules: byte-level-BPE unicode remaps inverted;
sentencepiece U+2581 and BPE 'G-dot' space markers -> 0x20; byte-fallback literals
``<0xNN>`` -> the single byte; special tokens are excluded from tries and masks.

``MockTokenizer`` is the deterministic test tokenizer: byte-fallback complete
(all 256 single-byte tokens) plus configurable multi-byte tokens; latin-1 is the
str<->bytes bridge so every byte string has a stable str form.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

_BYTE_FALLBACK = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


@dataclass
class MockTokenizer:
    """Pinned-protocol tokenizer over an explicit vocabulary (tests, mini-G5)."""

    extra_tokens: tuple[str, ...] = ()
    eos_token: str = "</s>"
    _vocab: dict[str, int] = field(default_factory=dict)
    _id_to_bytes: dict[int, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids: dict[str, int] = {}
        for i in range(256):
            ids[f"<0x{i:02X}>"] = len(ids)
        for tok in self.extra_tokens:
            if tok not in ids:
                ids[tok] = len(ids)
        ids[self.eos_token] = len(ids)
        self._vocab = ids
        for tok, tid in ids.items():
            m = _BYTE_FALLBACK.match(tok)
            if m:
                self._id_to_bytes[tid] = bytes([int(m.group(1), 16)])
            elif tok == self.eos_token:
                self._id_to_bytes[tid] = b""
            else:
                self._id_to_bytes[tid] = tok.encode("latin-1")

    # -- pinned Tokenizer protocol ------------------------------------------

    @property
    def vocabulary(self) -> dict[str, int]:
        return self._vocab

    @property
    def eos_token_id(self) -> int:
        return self._vocab[self.eos_token]

    @property
    def pad_token_id(self) -> int:
        return self.eos_token_id

    @property
    def special_tokens(self) -> set[str]:
        return {self.eos_token}

    def encode(self, prompt):
        if isinstance(prompt, list):
            return [self.encode(p)[0] for p in prompt], None
        data = prompt.encode("latin-1")
        return self.greedy_tokenize(data), None

    def decode(self, token_ids) -> list[str]:
        return ["".join(self._id_to_bytes[int(t)].decode("latin-1") for t in token_ids)]

    def convert_token_to_string(self, token: str) -> str:
        return token

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._vocab.items())))

    # -- GRID extensions (E6) ------------------------------------------------

    def token_bytes(self, token_id: int) -> bytes:
        return self._id_to_bytes[token_id]

    @property
    def special_token_ids(self) -> frozenset[int]:
        return frozenset(self._vocab[t] for t in self.special_tokens)

    def greedy_tokenize(self, data: bytes) -> list[int]:
        """Longest-match greedy tokenization (E4a cost model + Write rendering)."""
        by_bytes = self._bytes_to_id()
        out: list[int] = []
        i = 0
        max_len = max((len(b) for b in by_bytes), default=1)
        while i < len(data):
            for ln in range(min(max_len, len(data) - i), 0, -1):
                tid = by_bytes.get(data[i:i + ln])
                if tid is not None:
                    out.append(tid)
                    i += ln
                    break
            else:  # pragma: no cover - byte-fallback guarantees progress
                raise AssertionError("untokenizable byte")
        return out

    def _bytes_to_id(self) -> dict[bytes, int]:
        if not hasattr(self, "_b2i"):
            b2i: dict[bytes, int] = {}
            for tid, bs in self._id_to_bytes.items():
                if bs and (bs not in b2i or tid < b2i[bs]):
                    b2i[bs] = tid
            object.__setattr__(self, "_b2i", b2i)
        return self._b2i


def verify_byte_complete(adapter) -> bool:
    """E6 verify(): all 256 byte values reachable via token_bytes output.

    Returns True (VERIFIED_COMPLETE) or False (VERIFIED_DEGRADED, W-COMPLETENESS01):
    the completeness guarantee is formally void, soundness unaffected.
    """
    seen: set[int] = set()
    special = getattr(adapter, "special_token_ids", frozenset())
    for tid in adapter.vocabulary.values():
        if tid in special:
            continue
        bs = adapter.token_bytes(tid)
        if len(bs) == 1:
            seen.add(bs[0])
    if len(seen) == 256:
        return True
    warnings.warn(
        f"W-COMPLETENESS01: tokenizer lacks byte-fallback for {256 - len(seen)} byte values; "
        "completeness guarantee void (DESIGN.md E6)",
        stacklevel=2,
    )
    return False
