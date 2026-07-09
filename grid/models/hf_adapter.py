"""E6 for real tokenizers: HuggingFace adapter with the canonical token_bytes.

Normative rules (DESIGN.md E6):
- byte-level BPE (GPT-2 / Llama-3 / Qwen families): invert the GPT-2
  bytes<->unicode remap table per character;
- SentencePiece (Llama-1/2, Mistral): U+2581 -> 0x20, ``<0xNN>`` byte-fallback
  literals -> the single byte, else UTF-8 of the piece;
- special tokens excluded from tries and masks (EOS enters via SS6 step 7 only).

One definition, three consumers: trie build, fast path, ReferenceGuide.
"""

from __future__ import annotations

import re
from functools import lru_cache

_BYTE_FALLBACK = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
_SP_SPACE = "▁"  # '▁'


@lru_cache(maxsize=1)
def _gpt2_unicode_to_bytes() -> dict[str, int]:
    """Inverse of the standard GPT-2 bytes_to_unicode table."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


class HFTokenizerAdapter:
    """Wraps a HuggingFace tokenizer into the pinned Tokenizer protocol + GRID E6."""

    def __init__(self, hf_tokenizer) -> None:
        self.hf = hf_tokenizer
        self._vocab: dict[str, int] = dict(hf_tokenizer.get_vocab())
        self._special_ids = frozenset(int(i) for i in hf_tokenizer.all_special_ids)
        self._id_to_bytes: dict[int, bytes] = {}
        inv = _gpt2_unicode_to_bytes()
        sample = [t for t in list(self._vocab)[:512] if t not in hf_tokenizer.all_special_tokens]
        byte_level = sum(all(ch in inv for ch in t) for t in sample) > len(sample) * 0.9

        for token, tid in self._vocab.items():
            tid = int(tid)
            if tid in self._special_ids:
                self._id_to_bytes[tid] = b""
                continue
            m = _BYTE_FALLBACK.match(token)
            if m:
                self._id_to_bytes[tid] = bytes([int(m.group(1), 16)])
            elif byte_level:
                try:
                    self._id_to_bytes[tid] = bytes(inv[ch] for ch in token)
                except KeyError:  # added non-byte-level token (rare): utf-8 fallback
                    self._id_to_bytes[tid] = token.encode("utf-8")
            else:  # sentencepiece family
                self._id_to_bytes[tid] = token.replace(_SP_SPACE, " ").encode("utf-8")
        self._b2i: dict[bytes, int] = {}
        for tid, bs in self._id_to_bytes.items():
            if bs and (bs not in self._b2i or tid < self._b2i[bs]):
                self._b2i[bs] = tid
        self._max_tok_len = max((len(b) for b in self._b2i), default=1)

    # -- pinned Tokenizer protocol -------------------------------------------

    @property
    def vocabulary(self) -> dict[str, int]:
        return self._vocab

    @property
    def eos_token(self) -> str:
        return self.hf.eos_token

    @property
    def eos_token_id(self) -> int:
        return int(self.hf.eos_token_id)

    @property
    def pad_token_id(self) -> int:
        pid = self.hf.pad_token_id
        return int(pid) if pid is not None else self.eos_token_id

    @property
    def special_tokens(self) -> set[str]:
        return set(self.hf.all_special_tokens)

    def encode(self, prompt):
        if isinstance(prompt, list):
            return [self.encode(p)[0] for p in prompt], None
        return self.hf.encode(prompt, add_special_tokens=False), None

    def decode(self, token_ids) -> list[str]:
        return [self.hf.decode([int(t) for t in token_ids], skip_special_tokens=True)]

    def convert_token_to_string(self, token: str) -> str:
        return self.token_bytes(self._vocab[token]).decode("utf-8", errors="replace")

    def __hash__(self) -> int:
        return hash((type(self.hf).__name__, len(self._vocab), self.eos_token_id))

    # -- GRID extensions (E6) --------------------------------------------------

    def token_bytes(self, token_id: int) -> bytes:
        return self._id_to_bytes[token_id]

    @property
    def special_token_ids(self) -> frozenset[int]:
        return self._special_ids

    def greedy_tokenize(self, data: bytes) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(data):
            for ln in range(min(self._max_tok_len, len(data) - i), 0, -1):
                tid = self._b2i.get(data[i:i + ln])
                if tid is not None:
                    out.append(tid)
                    i += ln
                    break
            else:
                raise AssertionError(f"untokenizable byte 0x{data[i]:02X} (tokenizer not byte-complete)")
        return out


def load(name_or_path: str) -> HFTokenizerAdapter:
    from transformers import AutoTokenizer

    return HFTokenizerAdapter(AutoTokenizer.from_pretrained(name_or_path))
