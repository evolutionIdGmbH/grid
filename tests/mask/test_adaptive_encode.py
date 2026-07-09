"""adaptive_encode vectorization parity (STAGE 4).

The numpy implementation in grid/mask/cache.py must be BYTE-IDENTICAL to the
original per-int loop — the payload (and its tag byte) is hashed into
MaskCacheEntry.entry_id, and G10 audit replay depends on that hash never
moving for the same ci set. The reference implementation below IS the
original loop, kept verbatim as the executable specification.
"""

import random

import numpy as np

from grid.mask.cache import TAG_ACCEPT, TAG_BITSET, TAG_REJECT, adaptive_encode


def _reference_encode(ci_tokens, vocab_size: int) -> tuple[int, bytes]:
    """The pre-vectorization loop, verbatim (the byte-level specification)."""
    n = len(ci_tokens)
    size_accept = 4 * n
    size_reject = 4 * (vocab_size - n)
    size_bitset = (vocab_size + 7) // 8
    best = min((size_accept, TAG_ACCEPT), (size_reject, TAG_REJECT), (size_bitset, TAG_BITSET))
    tag = best[1]
    if tag == TAG_ACCEPT:
        payload = b"".join(t.to_bytes(4, "little") for t in sorted(ci_tokens))
    elif tag == TAG_REJECT:
        keep = set(ci_tokens)
        payload = b"".join(t.to_bytes(4, "little") for t in range(vocab_size) if t not in keep)
    else:
        bits = bytearray(size_bitset)
        for t in ci_tokens:
            bits[t >> 3] |= 1 << (t & 7)
        payload = bytes(bits)
    return tag, payload


def _cases():
    rng = random.Random(0xC1)
    vocab_sizes = (1, 8, 33, 1000, 50257, 151665)
    for v in vocab_sizes:
        yield v, ()                                   # empty -> accept, b""
        yield v, (0,)
        yield v, (v - 1,)
        yield v, tuple(range(v)[:7])                  # small -> accept-list
        if v >= 1000:
            # small / mid / huge random sets: hit accept, bitset, reject tags
            yield v, tuple(rng.sample(range(v), 5))
            yield v, tuple(rng.sample(range(v), v // 2))
            yield v, tuple(rng.sample(range(v), v - 3))
        yield v, tuple(range(v))                      # full vocab -> reject, b""


def test_numpy_matches_reference_loop():
    for vocab_size, ci in _cases():
        expected = _reference_encode(ci, vocab_size)
        got = adaptive_encode(ci, vocab_size)
        assert got[0] == expected[0], (vocab_size, len(ci), "tag")
        assert got[1] == expected[1], (vocab_size, len(ci), "payload bytes")


def test_tag_coverage():
    """The case list must actually exercise all three encodings."""
    tags = {_reference_encode(ci, v)[0] for v, ci in _cases()}
    assert tags == {TAG_ACCEPT, TAG_REJECT, TAG_BITSET}


def test_array_input_matches_tuple_input():
    """Kernel-walk entries pass ci as a read-only int32 ndarray (np.frombuffer
    over the i32-le FFI buffer); output must match the tuple path exactly."""
    rng = random.Random(0xC2)
    for vocab_size in (100, 151665):
        for n in (0, 4, vocab_size // 2, vocab_size - 2):
            ci = sorted(rng.sample(range(vocab_size), n))
            arr = np.frombuffer(np.asarray(ci, dtype=np.int32).tobytes(), dtype=np.int32)
            assert not arr.flags.writeable  # the FFI view really is read-only
            assert adaptive_encode(arr, vocab_size) == adaptive_encode(tuple(ci), vocab_size)
            assert adaptive_encode(arr, vocab_size) == _reference_encode(tuple(ci), vocab_size)


def test_duplicate_ids_preserved():
    """sorted() keeps duplicates in the accept payload; np.sort must too (and
    the tag choice counts duplicates via len on both paths)."""
    ci = (5, 1, 5, 3, 1)
    assert adaptive_encode(ci, 10_000) == _reference_encode(ci, 10_000)
