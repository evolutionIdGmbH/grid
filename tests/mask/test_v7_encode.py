"""Kernel v7 encode/hash cross-implementation vectors (red-team plan §4.1).

grid_core.encode_mask must be BYTE-IDENTICAL to grid/mask/cache.py
adaptive_encode (itself bound to the original per-int loop by
test_adaptive_encode's reference), and grid_core.entry_id_hex must reproduce
hashlib.blake2b(digest_size=16)(repr(key).encode() || tag || payload) — the
MaskCacheEntry.entry_id formula — for every key shape the producer emits:
generic raw/scoped, genN (including p=-1), ident, fp=None, and keys carrying
quote/escape/high bytes (Python's repr quote-preference rule is exercised by
the b"'" component). These vectors are what make register_blob's in-kernel
entry ids interchangeable with make_entry's (G10 replay invariant).
"""

import hashlib

import numpy as np
import pytest

import grid.trie.walk as W
from grid.mask.cache import TAG_ACCEPT, TAG_REJECT, adaptive_encode, make_entry
from tests.mask.test_adaptive_encode import _cases, _reference_encode

pytestmark = pytest.mark.skipif(
    not W._USE_RUST or not hasattr(W._grid_core, "encode_mask"),
    reason="grid_core v7 not installed (or disabled via GRID_NO_RUST)",
)


def _ci_bytes(ci) -> bytes:
    return np.asarray(list(ci), dtype=np.int32).tobytes()


def _rust_encode(ci, vocab_size):
    return W._grid_core.encode_mask(_ci_bytes(ci), vocab_size)


# ------------------------------------------------------------- encode vectors


def test_rust_matches_python_and_reference_over_cases():
    """The full test_adaptive_encode corpus (all vocab sizes, all three tags,
    empty and full-vocab edges) — Rust == numpy == the reference loop."""
    for vocab_size, ci in _cases():
        expected = _reference_encode(ci, vocab_size)
        got_py = adaptive_encode(ci, vocab_size)
        got_rs = _rust_encode(ci, vocab_size)
        assert got_rs == expected == got_py, (vocab_size, len(ci))


def test_duplicates_preserved():
    """n counts duplicates for the size comparison AND the accept payload
    keeps them, sorted (np.sort semantics)."""
    ci = (5, 1, 5, 3, 1)
    assert _rust_encode(ci, 10_000) == _reference_encode(ci, 10_000)
    assert _rust_encode(ci, 10_000)[0] == TAG_ACCEPT


def test_tie_breaks():
    """Size ties break by tag ACCEPT(0) < REJECT(1) < BITSET(2), exactly like
    the Python tuple-min."""
    # 4n == ceil(V/8): V=800, n=25 -> accept(100) ties bitset(100) -> ACCEPT
    ci = tuple(range(25))
    assert _rust_encode(ci, 800) == _reference_encode(ci, 800)
    assert _rust_encode(ci, 800)[0] == TAG_ACCEPT
    # 4(V-n) == ceil(V/8): V=800, n=775 -> reject(100) ties bitset(100) -> REJECT
    ci = tuple(range(775))
    assert _rust_encode(ci, 800) == _reference_encode(ci, 800)
    assert _rust_encode(ci, 800)[0] == TAG_REJECT
    # n == V-n (accept ties reject): both impls must agree on the winner
    # (bitset dominates any realizable n == V/2, so equality IS the assertion)
    for v, ci in ((2, (0,)), (8, (0, 1, 2, 3)), (64, tuple(range(32)))):
        assert _rust_encode(ci, v) == _reference_encode(ci, v), v


def test_giant_reject_case():
    """The literal-interior giant class from the H100 record: V=151665,
    n=148855 -> REJECT payload, byte-identical."""
    vocab = 151_665
    ci = tuple(t for t in range(vocab) if t % 54 != 0)[:148_855]
    expected = adaptive_encode(ci, vocab)
    assert expected[0] == TAG_REJECT
    assert _rust_encode(ci, vocab) == expected


def test_out_of_range_ids_hard_error():
    """Ids outside [0, vocab) are a hard error (the numpy reference raises
    from fancy indexing on the REJECT/BITSET paths; Rust rejects always)."""
    with pytest.raises(ValueError):
        _rust_encode((5,), 5)
    with pytest.raises(ValueError):
        _rust_encode((-1,), 5)
    with pytest.raises(ValueError):
        W._grid_core.encode_mask(b"\x01\x02\x03", 100)  # not a multiple of 4


# ----------------------------------------------------------- entry_id vectors


# every key shape cache_key can emit, plus repr-edge bytes: quotes (repr's
# quote-preference flips b"'" to double-quoted form), backslash, \xff
KEY_SHAPES = [
    ("generic", b"sel", (1, 2, 3), None),                 # legacy unscoped raw
    ("generic", b"select", (0, 5), "fp"),                 # v2 scoped raw fallback
    ("genN", -1, 109, b"", (1, 2, 3), "fp"),              # genN, p=-1 (no accept)
    ("genN", 5, 7, b"e", (2,), None),                     # genN, fp=None
    ("genN", -1, 3, b"'", (1,), "fp"),                    # single-quote byte
    ("genN", 2, 3, b'a"b\'c', (1, 4), "fp"),              # both quotes
    ("generic", b"q\\\xff\n", (0,), "fp"),                # backslash/high/ctrl
    ("ident", b"na", (4, 5), "fp"),                       # E11 ident key
    ("parity",),                                          # test-style key
]

CI_SETS = [
    (),                       # empty -> ACCEPT b""
    (0,),
    (3, 1, 2),
    tuple(range(950)),        # REJECT at V=1000
    tuple(range(0, 1000, 3)), # BITSET at V=1000
]


def test_entry_id_vectors_every_key_shape():
    vocab = 1000
    for key in KEY_SHAPES:
        key_repr = repr(key).encode()
        for ci in CI_SETS:
            tag, payload = adaptive_encode(ci, vocab)
            h = hashlib.blake2b(digest_size=16)
            h.update(key_repr)
            h.update(bytes([tag]))
            h.update(payload)
            got_hex, got_tag = W._grid_core.entry_id_hex(key_repr, _ci_bytes(ci), vocab)
            assert (got_hex, got_tag) == (h.hexdigest(), tag), (key, len(ci))
            # and the full make_entry formula (the normative Python producer)
            assert make_entry(key, list(ci), (), vocab).entry_id == got_hex, key


def test_blake2b_parameter_block_matches_hashlib():
    """digest_size=16 lives in the BLAKE2b parameter block (not a truncation)
    — a raw-vector pin against the blake2 crate's Blake2b<U16>."""
    got_hex, _tag = W._grid_core.entry_id_hex(b"abc", b"", 1)
    h = hashlib.blake2b(digest_size=16)
    h.update(b"abc")
    h.update(bytes([TAG_ACCEPT]))  # empty ci at V=1 -> ACCEPT, empty payload
    assert got_hex == h.hexdigest()
