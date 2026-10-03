"""Increment 5 tests: FR-6 (conditional Huffman bitstream serialization)."""

import numpy as np
import pytest

from asice.dp_tiling import Tile
from asice.conditional_huffman import (
    ConditionalSerializer,
    EntropyError,
    deserialize_tiles,
    huffman_decode,
    huffman_encode,
    serialize_tiles,
)


def _tiles_equal(a, b) -> bool:
    if len(a) != len(b):
        return False
    return all(
        t1.x == t2.x and t1.y == t2.y and t1.w == t2.w and t1.h == t2.h and t1.is_roi == t2.is_roi
        and np.allclose(t1.value, t2.value, atol=1.0)
        for t1, t2 in zip(a, b)
    )


# ---------------- raw Huffman round-trip ----------------
def test_huffman_round_trip_typical_data():
    data = bytes([1, 2, 3, 1, 1, 1, 2, 2, 5, 5, 5, 5, 5, 5, 250, 250, 0, 0, 0])
    assert huffman_decode(huffman_encode(data)) == data


def test_huffman_round_trip_single_symbol():
    """Degenerate case: only one distinct byte value in the whole input."""
    data = bytes([42] * 200)
    assert huffman_decode(huffman_encode(data)) == data


def test_huffman_round_trip_two_symbols():
    data = bytes([0, 1] * 100)
    assert huffman_decode(huffman_encode(data)) == data


def test_huffman_round_trip_empty():
    assert huffman_decode(huffman_encode(b"")) == b""


def test_huffman_round_trip_all_256_byte_values():
    data = bytes(range(256)) * 4
    assert huffman_decode(huffman_encode(data)) == data


def test_huffman_round_trip_random_bytes():
    rng = np.random.default_rng(0)
    data = bytes(rng.integers(0, 256, 5000, dtype=np.uint8))
    assert huffman_decode(huffman_encode(data)) == data


def test_huffman_shrinks_low_entropy_data():
    """Skewed byte frequencies should actually compress."""
    data = bytes([0] * 900 + [1] * 90 + [2] * 10)
    encoded = huffman_encode(data)
    assert len(encoded) < len(data)


def test_huffman_decode_rejects_truncated_input():
    data = bytes([1, 2, 3] * 50)
    encoded = huffman_encode(data)
    with pytest.raises(Exception):
        huffman_decode(encoded[: len(encoded) // 2])


# ---------------- tile serialization ----------------
def test_tile_serialize_round_trip():
    tiles = [
        Tile(x=0, y=0, w=10, h=10, value=np.array([100.0, 150.0, 200.0]), is_roi=False),
        Tile(x=10, y=0, w=1, h=1, value=np.array([255.0, 0.0, 0.0]), is_roi=True),
        Tile(x=0, y=10, w=5, h=5, value=np.array([0.0, 0.0, 0.0]), is_roi=False),
    ]
    back = deserialize_tiles(serialize_tiles(tiles))
    assert _tiles_equal(tiles, back)


def test_tile_serialize_rejects_oversized_dimension():
    tiles = [Tile(x=0, y=0, w=70000, h=1, value=np.array([0.0, 0.0, 0.0]), is_roi=False)]
    with pytest.raises(EntropyError):
        serialize_tiles(tiles)


def test_tile_deserialize_rejects_misaligned_bytes():
    with pytest.raises(EntropyError):
        deserialize_tiles(b"\x00" * 7)  # not a multiple of the 12-byte record


def test_tile_serialize_empty_list():
    assert serialize_tiles([]) == b""
    assert deserialize_tiles(b"") == []


# ---------------- ConditionalSerializer: the real FR-6 behaviour ----------------
def test_skips_huffman_when_target_cr_already_met():
    tiles = [Tile(x=0, y=0, w=100, h=100, value=np.array([128.0, 128.0, 128.0]), is_roi=False)]
    cs = ConditionalSerializer(target_cr=2.0)  # one giant tile trivially beats 2:1
    payload, meta = cs.serialize(tiles, raw_size_bytes=100 * 100 * 3)
    assert meta["huffman_used"] is False
    assert meta["final_cr"] >= 2.0


def test_uses_huffman_when_target_cr_not_yet_met():
    rng = np.random.default_rng(2)
    tiles = [
        Tile(x=i, y=0, w=1, h=1, value=np.array([float(rng.integers(0, 20))] * 3), is_roi=False)
        for i in range(2000)
    ]
    cs = ConditionalSerializer(target_cr=50.0)  # structural tiling alone can't reach this
    payload, meta = cs.serialize(tiles, raw_size_bytes=2000 * 3)
    assert meta["huffman_used"] is True


def test_falls_back_to_raw_when_huffman_does_not_help():
    """Tiny / already-high-entropy payloads: Huffman's tree overhead can
    exceed its savings. The serializer must not ship an inflated payload."""
    tiles = [Tile(x=0, y=0, w=1, h=1, value=np.array([7.0, 8.0, 9.0]), is_roi=False)]
    cs = ConditionalSerializer(target_cr=1000.0)  # forces a huffman attempt
    payload, meta = cs.serialize(tiles, raw_size_bytes=3)
    assert len(payload) <= len(serialize_tiles(tiles)) + 5  # magic+flag overhead only


def test_conditional_serializer_full_round_trip():
    tiles = [
        Tile(x=0, y=0, w=10, h=10, value=np.array([100.0, 150.0, 200.0]), is_roi=False),
        Tile(x=10, y=0, w=1, h=1, value=np.array([255.0, 0.0, 0.0]), is_roi=True),
    ]
    for target in (0.001, 1000.0):  # forces skip-path and huffman-path respectively
        cs = ConditionalSerializer(target_cr=target)
        payload, meta = cs.serialize(tiles, raw_size_bytes=10 * 10 * 3)
        back = cs.deserialize(payload)
        assert _tiles_equal(tiles, back), f"round-trip failed for target_cr={target}"


def test_target_cr_none_always_uses_huffman_path():
    tiles = [Tile(x=0, y=0, w=50, h=50, value=np.array([10.0, 10.0, 10.0]), is_roi=False)]
    cs = ConditionalSerializer(target_cr=None)
    payload, meta = cs.serialize(tiles, raw_size_bytes=50 * 50 * 3)
    # None disables the skip check entirely, so it must have attempted huffman
    # (it may still fall back to raw if huffman didn't help on this tiny input,
    # but "skipped because target already met" must not be the reason)
    assert meta["reason"] != "target CR already met by Stages 1-4"


def test_deserialize_rejects_bad_magic():
    cs = ConditionalSerializer()
    with pytest.raises(EntropyError):
        cs.deserialize(b"NOPE" + bytes([0]) + b"garbage")


def test_deserialize_rejects_unknown_flag():
    cs = ConditionalSerializer()
    from asice.conditional_huffman import MAGIC
    with pytest.raises(EntropyError):
        cs.deserialize(MAGIC + bytes([99]) + b"garbage")