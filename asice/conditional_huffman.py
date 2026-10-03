"""Increment 5: Conditional bitstream serialisation (FR-6).

Takes the Tile list from Stage 4 and turns it into actual bytes.

Two things happen here, matching the submitted sequence diagram:
    1. Serialise every tile's fields (x, y, w, h, value, is_roi) into a
       flat byte stream.
    2. Compute the compression ratio already achieved by Stages 1-4
       (raw size / serialised size). If it already meets --target-cr,
       skip Huffman entirely and store the stream raw (saves CPU, matches
       "If Current_CR >= Target Benchmark, skip Huffman" in the spec).
       Otherwise, Huffman-encode the byte stream to shrink it further.

This is a REAL Huffman coder: it builds a frequency table, a prefix-code
tree, and an actual bit-packed stream — and it decodes, because an
encoder nobody can decode isn't compression, it's data loss. The earlier
C++ prototype only estimated Huffman's size from entropy; this replaces
that with a working encoder/decoder pair.
"""

from __future__ import annotations

import heapq
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dp_tiling import Tile

MAGIC = b"ASCE"
FLAG_RAW = 0
FLAG_HUFFMAN = 1

# Per-tile fixed layout before Huffman: 4x uint16 (x,y,w,h) + 3x uint8 (RGB)
# + 1x uint8 (is_roi) = 12 bytes/tile. uint16 caps a single image dimension
# at 65535px, far above NFR-1's working range.
_TILE_STRUCT = struct.Struct("<HHHHBBBB")


class EntropyError(Exception):
    pass


# --------------------------------------------------------------------------
# Tile <-> raw bytes
# --------------------------------------------------------------------------

def serialize_tiles(tiles: List[Tile]) -> bytes:
    """Flatten tiles into a fixed-width byte record per tile."""
    out = bytearray()
    for t in tiles:
        if t.x > 65535 or t.y > 65535 or t.w > 65535 or t.h > 65535:
            raise EntropyError(f"tile dimension exceeds uint16 range: {t}")
        r, g, b = (int(round(v)) for v in t.value[:3])
        out += _TILE_STRUCT.pack(t.x, t.y, t.w, t.h, r, g, b, 1 if t.is_roi else 0)
    return bytes(out)


def deserialize_tiles(data: bytes) -> List[Tile]:
    """Inverse of serialize_tiles."""
    if len(data) % _TILE_STRUCT.size != 0:
        raise EntropyError(
            f"tile byte stream length {len(data)} is not a multiple of record size {_TILE_STRUCT.size}"
        )
    tiles = []
    for i in range(0, len(data), _TILE_STRUCT.size):
        x, y, w, h, r, g, b, is_roi = _TILE_STRUCT.unpack_from(data, i)
        tiles.append(
            Tile(x=x, y=y, w=w, h=h, value=np.array([r, g, b], dtype=np.float32), is_roi=bool(is_roi))
        )
    return tiles


# --------------------------------------------------------------------------
# Huffman coding over raw bytes
# --------------------------------------------------------------------------

@dataclass
class _HuffNode:
    freq: int
    byte: Optional[int] = None  # leaf: the byte value; internal: None
    left: Optional["_HuffNode"] = None
    right: Optional["_HuffNode"] = None

    def __lt__(self, other: "_HuffNode") -> bool:
        return self.freq < other.freq  # heapq needs a total order


def _build_tree(data: bytes) -> _HuffNode:
    if not data:
        raise EntropyError("cannot build a Huffman tree from empty data")

    counts: Dict[int, int] = {}
    for b in data:
        counts[b] = counts.get(b, 0) + 1

    heap: List[_HuffNode] = [_HuffNode(freq=f, byte=b) for b, f in counts.items()]
    if len(heap) == 1:
        # A single distinct byte value: still needs a valid tree with a
        # depth-1 code, or encoding would produce zero-length codewords.
        # Both children must be real leaves (the walk in _serialize_tree
        # only knows how to handle "leaf with a byte" or "internal with
        # two real children" — a bare freq=0/byte=None placeholder is
        # neither, and crashed _serialize_tree's recursion). The dummy
        # right child reuses the same byte value; it is never actually
        # reachable by any real encoded data (its code "1" is simply
        # unused), so its byte value doesn't matter for correctness.
        only = heap[0]
        return _HuffNode(freq=only.freq, left=only, right=_HuffNode(freq=0, byte=only.byte))

    heapq.heapify(heap)
    counter = 0  # tie-breaker so heapq never has to compare _HuffNode.byte
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        counter += 1
        heapq.heappush(heap, _HuffNode(freq=a.freq + b.freq, left=a, right=b))
    return heap[0]


def _build_codes(root: _HuffNode) -> Dict[int, str]:
    codes: Dict[int, str] = {}

    def walk(node: _HuffNode, prefix: str) -> None:
        if node.byte is not None:
            codes[node.byte] = prefix or "0"  # guard the single-symbol edge case
            return
        if node.left:
            walk(node.left, prefix + "0")
        if node.right:
            walk(node.right, prefix + "1")

    walk(root, "")
    return codes


def _serialize_tree(root: _HuffNode) -> bytes:
    """Pre-order encode the tree shape so the decoder can rebuild it
    without needing the original data. 1 bit per node (1=leaf+byte,
    0=internal), packed MSB-first, then a varint-free fixed byte per leaf."""
    bits: List[str] = []
    leaf_bytes: List[int] = []

    def walk(node: _HuffNode) -> None:
        if node.byte is not None:
            bits.append("1")
            leaf_bytes.append(node.byte)
            return
        bits.append("0")
        # internal nodes always have both children by construction above
        walk(node.left)
        walk(node.right)

    walk(root)
    bitstring = "".join(bits)
    packed = _pack_bits(bitstring)
    header = struct.pack("<III", len(bitstring), len(leaf_bytes), len(packed))
    return header + packed + bytes(leaf_bytes)


def _deserialize_tree(data: bytes, offset: int) -> Tuple[_HuffNode, int]:
    n_bits, n_leaves, packed_len = struct.unpack_from("<III", data, offset)
    offset += 12
    packed = data[offset : offset + packed_len]
    offset += packed_len
    leaf_bytes = data[offset : offset + n_leaves]
    offset += n_leaves

    bitstring = _unpack_bits(packed, n_bits)
    pos = [0]
    leaf_idx = [0]

    def walk() -> _HuffNode:
        bit = bitstring[pos[0]]
        pos[0] += 1
        if bit == "1":
            b = leaf_bytes[leaf_idx[0]]
            leaf_idx[0] += 1
            return _HuffNode(freq=0, byte=b)
        left = walk()
        right = walk()
        return _HuffNode(freq=0, left=left, right=right)

    root = walk()
    return root, offset


def _pack_bits(bitstring: str) -> bytes:
    pad = (-len(bitstring)) % 8
    bitstring = bitstring + "0" * pad
    out = bytearray(len(bitstring) // 8)
    for i in range(0, len(bitstring), 8):
        out[i // 8] = int(bitstring[i : i + 8], 2)
    return bytes(out)


def _unpack_bits(data: bytes, n_bits: int) -> str:
    bits = "".join(f"{byte:08b}" for byte in data)
    return bits[:n_bits]


def huffman_encode(data: bytes) -> bytes:
    """Encode raw bytes into: [tree][n_data_bits][packed data bits].
    Self-contained — huffman_decode needs nothing but this output."""
    if not data:
        return struct.pack("<I", 0)  # degenerate: zero-length payload, no tree needed

    tree = _build_tree(data)
    codes = _build_codes(tree)
    tree_bytes = _serialize_tree(tree)

    bitstring = "".join(codes[b] for b in data)
    packed = _pack_bits(bitstring)

    body = struct.pack("<I", len(data)) + tree_bytes + struct.pack("<I", len(bitstring)) + packed
    return body


def huffman_decode(data: bytes) -> bytes:
    (orig_len,) = struct.unpack_from("<I", data, 0)
    if orig_len == 0:
        return b""
    offset = 4
    tree, offset = _deserialize_tree(data, offset)
    (n_bits,) = struct.unpack_from("<I", data, offset)
    offset += 4
    packed = data[offset:]
    bitstring = _unpack_bits(packed, n_bits)

    out = bytearray()
    node = tree
    for bit in bitstring:
        node = node.left if bit == "0" else node.right
        if node.byte is not None:
            out.append(node.byte)
            node = tree
            if len(out) == orig_len:
                break
    if len(out) != orig_len:
        raise EntropyError(f"Huffman decode produced {len(out)} bytes, expected {orig_len}")
    return bytes(out)


# --------------------------------------------------------------------------
# Conditional serialisation (the actual Stage 5 entry point)
# --------------------------------------------------------------------------

@dataclass
class ConditionalSerializer:
    """Serialises a tile list, skipping Huffman if the target CR is already
    met by Stages 1-4 alone (FR-6's "skip Huffman coding to save processing
    time" branch).

    target_cr: desired raw_size / final_size ratio. None disables the
        skip logic entirely (always Huffman-encode).
    """

    target_cr: Optional[float] = 4.0

    def serialize(self, tiles: List[Tile], raw_size_bytes: int) -> Tuple[bytes, dict]:
        """Returns (archive_bytes, meta). meta reports which path was
        taken and the achieved ratio, for logging / the paper's results."""
        tile_bytes = serialize_tiles(tiles)
        structural_cr = raw_size_bytes / max(len(tile_bytes), 1)

        if self.target_cr is not None and structural_cr >= self.target_cr:
            payload = MAGIC + bytes([FLAG_RAW]) + tile_bytes
            meta = {
                "huffman_used": False,
                "structural_cr": structural_cr,
                "final_cr": raw_size_bytes / max(len(payload), 1),
                "reason": "target CR already met by Stages 1-4",
            }
            return payload, meta

        encoded = huffman_encode(tile_bytes)
        if len(encoded) < len(tile_bytes):
            payload = MAGIC + bytes([FLAG_HUFFMAN]) + encoded
            meta = {
                "huffman_used": True,
                "structural_cr": structural_cr,
                "final_cr": raw_size_bytes / max(len(payload), 1),
                "reason": "Huffman applied",
            }
        else:
            # Huffman lost (can happen on already near-random tile data —
            # e.g. a tiny or extremely detailed image); storing raw is
            # never worse, so fall back rather than inflate the output.
            payload = MAGIC + bytes([FLAG_RAW]) + tile_bytes
            meta = {
                "huffman_used": False,
                "structural_cr": structural_cr,
                "final_cr": raw_size_bytes / max(len(payload), 1),
                "reason": "Huffman did not shrink the data; stored raw",
            }
        return payload, meta

    def deserialize(self, payload: bytes) -> List[Tile]:
        if payload[:4] != MAGIC:
            raise EntropyError("not an asice tile payload (bad magic)")
        flag = payload[4]
        body = payload[5:]
        if flag == FLAG_RAW:
            tile_bytes = body
        elif flag == FLAG_HUFFMAN:
            tile_bytes = huffman_decode(body)
        else:
            raise EntropyError(f"unknown payload flag: {flag}")
        return deserialize_tiles(tile_bytes)