"""Increment 6: Batch storage & dataset aggregation (FR-7).

Wires Stages 1-5 together into one per-image run, and runs that across a
whole dataset into a single .asice archive.

Also defines the "strict / fast" mode switch:

                        ROI Mask
                            |
                   Quadtree Policy
                            |
              +-------------+-------------+
              |                           |
         STRICT MODE                 FAST MODE
       criterion="range"          criterion="variance"
        provable bound              heuristic bound
              |                           |
              +-------------+-------------+
                            |
                        DP Tiling
                            |
                  Conditional Huffman

"strict" and "fast" are the only pipeline-level knob most users need to
think about; internally each maps to a (quadtree criterion, DP tolerance
policy) pair, defined once here rather than letting the two stages drift
out of sync with each other. "range" (strict) gives the error bound the
paper actually claims; "variance" (fast) is faster but only statistically
bounded -- found, during dataset testing, to allow real per-pixel errors
several times past T on ordinary photographs (a single outlier pixel
diluted across a large uniform block can leave that block's variance
under threshold even though one pixel is far off). Both are kept as real,
supported options -- this project reports both rather than picking a
winner, which is the actual contribution: an explicit, named tradeoff
instead of a silent default.
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np

from .conditional_huffman import ConditionalSerializer
from .dp_tiling import DPTiler, reduction_ratio, tiles_to_image
from .io_buffer import DatasetAggregator, ImageBuffer
from .quadtree import QuadtreeDecomposer, leaf_count, max_background_error
from .roi import DEFAULT_MODEL_PATH, SaliencySegmenter

log = logging.getLogger("asice")

Mode = Literal["strict", "fast"]

ARCHIVE_MAGIC = b"ASCD"  # "ASice Compressed Dataset" -- distinct from the
                          # per-image ASCE magic in conditional_huffman.py
ARCHIVE_VERSION = 1

# Fixed per-image index record: name_len(H) + width(H) + height(H) +
# payload_offset(Q) + payload_len(I) = 2+2+2+8+4 = 18 bytes, plus the
# variable-length name itself (stored right after this record).
_INDEX_HEADER = struct.Struct("<HHHQI")


class PipelineError(Exception):
    pass


@dataclass
class ModeConfig:
    """What a mode name actually means, for both stages at once."""

    criterion: str  # "range" | "variance", passed to QuadtreeDecomposer
    label: str


_MODES: dict = {
    "strict": ModeConfig(criterion="range", label="Strict (range criterion, provable bound)"),
    "fast": ModeConfig(criterion="variance", label="Fast (variance criterion, heuristic bound)"),
}


def resolve_mode(mode: str) -> ModeConfig:
    if mode not in _MODES:
        raise PipelineError(f"unknown mode '{mode}', expected one of {list(_MODES)}")
    return _MODES[mode]


@dataclass
class ImageResult:
    """Everything worth knowing about one image's run, for logging, the
    paper's per-image table, and archive writing."""

    name: str
    width: int
    height: int
    mode: str
    roi_coverage: float
    roi_method_used: str
    quadtree_leaves: int
    dp_tiles: int
    dp_reduction: float
    huffman_used: bool
    structural_cr: float
    final_cr: float
    archive_bytes: int
    max_background_error: float
    roi_exact: bool
    roi_time_s: float
    quadtree_time_s: float
    dp_time_s: float
    serialize_time_s: float
    total_time_s: float
    payload: bytes = field(repr=False)


@dataclass
class PipelineConfig:
    """One place for every knob the CLI / app exposes, so a batch run and
    a single-image run can't drift apart in behaviour."""

    mode: str = "strict"
    threshold: float = 12.0
    min_block: int = 2
    merge_tolerance: Optional[float] = None  # None -> reuse threshold, per stage docs
    max_merge_run: int = 64
    target_cr: Optional[float] = 4.0
    roi_method: str = "classical"
    roi_model_path: Path = DEFAULT_MODEL_PATH
    roi_mask_dir: Optional[Path] = None
    roi_percentile: float = 90.0
    roi_dilate_px: int = 5


def compress_image(buf: ImageBuffer, config: PipelineConfig) -> ImageResult:
    """Run Stages 2-5 on one already-ingested image. Raises PipelineError
    on failure (caller decides whether to skip-and-continue for a batch,
    matching NFR-3's "one bad input never kills the whole run" principle
    at the pipeline level, not just at ingestion)."""
    mode_cfg = resolve_mode(config.mode)
    t_start = time.perf_counter()

    seg = SaliencySegmenter(
        method=config.roi_method,
        model_path=config.roi_model_path,
        mask_dir=config.roi_mask_dir,
        percentile=config.roi_percentile,
        dilate_px=config.roi_dilate_px,
    )
    t0 = time.perf_counter()
    try:
        mask, roi_meta = seg.generate_binary_values(buf)
    except Exception as exc:
        raise PipelineError(f"ROI extraction failed for {buf.source_path}: {exc}") from exc
    roi_time = time.perf_counter() - t0
    roi_coverage = float(mask.mean())

    dec = QuadtreeDecomposer(threshold=config.threshold, min_block=config.min_block, criterion=mode_cfg.criterion)
    t0 = time.perf_counter()
    try:
        root = dec.decompose(buf, mask)
    except Exception as exc:
        raise PipelineError(f"Quadtree decomposition failed for {buf.source_path}: {exc}") from exc
    qt_time = time.perf_counter() - t0
    n_leaves = leaf_count(root)
    bg_err = max_background_error(root, buf.rgb_matrix)

    tiler = DPTiler(tolerance=config.merge_tolerance, max_merge_run=config.max_merge_run)
    t0 = time.perf_counter()
    try:
        tiles = tiler.analyze_adjacent_leaf_nodes(
            root, default_tolerance=config.threshold, original_rgb=buf.rgb_matrix
        )
    except Exception as exc:
        raise PipelineError(f"DP tiling failed for {buf.source_path}: {exc}") from exc
    dp_time = time.perf_counter() - t0
    n_tiles = len(tiles)
    dp_reduction = reduction_ratio(n_leaves, n_tiles)

    serializer = ConditionalSerializer(target_cr=config.target_cr)
    t0 = time.perf_counter()
    try:
        payload, ser_meta = serializer.serialize(tiles, raw_size_bytes=buf.raw_size_bytes)
    except Exception as exc:
        raise PipelineError(f"Serialization failed for {buf.source_path}: {exc}") from exc
    ser_time = time.perf_counter() - t0

    # Round-trip check: decode what was just written and confirm the ROI
    # guarantee actually holds for the FINAL archive bytes, not just the
    # intermediate quadtree (this was the exact gap found during manual
    # testing -- the earlier app.py showed a quadtree-only error number
    # next to final-pipeline metrics, implying it covered the whole
    # pipeline when it didn't; compress_image does the real check instead
    # of leaving it to whichever caller remembers to).
    try:
        tiles_back = serializer.deserialize(payload)
        recon = tiles_to_image(tiles_back, buf.height, buf.width)
        roi_exact = bool(np.array_equal(buf.rgb_matrix[mask == 1], recon[mask == 1]))
    except Exception as exc:
        raise PipelineError(f"Round-trip verification failed for {buf.source_path}: {exc}") from exc

    total_time = time.perf_counter() - t_start

    return ImageResult(
        name=Path(buf.source_path).name,
        width=buf.width,
        height=buf.height,
        mode=config.mode,
        roi_coverage=roi_coverage,
        roi_method_used=roi_meta.get("method", config.roi_method),
        quadtree_leaves=n_leaves,
        dp_tiles=n_tiles,
        dp_reduction=dp_reduction,
        huffman_used=ser_meta["huffman_used"],
        structural_cr=ser_meta["structural_cr"],
        final_cr=ser_meta["final_cr"],
        archive_bytes=len(payload),
        max_background_error=bg_err,
        roi_exact=roi_exact,
        roi_time_s=roi_time,
        quadtree_time_s=qt_time,
        dp_time_s=dp_time,
        serialize_time_s=ser_time,
        total_time_s=total_time,
        payload=payload,
    )


@dataclass
class BatchResult:
    results: List[ImageResult] = field(default_factory=list)
    errors: List[tuple] = field(default_factory=list)  # (path, reason)
    total_time_s: float = 0.0

    @property
    def mean_cr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.final_cr for r in self.results) / len(self.results)

    @property
    def roi_exact_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.roi_exact) / len(self.results)


def compress_dataset(dataset_path: Path, archive_path: Path, config: PipelineConfig) -> BatchResult:
    """Compress every valid image under dataset_path into one .asice archive.

    One bad image (fails ingestion, ROI, quadtree, tiling, or serialization)
    is logged and skipped -- it does not stop the batch, matching NFR-3's
    guarantee extended across the whole pipeline, not just Stage 1 ingestion.
    """
    agg = DatasetAggregator(Path(dataset_path))
    batch = BatchResult()
    t_start = time.perf_counter()

    for buf in agg.parse_directory():
        try:
            result = compress_image(buf, config)
            batch.results.append(result)
        except PipelineError as exc:
            batch.errors.append((buf.source_path, str(exc)))
            log.error("Skipping %s: %s", buf.source_path, exc)

    for path, reason in agg.errors:  # ingestion failures from Stage 1
        batch.errors.append((path, reason))

    batch.total_time_s = time.perf_counter() - t_start
    _write_archive(archive_path, batch.results)
    return batch


def _write_archive(archive_path, results) -> None:
    """Write one .asice file: [magic][version][count][index table][payloads].

    Index table lets a reader seek directly to one image's payload without
    scanning the whole archive, which matters once a dataset has thousands
    of images (FR-7's "massive reduction in total directory footprint" is
    only useful if the archive itself stays efficiently navigable).
    """
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    index_bytes = bytearray()
    payload_bytes = bytearray()
    offset = 0
    for r in results:
        name_bytes = r.name.encode("utf-8")
        if len(name_bytes) > 65535:
            raise PipelineError(f"image name too long to index: {r.name!r}")
        index_bytes += _INDEX_HEADER.pack(len(name_bytes), r.width, r.height, offset, len(r.payload))
        index_bytes += name_bytes
        payload_bytes += r.payload
        offset += len(r.payload)

    with open(archive_path, "wb") as f:
        f.write(ARCHIVE_MAGIC)
        f.write(struct.pack("<BI", ARCHIVE_VERSION, len(results)))
        f.write(struct.pack("<I", len(index_bytes)))
        f.write(index_bytes)
        f.write(payload_bytes)


@dataclass
class ArchiveIndexEntry:
    name: str
    width: int
    height: int
    offset: int
    length: int


def read_archive_index(archive_path) -> List[ArchiveIndexEntry]:
    """Read just the index table, without loading any image payload --
    lets a caller list an archive's contents cheaply."""
    with open(archive_path, "rb") as f:
        magic = f.read(4)
        if magic != ARCHIVE_MAGIC:
            raise PipelineError(f"not an asice dataset archive (bad magic): {archive_path}")
        version, count = struct.unpack("<BI", f.read(5))
        if version != ARCHIVE_VERSION:
            raise PipelineError(f"unsupported archive version {version}")
        (index_len,) = struct.unpack("<I", f.read(4))
        index_bytes = f.read(index_len)

    entries = []
    pos = 0
    for _ in range(count):
        name_len, width, height, payload_offset, payload_len = _INDEX_HEADER.unpack_from(index_bytes, pos)
        pos += _INDEX_HEADER.size
        name = index_bytes[pos : pos + name_len].decode("utf-8")
        pos += name_len
        entries.append(ArchiveIndexEntry(name, width, height, payload_offset, payload_len))
    return entries


def decode_image_from_archive(archive_path, entry: ArchiveIndexEntry) -> np.ndarray:
    """Read and fully decode one image's payload from the archive, by its
    index entry (from read_archive_index), without touching any other
    image's data."""
    with open(archive_path, "rb") as f:
        f.seek(4)  # past magic
        version, count = struct.unpack("<BI", f.read(5))
        (index_len,) = struct.unpack("<I", f.read(4))
        payload_start = 4 + 5 + 4 + index_len
        f.seek(payload_start + entry.offset)
        payload = f.read(entry.length)

    serializer = ConditionalSerializer()
    tiles = serializer.deserialize(payload)
    return tiles_to_image(tiles, entry.height, entry.width)
