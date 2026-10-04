"""Increment 6 tests: FR-7 (pipeline orchestration, mode switch, .asice archive)."""

import cv2
import numpy as np
import pytest

from asice.io_buffer import ImageBuffer
from asice.pipeline import (
    ArchiveIndexEntry,
    BatchResult,
    PipelineConfig,
    PipelineError,
    compress_dataset,
    compress_image,
    decode_image_from_archive,
    read_archive_index,
    resolve_mode,
)


def _object_image(tmp_path, name="obj.png", h=150, w=200, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (40, 90, 160), np.uint8)
    cv2.circle(img, (w // 2, h // 2), min(h, w) // 4, (240, 220, 20), -1)
    img = np.clip(img.astype(int) + rng.integers(-2, 3, img.shape), 0, 255).astype(np.uint8)
    path = tmp_path / name
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


# ---------------- mode resolution ----------------
def test_resolve_mode_strict_uses_range():
    assert resolve_mode("strict").criterion == "range"


def test_resolve_mode_fast_uses_variance():
    assert resolve_mode("fast").criterion == "variance"


def test_resolve_mode_rejects_unknown():
    with pytest.raises(PipelineError):
        resolve_mode("bogus")


# ---------------- compress_image ----------------
def test_compress_image_strict_mode_end_to_end(tmp_path):
    path = _object_image(tmp_path)
    buf = ImageBuffer.ingest_uncompressed_image(path)
    cfg = PipelineConfig(mode="strict", threshold=12, roi_method="classical")
    result = compress_image(buf, cfg)
    assert result.roi_exact is True
    assert result.final_cr > 0
    assert result.mode == "strict"


def test_compress_image_fast_mode_end_to_end(tmp_path):
    path = _object_image(tmp_path)
    buf = ImageBuffer.ingest_uncompressed_image(path)
    cfg = PipelineConfig(mode="fast", threshold=12, roi_method="classical")
    result = compress_image(buf, cfg)
    assert result.roi_exact is True
    assert result.mode == "fast"


def test_compress_image_strict_gives_tighter_or_equal_error_than_fast(tmp_path):
    """Strict (range) should never report a worse bound than fast (variance)
    on the same image -- that's the entire point of the mode split."""
    path = _object_image(tmp_path, seed=3)
    buf = ImageBuffer.ingest_uncompressed_image(path)
    strict = compress_image(buf, PipelineConfig(mode="strict", threshold=10, roi_method="classical"))
    fast = compress_image(buf, PipelineConfig(mode="fast", threshold=10, roi_method="classical"))
    assert strict.max_background_error <= fast.max_background_error + 1e-6


def test_compress_image_round_trip_matches_reported_roi_exact(tmp_path):
    """result.roi_exact must reflect the FINAL decoded payload, not an
    intermediate stage -- regression guard for the exact mislabeling bug
    found in the earlier Streamlit app (quadtree-only error shown next to
    final-pipeline metrics)."""
    path = _object_image(tmp_path, seed=4)
    buf = ImageBuffer.ingest_uncompressed_image(path)
    cfg = PipelineConfig(mode="strict", threshold=10, roi_method="classical")
    result = compress_image(buf, cfg)

    from asice.conditional_huffman import ConditionalSerializer
    from asice.dp_tiling import tiles_to_image

    serializer = ConditionalSerializer()
    tiles = serializer.deserialize(result.payload)
    recon = tiles_to_image(tiles, buf.height, buf.width)

    seg_mask = None  # recompute mask the same way compress_image did, for the check
    from asice.roi import SaliencySegmenter
    seg = SaliencySegmenter(method="classical")
    mask, _ = seg.generate_binary_values(buf)
    actually_exact = bool(np.array_equal(buf.rgb_matrix[mask == 1], recon[mask == 1]))
    assert result.roi_exact == actually_exact


# ---------------- compress_dataset / batch ----------------
def test_compress_dataset_skips_bad_files_not_whole_batch(tmp_path):
    _object_image(tmp_path, name="good1.png", seed=1)
    _object_image(tmp_path, name="good2.png", seed=2)
    (tmp_path / "bad.png").write_bytes(b"not an image")

    archive_path = tmp_path / "out.asice"
    cfg = PipelineConfig(mode="fast", threshold=12, roi_method="classical")
    batch = compress_dataset(tmp_path, archive_path, cfg)

    assert len(batch.results) == 2
    assert len(batch.errors) == 1
    assert "bad.png" in batch.errors[0][0]
    assert archive_path.is_file()


def test_compress_dataset_empty_folder_produces_empty_archive(tmp_path):
    (tmp_path / "empty_dir").mkdir()
    archive_path = tmp_path / "out.asice"
    cfg = PipelineConfig(mode="fast", roi_method="classical")
    batch = compress_dataset(tmp_path / "empty_dir", archive_path, cfg)
    assert len(batch.results) == 0
    assert archive_path.is_file()
    assert read_archive_index(archive_path) == []


def test_batch_result_mean_cr_and_roi_exact_rate(tmp_path):
    _object_image(tmp_path, name="a.png", seed=1)
    _object_image(tmp_path, name="b.png", seed=2)
    archive_path = tmp_path / "out.asice"
    cfg = PipelineConfig(mode="fast", roi_method="classical")
    batch = compress_dataset(tmp_path, archive_path, cfg)
    assert batch.mean_cr > 0
    assert batch.roi_exact_rate == 1.0  # both should round-trip exact


def test_batch_result_empty_properties_do_not_crash():
    b = BatchResult()
    assert b.mean_cr == 0.0
    assert b.roi_exact_rate == 0.0


# ---------------- archive format ----------------
def test_archive_index_matches_compressed_images(tmp_path):
    _object_image(tmp_path, name="x.png", seed=5)
    _object_image(tmp_path, name="y.png", seed=6)
    archive_path = tmp_path / "out.asice"
    cfg = PipelineConfig(mode="fast", roi_method="classical")
    compress_dataset(tmp_path, archive_path, cfg)

    entries = read_archive_index(archive_path)
    names = sorted(e.name for e in entries)
    assert names == ["x.png", "y.png"]


def test_decode_image_from_archive_matches_original_shape(tmp_path):
    path = _object_image(tmp_path, name="z.png", h=120, w=160, seed=7)
    archive_path = tmp_path / "out.asice"
    cfg = PipelineConfig(mode="strict", threshold=10, roi_method="classical")
    compress_dataset(tmp_path, archive_path, cfg)

    entries = read_archive_index(archive_path)
    entry = next(e for e in entries if e.name == "z.png")
    decoded = decode_image_from_archive(archive_path, entry)
    assert decoded.shape == (120, 160, 3)


def test_decode_image_from_archive_roi_is_exact(tmp_path):
    path = _object_image(tmp_path, name="roi_check.png", h=100, w=140, seed=8)
    archive_path = tmp_path / "out.asice"
    cfg = PipelineConfig(mode="strict", threshold=10, roi_method="classical")
    compress_dataset(tmp_path, archive_path, cfg)

    original = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    from asice.roi import SaliencySegmenter
    seg = SaliencySegmenter(method="classical")
    buf = ImageBuffer.ingest_uncompressed_image(path)
    mask, _ = seg.generate_binary_values(buf)

    entries = read_archive_index(archive_path)
    entry = next(e for e in entries if e.name == "roi_check.png")
    decoded = decode_image_from_archive(archive_path, entry)
    assert np.array_equal(original[mask == 1], decoded[mask == 1])


def test_read_archive_index_rejects_bad_magic(tmp_path):
    bad_file = tmp_path / "notanarchive.asice"
    bad_file.write_bytes(b"NOPE" + b"\x00" * 20)
    with pytest.raises(PipelineError):
        read_archive_index(bad_file)


def test_multiple_images_indexed_at_correct_offsets(tmp_path):
    for i in range(3):
        _object_image(tmp_path, name=f"img{i}.png", seed=i)
    archive_path = tmp_path / "out.asice"
    cfg = PipelineConfig(mode="fast", roi_method="classical")
    compress_dataset(tmp_path, archive_path, cfg)

    entries = read_archive_index(archive_path)
    assert len(entries) == 3
    # offsets should be strictly increasing and non-overlapping
    entries_sorted = sorted(entries, key=lambda e: e.offset)
    for a, b in zip(entries_sorted, entries_sorted[1:]):
        assert a.offset + a.length <= b.offset


# ---------------- PipelineConfig defaults ----------------
def test_pipeline_config_defaults_are_strict_mode():
    cfg = PipelineConfig()
    assert cfg.mode == "strict"


def test_pipeline_config_merge_tolerance_defaults_to_none():
    """None means 'reuse threshold T', per the existing DPTiler contract."""
    cfg = PipelineConfig()
    assert cfg.merge_tolerance is None
