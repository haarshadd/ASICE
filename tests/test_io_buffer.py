"""Increment 1 tests: FR-1, NFR-3, NFR-4."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from asice.cli import main
from asice.io_buffer import MAX_DIM, DatasetAggregator, ImageBuffer, ImageError


def _write_rgb(path: Path, h=32, w=48, color=(200, 30, 60)) -> np.ndarray:
    """Write a solid RGB image; return the RGB array that was intended."""
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = color
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return rgb


# ---------------- FR-1: single image ----------------
def test_ingest_single_image_shape_and_dtype(tmp_path):
    _write_rgb(tmp_path / "a.png", h=32, w=48)
    buf = ImageBuffer.ingest_uncompressed_image(tmp_path / "a.png")
    assert (buf.height, buf.width) == (32, 48)
    assert buf.rgb_matrix.shape == (32, 48, 3)
    assert buf.rgb_matrix.dtype == np.uint8
    assert buf.raw_size_bytes == 32 * 48 * 3
    assert buf.original_bytes > 0


def test_colour_order_is_rgb_not_bgr(tmp_path):
    """A red image must come back with R=200 in channel 0 (guards BGR/RGB swap)."""
    _write_rgb(tmp_path / "red.png", color=(200, 30, 60))
    buf = ImageBuffer.ingest_uncompressed_image(tmp_path / "red.png")
    assert tuple(buf.rgb_matrix[0, 0]) == (200, 30, 60)


def test_lossless_png_roundtrips_exactly(tmp_path):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "n.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    buf = ImageBuffer.ingest_uncompressed_image(tmp_path / "n.png")
    assert np.array_equal(buf.rgb_matrix, rgb)


# ---------------- FR-1: folder / batch ----------------
def test_folder_yields_all_valid_images_sorted(tmp_path):
    for name in ["c.png", "a.png", "b.jpg"]:
        _write_rgb(tmp_path / name)
    agg = DatasetAggregator(tmp_path)
    got = [Path(b.source_path).name for b in agg.parse_directory()]
    assert got == ["a.png", "b.jpg", "c.png"]
    assert agg.errors == []


def test_recursive_and_non_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    _write_rgb(tmp_path / "top.png")
    _write_rgb(tmp_path / "sub" / "deep.png")
    assert DatasetAggregator(tmp_path, recursive=True).count_candidates() == 2
    assert DatasetAggregator(tmp_path, recursive=False).count_candidates() == 1


def test_single_file_path_is_accepted(tmp_path):
    _write_rgb(tmp_path / "one.png")
    agg = DatasetAggregator(tmp_path / "one.png")
    assert len(list(agg.parse_directory())) == 1


def test_missing_dataset_path_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        DatasetAggregator(tmp_path / "nope").list_files()


def test_non_image_files_are_ignored_not_errors(tmp_path):
    _write_rgb(tmp_path / "ok.png")
    (tmp_path / "notes.txt").write_text("hello")
    agg = DatasetAggregator(tmp_path)
    assert len(list(agg.parse_directory())) == 1
    assert agg.errors == []


# ---------------- NFR-3: bad input never crashes ----------------
def test_corrupted_image_is_rejected_with_clear_error(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"this is not a png at all")
    with pytest.raises(ImageError, match="Corrupted"):
        ImageBuffer.ingest_uncompressed_image(tmp_path / "bad.png")


def test_truncated_png_is_rejected(tmp_path):
    _write_rgb(tmp_path / "ok.png", h=64, w=64)
    data = (tmp_path / "ok.png").read_bytes()
    (tmp_path / "trunc.png").write_bytes(data[: len(data) // 3])
    with pytest.raises(ImageError):
        ImageBuffer.ingest_uncompressed_image(tmp_path / "trunc.png")


def test_grayscale_is_rejected(tmp_path):
    cv2.imwrite(str(tmp_path / "g.png"), np.zeros((10, 10), np.uint8))
    with pytest.raises(ImageError, match="grayscale"):
        ImageBuffer.ingest_uncompressed_image(tmp_path / "g.png")


def test_rgba_is_rejected(tmp_path):
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((10, 10, 4), np.uint8))
    with pytest.raises(ImageError, match="4 channels"):
        ImageBuffer.ingest_uncompressed_image(tmp_path / "a.png")


def test_16bit_is_rejected(tmp_path):
    cv2.imwrite(str(tmp_path / "s.png"), np.zeros((10, 10, 3), np.uint16))
    with pytest.raises(ImageError, match="8-bit"):
        ImageBuffer.ingest_uncompressed_image(tmp_path / "s.png")


def test_unsupported_extension_is_rejected(tmp_path):
    (tmp_path / "x.gif").write_bytes(b"GIF89a")
    with pytest.raises(ImageError, match="Unsupported file type"):
        ImageBuffer.ingest_uncompressed_image(tmp_path / "x.gif")


def test_oversize_image_is_rejected(tmp_path):
    big = np.zeros((MAX_DIM + 1, 8, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "big.png"), big)
    with pytest.raises(ImageError, match="exceeds"):
        ImageBuffer.ingest_uncompressed_image(tmp_path / "big.png")


def test_max_size_boundary_is_accepted(tmp_path):
    ok = np.zeros((MAX_DIM, 4, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "edge.png"), ok)
    buf = ImageBuffer.ingest_uncompressed_image(tmp_path / "edge.png")
    assert buf.height == MAX_DIM


def test_batch_survives_bad_files_and_records_errors(tmp_path):
    """The core NFR-3 guarantee: one bad file never stops the batch."""
    _write_rgb(tmp_path / "1_ok.png")
    (tmp_path / "2_bad.png").write_bytes(b"garbage")
    cv2.imwrite(str(tmp_path / "3_gray.png"), np.zeros((8, 8), np.uint8))
    _write_rgb(tmp_path / "4_ok.png")

    agg = DatasetAggregator(tmp_path)
    good = list(agg.parse_directory())

    assert [Path(b.source_path).name for b in good] == ["1_ok.png", "4_ok.png"]
    assert len(agg.errors) == 2
    assert {Path(p).name for p, _ in agg.errors} == {"2_bad.png", "3_gray.png"}


def test_direct_construction_validates_array():
    with pytest.raises(ImageError):
        ImageBuffer(rgb_matrix=np.zeros((5, 5), np.uint8))
    with pytest.raises(ImageError):
        ImageBuffer(rgb_matrix=np.zeros((5, 5, 3), np.float32))


# ---------------- NFR-4 / CLI ----------------
def test_cli_inspect_reports_and_writes_log(tmp_path, capsys):
    _write_rgb(tmp_path / "ok.png")
    (tmp_path / "bad.png").write_bytes(b"garbage")
    logf = tmp_path / "run.log"
    rc = main(["inspect", str(tmp_path), "--log-file", str(logf)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Valid RGB images : 1" in out
    assert "Rejected         : 1" in out
    assert "Corrupted" in logf.read_text()  # NFR-3: error recorded in execution log


def test_cli_inspect_missing_path_returns_error_code(tmp_path):
    assert main(["inspect", str(tmp_path / "missing")]) == 2


def test_cli_inspect_no_valid_images_returns_nonzero(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"garbage")
    assert main(["inspect", str(tmp_path)]) == 1


def test_cli_compress_accepts_parameters_without_recompiling():
    """NFR-4: the parameters exist and parse; pipeline itself lands in later increments."""
    from asice.cli import build_parser

    a = build_parser().parse_args(
        ["compress", "data", "--target-cr", "6", "-T", "20", "--roi-method", "classical"]
    )
    assert a.target_cr == 6.0 and a.threshold == 20.0 and a.roi_method == "classical"
    assert a.merge_tolerance is None  # defaults to T later


def test_cli_compress_rejects_bad_parameters(tmp_path):
    assert main(["compress", str(tmp_path), "--target-cr", "0"]) == 2
    assert main(["compress", str(tmp_path), "-T", "-1"]) == 2
    assert main(["compress", str(tmp_path), "--roi-method", "mask-dir"]) == 2
