"""Increment 2 tests: Saliency & ROI segmentation (FR-2, fallback mechanisms, user masks)."""

from pathlib import Path
import sys


import cv2
import numpy as np
import pytest

from asice.io_buffer import ImageBuffer
from asice.roi import (
    ROIError,
    SaliencySegmenter,
    _fill_enclosed_holes,
    _mask_from_saliency,
    _spectral_residual_saliency,
    classical_roi_mask,
    load_user_mask,
    roi_coverage,
)


def _make_buffer(
    tmp_path: Path, filename: str = "sample.png", h: int = 64, w: int = 64, draw_object: bool = True
) -> ImageBuffer:
    """Helper to create a temporary image buffer with optional high-contrast object."""
    rgb = np.full((h, w, 3), 40, dtype=np.uint8)  # dark background
    if draw_object:
        # Draw a bright white square in the center
        rgb[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 220

    img_path = tmp_path / filename
    cv2.imwrite(str(img_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return ImageBuffer.ingest_uncompressed_image(img_path)


# ---------------- FR-2: Mask Properties & Interface ----------------
def test_mask_dimensions_and_binary_values(tmp_path):
    buf = _make_buffer(tmp_path, h=32, w=48)
    segmenter = SaliencySegmenter(method="classical")
    mask, meta = segmenter.generate_binary_values(buf)

    assert mask.shape == (32, 48)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
    assert meta["method"] == "classical"


def test_roi_coverage_bounds(tmp_path):
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert roi_coverage(mask) == 0.0

    mask[0:5, :] = 1
    assert roi_coverage(mask) == 0.5

    mask[:] = 1
    assert roi_coverage(mask) == 1.0


# ---------------- Backend 1: Classical Spectral Residual ----------------
def test_classical_saliency_detects_contrast_region(tmp_path):
    buf = _make_buffer(tmp_path, h=64, w=64, draw_object=True)
    mask = classical_roi_mask(buf, percentile=90.0, dilate_px=2)

    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    # Center object should be covered by ROI
    assert mask[32, 32] == 1


def test_classical_saliency_flat_image_returns_empty_mask(tmp_path):
    """Uniform images should yield an empty mask (mean + k*std thresholding safeguard)."""
    buf = _make_buffer(tmp_path, h=32, w=32, draw_object=False)
    mask = classical_roi_mask(buf)

    assert np.all(mask == 0)
    assert roi_coverage(mask) == 0.0


def test_hole_filling_closes_enclosed_zeroes():
    """Test that _fill_enclosed_holes converts hollow ring masks to solid shapes."""
    # Create a 10x10 binary mask with a 1-pixel border and a hollow 4x4 center
    ring_mask = np.zeros((10, 10), dtype=np.uint8)
    ring_mask[2:8, 2:8] = 1
    ring_mask[4:6, 4:6] = 0  # hollow hole inside

    filled = _fill_enclosed_holes(ring_mask)

    # Hole inside should now be filled with 1s
    assert np.all(filled[4:6, 4:6] == 1)
    # Background outside outer boundary remains 0
    assert filled[0, 0] == 0


def test_mask_from_saliency_dilation():
    saliency = np.zeros((20, 20), dtype=np.float32)
    saliency[8:12, 8:12] = 1.0  # High saliency block

    mask_no_dilation = _mask_from_saliency(saliency, percentile=90.0, dilate_px=0)
    mask_dilated = _mask_from_saliency(saliency, percentile=90.0, dilate_px=3)

    assert np.sum(mask_dilated) > np.sum(mask_no_dilation)


# ---------------- Backend 2: U^2-Net Fallback Behavior ----------------
def test_u2net_falls_back_to_classical_when_weights_missing(tmp_path):
    """When model_path does not exist, system logs warning and falls back to classical."""
    buf = _make_buffer(tmp_path)
    missing_model = tmp_path / "non_existent_u2net.onnx"

    segmenter = SaliencySegmenter(method="u2net", model_path=missing_model)
    mask, meta = segmenter.generate_binary_values(buf)

    assert mask.shape == (buf.height, buf.width)
    assert set(np.unique(mask)).issubset({0, 1})
    assert meta["method"] == "classical"
    assert meta["fallback"] is True


def test_u2net_failed_load_persists_failure_state(tmp_path):
    """Once load fails, subsequent requests skip re-attempting file load."""
    buf = _make_buffer(tmp_path)
    missing_model = tmp_path / "missing.onnx"

    segmenter = SaliencySegmenter(method="u2net", model_path=missing_model)
    _ = segmenter.generate_binary_values(buf)

    assert segmenter._u2net_load_failed is True
    # Second call should immediately yield classical fallback without raising
    _, meta = segmenter.generate_binary_values(buf)
    assert meta["method"] == "classical"


# ---------------- Backend 3: User-Supplied Mask Directory ----------------
def test_load_user_mask_matches_by_filename_stem(tmp_path):
    buf = _make_buffer(tmp_path, filename="item_001.png", h=32, w=32)

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    # Create matching mask with same stem 'item_001'
    user_mask_img = np.zeros((32, 32), dtype=np.uint8)
    user_mask_img[10:20, 10:20] = 255
    cv2.imwrite(str(mask_dir / "item_001.png"), user_mask_img)

    mask = load_user_mask(mask_dir, buf)

    assert mask.shape == (32, 32)
    assert mask.dtype == np.uint8
    assert mask[15, 15] == 1
    assert mask[0, 0] == 0


def test_load_user_mask_resizes_mismatched_dimensions(tmp_path):
    buf = _make_buffer(tmp_path, filename="photo.jpg", h=64, w=64)

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    # User mask is 32x32, image is 64x64
    small_mask = np.full((32, 32), 255, dtype=np.uint8)
    cv2.imwrite(str(mask_dir / "photo.png"), small_mask)

    mask = load_user_mask(mask_dir, buf)

    assert mask.shape == (64, 64)
    assert np.all(mask == 1)


def test_load_user_mask_missing_mask_raises_roierror(tmp_path):
    buf = _make_buffer(tmp_path, filename="missing_mask_img.png")
    mask_dir = tmp_path / "empty_mask_dir"
    mask_dir.mkdir()

    with pytest.raises(ROIError, match="No mask found for 'missing_mask_img'"):
        load_user_mask(mask_dir, buf)


def test_mask_dir_via_segmenter_interface(tmp_path):
    buf = _make_buffer(tmp_path, filename="test.png", h=20, w=20)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    user_mask = np.ones((20, 20), dtype=np.uint8) * 255
    cv2.imwrite(str(mask_dir / "test.png"), user_mask)

    segmenter = SaliencySegmenter(method="mask-dir", mask_dir=mask_dir)
    mask, meta = segmenter.generate_binary_values(buf)

    assert np.all(mask == 1)
    assert meta == {"method": "mask-dir"}


# ---------------- Configuration & Error Handling ----------------
def test_invalid_roi_method_raises_roierror():
    with pytest.raises(ROIError, match="Unknown roi method"):
        SaliencySegmenter(method="invalid_backend")


def test_mask_dir_method_without_mask_dir_raises_roierror():
    with pytest.raises(ROIError, match="method='mask-dir' requires mask_dir"):
        SaliencySegmenter(method="mask-dir", mask_dir=None)