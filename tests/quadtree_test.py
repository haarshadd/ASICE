"""Increment 3 tests: FR-3 (ROI halts merging, exact), FR-4 (background variance threshold)."""

import cv2
import numpy as np
import pytest

from asice.io_buffer import ImageBuffer
from asice.quadtree import (
    QuadtreeDecomposer,
    QuadtreeError,
    leaf_count,
    max_background_error,
    reconstruct,
)


def _flat_image(h, w, color=(100, 100, 100)) -> ImageBuffer:
    img = np.full((h, w, 3), color, np.uint8)
    return ImageBuffer(rgb_matrix=img, source_path="flat.png")


def _object_image(h=128, w=128, cx=64, cy=64, r=25, seed=0) -> tuple[ImageBuffer, np.ndarray]:
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (40, 90, 160), np.uint8)
    cv2.circle(img, (cx, cy), r, (240, 220, 20), -1)
    img = np.clip(img.astype(int) + rng.integers(-2, 3, img.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), r + 3, 1, -1)  # slightly padded, like a real ROI mask would be
    return ImageBuffer(rgb_matrix=img, source_path="obj.png"), mask


# ---------------- FR-3: ROI is exact ----------------
def test_roi_region_reconstructs_pixel_exact():
    buf, mask = _object_image()
    dec = QuadtreeDecomposer(threshold=10)
    root = dec.decompose(buf, mask)
    recon = reconstruct(root, buf.height, buf.width)

    roi_pixels_original = buf.rgb_matrix[mask == 1]
    roi_pixels_recon = recon[mask == 1]
    assert np.array_equal(roi_pixels_original, roi_pixels_recon)


def test_full_roi_image_is_bit_exact_everywhere():
    rng = np.random.default_rng(3)
    noisy = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
    buf = ImageBuffer(rgb_matrix=noisy, source_path="n.png")
    full_mask = np.ones((48, 48), np.uint8)
    dec = QuadtreeDecomposer(threshold=50)  # high T must not matter for ROI
    root = dec.decompose(buf, full_mask)
    recon = reconstruct(root, 48, 48)
    assert np.array_equal(recon, noisy)


def test_roi_leaves_are_single_pixels():
    """FR-3: ROI halts merging immediately, i.e. splits to 1x1 regardless of T."""
    buf, mask = _object_image(h=64, w=64, cx=32, cy=32, r=15)
    dec = QuadtreeDecomposer(threshold=1000)  # absurdly high T; only ROI-ness should force splits
    root = dec.decompose(buf, mask)
    for leaf in root.leaves():
        if leaf.is_roi:
            assert (leaf.w, leaf.h) == (1, 1)


def test_high_threshold_still_keeps_roi_exact():
    """Even when T is so high the whole background collapses to one leaf, ROI must stay exact."""
    buf, mask = _object_image()
    dec = QuadtreeDecomposer(threshold=1e6)
    root = dec.decompose(buf, mask)
    recon = reconstruct(root, buf.height, buf.width)
    assert np.array_equal(buf.rgb_matrix[mask == 1], recon[mask == 1])


# ---------------- FR-4: background thresholding ----------------
def test_flat_background_collapses_to_one_leaf():
    buf = _flat_image(64, 64)
    dec = QuadtreeDecomposer(threshold=5)
    root = dec.decompose(buf, np.zeros((64, 64), np.uint8))
    assert leaf_count(root) == 1
    assert root.is_leaf


def test_background_error_never_exceeds_half_threshold():
    """Range-based merge guarantees max |reconstructed - original| <= T/2 on background."""
    buf, mask = _object_image(seed=7)
    T = 14
    dec = QuadtreeDecomposer(threshold=T, criterion="range")
    root = dec.decompose(buf, mask)
    err = max_background_error(root, buf.rgb_matrix)
    assert err <= T / 2 + 1e-6


def test_lower_threshold_never_increases_leaf_count():
    """Stricter T should split at least as much (monotonic merging)."""
    buf, mask = _object_image(seed=2)
    loose = QuadtreeDecomposer(threshold=40).decompose(buf, mask)
    strict = QuadtreeDecomposer(threshold=4).decompose(buf, mask)
    assert leaf_count(strict) >= leaf_count(loose)


def test_min_block_caps_recursion_on_mildly_noisy_background():
    """min_block should still cap ordinary mild noise (variance a bit over T),
    which is the case it's meant for. Severe non-uniformity (a real edge
    hiding inside a small block) is allowed past the floor — see
    test_severe_non_uniformity_overrides_min_block below."""
    rng = np.random.default_rng(5)
    mild = 120 + rng.integers(-3, 4, (64, 64, 3))  # small spread around 120
    buf = ImageBuffer(rgb_matrix=mild.astype(np.uint8), source_path="mild.png")
    # T chosen realistically (not so strict that ordinary noise looks like
    # a severe outlier relative to T*severity_factor); mirrors real usage.
    dec = QuadtreeDecomposer(threshold=8, min_block=4)
    root = dec.decompose(buf, np.zeros((64, 64), np.uint8))
    for leaf in root.leaves():
        assert leaf.w >= 4 and leaf.h >= 4


def test_severe_non_uniformity_overrides_min_block_floor():
    """A block whose variance is far past T (a real edge, not mild noise)
    must keep splitting past min_block rather than silently absorbing a
    large per-pixel error — this reproduces a real bug found via
    quadtree-preview on an actual image (an ROI-boundary anti-aliased
    pixel sitting inside an otherwise-uniform 2x2 background leaf gave a
    max per-pixel error of 160 against T=10 before this fix)."""
    img = np.full((16, 16, 3), 100, np.uint8)
    img[8, 8] = [255, 0, 0]  # one wildly different pixel inside a flat block
    buf = ImageBuffer(rgb_matrix=img, source_path="edge.png")
    dec = QuadtreeDecomposer(threshold=10, min_block=4)
    root = dec.decompose(buf, np.zeros((16, 16), np.uint8))
    err = max_background_error(root, img)
    # not a hard T/2 guarantee (criterion is variance, not range, by default)
    # but must be far below the ~150 it was before the floor-override fix
    assert err < 50


def test_variance_criterion_is_selectable():
    buf, mask = _object_image()
    dec = QuadtreeDecomposer(threshold=50, criterion="variance")
    root = dec.decompose(buf, mask)
    recon = reconstruct(root, buf.height, buf.width)
    assert np.array_equal(buf.rgb_matrix[mask == 1], recon[mask == 1])  # ROI still exact


# ---------------- Structural ----------------
def test_non_power_of_two_dimensions_reconstruct_to_original_shape():
    h, w = 137, 201
    buf = _flat_image(h, w)
    dec = QuadtreeDecomposer(threshold=10)
    root = dec.decompose(buf, np.zeros((h, w), np.uint8))
    recon = reconstruct(root, h, w)
    assert recon.shape == (h, w, 3)
    assert (root.x, root.y, root.w, root.h) == (0, 0, w, h)


def test_mask_shape_mismatch_raises():
    buf = _flat_image(32, 32)
    dec = QuadtreeDecomposer()
    with pytest.raises(QuadtreeError):
        dec.decompose(buf, np.zeros((16, 16), np.uint8))


def test_leaves_are_a_partition_no_gaps_no_overlap():
    """Every pixel is covered by exactly one leaf."""
    buf, mask = _object_image(h=96, w=96, seed=9)
    dec = QuadtreeDecomposer(threshold=12)
    root = dec.decompose(buf, mask)

    side = 128  # next pow2 >= 96
    coverage = np.zeros((side, side), np.int32)
    for leaf in root.leaves():
        region = coverage[leaf.y : leaf.y + leaf.h, leaf.x : leaf.x + leaf.w]
        assert (region == 0).all(), "overlap detected"
        coverage[leaf.y : leaf.y + leaf.h, leaf.x : leaf.x + leaf.w] = 1
    assert coverage.sum() == side * side


def test_evaluate_background_variance_is_zero_for_flat_background():
    buf = _flat_image(32, 32)
    dec = QuadtreeDecomposer()
    v = dec.evaluate_background_variance(buf, np.zeros((32, 32), np.uint8))
    assert v == pytest.approx(0.0, abs=1e-6)


def test_evaluate_background_variance_nonzero_for_noisy_background():
    rng = np.random.default_rng(4)
    noisy = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    buf = ImageBuffer(rgb_matrix=noisy, source_path="n.png")
    dec = QuadtreeDecomposer()
    v = dec.evaluate_background_variance(buf, np.zeros((32, 32), np.uint8))
    assert v > 100


def test_all_roi_pixels_get_is_roi_true_leaves_only():
    """No leaf should be flagged non-ROI if it lies entirely inside the mask."""
    buf, mask = _object_image(h=64, w=64, cx=32, cy=32, r=10, seed=6)
    dec = QuadtreeDecomposer(threshold=10)
    root = dec.decompose(buf, mask)
    for leaf in root.leaves():
        region = mask[leaf.y : leaf.y + leaf.h, leaf.x : leaf.x + leaf.w]
        if region.size and region.all():
            assert leaf.is_roi
        if region.size and not region.any():
            assert not leaf.is_roi


def test_default_criterion_is_variance_and_keeps_roi_exact():
    """Default criterion must stay usable at realistic image sizes without exact-range cost."""
    buf, mask = _object_image(h=256, w=256, cx=128, cy=128, r=50, seed=11)
    dec = QuadtreeDecomposer(threshold=12)  # default criterion
    assert dec.criterion == "variance"
    root = dec.decompose(buf, mask)
    recon = reconstruct(root, buf.height, buf.width)
    assert np.array_equal(buf.rgb_matrix[mask == 1], recon[mask == 1])


def test_range_criterion_available_as_opt_in():
    buf, mask = _object_image(seed=12)
    dec = QuadtreeDecomposer(threshold=14, criterion="range")
    root = dec.decompose(buf, mask)
    err = max_background_error(root, buf.rgb_matrix)
    assert err <= 14 / 2 + 1e-6