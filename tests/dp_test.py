"""Increment 4 tests: FR-5 (DP tiling merges adjacent background leaves,
never touches ROI, never introduces error beyond tolerance)."""

import cv2
import numpy as np
import pytest

from asice.dp_tiling import (
    DPTiler,
    DPTilingError,
    reduction_ratio,
    tiles_to_image,
    tiling_leaf_count,
)
from asice.io_buffer import ImageBuffer
from asice.quadtree import QuadtreeDecomposer, leaf_count, reconstruct


def _object_image(h=128, w=128, cx=64, cy=64, r=25, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (40, 90, 160), np.uint8)
    cv2.circle(img, (cx, cy), r, (240, 220, 20), -1)
    img = np.clip(img.astype(int) + rng.integers(-2, 3, img.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), r + 3, 1, -1)
    return ImageBuffer(rgb_matrix=img, source_path="obj.png"), mask


# ---------------- core merging behaviour ----------------
def test_merges_siblings_the_quadtree_could_not():
    """A flat background with one small feature in a corner forces the
    quadtree to split into 4 top-level quadrants; 3 of those are pure
    background and identical, but a quadtree can never re-merge siblings
    across that split. DP tiling should."""
    img = np.full((64, 64, 3), (135, 206, 235), np.uint8)
    img[0:8, 0:8] = (255, 255, 255)
    mask = np.zeros((64, 64), np.uint8)
    buf = ImageBuffer(rgb_matrix=img, source_path="sky.png")

    dec = QuadtreeDecomposer(threshold=5, min_block=2)
    root = dec.decompose(buf, mask)
    n_leaves = leaf_count(root)

    tiler = DPTiler()
    tiles = tiler.analyze_adjacent_leaf_nodes(root, default_tolerance=5, original_rgb=img)
    assert tiling_leaf_count(tiles) < n_leaves


def test_image_content_unchanged_by_tiling():
    """DP tiling must not alter what the image looks like — only how many
    records it takes to store it."""
    img = np.full((64, 64, 3), (135, 206, 235), np.uint8)
    img[0:8, 0:8] = (255, 255, 255)
    mask = np.zeros((64, 64), np.uint8)
    buf = ImageBuffer(rgb_matrix=img, source_path="sky.png")

    dec = QuadtreeDecomposer(threshold=5, min_block=2)
    root = dec.decompose(buf, mask)
    tiler = DPTiler()
    tiles = tiler.analyze_adjacent_leaf_nodes(root, default_tolerance=5, original_rgb=img)

    recon_quadtree = reconstruct(root, 64, 64)
    recon_tiles = tiles_to_image(tiles, 64, 64)
    assert np.array_equal(recon_quadtree, recon_tiles)


def test_dissimilar_neighbours_are_not_merged():
    """Two adjacent leaves with very different values must stay separate."""
    img = np.zeros((16, 32, 3), np.uint8)
    img[:, :16] = (10, 10, 10)
    img[:, 16:] = (250, 250, 250)
    mask = np.zeros((16, 32), np.uint8)
    buf = ImageBuffer(rgb_matrix=img, source_path="split.png")

    dec = QuadtreeDecomposer(threshold=2, min_block=1)
    root = dec.decompose(buf, mask)
    tiler = DPTiler()
    tiles = tiler.analyze_adjacent_leaf_nodes(root, default_tolerance=2, original_rgb=img)

    # no tile should span both the dark and light halves (tile covers
    # columns [x, x+w) — only a problem if it straddles the boundary,
    # i.e. starts before column 16 and ends after it)
    for t in tiles:
        if t.x < 16 < t.x + t.w:
            pytest.fail(f"tile spans the hard edge: x={t.x} w={t.w}")


# ---------------- FR-3: ROI is never touched ----------------
def test_roi_leaves_pass_through_unmerged_and_unmoved():
    buf, mask = _object_image(seed=1)
    dec = QuadtreeDecomposer(threshold=10)
    root = dec.decompose(buf, mask)
    tiler = DPTiler()
    tiles = tiler.analyze_adjacent_leaf_nodes(root, default_tolerance=10, original_rgb=buf.rgb_matrix)

    roi_tiles = [t for t in tiles if t.is_roi]
    assert all(t.w == 1 and t.h == 1 for t in roi_tiles)
    assert all(t.merged_from == 1 for t in roi_tiles)


def test_roi_stays_pixel_exact_after_tiling():
    buf, mask = _object_image(seed=2)
    dec = QuadtreeDecomposer(threshold=10)
    root = dec.decompose(buf, mask)
    tiler = DPTiler()
    tiles = tiler.analyze_adjacent_leaf_nodes(root, default_tolerance=10, original_rgb=buf.rgb_matrix)

    recon = tiles_to_image(tiles, buf.height, buf.width)
    assert np.array_equal(buf.rgb_matrix[mask == 1], recon[mask == 1])


def test_roi_leaf_count_matches_quadtree_roi_leaf_count():
    """Tiling must not add, drop, or merge any ROI leaf."""
    buf, mask = _object_image(seed=3)
    dec = QuadtreeDecomposer(threshold=10)
    root = dec.decompose(buf, mask)
    quadtree_roi_count = sum(1 for l in root.leaves() if l.is_roi)

    tiler = DPTiler()
    tiles = tiler.analyze_adjacent_leaf_nodes(root, default_tolerance=10, original_rgb=buf.rgb_matrix)
    tile_roi_count = sum(1 for t in tiles if t.is_roi)
    assert tile_roi_count == quadtree_roi_count


# ---------------- error bound ----------------
def test_merge_never_exceeds_tolerance_against_true_pixels():
    """Direct regression test: on a smooth gradient, checking merge
    tolerance against quadtree leaf VALUES alone (rather than the true
    original pixels) let merges compound error past tolerance — measured
    directly at 4.0 against tolerance=3 before this was fixed. Passing
    original_rgb closes that gap; this must hold even on a long, smoothly
    drifting gradient where many small in-tolerance steps could otherwise
    accumulate."""
    w = 512
    grad = np.linspace(0, 255, w).astype(np.uint8)
    img = np.tile(grad.reshape(1, w, 1), (32, 1, 3))
    mask = np.zeros((32, w), np.uint8)
    buf = ImageBuffer(rgb_matrix=img, source_path="grad.png")

    dec = QuadtreeDecomposer(threshold=3, min_block=1)
    root = dec.decompose(buf, mask)
    tiler = DPTiler(max_merge_run=16)
    tiles = tiler.analyze_adjacent_leaf_nodes(root, default_tolerance=3, original_rgb=img)

    worst = 0.0
    for t in tiles:
        block = img[t.y : t.y + t.h, t.x : t.x + t.w].astype(np.float64)
        worst = max(worst, float(np.abs(block - t.value).max()))
    assert worst <= 3.0 + 1e-6


def test_without_original_rgb_still_does_not_crash():
    """original_rgb is optional; omitting it falls back to the weaker
    leaf-value check rather than failing."""
    buf, mask = _object_image(seed=4)
    dec = QuadtreeDecomposer(threshold=10)
    root = dec.decompose(buf, mask)
    tiler = DPTiler()
    tiles = tiler.analyze_adjacent_leaf_nodes(root, default_tolerance=10)  # no original_rgb
    assert len(tiles) > 0


# ---------------- tiles never extend past true bounds ----------------
def test_tiles_respect_true_image_bounds_non_power_of_two():
    """Companion to the quadtree's own padding-leak regression test: a
    merged tile must never report w/h reaching into the quadtree's
    internal power-of-two padding."""
    h, w = 32, 512
    img = np.full((h, w, 3), 100, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    buf = ImageBuffer(rgb_matrix=img, source_path="flat.png")

    dec = QuadtreeDecomposer(threshold=3, min_block=1)
    root = dec.decompose(buf, mask)
    tiler = DPTiler(max_merge_run=16)
    tiles = tiler.analyze_adjacent_leaf_nodes(root, default_tolerance=3, original_rgb=img)

    for t in tiles:
        assert t.y + t.h <= h, f"tile y={t.y} h={t.h} exceeds true height {h}"
        assert t.x + t.w <= w, f"tile x={t.x} w={t.w} exceeds true width {w}"


# ---------------- max_merge_run safety cap ----------------
def test_max_merge_run_caps_a_single_horizontal_group():
    w = 512
    grad = np.linspace(0, 255, w).astype(np.uint8)
    img = np.tile(grad.reshape(1, w, 1), (8, 1, 3))
    mask = np.zeros((8, w), np.uint8)
    buf = ImageBuffer(rgb_matrix=img, source_path="grad.png")

    dec = QuadtreeDecomposer(threshold=5, min_block=1)
    root = dec.decompose(buf, mask)
    tiler = DPTiler(max_merge_run=8)
    tiles = tiler.analyze_adjacent_leaf_nodes(root, default_tolerance=5, original_rgb=img)

    for t in tiles:
        assert t.merged_from <= 8 * 2  # horizontal cap x vertical-pass headroom


# ---------------- degenerate inputs ----------------
def test_flat_background_collapses_to_one_tile():
    img = np.full((32, 32, 3), 100, np.uint8)
    mask = np.zeros((32, 32), np.uint8)
    buf = ImageBuffer(rgb_matrix=img, source_path="flat.png")
    dec = QuadtreeDecomposer(threshold=5)
    root = dec.decompose(buf, mask)
    tiler = DPTiler()
    tiles = tiler.analyze_adjacent_leaf_nodes(root, default_tolerance=5, original_rgb=img)
    assert tiling_leaf_count(tiles) == 1


def test_full_roi_image_produces_only_roi_tiles():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    mask = np.ones((16, 16), np.uint8)
    buf = ImageBuffer(rgb_matrix=img, source_path="full_roi.png")
    dec = QuadtreeDecomposer(threshold=10)
    root = dec.decompose(buf, mask)
    tiler = DPTiler()
    tiles = tiler.analyze_adjacent_leaf_nodes(root, default_tolerance=10, original_rgb=img)
    assert all(t.is_roi for t in tiles)
    assert len(tiles) == 16 * 16


def test_negative_tolerance_rejected():
    img = np.full((16, 16, 3), 100, np.uint8)
    mask = np.zeros((16, 16), np.uint8)
    buf = ImageBuffer(rgb_matrix=img, source_path="flat.png")
    dec = QuadtreeDecomposer(threshold=5)
    root = dec.decompose(buf, mask)
    tiler = DPTiler(tolerance=-1)
    with pytest.raises(DPTilingError):
        tiler.analyze_adjacent_leaf_nodes(root, default_tolerance=5)


# ---------------- reduction_ratio helper ----------------
def test_reduction_ratio_basic():
    assert reduction_ratio(100, 100) == 0.0
    assert reduction_ratio(100, 50) == 0.5
    assert reduction_ratio(0, 0) == 0.0


# ---------------- merge_boundary_tiles alias ----------------
def test_merge_boundary_tiles_alias_matches_analyze():
    buf, mask = _object_image(seed=6)
    dec = QuadtreeDecomposer(threshold=10)
    root = dec.decompose(buf, mask)
    tiler = DPTiler()
    a = tiler.analyze_adjacent_leaf_nodes(root, default_tolerance=10, original_rgb=buf.rgb_matrix)
    b = tiler.merge_boundary_tiles(root, default_tolerance=10, original_rgb=buf.rgb_matrix)
    assert len(a) == len(b)


# ---------------- explicit tolerance override ----------------
def test_explicit_tolerance_overrides_default():
    buf, mask = _object_image(seed=7)
    dec = QuadtreeDecomposer(threshold=10)
    root = dec.decompose(buf, mask)

    loose = DPTiler(tolerance=50).analyze_adjacent_leaf_nodes(
        root, default_tolerance=10, original_rgb=buf.rgb_matrix
    )
    strict = DPTiler(tolerance=1).analyze_adjacent_leaf_nodes(
        root, default_tolerance=10, original_rgb=buf.rgb_matrix
    )
    assert tiling_leaf_count(loose) <= tiling_leaf_count(strict)