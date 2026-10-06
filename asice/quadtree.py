"""Increment 3: Adaptive quadtree decomposition (FR-3, FR-4).

Splits an RGB image into a quadtree:
  - ROI blocks (from Stage 2's mask) always split down to individual
    pixels: zero loss, full depth (FR-3).
  - Background blocks split only while they're not "uniform enough";
    once a block passes the test it becomes a leaf storing one
    representative RGB value (FR-4).

Two uniformity criteria are available:
  - "variance" (default): per-channel variance <= T. Fast (built on
    cv2.resize box-filtering, see _StatsPyramid below) and the natural
    fit for a threshold named "variance threshold" in the spec. Its
    error bound is statistical, not a hard per-pixel guarantee: for
    pixel values roughly uniformly spread within a block, the worst
    single-pixel error is roughly bounded by sqrt(12*T) (derived from
    variance-of-a-uniform-distribution); it is reported per image via
    max_background_error() rather than assumed.
  - "range": max channel value - min channel value <= T. Gives an exact
    guarantee (|reconstructed - original| <= T for every background
    pixel) when leaves store the block mean. The implementation uses a
    multilevel 2x2 min/max pyramid built with OpenCV erosion/dilation, so
    each quadtree node gets its exact range in O(1) lookup time. This is now
    the correctness-first default; variance remains available as an
    explicitly selected fast/heuristic alternative.

A leaf's stored value is always the block's mean.

QuadtreeNode mirrors the submitted class diagram (x, y, variance, children).
QuadtreeDecomposer mirrors traverseImageBuffer / haltMergingForROI /
evaluateBackgroundVariance from the same diagram.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import cv2
import numpy as np

from .io_buffer import ImageBuffer

Criterion = Literal["variance", "range"]


class QuadtreeError(Exception):
    """Raised for invalid quadtree construction/reconstruction input."""


@dataclass
class QuadtreeNode:
    """One node of the tree. Leaves have children=None and a stored value.

    x, y, w, h: this node's block in the (padded) image, in pixels.
    is_roi: whether this block falls (fully or partly) inside the ROI mask.
    value: mean RGB colour of the block, always populated (used for both
        leaf storage and, transiently, for the parent's own uniformity test).
    children: None for a leaf; else exactly 4 sub-nodes in TL, TR, BL, BR
        order.
    """

    x: int
    y: int
    w: int
    h: int
    is_roi: bool
    value: np.ndarray  # shape (3,), float32 mean colour of this block
    variance: float = 0.0  # max per-channel variance, for inspection/paper stats
    children: Optional[List["QuadtreeNode"]] = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    def compute_variance(self) -> float:
        return self.variance

    def leaves(self) -> List["QuadtreeNode"]:
        """All leaf nodes under this node, in traversal order.

        Zero-area leaves are excluded: after clipping to the true image
        bounds (see _clip_leaves_to_bounds), a leaf that fell entirely
        within the power-of-two padding region ends up with w<=0 or
        h<=0, and carries no real image content at all.
        """
        if self.is_leaf:
            return [self] if (self.w > 0 and self.h > 0) else []
        out: List[QuadtreeNode] = []
        for c in self.children:
            out.extend(c.leaves())
        return out


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


class _StatsPyramid:
    """Precomputed per-block mean / variance / roi-any at every power-of-two
    block size, so each node visit during traversal is an O(1) array lookup
    instead of a fresh NumPy reduction over a freshly sliced sub-array.

    Mean and variance pyramids are built with cv2.resize(..., INTER_AREA),
    which computes an exact box-filter average in optimised C++ — this is
    dramatically faster than the equivalent done by hand with NumPy
    reshape+reduce (measured ~10x on a 2048x2048 image), because a naive
    reshape to interleave 2x2 blocks isn't contiguous in memory and forces
    a slow strided reduction. Variance per block is obtained from the
    identity Var(X) = E[X^2] - E[X]^2, using a second resize pyramid over
    the squared image — both pyramids together are still far cheaper than
    one reshape-based pass.

    Exact block ranges for the "range" criterion are provided by
    _RangePyramid below. It recursively applies 2x2 min/max pooling using
    OpenCV's optimized erosion/dilation kernels and stores only one scalar
    max-channel range per level, keeping memory substantially below storing
    full RGB min/max pyramids.
    """

    def __init__(self, rgb: np.ndarray, roi_mask: np.ndarray) -> None:
        side = rgb.shape[0]  # already padded to a power of two, square
        n_levels = int(np.log2(side)) + 1
        f = rgb.astype(np.float32)
        f_sq = f * f
        mask_f = roi_mask.astype(np.float32)

        self._mean: list[np.ndarray] = [None] * n_levels
        self._max_var: list[np.ndarray] = [None] * n_levels
        self._roi_any: list[np.ndarray] = [None] * n_levels

        self._mean[0] = f
        self._max_var[0] = np.zeros((side, side), np.float32)  # single pixel: variance 0
        self._roi_any[0] = roi_mask.astype(bool)

        mean_sq = f_sq
        for lvl in range(1, n_levels):
            size = side >> lvl
            block_mean = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
            block_mean_sq = cv2.resize(mean_sq, (size, size), interpolation=cv2.INTER_AREA)
            var = np.maximum(block_mean_sq - block_mean * block_mean, 0.0)

            self._mean[lvl] = block_mean.reshape(size, size, 3)
            self._max_var[lvl] = var.reshape(size, size, 3).max(axis=-1)

            # roi_any via max-pooling the mask (any pixel present -> > 0)
            block_roi = cv2.resize(mask_f, (size, size), interpolation=cv2.INTER_AREA)
            self._roi_any[lvl] = block_roi > 0.0
            mean_sq = block_mean_sq  # reuse: E[X^2] at this level feeds the next

        self.n_levels = n_levels

    def stats(self, x: int, y: int, size: int) -> tuple[np.ndarray, float]:
        """O(1) lookup: (mean RGB, max channel variance) for one block."""
        lvl = int(np.log2(size))
        bx, by = x // size, y // size
        return self._mean[lvl][by, bx], float(self._max_var[lvl][by, bx])

    def roi_any(self, x: int, y: int, size: int) -> bool:
        lvl = int(np.log2(size))
        bx, by = x // size, y // size
        return bool(self._roi_any[lvl][by, bx])


@dataclass
class QuadtreeDecomposer:
    """Builds an ROI-aware quadtree for one image.

    threshold: T. A background block is a leaf once its uniformity metric
        (variance or range, per `criterion`) is <= T; otherwise it splits.
    min_block: smallest allowed block side for *background* blocks, so a
        pathological image (e.g. random noise) can't recurse to 1x1 and
        blow up the leaf count / metadata cost. ROI blocks always go to
        1x1 regardless of this floor (FR-3 requires exact preservation).
    criterion: "variance" (default, fast) or "range" (exact per-pixel
        error bound, slower — see module docstring).
    """

    threshold: float = 12.0
    min_block: int = 2
    criterion: Criterion = "range"

    def decompose(self, image: ImageBuffer, roi_mask: np.ndarray) -> QuadtreeNode:
        """Build the tree. roi_mask must be uint8 {0,1}, same HxW as image."""
        rgb = image.rgb_matrix
        h, w = rgb.shape[:2]
        if roi_mask.shape != (h, w):
            raise QuadtreeError(
                f"roi_mask shape {roi_mask.shape} does not match image {(h, w)}"
            )

        # Pad to a power of two internally so every split is a clean bisection;
        # padding is never exposed outside this class (root.w/h below are the
        # *original* dimensions, and reconstruct() crops back to them).
        side = _next_pow2(max(h, w))
        if side != w or side != h:
            padded_rgb = np.zeros((side, side, 3), np.uint8)
            padded_rgb[:h, :w] = rgb
            if h > 0:
                padded_rgb[h:, :w] = rgb[h - 1 : h, :]  # edge-extend, avoids a fake black band
            if w > 0:
                padded_rgb[:, w:] = padded_rgb[:, w - 1 : w]
            padded_mask = np.zeros((side, side), np.uint8)
            padded_mask[:h, :w] = roi_mask
        else:
            padded_rgb, padded_mask = rgb, roi_mask

        pyr = _StatsPyramid(padded_rgb, padded_mask)
        range_cache = _RangePyramid(padded_rgb) if self.criterion == "range" else None

        root = self._split_node(pyr, range_cache, 0, 0, side)
        root.x, root.y, root.w, root.h = 0, 0, w, h  # report true (unpadded) extent at the root
        self._original_hw = (h, w)

        if side != w or side != h:
            # The padded region (rows >= h or columns >= w) is internal
            # bookkeeping only — edge-extended filler so every split is a
            # clean power-of-two bisection, never real image content. Every
            # node's x/y/w/h up to here is in padded coordinates, same as
            # the root was before the line above corrected it; clip every
            # leaf the same way, or a leaf that happens to span the
            # h/w boundary (e.g. a flat background merging rows 24-511 on
            # a 32-tall image) reports a height that reaches into padding,
            # and nothing downstream has any way to know that part of it
            # isn't real. Found via dp_tiling: a merged tile's declared
            # h=512 on a 32-row image silently produced a wrong stored
            # colour value, because the merge used the (fabricated)
            # padded rows in its weighted average.
            _clip_leaves_to_bounds(root, h, w)
        return root

    def _split_node(
        self, pyr: "_StatsPyramid", range_cache: "Optional[_RangePyramid]", x: int, y: int, size: int
    ) -> QuadtreeNode:
        mean, max_var = pyr.stats(x, y, size)
        block_is_roi = pyr.roi_any(x, y, size)

        if self.criterion == "range":
            metric = range_cache.max_range(x, y, size)
        else:
            metric = max_var
        uniform_enough = metric <= self.threshold
        can_still_split = size > 1 and (size // 2) >= 1

        # In strict range mode the threshold is a hard correctness bound, so
        # min_block must NEVER permit a background leaf whose range exceeds T.
        # The minimum-block floor remains useful for the variance/fast mode,
        # where the criterion is explicitly heuristic.
        must_split = block_is_roi and size > 1  # FR-3: ROI never stops before 1x1
        if self.criterion == "range":
            background_split = (not block_is_roi) and (not uniform_enough) and can_still_split
        else:
            severity_factor = 3.0
            at_floor_but_severe = (not uniform_enough) and metric > self.threshold * severity_factor
            background_split = (not uniform_enough and size > self.min_block) or (
                not uniform_enough and size <= self.min_block and at_floor_but_severe
            )
        should_split = can_still_split and (must_split or background_split)

        if not should_split:
            return QuadtreeNode(x, y, size, size, block_is_roi, mean, max_var)

        half = size // 2
        children = [
            self._split_node(pyr, range_cache, x, y, half),  # TL
            self._split_node(pyr, range_cache, x + half, y, half),  # TR
            self._split_node(pyr, range_cache, x, y + half, half),  # BL
            self._split_node(pyr, range_cache, x + half, y + half, half),  # BR
        ]
        return QuadtreeNode(x, y, size, size, block_is_roi, mean, max_var, children=children)

    def evaluate_background_variance(self, image: ImageBuffer, roi_mask: np.ndarray) -> float:
        """Quick pre-check: mean per-channel variance over background pixels only.

        Lets the CLI warn the user before running the full decomposition on
        an image whose background is too noisy for T to do much (see the
        "quadtree alone won't help busy backgrounds" caveat)."""
        rgb = image.rgb_matrix.astype(np.float32)
        bg = rgb[roi_mask == 0]
        if bg.size == 0:
            return 0.0
        return float(bg.reshape(-1, 3).var(axis=0).mean())


class _RangePyramid:
    """Exact max-channel range for every power-of-two block size.

    Each level contains one scalar per block: the maximum, over RGB
    channels, of (channel_max - channel_min). Starting from the pixel level,
    the next level is formed by exact non-overlapping 2x2 min/max pooling.
    OpenCV's erosion/dilation kernels execute those reductions in optimized
    native code. Only the scalar range map is retained for every level; the
    three-channel min/max images needed to construct the next level are
    released as soon as that level is complete.

    This changes the old range implementation from one NumPy reduction per
    visited tree node to O(1) lookup per node after a small, linear pyramid
    build, making the exact criterion practical as the default.
    """

    _KERNEL = np.ones((2, 2), dtype=np.uint8)

    def __init__(self, rgb: np.ndarray) -> None:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise QuadtreeError(f"range pyramid expects HxWx3 RGB input, got {rgb.shape}")
        side = rgb.shape[0]
        if side != rgb.shape[1] or side < 1 or (side & (side - 1)) != 0:
            raise QuadtreeError(f"range pyramid expects a non-empty power-of-two square, got {rgb.shape}")

        # Keep the transient min/max pyramids in the source uint8 domain.
        # RGB input is already bounded to [0,255], so float32 is unnecessary
        # here and would quadruple the working memory on 4K images.
        current_min = rgb.astype(np.uint8, copy=True)
        current_max = current_min.copy()
        n_levels = int(np.log2(side)) + 1
        self._max_range: list[np.ndarray] = [None] * n_levels

        # Pixel-level range is exactly zero.
        self._max_range[0] = np.zeros((side, side), dtype=np.float32)

        for lvl in range(1, n_levels):
            eroded = cv2.erode(
                current_min, self._KERNEL, anchor=(0, 0),
                borderType=cv2.BORDER_REPLICATE,
            )
            dilated = cv2.dilate(
                current_max, self._KERNEL, anchor=(0, 0),
                borderType=cv2.BORDER_REPLICATE,
            )
            next_min = eroded[::2, ::2]
            next_max = dilated[::2, ::2]
            # Cast before subtraction: uint8 subtraction would wrap at 255.
            self._max_range[lvl] = (
                next_max.astype(np.int16) - next_min.astype(np.int16)
            ).max(axis=2).astype(np.float32, copy=False)
            current_min = next_min
            current_max = next_max

        self.n_levels = n_levels

    def max_range(self, x: int, y: int, size: int) -> float:
        lvl = int(np.log2(size))
        bx, by = x // size, y // size
        return float(self._max_range[lvl][by, bx])


def _clip_leaves_to_bounds(node: QuadtreeNode, height: int, width: int) -> None:
    """Recursively clip every leaf's w/h so no leaf extends past the true
    (unpadded) image bounds; a leaf entirely outside those bounds has its
    children pruned to empty / is left with w<=0 or h<=0 so leaves()
    callers can filter it (see leaves() below, which drops zero-area nodes).
    Mutates in place.
    """
    if node.is_leaf:
        node.w = max(0, min(node.w, width - node.x))
        node.h = max(0, min(node.h, height - node.y))
        return
    for c in node.children:
        _clip_leaves_to_bounds(c, height, width)


def reconstruct(root: QuadtreeNode, height: int, width: int) -> np.ndarray:
    """Rebuild an RGB image from a quadtree. Exact on ROI leaves, mean-filled
    on background leaves. Used by tests (and later, decode) to verify the
    lossless-ROI / bounded-error-background guarantee, not just assert it."""
    side = _next_pow2(max(height, width))
    canvas = np.zeros((side, side, 3), np.float32)
    _paint(root, canvas)
    return np.clip(canvas[:height, :width], 0, 255).astype(np.uint8)


def _paint(node: QuadtreeNode, canvas: np.ndarray) -> None:
    if node.is_leaf:
        canvas[node.y : node.y + node.h, node.x : node.x + node.w] = node.value
        return
    for c in node.children:
        _paint(c, canvas)


def max_background_error(root: QuadtreeNode, original_rgb: np.ndarray) -> float:
    """Largest |reconstructed - original| over any background pixel, for tests/paper stats."""
    worst = 0.0
    for leaf in root.leaves():
        if leaf.is_roi:
            continue
        y0, y1 = leaf.y, min(leaf.y + leaf.h, original_rgb.shape[0])
        x0, x1 = leaf.x, min(leaf.x + leaf.w, original_rgb.shape[1])
        if y1 <= y0 or x1 <= x0:
            continue
        block = original_rgb[y0:y1, x0:x1].astype(np.float32)
        err = np.abs(block - leaf.value).max()
        worst = max(worst, float(err))
    return worst


def leaf_count(root: QuadtreeNode) -> int:
    return len(root.leaves())