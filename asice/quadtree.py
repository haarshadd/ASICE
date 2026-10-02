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
    guarantee (|reconstructed - original| <= T/2 for every background
    pixel) but needs true block min/max, which can't be computed via
    cv2.resize's box filter — only via an explicit per-level reduction.
    Kept for images/batches where the hard guarantee matters more than
    speed (the paper's ablation, or small datasets), not the default
    because that reduction is materially slower at 1080p+.

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
        """All leaf nodes under this node, in traversal order."""
        if self.is_leaf:
            return [self]
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

    True block min/max (needed for the "range" criterion) can't be produced
    this way — cv2.resize has no box-min/box-max mode — so that path uses
    a separate, slower reduction, built only down to the resolution the
    tree actually needs it at (see QuadtreeDecomposer._range_min_max).
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
    criterion: Criterion = "variance"

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
        range_cache = _RangeCache(padded_rgb) if self.criterion == "range" else None

        root = self._split_node(pyr, range_cache, 0, 0, side)
        root.x, root.y, root.w, root.h = 0, 0, w, h  # report true (unpadded) extent at the root
        self._original_hw = (h, w)
        return root

    def _split_node(
        self, pyr: "_StatsPyramid", range_cache: "Optional[_RangeCache]", x: int, y: int, size: int
    ) -> QuadtreeNode:
        mean, max_var = pyr.stats(x, y, size)
        block_is_roi = pyr.roi_any(x, y, size)

        if self.criterion == "range":
            metric = range_cache.max_range(x, y, size)
        else:
            metric = max_var
        uniform_enough = metric <= self.threshold
        can_still_split = size > 1 and (size // 2) >= 1

        # min_block caps recursion on backgrounds that are merely noisy
        # (many small variations, no single bound-breaking outlier) so a
        # pathological image can't blow up the leaf count. But capping
        # unconditionally at min_block let a leaf's worst-case pixel error
        # run far past T when the block's "noise" was actually a real hard
        # edge in disguise — e.g. a quadtree leaf straddling an object's
        # anti-aliased boundary, three similar pixels and one very
        # different one, averaging to a variance near T while the actual
        # per-pixel error was over 10x T (found via max_background_error()
        # on a real image, not a hypothetical). So the floor is only
        # honoured when the block is *merely* over threshold; a block many
        # times over threshold is treated as "this really is a sharp edge,
        # not noise" and allowed to keep splitting past min_block, down to
        # 1x1 if needed. severity_factor=3 was picked so ordinary photo
        # noise (which pushes variance a little over T) still gets capped,
        # while a hard edge (which pushes variance far over T) does not.
        severity_factor = 3.0
        at_floor_but_severe = (not uniform_enough) and metric > self.threshold * severity_factor

        must_split = block_is_roi and size > 1  # FR-3: ROI never stops before 1x1
        should_split = can_still_split and (
            must_split
            or (not uniform_enough and size > self.min_block)
            or (not uniform_enough and size <= self.min_block and at_floor_but_severe)
        )

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


class _RangeCache:
    """On-demand, memoised block min/max for the "range" criterion.

    No fast box-filter equivalent exists for min/max (unlike mean/variance,
    which cv2.resize computes natively), so each distinct block actually
    visited by the traversal is reduced directly with NumPy the first time
    it's asked for, and cached. This is still much cheaper than building a
    full min/max pyramid up front, because a typical image only visits a
    small fraction of all possible blocks.
    """

    def __init__(self, rgb: np.ndarray) -> None:
        self._rgb = rgb.astype(np.float32)
        self._cache: dict[tuple[int, int, int], float] = {}

    def max_range(self, x: int, y: int, size: int) -> float:
        key = (x, y, size)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        block = self._rgb[y : y + size, x : x + size].reshape(-1, 3)
        r = float((block.max(axis=0) - block.min(axis=0)).max())
        self._cache[key] = r
        return r


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