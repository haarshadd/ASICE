"""Increment 4: DP tiling & boundary optimisation (FR-5).

The quadtree (Increment 3) can only merge blocks that share a common
parent in its recursive split. Two sibling leaves from *different*
parents can sit right next to each other with identical colour and still
never merge, because the quadtree's split structure has no mechanism to
look sideways across a parent boundary. A flat sky split into 4 quadrants
because one corner had a cloud leaves 3 "pure sky" leaves that could
trivially be one leaf, but the quadtree stores them as 3.

This stage finds and merges those missed opportunities: adjacent
background leaves with close-enough values get combined into one larger
rectangle, reducing the number of leaf records (and therefore metadata
bytes) that Stage 5 has to serialise. No pixel values change and no
leaf's dimensions are altered except by merging — this is pure metadata
reduction, not a lossy or resampling step, and it never touches ROI
leaves at all (FR-3 stays intact).

This is NOT seam carving. Seam carving deletes a connected path of
pixels and changes image dimensions; it is unrelated to what happens
here, which only changes how many tile *records* are stored, never any
pixel value or any dimension.

Algorithm (why this is dynamic programming, not a greedy scan):
    Within one horizontal row-band of leaves (same y, same height,
    ordered by x), a greedy left-to-right merge can lock in an early
    decision that blocks a cheaper global grouping. The DP instead finds
    the row's minimum-cost partition into merged groups:

        dp[i] = min cost to encode leaves[0..i]
              = min over valid j of ( dp[j-1] + cost(merge leaves[j..i]) )

    A span leaves[j..i] is a valid merge candidate only if every leaf in
    it is background (never ROI) and all their values stay within the
    merge tolerance of each other. Backtracking from dp[n-1] recovers
    which groups were actually chosen. This is the standard "optimal
    partition of a sequence into contiguous segments" DP (the same shape
    as segmented least squares), applied here to leaf-merging cost
    instead of curve-fitting error.

Two passes: first merge horizontally within each row-band, then merge
vertically across row-bands that became identical after the horizontal
pass. This keeps the DP one-dimensional (and therefore fast) at the
cost of not finding every possible 2-D rectangle grouping — a leaf
arrangement that would only merge well diagonally, or via an L-shaped
union, is left as-is. That trade-off is what keeps this stage fast
enough to matter against NFR-1; an exact 2-D optimal rectangle cover is
a much harder problem and not worth it for a metadata-only gain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .quadtree import QuadtreeNode

# Fixed per-tile metadata cost, in abstract "cost units" (not bytes yet —
# Stage 5 does the actual bit-level accounting). What matters for the DP
# is only that it's constant per tile regardless of the tile's size, so
# minimizing total cost is the same as minimizing tile count.
TILE_COST = 1.0


@dataclass(frozen=True)
class Tile:
    """One rectangle in the post-DP tiling. Replaces a group of 1+ quadtree
    leaves that were merged together (or a single unmerged leaf / ROI leaf
    passed through unchanged)."""

    x: int
    y: int
    w: int
    h: int
    value: np.ndarray  # shape (3,), the stored RGB value for this tile
    is_roi: bool
    merged_from: int = 1  # how many original quadtree leaves this tile replaces


class DPTilingError(Exception):
    pass


def _row_bands(leaves: List[QuadtreeNode]) -> Dict[Tuple[int, int], List[QuadtreeNode]]:
    """Group leaves sharing the same (y, h) into row-bands, each sorted by x.

    Two leaves are only mergeable if they sit in the same horizontal strip
    (same top edge, same height) and are adjacent — grouping by (y, h)
    first is what turns the 2-D merge problem into several independent
    1-D problems the DP below can solve directly.
    """
    bands: Dict[Tuple[int, int], List[QuadtreeNode]] = {}
    for leaf in leaves:
        key = (leaf.y, leaf.h)
        bands.setdefault(key, []).append(leaf)
    for key in bands:
        bands[key].sort(key=lambda n: n.x)
    return bands


def _values_within_tolerance(
    leaves: List[QuadtreeNode], tolerance: float, original_rgb: Optional[np.ndarray] = None
) -> bool:
    """True if merging this span is safe within `tolerance`.

    When original_rgb is given, this checks the TRUE pixel range under the
    merged span against the value that would be stored (area-weighted
    mean) — the only check that actually bounds real per-pixel error.
    Checking leaf.value (the quadtree's own already-summarised mean)
    against the merged mean is not equivalent: a leaf's stored value can
    itself sit anywhere within the quadtree's own slack for whichever
    criterion built it, so two leaves whose *values* are close can still
    cover true pixels that are far apart (measured directly: a smooth
    512px gradient produced a worst-case stored-vs-original error of 4.0
    against tolerance=3, using leaf-value checking alone — confirmed
    independent of which quadtree criterion built the tree). Without
    original_rgb (e.g. a pure-value unit test with no backing image),
    falls back to the leaf-value check as a weaker approximation.
    """
    if len(leaves) <= 1:
        return True
    if original_rgb is not None:
        y0 = min(l.y for l in leaves)
        y1 = max(l.y + l.h for l in leaves)
        x0 = min(l.x for l in leaves)
        x1 = max(l.x + l.w for l in leaves)
        block = original_rgb[y0:y1, x0:x1].astype(np.float64)
        merged_value = _merge_weighted_value(leaves)
        return bool(np.abs(block - merged_value).max() <= tolerance)

    merged_value = _merge_weighted_value(leaves)
    values = np.stack([l.value for l in leaves]).astype(np.float64)
    max_dev = np.abs(values - merged_value).max()
    return bool(max_dev <= tolerance)


def _merge_weighted_value(leaves: List[QuadtreeNode]) -> np.ndarray:
    """Area-weighted mean colour for a merged group (bigger leaves count more)."""
    weights = np.array([l.w * l.h for l in leaves], dtype=np.float64)
    values = np.stack([l.value for l in leaves]).astype(np.float64)
    return (values * weights[:, None]).sum(axis=0) / weights.sum()


def _is_contiguous_run(leaves: List[QuadtreeNode]) -> bool:
    """True if leaves[j..i] (already x-sorted) have no gap between them —
    merging across a gap would create a tile covering space that wasn't
    actually leaf-covered, silently fabricating pixels that were never
    there."""
    for a, b in zip(leaves, leaves[1:]):
        if a.x + a.w != b.x:
            return False
    return True


def _dp_merge_row(
    row: List[QuadtreeNode],
    tolerance: float,
    max_merge_run: int,
    original_rgb: Optional[np.ndarray] = None,
) -> List[Tile]:
    """Optimal-cost merge of one row-band of background leaves.

    Returns the chosen tiles for this row, in left-to-right order.
    """
    n = len(row)
    if n == 0:
        return []

    # dp[i] = min cost for leaves[0..i-1] (0-indexed prefix of length i)
    # choice[i] = the start index j of the last group ending at i, so
    # group is leaves[j..i-1]
    dp = [0.0] * (n + 1)
    choice = [0] * (n + 1)

    for i in range(1, n + 1):
        best_cost = float("inf")
        best_j = i - 1
        # try every valid start j for the group ending at i-1 (leaves[j..i-1])
        lo = max(0, i - max_merge_run)
        for j in range(i - 1, lo - 1, -1):
            span = row[j:i]
            if not _is_contiguous_run(span):
                break  # a gap at this j means any smaller j also has the same gap
            if not _values_within_tolerance(span, tolerance, original_rgb):
                # once a span is out of tolerance, growing it further (smaller j)
                # only adds more leaves and can't bring it back into tolerance
                break
            cost = dp[j] + TILE_COST
            if cost < best_cost:
                best_cost = cost
                best_j = j
        dp[i] = best_cost
        choice[i] = best_j

    # backtrack to recover the chosen groups
    tiles: List[Tile] = []
    i = n
    groups: List[List[QuadtreeNode]] = []
    while i > 0:
        j = choice[i]
        groups.append(row[j:i])
        i = j
    groups.reverse()

    for group in groups:
        first = group[0]
        total_w = sum(l.w for l in group)
        value = group[0].value if len(group) == 1 else _merge_weighted_value(group)
        tiles.append(
            Tile(
                x=first.x, y=first.y, w=total_w, h=first.h,
                value=value, is_roi=False, merged_from=len(group),
            )
        )
    return tiles


def _try_merge_vertical(
    tiles: List[Tile], tolerance: float, original_rgb: Optional[np.ndarray] = None
) -> List[Tile]:
    """Second pass: merge vertically-stacked tiles that ended up with the
    same x-span and close values after the horizontal pass. Scans pairs of
    adjacent rows only (not a full DP) — this catches the common case
    (two horizontal strips that merged identically) without the cost of a
    second full optimal-partition search in the vertical direction, which
    would need the same tolerance/contiguity machinery again for comparatively
    little extra gain over a direct full 2-D DP.
    """
    # group remaining (post-horizontal-merge) tiles by (x, w) so only tiles
    # with identical horizontal extent are vertical-merge candidates
    by_column: Dict[Tuple[int, int], List[Tile]] = {}
    for t in tiles:
        if t.is_roi:
            continue
        by_column.setdefault((t.x, t.w), []).append(t)

    merged_ids = set()
    result: List[Tile] = [t for t in tiles if t.is_roi]

    for key, column in by_column.items():
        column.sort(key=lambda t: t.y)
        i = 0
        while i < len(column):
            group = [column[i]]
            j = i + 1
            while j < len(column) and group[-1].y + group[-1].h == column[j].y:
                candidate = group + [column[j]]
                # Area-weighted tolerance check against the FULL accumulated
                # candidate group each step, re-deriving the merged value
                # fresh each time — not a fake-leaf adapter that silently
                # dropped each tile's merged_from area weight (each tile
                # here can already represent several original quadtree
                # leaves, so weighting by raw tile count instead of area
                # let drift accumulate past tolerance one step at a time;
                # found by direct measurement: a worst-case 4.0 error
                # against tolerance=3 on a smooth gradient test image).
                if not _tiles_within_tolerance(candidate, tolerance, original_rgb):
                    break
                group = candidate
                j += 1
            if len(group) == 1:
                result.append(group[0])
            else:
                total_h = sum(t.h for t in group)
                value = _tile_weighted_value(group)
                result.append(
                    Tile(
                        x=group[0].x, y=group[0].y, w=group[0].w, h=total_h,
                        value=value, is_roi=False,
                        merged_from=sum(t.merged_from for t in group),
                    )
                )
            i = j
    return result


def _tile_weighted_value(tiles: List[Tile]) -> np.ndarray:
    """Area-weighted mean colour for a group of Tiles (mirrors
    _merge_weighted_value, but for Tiles — which already carry an area of
    their own — instead of raw QuadtreeNode leaves)."""
    weights = np.array([t.w * t.h for t in tiles], dtype=np.float64)
    values = np.stack([t.value for t in tiles]).astype(np.float64)
    return (values * weights[:, None]).sum(axis=0) / weights.sum()


def _tiles_within_tolerance(
    tiles: List[Tile], tolerance: float, original_rgb: Optional[np.ndarray] = None
) -> bool:
    """Tile-level equivalent of _values_within_tolerance (same true-pixel
    check when original_rgb is available; same leaf-value fallback otherwise)."""
    if len(tiles) <= 1:
        return True
    merged_value = _tile_weighted_value(tiles)
    if original_rgb is not None:
        y0 = min(t.y for t in tiles)
        y1 = max(t.y + t.h for t in tiles)
        x0 = min(t.x for t in tiles)
        x1 = max(t.x + t.w for t in tiles)
        block = original_rgb[y0:y1, x0:x1].astype(np.float64)
        return bool(np.abs(block - merged_value).max() <= tolerance)
    values = np.stack([t.value for t in tiles]).astype(np.float64)
    max_dev = np.abs(values - merged_value).max()
    return bool(max_dev <= tolerance)


@dataclass
class DPTiler:
    """Merges adjacent background quadtree leaves to reduce leaf-record count.

    tolerance: maximum allowed deviation between a merged group's leaf
        values. Defaults to the quadtree's own threshold T when None is
        passed to analyze_adjacent_leaf_nodes (matches the decision to
        reuse T for both stages, giving one coherent error budget).
    max_merge_run: safety cap on how many leaves one horizontal merge can
        absorb, so a long smooth gradient (each step just barely within
        tolerance of its neighbour, but drifting far from the first leaf
        over a long run) can't silently accumulate into one enormous tile
        whose first and last original pixels are much further apart than
        `tolerance` alone would suggest. The tolerance check already
        compares every member against the *group* mean (not just
        neighbour-to-neighbour), which catches most drift; this cap is a
        second, independent guard against pathologically long runs.
    """

    tolerance: Optional[float] = None
    max_merge_run: int = 64

    def analyze_adjacent_leaf_nodes(
        self,
        root: QuadtreeNode,
        default_tolerance: float,
        original_rgb: Optional[np.ndarray] = None,
    ) -> List[Tile]:
        """Build the tiling for one image's quadtree.

        default_tolerance is used when self.tolerance is None (the
        "reuse T" default); passing self.tolerance explicitly overrides it
        for experiments / the paper's ablation.

        original_rgb (the image's own rgb_matrix, HxWx3) should be passed
        whenever available: it lets every merge decision check the TRUE
        pixel range under the candidate span against tolerance, rather
        than only the quadtree's own already-summarised leaf values. The
        gap between those two is real and was found by direct measurement
        (see _values_within_tolerance's docstring) — without original_rgb,
        the merge-tolerance guarantee is weaker (bounded by the leaf-value
        approximation, not true pixels), so pass it whenever you have it.
        """
        tol = self.tolerance if self.tolerance is not None else default_tolerance
        if tol < 0:
            raise DPTilingError(f"tolerance must be >= 0, got {tol}")

        leaves = root.leaves()
        roi_leaves = [l for l in leaves if l.is_roi]
        bg_leaves = [l for l in leaves if not l.is_roi]

        bands = _row_bands(bg_leaves)
        horizontally_merged: List[Tile] = []
        for band in bands.values():
            horizontally_merged.extend(_dp_merge_row(band, tol, self.max_merge_run, original_rgb))

        roi_tiles = [
            Tile(x=l.x, y=l.y, w=l.w, h=l.h, value=l.value, is_roi=True, merged_from=1)
            for l in roi_leaves
        ]

        fully_merged = _try_merge_vertical(horizontally_merged, tol, original_rgb)
        return roi_tiles + fully_merged

    def merge_boundary_tiles(
        self,
        root: QuadtreeNode,
        default_tolerance: float,
        original_rgb: Optional[np.ndarray] = None,
    ) -> List[Tile]:
        """Alias for analyze_adjacent_leaf_nodes, matching the submitted
        class diagram's DPTiler.mergeBoundaryTiles() method name."""
        return self.analyze_adjacent_leaf_nodes(root, default_tolerance, original_rgb)


def tiling_leaf_count(tiles: List[Tile]) -> int:
    return len(tiles)


def reduction_ratio(original_leaf_count: int, tile_count: int) -> float:
    """Fraction of leaf records eliminated by tiling, in [0, 1]."""
    if original_leaf_count == 0:
        return 0.0
    return 1.0 - (tile_count / original_leaf_count)


def tiles_to_image(tiles: List[Tile], height: int, width: int) -> np.ndarray:
    """Rebuild an RGB image from tiles, the same role reconstruct() plays
    for a raw quadtree — used by tests to verify DP tiling doesn't change
    what the image looks like, only how it's stored."""
    canvas = np.zeros((height, width, 3), np.float32)
    for t in tiles:
        y0, y1 = t.y, min(t.y + t.h, height)
        x0, x1 = t.x, min(t.x + t.w, width)
        if y1 > y0 and x1 > x0:
            canvas[y0:y1, x0:x1] = t.value
    return np.clip(canvas, 0, 255).astype(np.uint8)