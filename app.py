"""ASICE live demo (Increments 1-5): I/O -> ROI -> Quadtree -> DP Tiling ->
Huffman, in the browser.

Run with:
    streamlit run app.py

Shows, for an uploaded image:
    1. Original (Increment 1: ImageBuffer ingestion)
    2. ROI mask overlay (Increment 2: SaliencySegmenter)
    3. Quadtree leaf boundaries (Increment 3: QuadtreeDecomposer)
    4. DP-tiled boundaries (Increment 4: DPTiler) — fewer, larger boxes than
       panel 3 where adjacent background leaves merged
    5. Decoded output (Increment 5: ConditionalSerializer, Huffman round
       trip) — what the archive actually reconstructs, so you can see the
       real end-to-end result, not just intermediate structures

Plus final metrics: compression ratio, whether Huffman actually ran, total
pipeline time, and whether ROI pixels survived the whole trip exactly.

This is a demo/visualisation tool, not part of the asice package itself —
it imports asice as a library, the same way any other client code would.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from asice.io_buffer import ImageBuffer, ImageError
from asice.quadtree import QuadtreeDecomposer, leaf_count, max_background_error, reconstruct
from asice.roi import DEFAULT_MODEL_PATH, SaliencySegmenter, roi_coverage
from asice.dp_tiling import DPTiler, tiling_leaf_count, reduction_ratio, tiles_to_image
from asice.conditional_huffman import ConditionalSerializer

st.set_page_config(page_title="ASICE — Increments 1-5", layout="wide")
st.title("ASICE Pipeline — live demo (Increments 1-5)")
st.caption(
    "Core I/O -> ROI / saliency segmentation -> adaptive quadtree decomposition "
    "-> DP tiling -> conditional Huffman serialization"
)

# ---------------------------------------------------------------- sidebar --

st.sidebar.header("Increment 2 - ROI")
roi_method = st.sidebar.selectbox(
    "ROI method",
    ["classical", "u2net", "mask-dir"],
    index=0,
    help="classical needs no model file and always works. u2net needs "
    "asice/models/u2netp.onnx to be present, and otherwise falls back "
    "to classical automatically.",
)
mask_dir_input = None
if roi_method == "mask-dir":
    mask_dir_input = st.sidebar.text_input("Mask folder path")
percentile = st.sidebar.slider(
    "Sensitivity (percentile)", 50.0, 99.0, 90.0, 1.0,
    help="Internally converted to a mean + k*std threshold on the saliency map.",
)
dilate_px = st.sidebar.slider("ROI safety margin (px)", 0, 30, 5)

if roi_method == "u2net" and not DEFAULT_MODEL_PATH.is_file():
    st.sidebar.warning(
        f"u2netp.onnx not found at {DEFAULT_MODEL_PATH.name} - "
        "will fall back to the classical method automatically."
    )

st.sidebar.header("Increment 3 - Quadtree")
criterion = st.sidebar.selectbox(
    "Uniformity criterion", ["variance", "range"], index=0,
    help="variance: fast, default. range: exact per-pixel error bound, slower.",
)
threshold = st.sidebar.slider("Threshold T", 1.0, 80.0, 12.0, 1.0)
min_block = st.sidebar.select_slider("Minimum background block size", options=[1, 2, 4, 8, 16], value=2)

st.sidebar.header("Increment 4 - DP Tiling")
merge_tolerance_override = st.sidebar.checkbox("Override merge tolerance (default: reuse T)", value=False)
merge_tolerance = (
    st.sidebar.slider("Merge tolerance", 0.0, 80.0, threshold, 1.0)
    if merge_tolerance_override else None
)
max_merge_run = st.sidebar.select_slider(
    "Max merge run (safety cap)", options=[8, 16, 32, 64, 128], value=64
)

st.sidebar.header("Increment 5 - Huffman")
target_cr = st.sidebar.slider(
    "Target compression ratio", 1.0, 20.0, 4.0, 0.5,
    help="If Stages 1-4 alone already reach this ratio, Huffman is skipped "
    "(FR-6) — saves time with no quality cost.",
)

st.sidebar.divider()
show_grid = st.sidebar.checkbox("Draw leaf/tile boundaries", value=True)
max_side = st.sidebar.select_slider(
    "Downscale before processing (px, longest side)",
    options=[256, 512, 768, 1024, 1920, 0],
    value=768,
    format_func=lambda v: "No downscale" if v == 0 else str(v),
    help="Demo only: keeps the browser responsive on large photos. "
    "0 processes the image at full resolution.",
)

# ----------------------------------------------------------------- upload --

uploaded = st.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
)

if uploaded is None:
    st.info("Upload an RGB image to run the live pipeline.")
    st.stop()

suffix = Path(uploaded.name).suffix or ".png"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = tmp.name


def _draw_boundaries(base: np.ndarray, boxes, height: int, width: int) -> np.ndarray:
    """boxes: iterable of objects with .x .y .w .h .is_roi (QuadtreeNode or Tile)."""
    panel = base.copy()
    for b in boxes:
        y0, y1 = b.y, min(b.y + b.h, height)
        x0, x1 = b.x, min(b.x + b.w, width)
        if y1 <= y0 or x1 <= x0:
            continue
        colour = (0, 255, 90) if b.is_roi else (255, 60, 60)
        panel[y0, x0:x1] = colour
        panel[y1 - 1, x0:x1] = colour
        panel[y0:y1, x0] = colour
        panel[y0:y1, x1 - 1] = colour
    return panel


try:
    # ---- Increment 1: ingest ----
    try:
        buf = ImageBuffer.ingest_uncompressed_image(tmp_path)
    except ImageError as exc:
        st.error(f"Could not load this image: {exc}")
        st.stop()

    if max_side and max(buf.height, buf.width) > max_side:
        scale = max_side / max(buf.height, buf.width)
        new_w, new_h = int(buf.width * scale), int(buf.height * scale)
        resized_rgb = cv2.resize(buf.rgb_matrix, (new_w, new_h), interpolation=cv2.INTER_AREA)
        buf = ImageBuffer(rgb_matrix=resized_rgb, source_path=buf.source_path)

    # ---- Increment 2: ROI ----
    if roi_method == "mask-dir" and not mask_dir_input:
        st.error("Enter a mask folder path in the sidebar, or choose a different ROI method.")
        st.stop()

    seg = SaliencySegmenter(
        method=roi_method,
        model_path=DEFAULT_MODEL_PATH,
        mask_dir=Path(mask_dir_input) if mask_dir_input else None,
        percentile=percentile,
        dilate_px=dilate_px,
    )
    t0 = time.perf_counter()
    try:
        mask, roi_meta = seg.generate_binary_values(buf)
    except Exception as exc:
        st.error(f"ROI extraction failed: {exc}")
        st.stop()
    roi_time = time.perf_counter() - t0
    coverage = roi_coverage(mask)

    # ---- Increment 3: quadtree ----
    dec = QuadtreeDecomposer(threshold=threshold, min_block=min_block, criterion=criterion)
    t0 = time.perf_counter()
    root = dec.decompose(buf, mask)
    qt_time = time.perf_counter() - t0

    recon_qt = reconstruct(root, buf.height, buf.width)
    n_leaves = leaf_count(root)
    bg_err_qt = max_background_error(root, buf.rgb_matrix)
    roi_exact_qt = bool(np.array_equal(buf.rgb_matrix[mask == 1], recon_qt[mask == 1]))

    # ---- Increment 4: DP tiling ----
    tiler = DPTiler(tolerance=merge_tolerance, max_merge_run=max_merge_run)
    t0 = time.perf_counter()
    tiles = tiler.analyze_adjacent_leaf_nodes(root, default_tolerance=threshold, original_rgb=buf.rgb_matrix)
    dp_time = time.perf_counter() - t0

    recon_dp = tiles_to_image(tiles, buf.height, buf.width)
    n_tiles = tiling_leaf_count(tiles)
    dp_reduction = reduction_ratio(n_leaves, n_tiles)
    roi_exact_dp = bool(np.array_equal(buf.rgb_matrix[mask == 1], recon_dp[mask == 1]))

    # ---- Increment 5: conditional Huffman serialization + decode ----
    serializer = ConditionalSerializer(target_cr=target_cr)
    t0 = time.perf_counter()
    payload, ser_meta = serializer.serialize(tiles, raw_size_bytes=buf.raw_size_bytes)
    ser_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    tiles_decoded = serializer.deserialize(payload)
    decode_time = time.perf_counter() - t0

    recon_final = tiles_to_image(tiles_decoded, buf.height, buf.width)
    roi_exact_final = bool(np.array_equal(buf.rgb_matrix[mask == 1], recon_final[mask == 1]))

    total_time = roi_time + qt_time + dp_time + ser_time + decode_time

    # ---- build display panels ----
    roi_overlay = buf.rgb_matrix.copy()
    roi_overlay[mask == 1] = (
        0.55 * roi_overlay[mask == 1].astype(np.float32) + 0.45 * np.array([255, 60, 60])
    ).astype(np.uint8)

    quadtree_panel = _draw_boundaries(recon_qt, root.leaves(), buf.height, buf.width) if show_grid else recon_qt
    dp_panel = _draw_boundaries(recon_dp, tiles, buf.height, buf.width) if show_grid else recon_dp

    # ------------------------------------------------------------- layout --
    row1 = st.columns(3)
    with row1[0]:
        st.subheader("1. Original")
        st.image(buf.rgb_matrix, use_container_width=True)
        st.caption(f"{buf.width}x{buf.height} px")

    with row1[1]:
        st.subheader("2. ROI mask")
        st.image(roi_overlay, use_container_width=True)
        method_used = roi_meta.get("method", roi_method)
        note = " (fell back from u2net)" if roi_meta.get("fallback") else ""
        st.caption(f"method={method_used}{note} - coverage={coverage:.1%} - {roi_time:.2f}s")

    with row1[2]:
        st.subheader("3. Quadtree")
        st.image(quadtree_panel, use_container_width=True)
        st.caption(f"{n_leaves:,} leaves - {qt_time:.2f}s")

    row2 = st.columns(3)
    with row2[0]:
        st.subheader("4. DP tiling")
        st.image(dp_panel, use_container_width=True)
        st.caption(f"{n_tiles:,} tiles ({dp_reduction:.1%} fewer than Stage 3) - {dp_time:.2f}s")

    with row2[1]:
        st.subheader("5. Decoded (final)")
        st.image(recon_final, use_container_width=True)
        hstate = "Huffman" if ser_meta["huffman_used"] else "stored raw (target CR already met)"
        st.caption(f"{hstate} - {len(payload):,} bytes - encode {ser_time:.2f}s / decode {decode_time:.2f}s")

    with row2[2]:
        st.subheader("Legend")
        st.caption("🟩 ROI leaf/tile (exact)")
        st.caption("🟥 Background leaf/tile (merged)")
        st.caption("Panel 5 is what the archive actually decodes to — compare it to Panel 1.")

    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Final compression ratio", f"{ser_meta['final_cr']:.2f}:1")
    m2.metric("Archive size", f"{len(payload):,} B", help=f"Raw size was {buf.raw_size_bytes:,} B")
    m3.metric("Total pipeline time", f"{total_time:.2f}s")
    m4.metric("ROI exact end-to-end", "Yes" if roi_exact_final else "No")
    m5.metric("Max background error", f"{bg_err_qt:.1f}", help="Measured after Stage 3, before tiling/Huffman (neither changes pixel values).")

    if not roi_exact_final:
        st.error(
            "ROI region did NOT reconstruct pixel-exact after the full round trip "
            "(encode -> decode). This breaks the core lossless-ROI guarantee — "
            "please note the image and settings used."
        )
    elif not roi_exact_qt or not roi_exact_dp:
        st.warning("ROI was exact at an earlier stage but not reported consistently — investigate.")

    with st.expander("Stage-by-stage detail"):
        st.markdown(
            f"- **Stage 3 -> 4**: {n_leaves:,} quadtree leaves became {n_tiles:,} DP tiles "
            f"({dp_reduction:.1%} reduction) by merging adjacent background leaves the "
            "quadtree's own recursive split couldn't merge on its own.\n"
            f"- **Stage 5**: {ser_meta['reason']}. Structural compression (Stages 1-4) alone "
            f"reached {ser_meta['structural_cr']:.2f}:1; target was {target_cr:g}:1.\n"
            "- **Panel 5 vs Panel 1**: this is the real test — the archive bytes were decoded "
            "back into tiles and repainted into an image with no shortcuts. If ROI pixels "
            "differ here, something is actually broken, not just visually subtle."
        )

finally:
    Path(tmp_path).unlink(missing_ok=True)