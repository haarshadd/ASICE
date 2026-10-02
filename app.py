"""ASICE live demo (Increments 1-3): I/O -> ROI -> Quadtree, in the browser.

Run with:
    streamlit run app.py

Shows, side by side, for an uploaded image:
    1. Original (Increment 1: ImageBuffer ingestion)
    2. ROI mask overlay (Increment 2: SaliencySegmenter)
    3. Quadtree leaf boundaries (Increment 3: QuadtreeDecomposer), drawn over
       the reconstructed image, so merged background blocks are visible as
       flat-colour rectangles and the ROI region is visibly left untouched.

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

st.set_page_config(page_title="ASICE — Increments 1-3", layout="wide")
st.title("ASICE Pipeline — live demo (Increments 1-3)")
st.caption("Core I/O -> ROI / saliency segmentation -> adaptive quadtree decomposition")

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

st.sidebar.divider()
show_grid = st.sidebar.checkbox("Draw quadtree leaf boundaries", value=True)
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

    recon = reconstruct(root, buf.height, buf.width)
    n_leaves = leaf_count(root)
    bg_err = max_background_error(root, buf.rgb_matrix)
    roi_exact = bool(np.array_equal(buf.rgb_matrix[mask == 1], recon[mask == 1]))

    # ---- build the three display panels ----
    roi_overlay = buf.rgb_matrix.copy()
    roi_overlay[mask == 1] = (
        0.55 * roi_overlay[mask == 1].astype(np.float32) + 0.45 * np.array([255, 60, 60])
    ).astype(np.uint8)

    quadtree_panel = recon.copy()
    if show_grid:
        for leaf in root.leaves():
            y0, y1 = leaf.y, min(leaf.y + leaf.h, buf.height)
            x0, x1 = leaf.x, min(leaf.x + leaf.w, buf.width)
            if y1 <= y0 or x1 <= x0:
                continue
            colour = (0, 255, 90) if leaf.is_roi else (255, 60, 60)
            quadtree_panel[y0, x0:x1] = colour
            quadtree_panel[y1 - 1, x0:x1] = colour
            quadtree_panel[y0:y1, x0] = colour
            quadtree_panel[y0:y1, x1 - 1] = colour

    # ------------------------------------------------------------- layout --
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Original")
        st.image(buf.rgb_matrix, use_container_width=True)
        st.caption(f"{buf.width}x{buf.height} px")

    with col2:
        st.subheader("2. ROI mask")
        st.image(roi_overlay, use_container_width=True)
        method_used = roi_meta.get("method", roi_method)
        note = " (fell back from u2net)" if roi_meta.get("fallback") else ""
        st.caption(f"method={method_used}{note} - coverage={coverage:.1%} - {roi_time:.2f}s")

    with col3:
        st.subheader("3. Quadtree")
        st.image(quadtree_panel, use_container_width=True)
        st.caption(f"{n_leaves:,} leaves - {qt_time:.2f}s")
        if show_grid:
            st.caption("Green = ROI leaf (exact)   Red = background leaf (merged)")

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROI coverage", f"{coverage:.1%}")
    m2.metric("Quadtree leaves", f"{n_leaves:,}")
    m3.metric("Max background error", f"{bg_err:.1f}", help="Largest |reconstructed - original| over any background pixel.")
    m4.metric("ROI exactly preserved", "Yes" if roi_exact else "No")

    if not roi_exact:
        st.warning(
            "ROI region did not reconstruct pixel-exact - this should not happen; "
            "please report this as a bug with the uploaded image."
        )

    with st.expander("What am I looking at?"):
        st.markdown(
            "- **ROI mask**: pixels the saliency method marked as important; "
            "these are preserved exactly by the quadtree (FR-3).\n"
            "- **Quadtree**: green boxes are ROI leaves (always 1x1 pixels - "
            "too small to see as boxes unless you zoom); red boxes are "
            "background leaves, merged wherever a block's colour "
            f"{'variance' if criterion == 'variance' else 'range'} was under "
            f"T={threshold:g} (FR-4). Fewer, larger red boxes = more compression.\n"
            "- **Max background error**: worst-case pixel difference introduced "
            "by merging background blocks. Should stay roughly proportional to T."
        )

finally:
    Path(tmp_path).unlink(missing_ok=True)
