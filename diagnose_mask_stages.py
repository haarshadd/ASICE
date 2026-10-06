"""Runs the REAL _mask_from_saliency logic step by step on one image's
actual u2net output, printing coverage after thresholding, after closing,
after hole-fill, and after dilation -- to find exactly which step inflates
coverage to 100%.

Usage: python diagnose_mask_stages.py path_to_image.jpg
"""
import sys
import numpy as np
import cv2
from asice.io_buffer import ImageBuffer
from asice.roi import _U2NetSession, DEFAULT_MODEL_PATH, _prepare_u2net_input, _normalize_u2net_output, _fill_enclosed_holes

path = sys.argv[1]
buf = ImageBuffer.ingest_uncompressed_image(path)
print(f"Image: {buf.width}x{buf.height}")

session = _U2NetSession.load(DEFAULT_MODEL_PATH)
chw = _prepare_u2net_input(buf.rgb_matrix)
if session.backend == "onnxruntime":
    raw_out = session._ort_session.run(None, {session._input_name: chw})
else:
    session._cv_net.setInput(chw)
    raw_out = session._cv_net.forward(session._output_names)

saliency = _normalize_u2net_output(raw_out, buf.height, buf.width)
print(f"saliency range: [{saliency.min():.4f}, {saliency.max():.4f}]")

percentile = 90.0
dilate_px = 5

s_min = float(saliency.min())
s_max = float(saliency.max())
threshold = float(np.percentile(saliency, percentile))
print(f"s_min={s_min:.6f} threshold={threshold:.6f}")

if threshold <= s_min + 1e-6:
    mask = (saliency > s_min + 1e-6).astype(np.uint8)
    print("branch: threshold<=s_min -> using '> s_min'")
else:
    mask = (saliency >= threshold).astype(np.uint8)
    print("branch: normal -> using '>= threshold'")

print(f"STAGE 1 (threshold only) coverage: {mask.mean():.4f}")
cv2.imwrite("stage1_threshold.png", mask * 255)

close_px = max(dilate_px, 3)
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1))
mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
print(f"STAGE 2 (after morphologyEx CLOSE, kernel={close_px*2+1}) coverage: {mask2.mean():.4f}")
cv2.imwrite("stage2_closed.png", mask2 * 255)

mask3 = _fill_enclosed_holes(mask2)
print(f"STAGE 3 (after _fill_enclosed_holes) coverage: {mask3.mean():.4f}")
cv2.imwrite("stage3_filled.png", mask3 * 255)

k_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
mask4 = cv2.dilate(mask3, k_el)
print(f"STAGE 4 (after final dilate) coverage: {mask4.mean():.4f}")
cv2.imwrite("stage4_final.png", mask4 * 255)

print("\nSaved stage1_threshold.png, stage2_closed.png, stage3_filled.png, stage4_final.png")
print("Please paste the printed coverage numbers back.")