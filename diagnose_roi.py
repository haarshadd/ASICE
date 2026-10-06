r"""Run this on your machine to capture exactly what u2net outputs for one
of the failing images, before any thresholding. Paste the printed output
back to me.

Usage:
    python diagnose_roi.py "C:\Users\haars\OneDrive\Desktop\wallpapers\833508.jpg"
"""
import sys
import numpy as np
import cv2
from asice.io_buffer import ImageBuffer
from asice.roi import _U2NetSession, DEFAULT_MODEL_PATH, _prepare_u2net_input, _normalize_u2net_output

path = sys.argv[1]
buf = ImageBuffer.ingest_uncompressed_image(path)
print(f"Image: {buf.width}x{buf.height}")

session = _U2NetSession.load(DEFAULT_MODEL_PATH)
print(f"Backend: {session.backend}")

chw = _prepare_u2net_input(buf.rgb_matrix)
print(f"Input shape: {chw.shape}, dtype: {chw.dtype}")
print(f"Input range: [{chw.min():.3f}, {chw.max():.3f}], mean: {chw.mean():.3f}")

if session.backend == "onnxruntime":
    raw_out = session._ort_session.run(None, {session._input_name: chw})
else:
    session._cv_net.setInput(chw)
    raw_out = session._cv_net.forward(session._output_names)

out0 = np.asarray(raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out)
out0 = np.squeeze(out0)
print(f"\nRaw model output shape: {out0.shape}")
print(f"Raw output range: [{out0.min():.4f}, {out0.max():.4f}]")
print(f"Raw output mean: {out0.mean():.4f}, std: {out0.std():.4f}")
print(f"Raw output percentiles: p10={np.percentile(out0,10):.4f} p50={np.percentile(out0,50):.4f} p90={np.percentile(out0,90):.4f}")

normalized = _normalize_u2net_output(raw_out, buf.height, buf.width)
print(f"\nNormalized output range: [{normalized.min():.4f}, {normalized.max():.4f}]")
print(f"Normalized mean: {normalized.mean():.4f}")

thresh_90 = np.percentile(normalized, 90.0)
print(f"\n90th percentile threshold value: {thresh_90:.4f}")
print(f"Fraction >= threshold: {(normalized >= thresh_90).mean():.4f}")

# Save a visualization
viz = (normalized * 255).astype(np.uint8)
cv2.imwrite("saliency_debug.png", viz)
print("\nSaved raw saliency map to saliency_debug.png -- please view and describe what it looks like")