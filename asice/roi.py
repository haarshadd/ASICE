"""Increment 2: Saliency & ROI segmentation (FR-2).

Produces a binary mask the same H x W as the input image: 1 = important
(preserve losslessly in Stage 3), 0 = background (may be compressed).

Three interchangeable backends, selected by name so the CLI can expose them
without importing heavy dependencies unless they're actually used:

    "classical"  - spectral-residual saliency, implemented with core OpenCV
                   and NumPy only (no opencv-contrib dependency). Always
                   available, works on any RGB image, no model file needed.
    "u2net"      - a bundled U^2-Net-small ONNX model, run with onnxruntime
                   (falling back to cv2.dnn if onnxruntime is unavailable).
                   Falls back to "classical" if no model file is present,
                   so the pipeline never hard-fails for a missing weight.
    "mask-dir"   - user supplies their own binary masks, matched to images
                   by filename stem.

All backends return a (mask, meta) pair. `mask` is uint8 {0,1}, `meta` is a
small dict recording which method actually ran and why, for logging/paper
ablations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .io_buffer import ImageBuffer

log = logging.getLogger("asice")

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "u2netp.onnx"
U2NET_INPUT_SIZE = 320  # standard U^2-Net(p) input resolution


class ROIError(Exception):
    """Raised when ROI extraction cannot proceed at all (e.g. bad mask-dir setup)."""


# --------------------------------------------------------------------------
# Backend 1: classical spectral-residual saliency (no extra dependencies)
# --------------------------------------------------------------------------

def _spectral_residual_saliency(rgb: np.ndarray) -> np.ndarray:
    """Hou & Zhang (2007) spectral residual saliency, implemented directly.

    Returns a float32 saliency map in [0, 1], same H x W as the input.
    Reimplemented rather than using cv2.saliency so the package only needs
    opencv-python-headless (no contrib build required).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Work at a small fixed size: this algorithm is about the coarse
    # frequency spectrum, not per-pixel detail, and it keeps this fast
    # and size-independent (important for NFR-1 on 1920x1080 batches).
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

    fft = np.fft.fft2(small)
    magnitude = np.abs(fft)
    log_amplitude = np.log(magnitude + 1e-8)
    phase = np.angle(fft)

    # "Residual" = log spectrum minus its local average (a 3x3 box filter
    # stands in for the averaging filter in the original paper).
    avg_log_amplitude = cv2.blur(log_amplitude, (3, 3))
    spectral_residual = log_amplitude - avg_log_amplitude

    reconstructed = np.exp(spectral_residual) * np.exp(1j * phase)
    saliency = np.abs(np.fft.ifft2(reconstructed)) ** 2

    # Smooth to suppress single-pixel noise, then rescale to [0, 1].
    saliency = cv2.GaussianBlur(saliency, (5, 5), sigmaX=3)
    saliency = cv2.resize(saliency, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    s_min, s_max = float(saliency.min()), float(saliency.max())
    if s_max - s_min < 1e-6:
        return np.zeros_like(saliency, dtype=np.float32)
    return ((saliency - s_min) / (s_max - s_min)).astype(np.float32)


def _mask_from_saliency(
    saliency: np.ndarray, percentile: float = 90.0, dilate_px: int = 5
) -> np.ndarray:
    """Threshold a [0,1] saliency map into a binary mask, then pad it out.

    Uses mean + k*std (the threshold from Hou & Zhang's original paper),
    not a plain percentile. A percentile always marks exactly (100-p)% of
    pixels as ROI even when the map is almost uniformly near-zero (e.g. a
    flat background with only faint numerical ripple), which was pulling
    in spurious low-saliency regions on synthetic/flat-background images.
    mean + k*std instead adapts to how much real contrast is in the map:
    a genuinely flat background yields an (almost) empty mask.

    `percentile` is kept as the tuning knob for backward compatibility /
    CLI naming, translated internally to an equivalent k in mean+k*std.

    Padding (dilation) is deliberate: a saliency map that clips the object's
    edge would let the quadtree compress right up to the boundary, which is
    the one place errors are most visible. A small safety margin costs a
    little compression but protects the actual object.
    """
    # percentile=90 (default) -> k=3, matching the classic spectral-residual
    # paper's "mean + 3*std" rule; higher percentile => stricter (higher k).
    k = 3.0 * (percentile / 90.0)
    thresh = float(saliency.mean() + k * saliency.std())
    mask = (saliency >= thresh).astype(np.uint8)

    # Spectral-residual saliency responds to contrast, so it lights up an
    # object's *edges* strongly but its flat interior only weakly — a solid
    # circle can threshold into a ring, not a disc. Close small gaps and
    # fill fully-enclosed holes so the ROI covers the whole object, not
    # just its outline. This is a real behaviour of the algorithm, not a
    # rare edge case, so it's applied unconditionally rather than as an
    # opt-in flag.
    close_px = max(dilate_px, 3)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = _fill_enclosed_holes(mask)

    if dilate_px > 0:
        k_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, k_el)

    return mask


def _fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions fully enclosed by ROI (flood-fill from the border)."""
    h, w = mask.shape
    flood = mask.copy()
    fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, fill_mask, (0, 0), 1)  # mark background reachable from the border
    # Anything that's 0 in `mask` but wasn't reached by the flood is enclosed -> fill it.
    enclosed = (flood == 0)
    return np.where(enclosed, 1, mask).astype(np.uint8)


def classical_roi_mask(image: ImageBuffer, percentile: float = 90.0, dilate_px: int = 5) -> np.ndarray:
    """Full classical pipeline: saliency map -> threshold -> dilate."""
    saliency = _spectral_residual_saliency(image.rgb_matrix)
    return _mask_from_saliency(saliency, percentile=percentile, dilate_px=dilate_px)


# --------------------------------------------------------------------------
# Backend 2: bundled U^2-Net(p) ONNX model
# --------------------------------------------------------------------------

@dataclass
class _U2NetSession:
    """Lazily-loaded ONNX session, shared across images in a batch.

    Loading and provider selection happen once; extraction proper is a call
    to .run() per image. Kept separate from SaliencySegmenter so a failed
    load can be caught and reported once, not per image.
    """

    path: Path
    backend: str  # "onnxruntime" | "cv2.dnn"
    _ort_session: object = None
    _cv_net: object = None

    @classmethod
    def load(cls, model_path: Path) -> "_U2NetSession":
        if not model_path.is_file():
            raise ROIError(f"U^2-Net model not found at {model_path}")

        # Prefer onnxruntime: broader operator support, faster on CPU.
        try:
            # Import lazily because onnxruntime is an optional dependency.
            import importlib

            ort = importlib.import_module("onnxruntime")

            sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            obj = cls(path=model_path, backend="onnxruntime")
            obj._ort_session = sess
            return obj
        except Exception as exc:
            log.warning("onnxruntime unavailable/failed (%s); trying cv2.dnn", exc)

        # Fallback: OpenCV's built-in ONNX runner needs no extra dependency,
        # but doesn't support every op U^2-Net variants can use.
        try:
            net = cv2.dnn.readNetFromONNX(str(model_path))
            obj = cls(path=model_path, backend="cv2.dnn")
            obj._cv_net = net
            return obj
        except Exception as exc:
            raise ROIError(f"Could not load {model_path} with onnxruntime or cv2.dnn: {exc}") from exc

    def run(self, rgb: np.ndarray) -> np.ndarray:
        """Run the model on one RGB image, return a [0,1] saliency map at input resolution."""
        h, w = rgb.shape[:2]
        resized = cv2.resize(rgb, (U2NET_INPUT_SIZE, U2NET_INPUT_SIZE), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        x = (x - 0.485) / 0.229  # ImageNet-ish normalisation used by U^2-Net training
        chw = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)

        if self.backend == "onnxruntime":
            input_name = self._ort_session.get_inputs()[0].name
            out = self._ort_session.run(None, {input_name: chw})[0]
        else:
            self._cv_net.setInput(chw)
            out = self._cv_net.forward()

        pred = np.squeeze(out)  # (320, 320), higher = more salient
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

        p_min, p_max = float(pred.min()), float(pred.max())
        if p_max - p_min < 1e-6:
            return np.zeros((h, w), dtype=np.float32)
        return ((pred - p_min) / (p_max - p_min)).astype(np.float32)


# --------------------------------------------------------------------------
# Backend 3: user-supplied masks
# --------------------------------------------------------------------------

def load_user_mask(mask_dir: Path, image: ImageBuffer) -> np.ndarray:
    """Find a mask matching `image` by filename stem, load and binarise it."""
    stem = Path(image.source_path).stem
    candidates = list(Path(mask_dir).glob(f"{stem}.*"))
    if not candidates:
        raise ROIError(f"No mask found for '{stem}' in {mask_dir}")

    raw = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ROIError(f"Could not read mask file: {candidates[0]}")

    if raw.shape != (image.height, image.width):
        raw = cv2.resize(raw, (image.width, image.height), interpolation=cv2.INTER_NEAREST)

    return (raw > 127).astype(np.uint8)


# --------------------------------------------------------------------------
# Unified interface
# --------------------------------------------------------------------------

@dataclass
class SaliencySegmenter:
    """Produces an ROI mask for an ImageBuffer using the configured method.

    method: "classical" | "u2net" | "mask-dir"
    model_path: override for the bundled U^2-Net weights
    mask_dir: required when method == "mask-dir"
    percentile: classical/u2net threshold — top (100 - percentile)% of the
        saliency map is kept as ROI
    dilate_px: safety margin added around the thresholded region
    """

    method: str = "u2net"
    model_path: Path = DEFAULT_MODEL_PATH
    mask_dir: Optional[Path] = None
    percentile: float = 90.0
    dilate_px: int = 5

    _session: Optional[_U2NetSession] = None
    _u2net_load_failed: bool = False

    def __post_init__(self) -> None:
        if self.method not in ("classical", "u2net", "mask-dir"):
            raise ROIError(f"Unknown roi method: {self.method}")
        if self.method == "mask-dir" and not self.mask_dir:
            raise ROIError("method='mask-dir' requires mask_dir to be set")

    def generate_binary_values(self, image: ImageBuffer) -> Tuple[np.ndarray, dict]:
        """Return (mask uint8 {0,1} of shape HxW, meta dict) for one image."""
        if self.method == "mask-dir":
            mask = load_user_mask(self.mask_dir, image)
            return mask, {"method": "mask-dir"}

        if self.method == "u2net":
            mask = self._try_u2net(image)
            if mask is not None:
                return mask, {"method": "u2net", "backend": self._session.backend}
            # Model missing or failed to load: fall back rather than crash
            # the whole batch, matching NFR-3's "never crash on bad input"
            # spirit — a missing weight file is exactly this kind of case.
            log.warning("Falling back to classical saliency for %s", image.source_path)

        saliency = _spectral_residual_saliency(image.rgb_matrix)
        mask = _mask_from_saliency(saliency, percentile=self.percentile, dilate_px=self.dilate_px)
        return mask, {"method": "classical", "fallback": self.method == "u2net"}

    def _try_u2net(self, image: ImageBuffer) -> Optional[np.ndarray]:
        if self._u2net_load_failed:
            return None
        if self._session is None:
            try:
                self._session = _U2NetSession.load(self.model_path)
            except ROIError as exc:
                log.warning("%s", exc)
                self._u2net_load_failed = True
                return None
        saliency = self._session.run(image.rgb_matrix)
        return _mask_from_saliency(saliency, percentile=self.percentile, dilate_px=self.dilate_px)


def roi_coverage(mask: np.ndarray) -> float:
    """Fraction of pixels marked as ROI, in [0, 1]. Used for CR estimates and logging."""
    return float(mask.mean())