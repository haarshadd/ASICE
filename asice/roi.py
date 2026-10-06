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
    """Convert a finite saliency map into a binary ROI mask.

    ``percentile`` is a real percentile now: ``90`` keeps pixels at or above
    the 90th percentile of the saliency distribution. A near-constant map is
    treated as having no detected ROI; otherwise a constant all-zero map would
    pass a ``>= 0`` threshold and incorrectly mark the whole image as ROI.

    The optional closing/hole-fill/dilation stage adds a safety margin around
    the detected salient region. It is applied after thresholding so the same
    post-processing is used for classical and U^2-Net saliency.
    """
    saliency = np.asarray(saliency, dtype=np.float32)
    if saliency.ndim != 2:
        raise ROIError(f"Saliency map must be 2-D, got shape {saliency.shape}")
    if not np.isfinite(saliency).all():
        raise ROIError("Saliency map contains NaN or infinite values")
    if not 0.0 <= percentile <= 100.0:
        raise ROIError("Saliency percentile must be between 0 and 100")
    if dilate_px < 0:
        raise ROIError("ROI dilation must be >= 0")

    s_min = float(saliency.min())
    s_max = float(saliency.max())
    if s_max - s_min <= 1e-6:
        return np.zeros(saliency.shape, dtype=np.uint8)

    threshold = float(np.percentile(saliency, percentile))
    # When the requested percentile lands exactly on the minimum (common for
    # sparse saliency maps), ``>= threshold`` would classify the entire low
    # saliency background as ROI. In that case keep only values above the
    # minimum; a truly constant map was already handled above.
    if threshold <= s_min + 1e-6:
        mask = (saliency > s_min + 1e-6).astype(np.uint8)
    else:
        mask = (saliency >= threshold).astype(np.uint8)

    # Spectral-residual saliency responds to contrast, so it can emphasize an
    # object's boundary while leaving its interior weak. Closing and filling
    # enclosed holes makes the protected region spatially more coherent.
    close_px = max(dilate_px, 3)
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = _fill_enclosed_holes(mask)

    if dilate_px > 0:
        k_el = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        mask = cv2.dilate(mask, k_el)

    return mask.astype(np.uint8, copy=False)


def _fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions fully enclosed by ROI.

    Uses connected-component labelling on the BACKGROUND (not a single
    floodFill seed at (0,0)): any background component that touches the
    image border is real background and stays 0; any background component
    that touches no border is enclosed by ROI on all sides and gets filled.

    A single-seed floodFill at (0,0) was used previously, which silently
    breaks whenever the mask itself touches pixel (0,0) -- not a rare
    edge case in practice: on a real wallpaper dataset, a u2net mask with
    several scattered salient regions connected by morphological closing
    reached the image's top-left corner, so the seed point was foreground
    rather than background. floodFill then filled nothing (0 reachable
    background pixels), so the old code treated the ENTIRE rest of the
    image as "enclosed" and filled it to 1 -- the exact mechanism behind
    several dataset-run images reporting 100% ROI coverage with max
    background error 0.0, when the correct coverage was ~10%.
    Connected-component border-touching is border-agnostic: it doesn't
    matter whether the mask happens to touch (0,0) or the image centre.
    """
    h, w = mask.shape
    background = (mask == 0).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(background, connectivity=8)

    # Collect every label that touches any edge of the image -- those
    # components are real, unenclosed background.
    border_labels = set(labels[0, :].tolist()) | set(labels[-1, :].tolist())
    border_labels |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    border_labels.discard(0)  # label 0 is foreground (mask==1), not a background component

    filled = mask.copy()
    for label in range(1, n_labels):
        if label not in border_labels:
            filled[labels == label] = 1  # enclosed on all sides -> fill
    return filled.astype(np.uint8)


def classical_roi_mask(image: ImageBuffer, percentile: float = 90.0, dilate_px: int = 5) -> np.ndarray:
    """Full classical pipeline: saliency map -> threshold -> dilate."""
    saliency = _spectral_residual_saliency(image.rgb_matrix)
    return _mask_from_saliency(saliency, percentile=percentile, dilate_px=dilate_px)


# --------------------------------------------------------------------------
# Backend 2: bundled U^2-Net(p) ONNX model
# --------------------------------------------------------------------------

# These are the channel-wise ImageNet statistics used by the official
# U^2-Net RGB inference preprocessing (ToTensorLab(flag=0)).  The previous
# implementation applied the red-channel statistics to all three channels,
# which changes the model input and can materially degrade the saliency map.
U2NET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
U2NET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _prepare_u2net_input(rgb: np.ndarray) -> np.ndarray:
    """Prepare an RGB image exactly for the bundled fixed-size U^2-Net model.

    The shipped u2netp ONNX graph expects ``NCHW`` float32 input at 320x320.
    U^2-Net's reference preprocessing resizes to a square and applies
    channel-wise RGB normalization before transposing to NCHW.
    """
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ROIError("U^2-Net expects an RGB HxWx3 image")
    if rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
        raise ROIError("U^2-Net cannot process an empty image")

    resized = cv2.resize(
        rgb, (U2NET_INPUT_SIZE, U2NET_INPUT_SIZE), interpolation=cv2.INTER_AREA
    )
    x = resized.astype(np.float32, copy=False) / 255.0
    x = (x - U2NET_MEAN) / U2NET_STD
    chw = np.transpose(x, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(chw, dtype=np.float32)


def _normalize_u2net_output(output: object, height: int, width: int) -> np.ndarray:
    """Convert the primary U^2-Net output to a finite [0,1] HxW map.

    The official U^2-Net inference code normalizes its first (d1) output with
    min-max normalization before resizing it back to the original image size.
    The ONNX exporter may expose that output either directly or inside a list
    of outputs, so output extraction is handled separately from normalization.
    """
    if isinstance(output, (list, tuple)):
        if not output:
            raise ROIError("U^2-Net returned no outputs")
        output = output[0]

    pred = np.asarray(output)
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise ROIError(f"Unexpected U^2-Net output shape: {pred.shape}")
    if not np.isfinite(pred).all():
        raise ROIError("U^2-Net returned NaN or infinite saliency values")

    # Match the official U^2-Net inference order: normalize the network
    # prediction first, then resize the normalized probability map back to the
    # original image dimensions. Resizing first can change the extrema and
    # therefore change the min-max normalization itself.
    p_min = float(pred.min())
    p_max = float(pred.max())
    if p_max - p_min <= 1e-6:
        # A constant prediction contains no usable foreground/background
        # separation. Treat it as an empty ROI instead of turning every pixel
        # into ROI through a zero threshold.
        return np.zeros((height, width), dtype=np.float32)

    normalized = (pred.astype(np.float32, copy=False) - p_min) / (p_max - p_min)
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)
    return cv2.resize(normalized, (width, height), interpolation=cv2.INTER_LINEAR).astype(
        np.float32, copy=False
    )


@dataclass
class _U2NetSession:
    """Lazily-loaded ONNX session, shared across images in a batch.

    Loading and provider selection happen once; extraction proper is a call
    to .run() per image. The input/output names are cached so repeated images
    do not repeatedly query the inference backend.
    """

    path: Path
    backend: str  # "onnxruntime" | "cv2.dnn"
    _ort_session: object = None
    _cv_net: object = None
    _input_name: Optional[str] = None
    _output_names: Optional[list[str]] = None

    @classmethod
    def load(cls, model_path: Path) -> "_U2NetSession":
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise ROIError(f"U^2-Net model not found at {model_path}")

        # Prefer onnxruntime: broader operator support and a cleaner ONNX
        # execution path. Import remains lazy because it is optional.
        try:
            import importlib

            ort = importlib.import_module("onnxruntime")
            sess = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            inputs = sess.get_inputs()
            if not inputs:
                raise ROIError("U^2-Net ONNX model has no inputs")
            return cls(
                path=model_path,
                backend="onnxruntime",
                _ort_session=sess,
                _input_name=inputs[0].name,
            )
        except ROIError:
            raise
        except Exception as exc:
            log.warning("onnxruntime unavailable/failed (%s); trying cv2.dnn", exc)

        # OpenCV is retained as a dependency-free fallback. Some U^2-Net ONNX
        # exports contain operators/weights that OpenCV DNN cannot import; in
        # that case the caller falls back to the classical saliency backend.
        try:
            net = cv2.dnn.readNetFromONNX(str(model_path))
            names = list(net.getUnconnectedOutLayersNames())
            if not names:
                raise ROIError("U^2-Net OpenCV graph has no output nodes")
            return cls(
                path=model_path,
                backend="cv2.dnn",
                _cv_net=net,
                _output_names=names,
            )
        except ROIError:
            raise
        except Exception as exc:
            raise ROIError(
                f"Could not load {model_path} with onnxruntime or cv2.dnn: {exc}"
            ) from exc

    def run(self, rgb: np.ndarray) -> np.ndarray:
        """Run U^2-Net and return a [0,1] saliency map at input resolution."""
        h, w = rgb.shape[:2]
        chw = _prepare_u2net_input(rgb)

        try:
            if self.backend == "onnxruntime":
                out = self._ort_session.run(None, {self._input_name: chw})
            elif self.backend == "cv2.dnn":
                self._cv_net.setInput(chw)
                out = self._cv_net.forward(self._output_names)
            else:  # pragma: no cover - guarded by construction
                raise ROIError(f"Unknown U^2-Net backend: {self.backend}")
        except ROIError:
            raise
        except Exception as exc:
            raise ROIError(f"U^2-Net inference failed: {exc}") from exc

        return _normalize_u2net_output(out, h, w)


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
    percentile: classical/u2net saliency percentile. For example, 90 keeps
        pixels at or above the 90th percentile before ROI morphology.
    dilate_px: safety margin added around the thresholded region
    """

    method: str = "u2net"
    model_path: Path = DEFAULT_MODEL_PATH
    mask_dir: Optional[Path] = None
    percentile: float = 90.0
    dilate_px: int = 5

    _session: Optional[_U2NetSession] = None
    _u2net_load_failed: bool = False
    _u2net_failure_reason: Optional[str] = None

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
                return mask, {
                    "method": "u2net",
                    "backend": self._session.backend,
                    "percentile": self.percentile,
                    "dilate_px": self.dilate_px,
                }
            # Model missing or inference failed: fall back rather than crash
            # the whole batch. The metadata records the fallback so a paper
            # benchmark cannot accidentally count a classical run as U^2-Net.
            log.warning("Falling back to classical saliency for %s", image.source_path)

        saliency = _spectral_residual_saliency(image.rgb_matrix)
        mask = _mask_from_saliency(
            saliency, percentile=self.percentile, dilate_px=self.dilate_px
        )
        meta = {
            "method": "classical",
            "fallback": self.method == "u2net",
            "percentile": self.percentile,
            "dilate_px": self.dilate_px,
        }
        if self.method == "u2net" and self._u2net_failure_reason:
            meta["fallback_reason"] = self._u2net_failure_reason
        return mask, meta

    def _try_u2net(self, image: ImageBuffer) -> Optional[np.ndarray]:
        if self._u2net_load_failed:
            return None
        if self._session is None:
            try:
                self._session = _U2NetSession.load(self.model_path)
            except ROIError as exc:
                log.warning("%s", exc)
                self._u2net_load_failed = True
                self._u2net_failure_reason = str(exc)
                return None
        try:
            saliency = self._session.run(image.rgb_matrix)
            return _mask_from_saliency(
                saliency, percentile=self.percentile, dilate_px=self.dilate_px
            )
        except ROIError as exc:
            log.warning("U^2-Net failed for %s: %s", image.source_path, exc)
            self._u2net_load_failed = True
            self._u2net_failure_reason = str(exc)
            return None


def roi_coverage(mask: np.ndarray) -> float:
    """Fraction of pixels marked as ROI, in [0, 1]. Used for CR estimates and logging."""
    return float(mask.mean())