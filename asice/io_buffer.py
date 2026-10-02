"""Increment 1: Core I/O & colour matrix buffer.

Implements FR-1 (accept single images or folders, max 4096x4096) and NFR-3
(unsupported / corrupted / non-RGB images are reported, never crash the run).

Classes mirror the submitted class diagram:
    ImageBuffer        - one RGB image as an HxWx3 uint8 matrix
    DatasetAggregator  - walks a directory and yields ImageBuffers lazily
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np

log = logging.getLogger("asice")

MAX_DIM = 4096  # FR-1: maximum supported side length in pixels
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageError(Exception):
    """Raised when a single image cannot be ingested (NFR-3)."""


@dataclass
class ImageBuffer:
    """One uncompressed RGB image held as an HxWx3 uint8 NumPy array."""

    rgb_matrix: np.ndarray
    source_path: str = ""
    original_bytes: int = 0  # size of the source file on disk, for CR reporting

    def __post_init__(self) -> None:
        m = self.rgb_matrix
        if m.ndim != 3 or m.shape[2] != 3 or m.dtype != np.uint8:
            raise ImageError(
                f"ImageBuffer needs an HxWx3 uint8 array, got shape={m.shape} dtype={m.dtype}"
            )

    @property
    def height(self) -> int:
        return int(self.rgb_matrix.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb_matrix.shape[1])

    @property
    def raw_size_bytes(self) -> int:
        """Uncompressed in-memory footprint (H * W * 3)."""
        return self.height * self.width * 3

    @classmethod
    def ingest_uncompressed_image(cls, file_path: str | Path) -> "ImageBuffer":
        """Load one image from disk. Raises ImageError with a clear message on failure."""
        path = Path(file_path)

        if not path.is_file():
            raise ImageError(f"File not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ImageError(f"Unsupported file type '{path.suffix}': {path.name}")

        # IMREAD_UNCHANGED lets us see the real channel count, so we can reject
        # grayscale / RGBA explicitly instead of silently converting them.
        # np.fromfile + imdecode also handles non-ASCII paths on Windows.
        try:
            raw = np.fromfile(str(path), dtype=np.uint8)
            decoded = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        except Exception as exc:  # pragma: no cover - defensive
            raise ImageError(f"Could not read {path.name}: {exc}") from exc

        if decoded is None:
            raise ImageError(f"Corrupted or undecodable image: {path.name}")

        if decoded.dtype != np.uint8:
            raise ImageError(
                f"Unsupported bit depth ({decoded.dtype}); only 8-bit images allowed: {path.name}"
            )
        if decoded.ndim == 2:
            raise ImageError(f"Non-RGB image (grayscale) rejected: {path.name}")
        if decoded.shape[2] != 3:
            raise ImageError(
                f"Non-RGB image ({decoded.shape[2]} channels) rejected: {path.name}"
            )

        h, w = decoded.shape[:2]
        if h > MAX_DIM or w > MAX_DIM:
            raise ImageError(
                f"Image {w}x{h} exceeds the {MAX_DIM}x{MAX_DIM} limit: {path.name}"
            )
        if h < 1 or w < 1:
            raise ImageError(f"Empty image: {path.name}")

        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)  # OpenCV is BGR; we store RGB
        return cls(
            rgb_matrix=np.ascontiguousarray(rgb),
            source_path=str(path),
            original_bytes=path.stat().st_size,
        )


@dataclass
class DatasetAggregator:
    """Walks a file or directory and yields ImageBuffers one at a time.

    Bad files are recorded in ``errors`` and skipped, never raised (NFR-3).
    """

    dataset_path: Path
    recursive: bool = True
    errors: List[Tuple[str, str]] = field(default_factory=list)  # (path, reason)

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)

    def list_files(self) -> List[Path]:
        """All candidate files, sorted so runs are deterministic."""
        p = self.dataset_path
        if p.is_file():
            return [p]
        if not p.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {p}")
        pattern = p.rglob("*") if self.recursive else p.glob("*")
        return sorted(
            f for f in pattern if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def parse_directory(self) -> Iterator[ImageBuffer]:
        """Lazily yield valid images. Invalid ones are logged and collected."""
        for f in self.list_files():
            try:
                yield ImageBuffer.ingest_uncompressed_image(f)
            except ImageError as exc:
                self.errors.append((str(f), str(exc)))
                log.error("Skipping %s: %s", f, exc)

    def count_candidates(self) -> int:
        return len(self.list_files())
