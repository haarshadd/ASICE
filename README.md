# ASICE — Adaptive Spatial Image Compression Engine

Shrink large image datasets for developers with limited storage, keeping the
important regions (ROI) pixel-exact and compressing the background.

**Status:** Increment 1 of 6 complete (Core I/O & CLI).

| Inc. | Module | Status |
|------|--------|--------|
| 1 | `io_buffer.py`, `cli.py` — RGB ingestion, batch iteration, CLI | done |
| 2 | `roi.py` — saliency / ROI mask | done |
| 3 | `quadtree.py` — ROI-aware quadtree |done |
| 4 | `dp_tiling.py` — DP merge of adjacent leaves |done |
| 5 | `entropy.py` — Huffman, conditional on target CR |done|
| 6 | `archive.py`, `pipeline.py` — `.asice` container |next|


