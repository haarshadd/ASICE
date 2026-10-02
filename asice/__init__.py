"""ASICE: Adaptive Spatial Image Compression Engine.

Pipeline (see docs, six-increment plan):
    1. io_buffer   - Core I/O & colour matrix buffer      (FR-1, NFR-3)
    2. roi         - Saliency & ROI segmentation          (FR-2)
    3. quadtree    - Adaptive quadtree decomposition      (FR-3, FR-4)
    4. dp_tiling   - DP tiling & boundary optimisation    (FR-5)
    5. entropy     - Conditional bitstream serialisation  (FR-6)
    6. archive     - Batch storage & dataset aggregation  (FR-7)
"""

__version__ = "0.1.0"
