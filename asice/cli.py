"""Command-line interface (NFR-4).

All important settings are CLI arguments; nothing needs recompiling or editing.

    asice compress ./dataset -o out.asice --target-cr 4 --threshold 12
    asice inspect  ./dataset            # Increment 1: validate & summarise input
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from . import __version__
from .io_buffer import DatasetAggregator

log = logging.getLogger("asice")


def _setup_logging(log_file: Optional[str], verbose: bool) -> None:
    """Console + optional file logging; errors always reach the execution log (NFR-3)."""
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="asice",
        description="Adaptive Spatial Image Compression Engine: shrink image datasets "
        "while keeping important regions lossless.",
    )
    p.add_argument("--version", action="version", version=f"asice {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # ---- inspect: Increment 1 stand-alone command ----
    ins = sub.add_parser("inspect", help="Validate a dataset and print a summary.")
    ins.add_argument("dataset", help="Image file or folder of images.")
    ins.add_argument("--no-recursive", action="store_true", help="Do not descend into subfolders.")
    ins.add_argument("--log-file", default=None, help="Also write the execution log here.")
    ins.add_argument("-v", "--verbose", action="store_true")

    # ---- compress: full pipeline (stages land increment by increment) ----
    c = sub.add_parser("compress", help="Compress a dataset into one .asice archive.")
    c.add_argument("dataset", help="Image file or folder of images.")
    c.add_argument("-o", "--output", default="dataset.asice", help="Output archive path.")
    c.add_argument("--target-cr", type=float, default=4.0,
                   help="Target compression ratio N:1. Huffman is skipped once reached (default 4).")
    c.add_argument("-T", "--threshold", type=float, default=12.0,
                   help="Variance threshold T for background merging (default 12).")
    c.add_argument("--merge-tolerance", type=float, default=None,
                   help="DP tiling merge tolerance; defaults to --threshold.")
    c.add_argument("--roi-method", choices=["u2net", "classical", "mask-dir"], default="u2net",
                   help="How important regions are found (default u2net).")
    c.add_argument("--mask-dir", default=None,
                   help="Folder of user-supplied binary masks (with --roi-method mask-dir).")
    c.add_argument("--no-recursive", action="store_true")
    c.add_argument("--log-file", default=None)
    c.add_argument("-v", "--verbose", action="store_true")
    return p


def cmd_inspect(args: argparse.Namespace) -> int:
    agg = DatasetAggregator(Path(args.dataset), recursive=not args.no_recursive)
    t0 = time.perf_counter()
    try:
        total = agg.count_candidates()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2

    n_ok = 0
    raw_bytes = 0
    disk_bytes = 0
    for buf in agg.parse_directory():
        n_ok += 1
        raw_bytes += buf.raw_size_bytes
        disk_bytes += buf.original_bytes
    dt = time.perf_counter() - t0

    print(f"Dataset          : {args.dataset}")
    print(f"Candidate files  : {total}")
    print(f"Valid RGB images : {n_ok}")
    print(f"Rejected         : {len(agg.errors)}")
    print(f"On disk          : {disk_bytes:,} bytes")
    print(f"Raw (HxWx3)      : {raw_bytes:,} bytes")
    print(f"Time             : {dt:.2f} s")
    for path, reason in agg.errors:
        print(f"  ! {reason}")
    return 0 if n_ok > 0 else 1


def cmd_compress(args: argparse.Namespace) -> int:
    # Validate parameters up front so bad values fail fast with a clear message.
    if args.target_cr <= 0:
        log.error("--target-cr must be > 0")
        return 2
    if args.threshold < 0:
        log.error("--threshold must be >= 0")
        return 2
    if args.roi_method == "mask-dir" and not args.mask_dir:
        log.error("--roi-method mask-dir requires --mask-dir")
        return 2

    log.error("The compression pipeline is not implemented yet "
              "(Increment 1 provides I/O and CLI only). Try: asice inspect %s", args.dataset)
    return 3


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(getattr(args, "log_file", None), getattr(args, "verbose", False))
    if args.command == "inspect":
        return cmd_inspect(args)
    if args.command == "compress":
        return cmd_compress(args)
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
