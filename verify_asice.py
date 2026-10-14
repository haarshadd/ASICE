#!/usr/bin/env python3
"""Verify a final ASICE dataset archive against its original image folder.

Checks the actual .asice artifact, not an in-memory reconstruction:
  1. archive/index integrity and image-name coverage
  2. decoded dimensions
  3. ROI pixel-exactness using the same ROI configuration as compression
  4. strict background max-error bound (default T=12)
  5. full-image equality as an informational metric (normally expected False)
  6. per-image + summary CSV/JSON reports

Run from the ASICE repository root, for example:

    python verify_asice.py \
      --original "C:\\Users\\haars\\OneDrive\\Desktop\\train2017" \
      --archive "C:\\Users\\haars\\Downloads\\asice_pbl\\output.asice"

For the final strict benchmark, keep the defaults unless your compression run
used different ROI settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from asice.io_buffer import DatasetAggregator
from asice.pipeline import (
    PipelineError,
    decode_image_from_archive,
    read_archive_index,
)
from asice.roi import DEFAULT_MODEL_PATH, SaliencySegmenter


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify an ASICE archive against its original image folder."
    )
    p.add_argument("--original", required=True, type=Path, help="Original image folder")
    p.add_argument("--archive", required=True, type=Path, help=".asice archive to verify")
    p.add_argument(
        "--threshold",
        type=float,
        default=12.0,
        help="Configured strict background error bound T (default: 12)",
    )
    p.add_argument(
        "--roi-method",
        choices=["u2net", "classical", "mask-dir"],
        default="u2net",
        help="ROI method used by the compression run (default: u2net)",
    )
    p.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="ROI saliency percentile used during compression (default: 90)",
    )
    p.add_argument(
        "--dilate-px",
        type=int,
        default=5,
        help="ROI dilation in pixels used during compression (default: 5)",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="U2-Net ONNX model path",
    )
    p.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Mask directory when --roi-method mask-dir is used",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for verification_results.csv/json (default: archive folder)",
    )
    return p.parse_args()


def _safe_float(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def _build_original_map(original_dir: Path):
    """Load originals through the project's own ImageBuffer ingestion path."""
    agg = DatasetAggregator(original_dir)
    originals: Dict[str, object] = {}
    ingest_errors: List[dict] = []

    for buf in agg.parse_directory():
        name = Path(buf.source_path).name
        if name in originals:
            ingest_errors.append(
                {
                    "image": name,
                    "status": "FAIL",
                    "reason": "duplicate basename in original folder; archive stores basename only",
                }
            )
            continue
        originals[name] = buf

    for path, reason in agg.errors:
        ingest_errors.append(
            {"image": Path(path).name, "status": "FAIL", "reason": str(reason)}
        )

    return originals, ingest_errors


def _validate_archive_structure(archive: Path, entries) -> List[str]:
    errors: List[str] = []
    archive_size = archive.stat().st_size

    names = set()
    for entry in entries:
        if entry.name in names:
            errors.append(f"duplicate archive filename: {entry.name}")
        names.add(entry.name)

        if entry.width <= 0 or entry.height <= 0:
            errors.append(
                f"invalid dimensions for {entry.name}: {entry.width}x{entry.height}"
            )
        if entry.offset < 0 or entry.length < 0:
            errors.append(f"negative payload metadata for {entry.name}")

    # read_archive_index exposes payload offsets relative to the payload area,
    # so validate against the total payload span rather than the raw file size.
    total_payload = sum(int(e.length) for e in entries)
    for entry in entries:
        if entry.offset + entry.length > total_payload:
            errors.append(
                f"payload range exceeds payload area for {entry.name}: "
                f"offset={entry.offset}, length={entry.length}, total={total_payload}"
            )

    if archive_size <= 0:
        errors.append("archive file is empty")

    return errors


def verify(args: argparse.Namespace) -> int:
    original_dir = args.original.resolve()
    archive = args.archive.resolve()

    if not original_dir.is_dir():
        print(f"ERROR: original folder does not exist: {original_dir}")
        return 2
    if not archive.is_file():
        print(f"ERROR: archive does not exist: {archive}")
        return 2
    if args.threshold < 0:
        print("ERROR: threshold must be non-negative")
        return 2
    if args.roi_method == "mask-dir" and args.mask_dir is None:
        print("ERROR: --mask-dir is required with --roi-method mask-dir")
        return 2

    print("=" * 72)
    print("ASICE LOSSLESSNESS / CORRECTNESS VERIFICATION")
    print("=" * 72)
    print(f"Original folder : {original_dir}")
    print(f"Archive         : {archive}")
    print(f"ROI method      : {args.roi_method}")
    print(f"ROI percentile  : {args.percentile}")
    print(f"ROI dilation    : {args.dilate_px}px")
    print(f"Background T    : {args.threshold:g}")
    print()

    try:
        entries = read_archive_index(archive)
    except Exception as exc:
        print(f"ARCHIVE INTEGRITY: FAIL -- {exc}")
        return 1

    structure_errors = _validate_archive_structure(archive, entries)
    if structure_errors:
        print("ARCHIVE STRUCTURE: FAIL")
        for err in structure_errors:
            print(f"  - {err}")
        return 1

    print(f"Archive entries : {len(entries)}")

    originals, ingest_errors = _build_original_map(original_dir)
    print(f"Original images : {len(originals)} successfully ingested")
    if ingest_errors:
        print(f"Original ingest errors: {len(ingest_errors)}")
        for err in ingest_errors:
            print(f"  - {err['image']}: {err['reason']}")

    entry_names = {e.name for e in entries}
    original_names = set(originals)
    missing_from_archive = sorted(original_names - entry_names)
    extra_in_archive = sorted(entry_names - original_names)

    if missing_from_archive:
        print(f"Missing from archive: {len(missing_from_archive)}")
        for name in missing_from_archive:
            print(f"  - {name}")
    if extra_in_archive:
        print(f"Archive entries without original: {len(extra_in_archive)}")
        for name in extra_in_archive:
            print(f"  - {name}")

    # One segmenter is reused across the verification run, just as the final
    # batch pipeline reuses one U2-Net session.
    segmenter = SaliencySegmenter(
        method=args.roi_method,
        model_path=args.model_path,
        mask_dir=args.mask_dir,
        percentile=args.percentile,
        dilate_px=args.dilate_px,
    )

    rows: List[dict] = []
    decode_failures = 0
    roi_failures = 0
    dimension_failures = 0
    roi_exact_count = 0
    background_bound_count = 0
    full_image_exact_count = 0
    max_observed_background_error = 0.0

    entry_by_name = {e.name: e for e in entries}

    for i, entry in enumerate(entries, start=1):
        name = entry.name
        row = {
            "image": name,
            "width": entry.width,
            "height": entry.height,
            "decode_ok": False,
            "dimensions_ok": False,
            "roi_pixels": 0,
            "roi_exact_pixels": 0,
            "roi_exact_percent": 0.0,
            "roi_exact": False,
            "background_pixels": 0,
            "max_background_error": None,
            "background_bound": args.threshold,
            "background_bound_ok": False,
            "full_image_exact": False,
            "roi_method_used": None,
            "status": "FAIL",
            "reason": "",
        }

        print(f"[{i:>3}/{len(entries)}] {name} ...", end=" ", flush=True)

        buf = originals.get(name)
        if buf is None:
            row["reason"] = "original image not found/ingested"
            rows.append(row)
            print("FAIL (original missing)")
            continue

        if buf.width != entry.width or buf.height != entry.height:
            dimension_failures += 1
            row["reason"] = (
                f"dimension mismatch: original={buf.width}x{buf.height}, "
                f"archive={entry.width}x{entry.height}"
            )
            rows.append(row)
            print("FAIL (dimensions)")
            continue
        row["dimensions_ok"] = True

        try:
            decoded = decode_image_from_archive(archive, entry)
            row["decode_ok"] = True
        except Exception as exc:
            decode_failures += 1
            row["reason"] = f"decode failed: {exc}"
            rows.append(row)
            print("FAIL (decode)")
            continue

        try:
            mask, roi_meta = segmenter.generate_binary_values(buf)
            row["roi_method_used"] = roi_meta.get("method", args.roi_method)

            # Formal verification must not silently accept a U2-Net fallback.
            if args.roi_method == "u2net" and roi_meta.get("method") != "u2net":
                raise RuntimeError(
                    "requested U2-Net ROI verification but ROI segmenter fell back "
                    "to classical saliency"
                )

            roi = mask.astype(bool)
            bg = ~roi
            roi_pixels = int(roi.sum())
            bg_pixels = int(bg.sum())
            row["roi_pixels"] = roi_pixels
            row["background_pixels"] = bg_pixels

            roi_original = buf.rgb_matrix[roi]
            roi_decoded = decoded[roi]
            exact_pixels = int(np.all(roi_original == roi_decoded, axis=1).sum()) if roi_pixels else 0
            row["roi_exact_pixels"] = exact_pixels
            row["roi_exact_percent"] = (
                100.0 * exact_pixels / roi_pixels if roi_pixels else 100.0
            )
            row["roi_exact"] = bool(exact_pixels == roi_pixels)

            if row["roi_exact"]:
                roi_exact_count += 1

            if bg_pixels:
                diff = np.abs(
                    buf.rgb_matrix[bg].astype(np.int16) - decoded[bg].astype(np.int16)
                )
                max_bg = float(diff.max())
                row["max_background_error"] = max_bg
                max_observed_background_error = max(max_observed_background_error, max_bg)
                row["background_bound_ok"] = bool(max_bg <= args.threshold + 1e-6)
            else:
                row["max_background_error"] = 0.0
                row["background_bound_ok"] = True

            if row["background_bound_ok"]:
                background_bound_count += 1

            row["full_image_exact"] = bool(np.array_equal(buf.rgb_matrix, decoded))
            if row["full_image_exact"]:
                full_image_exact_count += 1

            if row["roi_exact"] and row["background_bound_ok"]:
                row["status"] = "PASS"
                row["reason"] = "ROI exact and background bound satisfied"
                print(
                    f"PASS | ROI 100.000% | max BG error "
                    f"{row['max_background_error']:g}"
                )
            else:
                reasons = []
                if not row["roi_exact"]:
                    reasons.append("ROI not exact")
                if not row["background_bound_ok"]:
                    reasons.append("background bound exceeded")
                row["reason"] = "; ".join(reasons)
                print(
                    f"FAIL | ROI {row['roi_exact_percent']:.3f}% | max BG error "
                    f"{row['max_background_error']:g}"
                )

        except Exception as exc:
            roi_failures += 1
            row["reason"] = f"ROI/comparison failed: {exc}"
            print("FAIL (comparison)")

        rows.append(row)

    passed = sum(1 for r in rows if r["status"] == "PASS")
    total = len(rows)

    archive_structure_ok = not structure_errors
    coverage_ok = not missing_from_archive and not extra_in_archive
    roi_ok = roi_exact_count == total and total > 0
    background_ok = background_bound_count == total and total > 0
    decode_ok = decode_failures == 0 and total > 0

    overall_pass = archive_structure_ok and coverage_ok and decode_ok and roi_ok and background_ok

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Archive entries             : {len(entries)}")
    print(f"Per-image verification      : {passed}/{total} PASS")
    print(f"Decode failures             : {decode_failures}")
    print(f"Dimension failures          : {dimension_failures}")
    print(f"ROI comparison failures     : {roi_failures}")
    print(f"ROI exact images             : {roi_exact_count}/{total}")

    total_roi_pixels = sum(int(r["roi_pixels"]) for r in rows)
    total_roi_exact = sum(int(r["roi_exact_pixels"]) for r in rows)
    overall_roi_percent = 100.0 * total_roi_exact / total_roi_pixels if total_roi_pixels else 100.0
    print(f"ROI exact pixels             : {total_roi_exact}/{total_roi_pixels}")
    print(f"ROI exactness                : {overall_roi_percent:.6f}%")
    print(f"Background bound             : <= {args.threshold:g}")
    print(f"Background-bound images      : {background_bound_count}/{total}")
    print(f"Worst observed BG error      : {max_observed_background_error:g}")
    print(f"Full-image exact images      : {full_image_exact_count}/{total} (informational)")
    print()
    print(f"ARCHIVE INTEGRITY             : {'PASS' if archive_structure_ok else 'FAIL'}")
    print(f"ARCHIVE/ORIGINAL COVERAGE     : {'PASS' if coverage_ok else 'FAIL'}")
    print(f"DECODE                         : {'PASS' if decode_ok else 'FAIL'}")
    print(f"ROI LOSSLESSNESS               : {'PASS' if roi_ok else 'FAIL'}")
    print(f"BACKGROUND ERROR BOUND         : {'PASS' if background_ok else 'FAIL'}")
    print("-" * 72)
    print(f"OVERALL VERIFICATION           : {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 72)

    output_dir = (args.output_dir or archive.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "verification_results.csv"
    json_path = output_dir / "verification_results.json"

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "archive": str(archive),
        "original_folder": str(original_dir),
        "roi_method_requested": args.roi_method,
        "roi_percentile": args.percentile,
        "roi_dilate_px": args.dilate_px,
        "background_error_bound": args.threshold,
        "archive_entries": len(entries),
        "original_images_ingested": len(originals),
        "missing_from_archive": missing_from_archive,
        "extra_in_archive": extra_in_archive,
        "decode_failures": decode_failures,
        "dimension_failures": dimension_failures,
        "roi_comparison_failures": roi_failures,
        "roi_exact_images": roi_exact_count,
        "roi_exact_image_total": total,
        "roi_exact_pixels": total_roi_exact,
        "roi_pixels": total_roi_pixels,
        "roi_exact_percent": overall_roi_percent,
        "background_bound_images": background_bound_count,
        "worst_background_error": max_observed_background_error,
        "full_image_exact_images": full_image_exact_count,
        "overall_pass": overall_pass,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "images": rows}, f, indent=2)

    print()
    print(f"CSV report : {csv_path}")
    print(f"JSON report: {json_path}")

    return 0 if overall_pass else 1


def main() -> int:
    try:
        return verify(_parse_args())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
