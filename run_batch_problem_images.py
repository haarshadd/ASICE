from pathlib import Path
import time

from asice.pipeline import compress_dataset, PipelineConfig


cfg = PipelineConfig(
    mode="fast",
    threshold=12,
    target_cr=4.0,
    roi_method="u2net",  # Updated: use the fixed U²-Net ROI pipeline
)

dataset_path = Path(r"C:\Users\haars\OneDrive\Desktop\wallpapers")
archive_path = Path("output_problem_images.asice")

# Re-run only the three problem images from the previous batch.
problem_images = [
    "307465-3840x2160-desktop-4k-the-matrix-wallpaper-photo.jpg",
    "833508.jpg",
    "images (10).jpg",
]

print("=" * 80)
print("ASICE BATCH COMPRESSION")
print("=" * 80)
print(f"Dataset : {dataset_path}")
print(f"Archive : {archive_path}")
print(f"Mode    : {cfg.mode}")
print(f"ROI     : {cfg.roi_method}")
print(f"Target CR: {cfg.target_cr:.2f}:1")
print(f"Threshold: {cfg.threshold}")
print("=" * 80)

batch_start = time.perf_counter()

import shutil
import tempfile

with tempfile.TemporaryDirectory(prefix="asice_problem_images_") as tmp:
    test_dir = Path(tmp)
    for name in problem_images:
        source = dataset_path / name
        if not source.is_file():
            raise FileNotFoundError(f"Problem image not found: {source}")
        shutil.copy2(source, test_dir / name)

    batch = compress_dataset(
        dataset_path=test_dir,
        archive_path=archive_path,
        config=cfg,
    )

batch_elapsed = time.perf_counter() - batch_start

print()
print("=" * 80)
print("PER-IMAGE RESULTS")
print("=" * 80)

for i, r in enumerate(batch.results, start=1):
    print(f"\n[{i}/{len(batch.results)}] {r.name}")
    print("-" * 80)
    print(f"  Compression ratio : {r.final_cr:.2f}:1")
    print(f"  Quadtree leaves   : {r.quadtree_leaves}")
    print(f"  DP tiles          : {r.dp_tiles}")
    print(f"  Max background err: {r.max_background_error:.1f}")
    print(f"  ROI coverage      : {r.roi_coverage:.2%}")
    print(f"  ROI exact         : {'YES' if r.roi_exact else 'NO'}")
    print(f"  Processing time   : {r.total_time_s:.2f}s")

if batch.errors:
    print()
    print("=" * 80)
    print("FAILED IMAGES")
    print("=" * 80)
    for i, error in enumerate(batch.errors, start=1):
        print(f"[{i}] {error}")

print()
print("=" * 80)
print("BATCH SUMMARY")
print("=" * 80)
print(f"Compressed         : {len(batch.results)}")
print(f"Failed             : {len(batch.errors)}")
print(f"Mean CR            : {batch.mean_cr:.2f}:1")
print(f"ROI exact rate     : {batch.roi_exact_rate:.0%}")
print(f"Total pipeline time: {batch.total_time_s:.2f}s")
print(f"Wall-clock time    : {batch_elapsed:.2f}s")
print("=" * 80)
