"""
Step 6: Compaction + S3 re-benchmark + worker scaling test.

1. Compact 344 small Lance files → 1 merged file, upload to S3
2. Re-benchmark S3 reads: compacted vs uncompacted
3. Scaling test: vary worker count for parallel read and write
"""

import json
import os
import random
import shutil
import time
from typing import Callable, List

import lance
import pyarrow as pa
import ray

# --- Config ---
SOURCE_DIR = "/mnt/amlfs-07/shared/datasets/egocentric_human/vendor_raw/mecka/feb2_small_batch"
LOCAL_DATA = "/mnt/amlfs-03/shared/jingwang/WORKSPACE/PlayRayDeltaCAT/output/data"
LOCAL_MERGED = os.path.join(LOCAL_DATA, "_merged.lance")

S3_BASE = "s3://GearHome/jingwang/datasets/feb2_small_batch_deltacat/data"
S3_COMPACTED = f"{S3_BASE}/_compacted.lance"
S3_MERGED = f"{S3_BASE}/_merged.lance"

STORAGE_OPTIONS = {
    "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
    "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    "aws_endpoint": "https://pdx.s8k.io",
    "aws_region": "us-east-1",
    "allow_http": "false",
}

SCHEMA = pa.schema([
    pa.field("episode_id", pa.string()),
    pa.field("user_id", pa.string()),
    pa.field("environment_id", pa.string()),
    pa.field("scene_id", pa.string()),
    pa.field("task_id", pa.string()),
    pa.field("duration", pa.float64()),
    pa.field("video_bytes", pa.large_binary()),
    pa.field("preannotations_json", pa.string()),
])


def benchmark(name: str, fn: Callable, iterations: int = 3, warmup: int = 0):
    for _ in range(warmup):
        fn()
    times = []
    result = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f"  {name}: avg={avg:.3f}s  min={min(times):.3f}s  max={max(times):.3f}s")
    return avg, result


# =========================================================================
# Part 1: Compaction — write compacted dataset to S3
# =========================================================================

def do_compaction():
    """Read local merged Lance, write as compacted Lance directly to S3."""
    print("=" * 60)
    print("Part 1: Compaction (local merged → S3 compacted)")
    print("=" * 60)

    # Write compacted to S3 directly
    print("Reading local merged dataset...")
    t0 = time.perf_counter()
    ds = lance.dataset(LOCAL_MERGED)
    full_table = ds.to_table()
    t_read = time.perf_counter() - t0
    print(f"  Read {full_table.num_rows} rows, {full_table.nbytes / 1e9:.2f} GB in {t_read:.2f}s")

    print("Writing compacted dataset to S3...")
    t1 = time.perf_counter()
    lance.write_dataset(full_table, S3_COMPACTED, mode="overwrite",
                        storage_options=STORAGE_OPTIONS)
    t_write = time.perf_counter() - t1
    print(f"  Wrote compacted to S3 in {t_write:.2f}s ({full_table.nbytes / t_write / 1e6:.1f} MB/s)")
    print()


# =========================================================================
# Part 2: S3 compacted vs uncompacted (merged was already uploaded)
# =========================================================================

def s3_comparison():
    print("=" * 60)
    print("Part 2: S3 Compacted vs Uncompacted (344 files)")
    print("=" * 60)

    # Get individual episode paths
    ds_comp = lance.dataset(S3_COMPACTED, storage_options=STORAGE_OPTIONS)
    ids = ds_comp.to_table(columns=["episode_id"])["episode_id"].to_pylist()
    individual_paths = [f"{S3_BASE}/{eid}.lance" for eid in ids]

    print(f"\nDataset: {len(ids)} episodes")
    print()

    # --- Compacted reads ---
    print("--- COMPACTED (1 file on S3) ---")

    def read_compacted_full():
        return lance.dataset(S3_COMPACTED, storage_options=STORAGE_OPTIONS).to_table().num_rows

    def read_compacted_meta():
        return lance.dataset(S3_COMPACTED, storage_options=STORAGE_OPTIONS).to_table(
            columns=["episode_id", "duration", "environment_id"]).num_rows

    def read_compacted_take10():
        ds = lance.dataset(S3_COMPACTED, storage_options=STORAGE_OPTIONS)
        return len(ds.take(random.sample(range(344), 10)))

    def read_compacted_col2():
        return lance.dataset(S3_COMPACTED, storage_options=STORAGE_OPTIONS).to_table(
            columns=["episode_id", "duration"]).num_rows

    benchmark("Eager full (all cols + video)", read_compacted_full)
    benchmark("Eager metadata (3 cols)", read_compacted_meta)
    benchmark("Lazy take(10 random)", read_compacted_take10)
    benchmark("Lazy column projection (2 cols)", read_compacted_col2)

    # --- Uncompacted reads (344 individual files) ---
    print()
    print("--- UNCOMPACTED (344 files on S3) ---")

    def read_uncompacted_meta():
        tables = []
        for p in individual_paths:
            ds = lance.dataset(p, storage_options=STORAGE_OPTIONS)
            tables.append(ds.to_table(columns=["episode_id", "duration", "environment_id"]))
        return pa.concat_tables(tables).num_rows

    def read_uncompacted_take10():
        # Pick 10 random episode files, take 1 row each
        sample_paths = random.sample(individual_paths, 10)
        rows = 0
        for p in sample_paths:
            ds = lance.dataset(p, storage_options=STORAGE_OPTIONS)
            rows += len(ds.take([0]))
        return rows

    benchmark("Eager metadata (3 cols, 344 files)", read_uncompacted_meta, iterations=1)
    benchmark("Lazy take(10 random, 10 files)", read_uncompacted_take10)
    print()


# =========================================================================
# Part 3: Worker scaling — read and write with different worker counts
# =========================================================================

def scaling_test():
    print("=" * 60)
    print("Part 3: Worker Scaling (compacted S3 dataset)")
    print("=" * 60)

    worker_counts = [1, 2, 5, 10, 20, 50]

    # ----- READ SCALING -----
    print("\n--- READ SCALING (metadata from S3 compacted) ---")
    print(f"{'Workers':>8} | {'Time (s)':>10} | {'Rows/s':>10} | {'Speedup':>8}")
    print("-" * 48)

    base_time = None
    for n_workers in worker_counts:
        @ray.remote
        def read_worker(indices, storage_options):
            ds = lance.dataset(S3_COMPACTED, storage_options=storage_options)
            return ds.take(indices).num_rows

        # Split 344 rows across workers
        all_indices = list(range(344))
        chunk = len(all_indices) // n_workers + 1
        chunks = [all_indices[i * chunk:(i + 1) * chunk] for i in range(n_workers)]
        chunks = [c for c in chunks if c]  # remove empty

        t0 = time.perf_counter()
        futures = [read_worker.remote(c, STORAGE_OPTIONS) for c in chunks]
        total = sum(ray.get(futures))
        elapsed = time.perf_counter() - t0

        if base_time is None:
            base_time = elapsed
        speedup = base_time / elapsed

        print(f"{n_workers:>8} | {elapsed:>10.3f} | {total / elapsed:>10.0f} | {speedup:>7.2f}x")

    # ----- WRITE SCALING -----
    print("\n--- WRITE SCALING (parallel Lance write to local FS) ---")
    print(f"{'Workers':>8} | {'Time (s)':>10} | {'MB/s':>10} | {'Speedup':>8}")
    print("-" * 48)

    episode_dirs = sorted([
        os.path.join(SOURCE_DIR, d)
        for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ])

    @ray.remote
    def write_worker(ep_dirs, out_dir):
        total_bytes = 0
        for ep_dir in ep_dirs:
            with open(os.path.join(ep_dir, "metadata.json")) as f:
                meta = json.load(f)
            with open(os.path.join(ep_dir, "video.mp4"), "rb") as f:
                video = f.read()
            total_bytes += len(video)
            table = pa.table({
                "episode_id": [meta["episodeId"]],
                "user_id": [meta["userId"]],
                "environment_id": [meta["environmentId"]],
                "scene_id": [meta["sceneId"]],
                "task_id": [meta["taskId"]],
                "duration": [meta["duration"]],
                "video_bytes": [video],
                "preannotations_json": [json.dumps(meta.get("preannotations", []))],
            }, schema=SCHEMA)
            path = os.path.join(out_dir, f"{meta['episodeId']}.lance")
            lance.write_dataset(table, path, mode="overwrite")
        return total_bytes

    base_time = None
    total_data_bytes = None
    for n_workers in worker_counts:
        out_dir = f"/mnt/amlfs-03/shared/jingwang/WORKSPACE/PlayRayDeltaCAT/output/bench/scale_{n_workers}"
        os.makedirs(out_dir, exist_ok=True)

        chunk = len(episode_dirs) // n_workers + 1
        chunks = [episode_dirs[i * chunk:(i + 1) * chunk] for i in range(n_workers)]
        chunks = [c for c in chunks if c]

        t0 = time.perf_counter()
        futures = [write_worker.remote(c, out_dir) for c in chunks]
        bytes_list = ray.get(futures)
        elapsed = time.perf_counter() - t0
        total_bytes = sum(bytes_list)

        if base_time is None:
            base_time = elapsed
            total_data_bytes = total_bytes
        speedup = base_time / elapsed

        print(f"{n_workers:>8} | {elapsed:>10.3f} | {total_bytes / elapsed / 1e6:>10.1f} | {speedup:>7.2f}x")

        shutil.rmtree(out_dir, ignore_errors=True)

    print()


def main():
    ray.init(address="auto")
    do_compaction()
    s3_comparison()
    scaling_test()
    print("All benchmarks complete!")


if __name__ == "__main__":
    main()
