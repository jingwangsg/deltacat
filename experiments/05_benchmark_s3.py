"""
Step 6: Read/profile from S3 (SwiftStack pdx.s8k.io).

Lance's Rust object_store needs credentials via storage_options.
We use environment variables (set in runtime_env.yaml) for credential bridging.
"""

import os
import time
from typing import Callable

import lance
import pyarrow as pa
import ray

S3_BASE = "s3://GearHome/jingwang/datasets/feb2_small_batch_deltacat/data"
S3_MERGED = f"{S3_BASE}/_merged.lance"

STORAGE_OPTIONS = {
    "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
    "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    "aws_endpoint": "https://pdx.s8k.io",
    "aws_region": "us-east-1",
    "allow_http": "false",
}

# List of episode IDs to construct S3 paths
# We'll get them from the merged dataset
EPISODE_LANCE_PATHS = None


def get_episode_paths():
    """Get S3 Lance paths by listing the merged dataset for episode IDs."""
    global EPISODE_LANCE_PATHS
    if EPISODE_LANCE_PATHS is not None:
        return EPISODE_LANCE_PATHS

    ds = lance.dataset(S3_MERGED, storage_options=STORAGE_OPTIONS)
    ids = ds.to_table(columns=["episode_id"])["episode_id"].to_pylist()
    EPISODE_LANCE_PATHS = [f"{S3_BASE}/{eid}.lance" for eid in ids]
    return EPISODE_LANCE_PATHS


def benchmark(name: str, fn: Callable, iterations: int = 3):
    times = []
    result = None
    for i in range(iterations):
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    avg = sum(times) / len(times)
    print(f"  {name}: avg={avg:.3f}s  min={min(times):.3f}s  max={max(times):.3f}s")
    return avg, result


# ===== READ BENCHMARKS =====

def bench_s3_eager_full():
    """S3 eager full load — all columns from merged dataset."""
    ds = lance.dataset(S3_MERGED, storage_options=STORAGE_OPTIONS)
    result = ds.to_table()
    return result.num_rows


def bench_s3_eager_metadata():
    """S3 eager metadata-only — 3 columns, no video."""
    ds = lance.dataset(S3_MERGED, storage_options=STORAGE_OPTIONS)
    result = ds.to_table(columns=["episode_id", "duration", "environment_id"])
    return result.num_rows


def bench_s3_lazy_take():
    """S3 lazy random access — take 10 rows."""
    import random
    ds = lance.dataset(S3_MERGED, storage_options=STORAGE_OPTIONS)
    n = ds.count_rows()
    indices = random.sample(range(n), min(10, n))
    batch = ds.take(indices)
    return len(batch)


def bench_s3_lazy_column_proj():
    """S3 lazy column projection — 2 columns from merged."""
    ds = lance.dataset(S3_MERGED, storage_options=STORAGE_OPTIONS)
    result = ds.to_table(columns=["episode_id", "duration"])
    return result.num_rows


def bench_s3_distributed():
    """S3 distributed read — 10 Ray workers, metadata only."""
    @ray.remote
    def worker(paths, storage_options):
        tables = []
        for p in paths:
            ds = lance.dataset(p, storage_options=storage_options)
            tables.append(ds.to_table(columns=["episode_id", "duration"]))
        return pa.concat_tables(tables).num_rows

    paths = get_episode_paths()
    n = 10
    chunk = len(paths) // n + 1
    futures = [
        worker.remote(paths[i * chunk:(i + 1) * chunk], STORAGE_OPTIONS)
        for i in range(n)
        if paths[i * chunk:(i + 1) * chunk]
    ]
    return sum(ray.get(futures))


def main():
    ray.init(address="auto")

    print("=" * 60)
    print("DeltaCAT + Lance Benchmark (S3: pdx.s8k.io)")
    print("=" * 60)
    print(f"S3 base: {S3_BASE}")
    print(f"Merged dataset: {S3_MERGED}")
    print()

    # Warm up — get episode paths
    print("Warming up (listing episode IDs from S3)...")
    t0 = time.perf_counter()
    paths = get_episode_paths()
    print(f"  Found {len(paths)} episodes in {time.perf_counter() - t0:.2f}s")
    print()

    print("--- S3 READ BENCHMARKS (3 iterations) ---")
    benchmark("S3 Eager full load (all cols + video)", bench_s3_eager_full)
    benchmark("S3 Eager metadata-only (3 cols)", bench_s3_eager_metadata)
    benchmark("S3 Lazy take(10 random)", bench_s3_lazy_take)
    benchmark("S3 Lazy column projection (2 cols)", bench_s3_lazy_column_proj)
    benchmark("S3 Distributed (10 workers, metadata)", bench_s3_distributed)

    print()
    print("S3 Benchmark complete!")


if __name__ == "__main__":
    main()
