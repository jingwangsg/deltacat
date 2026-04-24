"""
Test read scaling with multiple Lance shards on S3.

Hypothesis: single Lance file = single manifest = no read parallelism.
Multiple Lance files = multiple manifest endpoints = better scaling.

We create shards of [1, 2, 5, 10, 20, 50] Lance files,
then test read performance with matching worker counts.
"""

import math
import os
import time

import lance
import pyarrow as pa
import ray

LOCAL_MERGED = "/mnt/amlfs-03/shared/jingwang/WORKSPACE/PlayRayDeltaCAT/output/data/_merged.lance"
S3_BASE = "s3://GearHome/jingwang/datasets/feb2_small_batch_deltacat/shards"

STORAGE_OPTIONS = {
    "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
    "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    "aws_endpoint": "https://pdx.s8k.io",
    "aws_region": "us-east-1",
    "allow_http": "false",
}


def create_shards():
    """Split merged dataset into different shard counts, upload to S3."""
    print("Loading local merged dataset...")
    ds = lance.dataset(LOCAL_MERGED)
    full_table = ds.to_table()
    n_rows = full_table.num_rows
    print(f"  {n_rows} rows, {full_table.nbytes / 1e9:.2f} GB")

    shard_counts = [1, 2, 5, 10, 20, 50]

    for n_shards in shard_counts:
        shard_dir = f"{S3_BASE}/n{n_shards:03d}"
        rows_per_shard = math.ceil(n_rows / n_shards)

        print(f"\nCreating {n_shards} shards ({rows_per_shard} rows each)...")
        t0 = time.perf_counter()

        for i in range(n_shards):
            start = i * rows_per_shard
            end = min(start + rows_per_shard, n_rows)
            if start >= n_rows:
                break
            shard_table = full_table.slice(start, end - start)
            shard_path = f"{shard_dir}/shard_{i:04d}.lance"
            lance.write_dataset(shard_table, shard_path, mode="overwrite",
                                storage_options=STORAGE_OPTIONS)

        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s ({full_table.nbytes / elapsed / 1e6:.1f} MB/s)")


def read_scaling_test():
    """Test read perf: N shards x M workers."""
    print("\n" + "=" * 70)
    print("READ SCALING: shards x workers (S3)")
    print("=" * 70)

    shard_counts = [1, 2, 5, 10, 20, 50]
    worker_counts = [1, 2, 5, 10, 20, 50]

    @ray.remote
    def read_worker(shard_paths, storage_options):
        """Read assigned shards, return total rows and bytes."""
        total_rows = 0
        total_bytes = 0
        for path in shard_paths:
            ds = lance.dataset(path, storage_options=storage_options)
            tbl = ds.to_table()
            total_rows += tbl.num_rows
            total_bytes += tbl.nbytes
        return total_rows, total_bytes

    # Header
    print(f"\n{'Shards':>7} | {'Workers':>8} | {'Time (s)':>9} | {'MB/s':>8} | {'Rows/s':>8} | {'Speedup':>8}")
    print("-" * 65)

    # Baseline: 1 shard, 1 worker
    baseline_time = None

    for n_shards in shard_counts:
        shard_dir = f"{S3_BASE}/n{n_shards:03d}"
        shard_paths = [f"{shard_dir}/shard_{i:04d}.lance" for i in range(n_shards)]

        for n_workers in worker_counts:
            if n_workers > n_shards * 5:
                # Skip obviously over-subscribed configs
                continue

            # Assign shards to workers (round-robin)
            assignments = [[] for _ in range(n_workers)]
            for i, path in enumerate(shard_paths):
                assignments[i % n_workers].append(path)
            assignments = [a for a in assignments if a]

            t0 = time.perf_counter()
            futures = [read_worker.remote(a, STORAGE_OPTIONS) for a in assignments]
            results = ray.get(futures)
            elapsed = time.perf_counter() - t0

            total_rows = sum(r[0] for r in results)
            total_bytes = sum(r[1] for r in results)

            if baseline_time is None:
                baseline_time = elapsed
            speedup = baseline_time / elapsed

            print(f"{n_shards:>7} | {n_workers:>8} | {elapsed:>9.3f} | "
                  f"{total_bytes / elapsed / 1e6:>8.1f} | {total_rows / elapsed:>8.0f} | "
                  f"{speedup:>7.2f}x")

        print()  # blank line between shard groups


def read_scaling_metadata_only():
    """Same test but metadata-only (3 small columns, skip video)."""
    print("\n" + "=" * 70)
    print("READ SCALING (metadata only, no video): shards x workers (S3)")
    print("=" * 70)

    shard_counts = [1, 2, 5, 10, 20, 50]
    worker_counts = [1, 2, 5, 10, 20, 50]

    @ray.remote
    def read_meta_worker(shard_paths, storage_options):
        total_rows = 0
        for path in shard_paths:
            ds = lance.dataset(path, storage_options=storage_options)
            tbl = ds.to_table(columns=["episode_id", "duration", "environment_id"])
            total_rows += tbl.num_rows
        return total_rows

    print(f"\n{'Shards':>7} | {'Workers':>8} | {'Time (s)':>9} | {'Rows/s':>8} | {'Speedup':>8}")
    print("-" * 52)

    baseline_time = None

    for n_shards in shard_counts:
        shard_dir = f"{S3_BASE}/n{n_shards:03d}"
        shard_paths = [f"{shard_dir}/shard_{i:04d}.lance" for i in range(n_shards)]

        for n_workers in worker_counts:
            if n_workers > n_shards * 5:
                continue

            assignments = [[] for _ in range(n_workers)]
            for i, path in enumerate(shard_paths):
                assignments[i % n_workers].append(path)
            assignments = [a for a in assignments if a]

            t0 = time.perf_counter()
            futures = [read_meta_worker.remote(a, STORAGE_OPTIONS) for a in assignments]
            results = ray.get(futures)
            elapsed = time.perf_counter() - t0

            total_rows = sum(results)
            if baseline_time is None:
                baseline_time = elapsed
            speedup = baseline_time / elapsed

            print(f"{n_shards:>7} | {n_workers:>8} | {elapsed:>9.3f} | "
                  f"{total_rows / elapsed:>8.0f} | {speedup:>7.2f}x")

        print()


def main():
    ray.init(address="auto")

    create_shards()
    read_scaling_test()
    read_scaling_metadata_only()

    print("\nShard scaling test complete!")


if __name__ == "__main__":
    main()
