"""
mar_6_umi_lerobot: S3 read scaling test.
54M rows, 18.5 GB, 12 columns (state vectors).
"""

import os
import time

import lance
import ray

S3_BASE = "s3://GearHome/jingwang/datasets/mar6_umi_deltacat/shards"

STORAGE_OPTIONS = {
    "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
    "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    "aws_endpoint": "https://pdx.s8k.io",
    "aws_region": "us-east-1",
    "allow_http": "false",
}

META_COLUMNS = ["index", "episode_index", "frame_index", "timestamp"]


@ray.remote(num_cpus=1)
def read_full(shard_paths, storage_options):
    total_rows = 0
    total_bytes = 0
    for path in shard_paths:
        ds = lance.dataset(path, storage_options=storage_options)
        tbl = ds.to_table()
        total_rows += tbl.num_rows
        total_bytes += tbl.nbytes
    return total_rows, total_bytes


@ray.remote(num_cpus=1)
def read_meta(shard_paths, storage_options, columns):
    total_rows = 0
    for path in shard_paths:
        ds = lance.dataset(path, storage_options=storage_options)
        tbl = ds.to_table(columns=columns)
        total_rows += tbl.num_rows
    return total_rows


def run_test(label, worker_fn, shard_counts, worker_counts, extra_args=()):
    print(f"\n{'=' * 75}")
    print(f"{label}")
    print(f"{'=' * 75}")

    is_full = "FULL" in label
    if is_full:
        print(f"\n{'Shards':>7} | {'Workers':>8} | {'Time (s)':>9} | {'MB/s':>8} | {'Rows/s':>10} | {'Speedup':>8}")
    else:
        print(f"\n{'Shards':>7} | {'Workers':>8} | {'Time (s)':>9} | {'Rows/s':>10} | {'Speedup':>8}")
    print("-" * 70)

    baseline = None

    for n_shards in shard_counts:
        shard_paths = [f"{S3_BASE}/n{n_shards:03d}/shard_{i:04d}.lance" for i in range(n_shards)]

        for n_workers in worker_counts:
            if n_workers > n_shards * 5:
                continue

            assignments = [[] for _ in range(n_workers)]
            for i, p in enumerate(shard_paths):
                assignments[i % n_workers].append(p)
            assignments = [a for a in assignments if a]

            t0 = time.perf_counter()
            futures = [worker_fn.remote(a, STORAGE_OPTIONS, *extra_args) for a in assignments]
            results = ray.get(futures)
            elapsed = time.perf_counter() - t0

            if isinstance(results[0], tuple):
                rows = sum(r[0] for r in results)
                bts = sum(r[1] for r in results)
            else:
                rows = sum(results)
                bts = 0

            if baseline is None:
                baseline = elapsed
            sp = baseline / elapsed

            if is_full:
                print(f"{n_shards:>7} | {n_workers:>8} | {elapsed:>9.1f} | "
                      f"{bts / elapsed / 1e6:>8.1f} | {rows / elapsed:>10.0f} | {sp:>7.2f}x")
            else:
                print(f"{n_shards:>7} | {n_workers:>8} | {elapsed:>9.2f} | "
                      f"{rows / elapsed:>10.0f} | {sp:>7.2f}x")
        print()


def main():
    ray.init(address="auto")

    shards = [1, 5, 10, 20, 50, 100]
    workers = [1, 2, 5, 10, 20, 50, 100]

    run_test("FULL READ (S3, mar6, 18.5GB, 54M rows)", read_full, shards, workers)
    run_test("METADATA-ONLY READ (S3, mar6, 4 cols)", read_meta, shards, workers, extra_args=(META_COLUMNS,))

    print("\nmar6 read scaling complete!")


if __name__ == "__main__":
    main()
