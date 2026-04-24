"""
Phase 4 only: Read scaling test on S3 (large dataset, ~35GB, 54M rows).
Shards already uploaded by s5cmd.
"""

import os
import time

import lance
import ray

S3_BASE = "s3://GearHome/jingwang/datasets/feb_08_500hr_deltacat/shards"

STORAGE_OPTIONS = {
    "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
    "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    "aws_endpoint": "https://pdx.s8k.io",
    "aws_region": "us-east-1",
    "allow_http": "false",
}

META_COLUMNS = ["index", "episode_index", "frame_index", "timestamp", "task_index", "subtask_index"]


@ray.remote(num_cpus=1)
def read_worker_full(shard_paths, storage_options):
    total_rows = 0
    total_bytes = 0
    for path in shard_paths:
        ds = lance.dataset(path, storage_options=storage_options)
        tbl = ds.to_table()
        total_rows += tbl.num_rows
        total_bytes += tbl.nbytes
    return total_rows, total_bytes


@ray.remote(num_cpus=1)
def read_worker_meta(shard_paths, storage_options, columns):
    total_rows = 0
    for path in shard_paths:
        ds = lance.dataset(path, storage_options=storage_options)
        tbl = ds.to_table(columns=columns)
        total_rows += tbl.num_rows
    return total_rows


def run_scaling(label, worker_fn, shard_counts, worker_counts, extra_args=()):
    print(f"\n{'=' * 75}")
    print(f"{label}")
    print(f"{'=' * 75}")

    if "FULL" in label:
        print(f"\n{'Shards':>7} | {'Workers':>8} | {'Time (s)':>9} | {'MB/s':>8} | {'Rows/s':>10} | {'Speedup':>8}")
    else:
        print(f"\n{'Shards':>7} | {'Workers':>8} | {'Time (s)':>9} | {'Rows/s':>10} | {'Speedup':>8}")
    print("-" * 70)

    baseline_time = None

    for n_shards in shard_counts:
        shard_dir = f"{S3_BASE}/n{n_shards:03d}"
        shard_paths = [f"{shard_dir}/shard_{i:04d}.lance" for i in range(n_shards)]

        for n_workers in worker_counts:
            # Skip configs where workers >> shards (nothing to parallelize)
            if n_workers > n_shards * 5:
                continue

            assignments = [[] for _ in range(n_workers)]
            for i, path in enumerate(shard_paths):
                assignments[i % n_workers].append(path)
            assignments = [a for a in assignments if a]

            t0 = time.perf_counter()
            futures = [worker_fn.remote(a, STORAGE_OPTIONS, *extra_args) for a in assignments]
            results = ray.get(futures)
            elapsed = time.perf_counter() - t0

            if isinstance(results[0], tuple):
                total_rows = sum(r[0] for r in results)
                total_bytes = sum(r[1] for r in results)
            else:
                total_rows = sum(results)
                total_bytes = 0

            if baseline_time is None:
                baseline_time = elapsed
            speedup = baseline_time / elapsed

            if total_bytes > 0:
                print(f"{n_shards:>7} | {n_workers:>8} | {elapsed:>9.1f} | "
                      f"{total_bytes / elapsed / 1e6:>8.1f} | {total_rows / elapsed:>10.0f} | "
                      f"{speedup:>7.2f}x")
            else:
                print(f"{n_shards:>7} | {n_workers:>8} | {elapsed:>9.2f} | "
                      f"{total_rows / elapsed:>10.0f} | {speedup:>7.2f}x")

        print()


def main():
    ray.init(address="auto")

    shard_counts = [1, 5, 10, 20, 50, 100]
    worker_counts = [1, 2, 5, 10, 20, 50, 100]

    # Full read
    run_scaling(
        "FULL READ SCALING: shards x workers (S3, ~35GB, 54M rows)",
        read_worker_full, shard_counts, worker_counts,
    )

    # Metadata-only read
    run_scaling(
        "METADATA-ONLY READ SCALING: shards x workers (S3, 6 cols)",
        read_worker_meta, shard_counts, worker_counts,
        extra_args=(META_COLUMNS,),
    )

    print("\nRead scaling test complete!")


if __name__ == "__main__":
    main()
