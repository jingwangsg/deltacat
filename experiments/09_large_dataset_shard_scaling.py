"""
Shard scaling on large dataset: feb_08_500hr_lerobot_updated_fixed
~35 GB, 54M rows.

Phase 1: Convert parquet → merged Lance locally (incremental append)
Phase 2: Create local shard configs [1, 5, 10, 20, 50, 100]
Phase 3: Upload all shards to S3 via mscsync (parallel)
Phase 4: Read scaling test: shards x workers on S3
"""

import math
import os
import shutil
import subprocess
import time

import lance
import pyarrow.parquet as pq
import ray

# --- Config ---
SOURCE_BASE = "/mnt/amlfs-07/shared/datasets/egocentric_human/vendor_raw/mecka/feb_08_500hr_lerobot_updated_fixed"
LOCAL_OUT = "/mnt/amlfs-03/shared/jingwang/WORKSPACE/PlayRayDeltaCAT/output/large"
LOCAL_SHARDS = os.path.join(LOCAL_OUT, "shards")
S3_BASE = "s3://GearHome/jingwang/datasets/feb_08_500hr_deltacat/shards"

STORAGE_OPTIONS = {
    "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
    "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    "aws_endpoint": "https://pdx.s8k.io",
    "aws_region": "us-east-1",
    "allow_http": "false",
}

META_COLUMNS = ["index", "episode_index", "frame_index", "timestamp", "task_index", "subtask_index"]

MSCSYNC_SH = os.path.expanduser("~/WORKSPACE/GearLaunchV2/scripts/msc/mscsync.sh")


def discover_parquets():
    parquets = []
    for shard in sorted(os.listdir(SOURCE_BASE)):
        data_dir = os.path.join(SOURCE_BASE, shard, "data")
        if not os.path.isdir(data_dir):
            continue
        for chunk in sorted(os.listdir(data_dir)):
            chunk_dir = os.path.join(data_dir, chunk)
            if not os.path.isdir(chunk_dir):
                continue
            for f in sorted(os.listdir(chunk_dir)):
                if f.endswith(".parquet"):
                    parquets.append(os.path.join(chunk_dir, f))
    return parquets


# =========================================================================
# Phase 1: parquet → merged Lance (local, incremental)
# =========================================================================

def phase1_convert():
    merged_path = os.path.join(LOCAL_OUT, "merged.lance")
    if os.path.exists(merged_path):
        ds = lance.dataset(merged_path)
        n = ds.count_rows()
        print(f"Phase 1: merged.lance already exists ({n:,} rows)")
        return merged_path, n

    print("Phase 1: Converting parquet → merged Lance (local, incremental)...")
    parquets = discover_parquets()
    print(f"  Found {len(parquets)} parquet files")

    os.makedirs(LOCAL_OUT, exist_ok=True)
    t0 = time.perf_counter()
    total_rows = 0
    total_bytes = 0

    for i, pf in enumerate(parquets):
        tbl = pq.read_table(pf)
        new_names = [name.replace(".", "__") for name in tbl.column_names]
        tbl = tbl.rename_columns(new_names)

        mode = "create" if i == 0 else "append"
        lance.write_dataset(tbl, merged_path, mode=mode, max_rows_per_file=1_000_000)
        total_rows += tbl.num_rows
        total_bytes += tbl.nbytes
        if (i + 1) % 20 == 0:
            print(f"  Written {i+1}/{len(parquets)} files, {total_rows:,} rows, "
                  f"{total_bytes / 1e9:.1f} GB...")

    elapsed = time.perf_counter() - t0
    print(f"  Complete: {total_rows:,} rows, {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s "
          f"({total_bytes / elapsed / 1e6:.0f} MB/s)")
    return merged_path, total_rows


# =========================================================================
# Phase 2: Create local shards (Ray parallel)
# =========================================================================

@ray.remote(num_cpus=1)
def create_shard_local(merged_path, shard_idx, start_row, num_rows, out_path):
    """Read a slice from merged Lance, write as local shard."""
    ds = lance.dataset(merged_path)
    end_row = min(start_row + num_rows, ds.count_rows())
    indices = list(range(start_row, end_row))
    shard_table = ds.take(indices)
    lance.write_dataset(shard_table, out_path, mode="create", max_rows_per_file=1_000_000)
    return {"shard_idx": shard_idx, "rows": len(indices)}


def phase2_create_local_shards(merged_path, total_rows, shard_counts):
    print(f"\nPhase 2: Creating local shards")
    os.makedirs(LOCAL_SHARDS, exist_ok=True)

    for n_shards in shard_counts:
        shard_dir = os.path.join(LOCAL_SHARDS, f"n{n_shards:03d}")
        done_marker = os.path.join(shard_dir, ".done")
        if os.path.exists(done_marker):
            print(f"  n{n_shards:03d}: already done, skipping")
            continue

        if os.path.exists(shard_dir):
            shutil.rmtree(shard_dir)
        os.makedirs(shard_dir)

        rows_per_shard = math.ceil(total_rows / n_shards)
        print(f"  Creating n{n_shards:03d}: {n_shards} shards x {rows_per_shard:,} rows...")

        t0 = time.perf_counter()
        futures = []
        for i in range(n_shards):
            start = i * rows_per_shard
            if start >= total_rows:
                break
            out_path = os.path.join(shard_dir, f"shard_{i:04d}.lance")
            futures.append(create_shard_local.remote(merged_path, i, start, rows_per_shard, out_path))

        results = ray.get(futures)
        elapsed = time.perf_counter() - t0
        total_written = sum(r["rows"] for r in results)
        print(f"    Done: {len(results)} shards, {total_written:,} rows, {elapsed:.1f}s")

        # Mark done
        open(done_marker, "w").close()


# =========================================================================
# Phase 3: Upload to S3 via mscsync
# =========================================================================

def phase3_upload(shard_counts):
    print(f"\nPhase 3: Uploading shards to S3 via s5cmd")
    creds_file = os.path.expanduser("~/.gear/aws_credentials")
    endpoint = "https://pdx.s8k.io"

    for n_shards in shard_counts:
        local_dir = os.path.join(LOCAL_SHARDS, f"n{n_shards:03d}")
        s3_dir = f"{S3_BASE}/n{n_shards:03d}/"

        print(f"  Uploading n{n_shards:03d} → {s3_dir} ...")
        t0 = time.perf_counter()

        result = subprocess.run(
            ["s5cmd", "--credentials-file", creds_file, "--endpoint-url", endpoint,
             "cp", "--concurrency", "50", f"{local_dir}/*", s3_dir],
            capture_output=True, text=True, timeout=3600,
        )
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            # Check if it's just warnings
            err_lines = [l for l in result.stderr.split("\n") if "ERROR" in l]
            if err_lines:
                print(f"    FAILED ({elapsed:.1f}s): {err_lines[0][:150]}")
            else:
                print(f"    Done: {elapsed:.1f}s (with warnings)")
        else:
            # Count uploaded files
            cp_lines = [l for l in result.stdout.split("\n") if l.startswith("cp ")]
            print(f"    Done: {len(cp_lines)} files in {elapsed:.1f}s")


# =========================================================================
# Phase 4: Read scaling on S3
# =========================================================================

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


def phase4_read_scaling(shard_counts, worker_counts):
    # --- Full read ---
    print("\n" + "=" * 75)
    print("FULL READ SCALING: shards x workers (S3, ~35GB)")
    print("=" * 75)
    print(f"\n{'Shards':>7} | {'Workers':>8} | {'Time (s)':>9} | {'MB/s':>8} | {'Rows/s':>10} | {'Speedup':>8}")
    print("-" * 70)

    baseline_time = None

    for n_shards in shard_counts:
        shard_dir = f"{S3_BASE}/n{n_shards:03d}"
        shard_paths = [f"{shard_dir}/shard_{i:04d}.lance" for i in range(n_shards)]

        for n_workers in worker_counts:
            if n_workers > n_shards * 10:
                continue

            assignments = [[] for _ in range(n_workers)]
            for i, path in enumerate(shard_paths):
                assignments[i % n_workers].append(path)
            assignments = [a for a in assignments if a]

            t0 = time.perf_counter()
            futures = [read_worker_full.remote(a, STORAGE_OPTIONS) for a in assignments]
            results = ray.get(futures)
            elapsed = time.perf_counter() - t0

            total_rows = sum(r[0] for r in results)
            total_bytes = sum(r[1] for r in results)

            if baseline_time is None:
                baseline_time = elapsed
            speedup = baseline_time / elapsed

            print(f"{n_shards:>7} | {n_workers:>8} | {elapsed:>9.1f} | "
                  f"{total_bytes / elapsed / 1e6:>8.1f} | {total_rows / elapsed:>10.0f} | "
                  f"{speedup:>7.2f}x")
        print()

    # --- Metadata-only ---
    print("\n" + "=" * 75)
    print("METADATA-ONLY READ: shards x workers (S3)")
    print("=" * 75)
    print(f"\n{'Shards':>7} | {'Workers':>8} | {'Time (s)':>9} | {'Rows/s':>10} | {'Speedup':>8}")
    print("-" * 55)

    baseline_time = None

    for n_shards in shard_counts:
        shard_dir = f"{S3_BASE}/n{n_shards:03d}"
        shard_paths = [f"{shard_dir}/shard_{i:04d}.lance" for i in range(n_shards)]

        for n_workers in worker_counts:
            if n_workers > n_shards * 10:
                continue

            assignments = [[] for _ in range(n_workers)]
            for i, path in enumerate(shard_paths):
                assignments[i % n_workers].append(path)
            assignments = [a for a in assignments if a]

            t0 = time.perf_counter()
            futures = [read_worker_meta.remote(a, STORAGE_OPTIONS, META_COLUMNS) for a in assignments]
            results = ray.get(futures)
            elapsed = time.perf_counter() - t0

            total_rows = sum(results)
            if baseline_time is None:
                baseline_time = elapsed
            speedup = baseline_time / elapsed

            print(f"{n_shards:>7} | {n_workers:>8} | {elapsed:>9.2f} | "
                  f"{total_rows / elapsed:>10.0f} | {speedup:>7.2f}x")
        print()


def main():
    ray.init(address="auto")

    shard_counts = [1, 5, 10, 20, 50, 100]
    worker_counts = [1, 2, 5, 10, 20, 50, 100]

    # Phase 1
    merged_path, total_rows = phase1_convert()

    # Phase 2: local shards
    phase2_create_local_shards(merged_path, total_rows, shard_counts)

    # Phase 3: upload via mscsync
    phase3_upload(shard_counts)

    # Phase 4: read scaling
    phase4_read_scaling(shard_counts, worker_counts)

    print("\nLarge dataset shard scaling complete!")


if __name__ == "__main__":
    main()
