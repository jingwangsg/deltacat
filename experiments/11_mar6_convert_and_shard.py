"""
mar_6_umi_lerobot: Convert parquet → Lance, create local shards.
54M rows, 17 GB tabular data, 360 parquet files.
"""

import math
import os
import shutil
import time

import lance
import pyarrow.parquet as pq
import ray

SOURCE_BASE = "/mnt/amlfs-07/shared/datasets/egocentric_human/vendor_raw/mecka/mar_6_umi_lerobot"
LOCAL_OUT = "/mnt/amlfs-03/shared/jingwang/WORKSPACE/PlayRayDeltaCAT/output/mar6"
LOCAL_SHARDS = os.path.join(LOCAL_OUT, "shards")


def discover_parquets():
    parquets = []
    data_dir = os.path.join(SOURCE_BASE, "data")
    for chunk in sorted(os.listdir(data_dir)):
        chunk_dir = os.path.join(data_dir, chunk)
        if not os.path.isdir(chunk_dir):
            continue
        for f in sorted(os.listdir(chunk_dir)):
            if f.endswith(".parquet"):
                parquets.append(os.path.join(chunk_dir, f))
    return parquets


def phase1_convert():
    merged_path = os.path.join(LOCAL_OUT, "merged.lance")
    if os.path.exists(merged_path):
        ds = lance.dataset(merged_path)
        n = ds.count_rows()
        print(f"Phase 1: merged.lance already exists ({n:,} rows)")
        return merged_path, n

    print("Phase 1: Converting parquet → merged Lance...")
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
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(parquets)} files, {total_rows:,} rows, {total_bytes / 1e9:.1f} GB...")

    elapsed = time.perf_counter() - t0
    print(f"  Complete: {total_rows:,} rows, {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s "
          f"({total_bytes / elapsed / 1e6:.0f} MB/s)")
    return merged_path, total_rows


@ray.remote(num_cpus=1)
def create_shard_local(merged_path, shard_idx, start_row, num_rows, out_path):
    ds = lance.dataset(merged_path)
    end_row = min(start_row + num_rows, ds.count_rows())
    indices = list(range(start_row, end_row))
    shard_table = ds.take(indices)
    lance.write_dataset(shard_table, out_path, mode="create", max_rows_per_file=1_000_000)
    return {"shard_idx": shard_idx, "rows": len(indices)}


def phase2_create_shards(merged_path, total_rows, shard_counts):
    print(f"\nPhase 2: Creating local shards")
    os.makedirs(LOCAL_SHARDS, exist_ok=True)

    for n_shards in shard_counts:
        shard_dir = os.path.join(LOCAL_SHARDS, f"n{n_shards:03d}")
        done_marker = os.path.join(shard_dir, ".done")
        if os.path.exists(done_marker):
            print(f"  n{n_shards:03d}: already done")
            continue

        if os.path.exists(shard_dir):
            shutil.rmtree(shard_dir)
        os.makedirs(shard_dir)

        rows_per_shard = math.ceil(total_rows / n_shards)
        print(f"  n{n_shards:03d}: {n_shards} shards x {rows_per_shard:,} rows...")

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
        open(done_marker, "w").close()


def main():
    ray.init(address="auto")
    shard_counts = [1, 5, 10, 20, 50, 100]

    merged_path, total_rows = phase1_convert()
    phase2_create_shards(merged_path, total_rows, shard_counts)
    print("\nConversion and sharding complete!")


if __name__ == "__main__":
    main()
