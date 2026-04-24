"""
Step 3: Read from local DeltaCAT catalog — 3 patterns.

The open-source deltacat doesn't have Lance-aware read dispatch yet,
so we resolve manifest entries (paths) from the catalog, then read
the Lance datasets directly. This mirrors what deltacat-dev does internally.

Pattern A: Eager (full PyArrow load from all Lance files)
Pattern B: Lazy (O(1) random access via unified Lance dataset)
Pattern C: Distributed (Ray workers each read a subset)
"""

import os
import time

import lance
import pyarrow as pa
import ray

import deltacat as dc
from deltacat import Catalog, CatalogProperties

CATALOG_ROOT = "/mnt/amlfs-03/shared/jingwang/WORKSPACE/PlayRayDeltaCAT/output/catalog"
DATA_DIR = "/mnt/amlfs-03/shared/jingwang/WORKSPACE/PlayRayDeltaCAT/output/data"
CATALOG_NAME = "tutorial"
NAMESPACE = "mecka"
TABLE_NAME = "feb2_small_batch"


def get_lance_paths():
    """Get all Lance dataset paths from the data directory."""
    return sorted([
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if d.endswith(".lance")
    ])


def pattern_a_eager():
    """Eager read — load all Lance files into a single PyArrow table."""
    print("\n=== Pattern A: Eager Read (PyArrow) ===")
    lance_paths = get_lance_paths()

    t0 = time.perf_counter()
    tables = []
    for path in lance_paths:
        ds = lance.dataset(path)
        tables.append(ds.to_table())
    result = pa.concat_tables(tables)
    elapsed = time.perf_counter() - t0

    print(f"  Lance files: {len(lance_paths)}")
    print(f"  Rows: {result.num_rows}")
    print(f"  Columns: {result.column_names}")
    print(f"  Size: {result.nbytes / 1e9:.2f} GB")
    print(f"  Time: {elapsed:.2f}s ({result.nbytes / elapsed / 1e6:.1f} MB/s)")

    sample = result.slice(0, 3).to_pandas()[["episode_id", "environment_id", "task_id", "duration"]]
    print(f"  Sample:\n{sample.to_string(index=False)}")
    return result


def pattern_b_lazy():
    """Lazy read — O(1) random access via Lance.

    Merge all small Lance datasets into one for efficient random access.
    In production, compaction does this; here we do it ad-hoc.
    """
    print("\n=== Pattern B: Lazy Read (Lance O(1) access) ===")
    lance_paths = get_lance_paths()

    # Merge into a single Lance dataset for efficient random access
    merged_path = os.path.join(DATA_DIR, "_merged.lance")
    if not os.path.exists(merged_path):
        print("  Merging Lance files for random access...")
        t0 = time.perf_counter()
        tables = [lance.dataset(p).to_table() for p in lance_paths]
        merged = pa.concat_tables(tables)
        lance.write_dataset(merged, merged_path, mode="create")
        print(f"  Merge time: {time.perf_counter() - t0:.2f}s")

    ds = lance.dataset(merged_path)
    print(f"  Total rows: {ds.count_rows()}")

    # O(1) random access
    t1 = time.perf_counter()
    sample = ds.take([0, 100, 200])
    t_take = time.perf_counter() - t1
    print(f"  take([0, 100, 200]) time: {t_take:.4f}s")
    print(f"  Episode IDs: {sample['episode_id'].to_pylist()}")

    # Column projection — only read metadata, skip video_bytes
    t2 = time.perf_counter()
    meta_only = ds.to_table(columns=["episode_id", "duration", "environment_id"])
    t_proj = time.perf_counter() - t2
    print(f"  Column projection (3 cols, no video): {t_proj:.4f}s, {meta_only.num_rows} rows")
    return ds


def pattern_c_distributed():
    """Distributed read — Ray workers each read a subset of Lance files."""
    print("\n=== Pattern C: Distributed Read (Ray workers) ===")

    @ray.remote
    def worker_read(lance_paths_subset):
        """Each worker reads a chunk of Lance files."""
        tables = []
        for path in lance_paths_subset:
            ds = lance.dataset(path)
            tables.append(ds.to_table(columns=["episode_id", "duration", "environment_id"]))
        result = pa.concat_tables(tables)
        return {
            "num_rows": result.num_rows,
            "episode_ids": result["episode_id"].to_pylist()[:3],
        }

    ray.init(address="auto")

    lance_paths = get_lance_paths()
    num_workers = 10
    chunk_size = len(lance_paths) // num_workers + 1

    t0 = time.perf_counter()
    futures = []
    for i in range(num_workers):
        chunk = lance_paths[i * chunk_size : (i + 1) * chunk_size]
        if chunk:
            futures.append(worker_read.remote(chunk))

    results = ray.get(futures)
    elapsed = time.perf_counter() - t0
    total_rows = sum(r["num_rows"] for r in results)
    print(f"  {len(results)} workers, {total_rows} total rows, {elapsed:.2f}s")
    print(f"  Sample from worker 0: {results[0]['episode_ids']}")


def main():
    pattern_a_eager()
    pattern_b_lazy()
    pattern_c_distributed()
    print("\nAll read patterns complete!")


if __name__ == "__main__":
    main()
