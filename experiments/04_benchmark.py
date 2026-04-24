"""
Step 5: Profile all read/write patterns (local filesystem).

Write Patterns:
  1. Thin-Client (parallel Lance + manifest register)
  2. ADD mode (column append)
  3. Direct dc.write_to_table with PyArrow data

Read Patterns:
  1. Eager full load (PyArrow)
  2. Eager metadata-only (column projection)
  3. Lazy random access (Lance take)
  4. Distributed (Ray workers)
"""

import json
import os
import shutil
import time
from typing import Callable

import lance
import pyarrow as pa
import ray

import deltacat as dc
from deltacat import Catalog, CatalogProperties
from deltacat.types.media import ContentType
from deltacat.types.tables import TableWriteMode
from deltacat.storage.model.manifest import Manifest
from deltacat.utils.lance_utils import build_manifest_entries

SOURCE_DIR = "/mnt/amlfs-07/shared/datasets/egocentric_human/vendor_raw/mecka/feb2_small_batch"
BENCH_DIR = "/mnt/amlfs-03/shared/jingwang/WORKSPACE/PlayRayDeltaCAT/output/bench"
DATA_DIR = "/mnt/amlfs-03/shared/jingwang/WORKSPACE/PlayRayDeltaCAT/output/data"
MERGED_PATH = os.path.join(DATA_DIR, "_merged.lance")

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


def benchmark(name: str, fn: Callable, iterations: int = 3):
    """Run fn multiple times and report stats."""
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


def get_lance_paths():
    return sorted([
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if d.endswith(".lance") and not d.startswith("_")
    ])


def load_episodes():
    """Load all episodes into a single PyArrow table."""
    episode_dirs = sorted([
        os.path.join(SOURCE_DIR, d)
        for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ])
    rows = []
    for ep_dir in episode_dirs:
        with open(os.path.join(ep_dir, "metadata.json")) as f:
            meta = json.load(f)
        with open(os.path.join(ep_dir, "video.mp4"), "rb") as f:
            video = f.read()
        rows.append({
            "episode_id": meta["episodeId"],
            "user_id": meta["userId"],
            "environment_id": meta["environmentId"],
            "scene_id": meta["sceneId"],
            "task_id": meta["taskId"],
            "duration": meta["duration"],
            "video_bytes": video,
            "preannotations_json": json.dumps(meta.get("preannotations", [])),
        })
    return pa.table(
        {k: [r[k] for r in rows] for k in rows[0]},
        schema=SCHEMA,
    )


# ===== WRITE BENCHMARKS =====

def bench_write_thin_client():
    """Thin-client: parallel Ray Lance writes + manifest register."""
    @ray.remote
    def write_one(episode_dir, out_dir):
        with open(os.path.join(episode_dir, "metadata.json")) as f:
            meta = json.load(f)
        with open(os.path.join(episode_dir, "video.mp4"), "rb") as f:
            video = f.read()
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
        lance.write_dataset(table, path, mode="create")
        return path

    out_dir = os.path.join(BENCH_DIR, "thin_client")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    episode_dirs = sorted([
        os.path.join(SOURCE_DIR, d)
        for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ])
    futures = [write_one.remote(d, out_dir) for d in episode_dirs]
    ray.get(futures)
    return len(episode_dirs)


def bench_write_direct_lance():
    """Direct lance.write_dataset — single writer, full table."""
    pa_table = load_episodes()
    out_path = os.path.join(BENCH_DIR, "direct.lance")
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    lance.write_dataset(pa_table, out_path, mode="create")
    return pa_table.num_rows


# ===== READ BENCHMARKS =====

def bench_read_eager_full():
    """Eager full load — all columns including video."""
    paths = get_lance_paths()
    tables = [lance.dataset(p).to_table() for p in paths]
    result = pa.concat_tables(tables)
    return result.num_rows


def bench_read_eager_metadata():
    """Eager metadata-only — column projection, no video."""
    paths = get_lance_paths()
    tables = [lance.dataset(p).to_table(columns=["episode_id", "duration", "environment_id"]) for p in paths]
    result = pa.concat_tables(tables)
    return result.num_rows


def bench_read_lazy_take():
    """Lazy random access — take 10 random rows."""
    import random
    ds = lance.dataset(MERGED_PATH)
    indices = random.sample(range(ds.count_rows()), 10)
    batch = ds.take(indices)
    return len(batch)


def bench_read_lazy_column_proj():
    """Lazy column projection — metadata from merged dataset."""
    ds = lance.dataset(MERGED_PATH)
    result = ds.to_table(columns=["episode_id", "duration"])
    return result.num_rows


def bench_read_distributed():
    """Distributed read — 10 Ray workers, metadata columns only."""
    @ray.remote
    def worker(paths):
        tables = [lance.dataset(p).to_table(columns=["episode_id", "duration"]) for p in paths]
        return pa.concat_tables(tables).num_rows

    paths = get_lance_paths()
    n = 10
    chunk = len(paths) // n + 1
    futures = [worker.remote(paths[i * chunk:(i + 1) * chunk]) for i in range(n) if paths[i * chunk:(i + 1) * chunk]]
    return sum(ray.get(futures))


def main():
    ray.init(address="auto")
    os.makedirs(BENCH_DIR, exist_ok=True)

    print("=" * 60)
    print("DeltaCAT + Lance Benchmark (Local Filesystem)")
    print("=" * 60)
    print(f"Dataset: 344 episodes, ~7.15 GB total")
    print()

    print("--- WRITE BENCHMARKS (3 iterations) ---")
    benchmark("Thin-Client (parallel Ray)", bench_write_thin_client)
    benchmark("Direct lance.write_dataset (single)", bench_write_direct_lance)

    print()
    print("--- READ BENCHMARKS (3 iterations) ---")
    benchmark("Eager full load (all cols + video)", bench_read_eager_full)
    benchmark("Eager metadata-only (3 cols)", bench_read_eager_metadata)
    benchmark("Lazy take(10 random)", bench_read_lazy_take)
    benchmark("Lazy column projection (2 cols)", bench_read_lazy_column_proj)
    benchmark("Distributed (10 workers, metadata)", bench_read_distributed)

    print()
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
