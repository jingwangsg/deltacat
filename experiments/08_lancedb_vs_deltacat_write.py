"""
Experiment: LanceDB concurrent write vs DeltaCAT thin-client write.

Validates the claim from DeltaCAT note:
  "LanceDB: 多 writer 争抢同一个 manifest（串行锁）"
  "DeltaCAT: 每个 writer 写独立 Lance 目录（无锁）"

Approach A — LanceDB style: N workers all append to the SAME Lance dataset
  (simulating concurrent writers sharing one manifest)

Approach B — DeltaCAT style: N workers each write an INDEPENDENT Lance dataset
  (simulating thin-client pattern, zero contention)

Approach C — LanceDB with retries: Same as A, but handle ConflictError

We measure:
  - Wall-clock time
  - Conflict/retry count (A and C)
  - Data integrity (all rows present after write)

Tests on both local FS and S3.
"""

import json
import os
import shutil
import time
import traceback

import lance
import pyarrow as pa
import ray

SOURCE_DIR = "/mnt/amlfs-07/shared/datasets/egocentric_human/vendor_raw/mecka/feb2_small_batch"
BENCH_DIR = "/mnt/amlfs-03/shared/jingwang/WORKSPACE/PlayRayDeltaCAT/output/bench_write_contention"
S3_BENCH = "s3://GearHome/jingwang/datasets/feb2_small_batch_deltacat/bench_write"

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


def load_episode(episode_dir):
    with open(os.path.join(episode_dir, "metadata.json")) as f:
        meta = json.load(f)
    with open(os.path.join(episode_dir, "video.mp4"), "rb") as f:
        video = f.read()
    return pa.table({
        "episode_id": [meta["episodeId"]],
        "user_id": [meta["userId"]],
        "environment_id": [meta["environmentId"]],
        "scene_id": [meta["sceneId"]],
        "task_id": [meta["taskId"]],
        "duration": [meta["duration"]],
        "video_bytes": [video],
        "preannotations_json": [json.dumps(meta.get("preannotations", []))],
    }, schema=SCHEMA)


# =========================================================================
# Approach A: LanceDB style — all workers append to same dataset
# =========================================================================

@ray.remote
def lancedb_style_worker(episode_dirs, dataset_path, storage_options=None):
    """Each worker appends to the SAME Lance dataset (shared manifest)."""
    written = 0
    conflicts = 0
    errors = []

    for ep_dir in episode_dirs:
        table = load_episode(ep_dir)
        max_retries = 10
        for attempt in range(max_retries):
            try:
                lance.write_dataset(
                    table, dataset_path, mode="append",
                    storage_options=storage_options,
                )
                written += 1
                break
            except Exception as e:
                err_str = str(e)
                if "conflict" in err_str.lower() or "commit" in err_str.lower():
                    conflicts += 1
                    time.sleep(0.1 * (attempt + 1))  # backoff
                else:
                    errors.append(err_str)
                    break

    return {"written": written, "conflicts": conflicts, "errors": errors[:5]}


# =========================================================================
# Approach B: DeltaCAT style — each worker writes independent files
# =========================================================================

@ray.remote
def deltacat_style_worker(episode_dirs, output_dir, storage_options=None):
    """Each worker writes to its OWN Lance dataset (zero contention)."""
    written = 0

    for ep_dir in episode_dirs:
        table = load_episode(ep_dir)
        meta_path = os.path.join(ep_dir, "metadata.json")
        with open(meta_path) as f:
            eid = json.load(f)["episodeId"]

        path = os.path.join(output_dir, f"{eid}.lance")
        lance.write_dataset(
            table, path, mode="create",
            storage_options=storage_options,
        )
        written += 1

    return {"written": written, "conflicts": 0, "errors": []}


def run_experiment(name, worker_fn, n_workers, episode_dirs, target_path,
                   storage_options=None, is_shared=False):
    """Run one experiment configuration."""
    chunk = len(episode_dirs) // n_workers + 1
    chunks = [episode_dirs[i * chunk:(i + 1) * chunk] for i in range(n_workers)]
    chunks = [c for c in chunks if c]

    if is_shared:
        # For shared dataset, create it first with 0 rows (schema only)
        empty = pa.table({f.name: pa.array([], type=f.type) for f in SCHEMA}, schema=SCHEMA)
        lance.write_dataset(empty, target_path, mode="overwrite",
                            storage_options=storage_options)

    t0 = time.perf_counter()
    futures = [worker_fn.remote(c, target_path, storage_options) for c in chunks]
    results = ray.get(futures)
    elapsed = time.perf_counter() - t0

    total_written = sum(r["written"] for r in results)
    total_conflicts = sum(r["conflicts"] for r in results)
    all_errors = []
    for r in results:
        all_errors.extend(r["errors"])

    # Verify data integrity
    if is_shared:
        try:
            ds = lance.dataset(target_path, storage_options=storage_options)
            actual_rows = ds.count_rows()
        except Exception as e:
            actual_rows = f"ERROR: {e}"
    else:
        actual_rows = total_written  # each file has 1 row

    return {
        "name": name,
        "workers": n_workers,
        "time": elapsed,
        "written": total_written,
        "conflicts": total_conflicts,
        "errors": len(all_errors),
        "error_samples": all_errors[:3],
        "actual_rows": actual_rows,
        "expected_rows": len(episode_dirs),
    }


def print_result(r):
    status = "✓" if r["actual_rows"] == r["expected_rows"] else "✗"
    print(f"  {r['name']:45s} | {r['workers']:>3}w | {r['time']:>7.2f}s | "
          f"written={r['written']:>3} | conflicts={r['conflicts']:>4} | "
          f"errors={r['errors']:>2} | rows={r['actual_rows']}/{r['expected_rows']} {status}")
    if r["error_samples"]:
        for e in r["error_samples"]:
            print(f"    ERROR: {e[:120]}")


def main():
    ray.init(address="auto")

    episode_dirs = sorted([
        os.path.join(SOURCE_DIR, d)
        for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ])
    n_episodes = len(episode_dirs)
    print(f"Dataset: {n_episodes} episodes")

    worker_counts = [1, 2, 5, 10, 20, 50]

    # =========================
    # LOCAL FILESYSTEM
    # =========================
    print("\n" + "=" * 100)
    print("LOCAL FILESYSTEM: LanceDB-style (shared) vs DeltaCAT-style (independent)")
    print("=" * 100)
    print(f"  {'Approach':45s} | {'W':>3}  | {'Time':>7}  | {'Written':>11} | {'Conflicts':>10} | "
          f"{'Errors':>6} | Integrity")
    print("-" * 100)

    for n_workers in worker_counts:
        # Clean
        shared_path = os.path.join(BENCH_DIR, f"shared_{n_workers}w.lance")
        indep_dir = os.path.join(BENCH_DIR, f"independent_{n_workers}w")
        for p in [shared_path, indep_dir]:
            if os.path.exists(p):
                shutil.rmtree(p)
        os.makedirs(indep_dir, exist_ok=True)
        os.makedirs(os.path.dirname(shared_path), exist_ok=True)

        # LanceDB-style: shared dataset
        r1 = run_experiment(
            f"LanceDB-style (shared, local)",
            lancedb_style_worker, n_workers, episode_dirs,
            shared_path, storage_options=None, is_shared=True,
        )
        print_result(r1)

        # DeltaCAT-style: independent files
        r2 = run_experiment(
            f"DeltaCAT-style (independent, local)",
            deltacat_style_worker, n_workers, episode_dirs,
            indep_dir, storage_options=None, is_shared=False,
        )
        print_result(r2)
        print()

    # =========================
    # S3 (SwiftStack)
    # =========================
    print("\n" + "=" * 100)
    print("S3 (SwiftStack): LanceDB-style (shared) vs DeltaCAT-style (independent)")
    print("=" * 100)
    print(f"  {'Approach':45s} | {'W':>3}  | {'Time':>7}  | {'Written':>11} | {'Conflicts':>10} | "
          f"{'Errors':>6} | Integrity")
    print("-" * 100)

    # Use fewer workers for S3 (slow)
    s3_worker_counts = [1, 2, 5, 10]

    for n_workers in s3_worker_counts:
        shared_path = f"{S3_BENCH}/shared_{n_workers}w.lance"
        indep_dir = f"{S3_BENCH}/independent_{n_workers}w"

        # LanceDB-style: shared dataset on S3
        r1 = run_experiment(
            f"LanceDB-style (shared, S3)",
            lancedb_style_worker, n_workers, episode_dirs,
            shared_path, storage_options=STORAGE_OPTIONS, is_shared=True,
        )
        print_result(r1)

        # DeltaCAT-style: independent files on S3
        r2 = run_experiment(
            f"DeltaCAT-style (independent, S3)",
            deltacat_style_worker, n_workers, episode_dirs,
            indep_dir, storage_options=STORAGE_OPTIONS, is_shared=False,
        )
        print_result(r2)
        print()

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
