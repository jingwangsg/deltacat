"""
Step 2: High-performance write using Thin-Client pattern.

Reads 346 episodes from feb2_small_batch, writes Lance files in parallel
via Ray tasks, then registers them with DeltaCAT catalog.
"""

import json
import os
import time
from pathlib import Path

import pyarrow as pa
import lance
import ray

import deltacat as dc
from deltacat import Catalog, CatalogProperties
from deltacat.types.media import ContentType
from deltacat.types.tables import TableWriteMode
from deltacat.storage.model.manifest import Manifest
from deltacat.utils.lance_utils import build_manifest_entries

# --- Config ---
SOURCE_DIR = "/mnt/amlfs-07/shared/datasets/egocentric_human/vendor_raw/mecka/feb2_small_batch"
OUTPUT_BASE = "/mnt/amlfs-03/shared/jingwang/WORKSPACE/PlayRayDeltaCAT/output"
DATA_DIR = os.path.join(OUTPUT_BASE, "data")
CATALOG_ROOT = os.path.join(OUTPUT_BASE, "catalog")
CATALOG_NAME = "tutorial"
NAMESPACE = "mecka"
TABLE_NAME = "feb2_small_batch"

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


@ray.remote
def process_episode(episode_dir: str, data_dir: str) -> dict:
    """Read one episode and write it as a Lance dataset."""
    meta_path = os.path.join(episode_dir, "metadata.json")
    video_path = os.path.join(episode_dir, "video.mp4")

    with open(meta_path) as f:
        meta = json.load(f)
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    episode_id = meta["episodeId"]
    table = pa.table({
        "episode_id": [episode_id],
        "user_id": [meta["userId"]],
        "environment_id": [meta["environmentId"]],
        "scene_id": [meta["sceneId"]],
        "task_id": [meta["taskId"]],
        "duration": [meta["duration"]],
        "video_bytes": [video_bytes],
        "preannotations_json": [json.dumps(meta.get("preannotations", []))],
    }, schema=SCHEMA)

    lance_path = os.path.join(data_dir, f"{episode_id}.lance")
    lance.write_dataset(table, lance_path, mode="create")

    return {
        "lance_path": lance_path,
        "episode_id": episode_id,
        "video_size": len(video_bytes),
    }


def main():
    ray.init(address="auto")

    # Clean previous output
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CATALOG_ROOT, exist_ok=True)

    # List episode directories
    episode_dirs = sorted([
        os.path.join(SOURCE_DIR, d)
        for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ])
    print(f"Found {len(episode_dirs)} episodes")

    # Phase 1: Parallel Lance writes via Ray
    t0 = time.perf_counter()
    futures = [process_episode.remote(d, DATA_DIR) for d in episode_dirs]
    results = ray.get(futures)
    t_write = time.perf_counter() - t0
    total_bytes = sum(r["video_size"] for r in results)
    print(f"Phase 1 (parallel Lance write): {t_write:.2f}s, "
          f"{len(results)} episodes, {total_bytes / 1e9:.2f} GB, "
          f"{total_bytes / t_write / 1e6:.1f} MB/s")

    # Phase 2: Build manifest entries and register with DeltaCAT
    t1 = time.perf_counter()

    dc.init()
    cat_props = CatalogProperties(root=CATALOG_ROOT)
    dc.put_catalog(CATALOG_NAME, catalog=Catalog(config=cat_props))
    dc.create_namespace(NAMESPACE, catalog=CATALOG_NAME)

    all_entries = []
    for r in results:
        entries = build_manifest_entries(lance_path=r["lance_path"])
        all_entries.extend(entries)

    from deltacat.storage.model.manifest import ManifestEntryList
    merged = ManifestEntryList.of(all_entries)

    dc.write_to_table(
        data=None,
        table=TABLE_NAME,
        namespace=NAMESPACE,
        manifest=Manifest.of(entries=merged),
        content_type=ContentType.LANCE,
        mode=TableWriteMode.CREATE,
        catalog=CATALOG_NAME,
    )

    t_register = time.perf_counter() - t1
    print(f"Phase 2 (manifest registration): {t_register:.2f}s")
    print(f"Total: {time.perf_counter() - t0:.2f}s")
    print(f"Written to catalog: {CATALOG_ROOT}")


if __name__ == "__main__":
    main()
