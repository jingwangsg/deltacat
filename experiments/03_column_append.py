"""
Step 4: Column-append write using DeltaCAT ADD mode.

Adds a computed column (video_size_bytes) to the existing table.
Uses ADD mode with the thin-client pattern: write Lance, register manifest.
This creates a new Delta alongside existing data (MOR).
"""

import os
import time

import lance
import pyarrow as pa

import deltacat as dc
from deltacat import Catalog, CatalogProperties
from deltacat.types.media import ContentType
from deltacat.types.tables import TableWriteMode
from deltacat.storage.model.manifest import Manifest
from deltacat.utils.lance_utils import build_manifest_entries

CATALOG_ROOT = "/mnt/amlfs-03/shared/jingwang/WORKSPACE/PlayRayDeltaCAT/output/catalog"
DATA_DIR = "/mnt/amlfs-03/shared/jingwang/WORKSPACE/PlayRayDeltaCAT/output/data"
CATALOG_NAME = "tutorial"
NAMESPACE = "mecka"
TABLE_NAME = "feb2_small_batch"


def main():
    # Step 1: Read existing data to compute new column
    print("Reading existing data...")
    t0 = time.perf_counter()
    lance_paths = sorted([
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if d.endswith(".lance") and not d.startswith("_")
    ])

    episode_ids = []
    video_sizes = []
    for path in lance_paths:
        ds = lance.dataset(path)
        tbl = ds.to_table(columns=["episode_id", "video_bytes"])
        for i in range(tbl.num_rows):
            episode_ids.append(tbl["episode_id"][i].as_py())
            video_sizes.append(len(tbl["video_bytes"][i].as_py()))

    print(f"  Read {len(episode_ids)} episodes in {time.perf_counter() - t0:.2f}s")
    print(f"  Video sizes: min={min(video_sizes)}, max={max(video_sizes)}, "
          f"avg={sum(video_sizes) // len(video_sizes)}")

    # Step 2: Write new column as a Lance dataset
    new_col_table = pa.table({
        "episode_id": pa.array(episode_ids, type=pa.string()),
        "video_size_bytes": pa.array(video_sizes, type=pa.int64()),
    })

    new_lance_path = os.path.join(DATA_DIR, "_column_video_size.lance")
    if os.path.exists(new_lance_path):
        import shutil
        shutil.rmtree(new_lance_path)

    t1 = time.perf_counter()
    lance.write_dataset(new_col_table, new_lance_path, mode="create")
    print(f"  Wrote new column Lance dataset in {time.perf_counter() - t1:.3f}s")

    # Step 3: Register with DeltaCAT as a new Delta (ADD mode)
    t2 = time.perf_counter()
    dc.init()
    cat_props = CatalogProperties(root=CATALOG_ROOT)
    dc.put_catalog(CATALOG_NAME, catalog=Catalog(config=cat_props))

    entries = build_manifest_entries(lance_path=new_lance_path)
    dc.write_to_table(
        data=None,
        table=TABLE_NAME,
        namespace=NAMESPACE,
        manifest=Manifest.of(entries=entries),
        content_type=ContentType.LANCE,
        mode=TableWriteMode.ADD,
        catalog=CATALOG_NAME,
    )
    print(f"  Registered new Delta (ADD) in {time.perf_counter() - t2:.2f}s")

    # Step 4: Verify — read back the new column
    ds = lance.dataset(new_lance_path)
    verify = ds.to_table()
    print(f"\n  Verification: {verify.num_rows} rows, columns={verify.column_names}")
    print(f"  Sample:\n{verify.slice(0, 5).to_pandas().to_string(index=False)}")
    print(f"\nColumn append complete! New Delta registered in catalog.")


if __name__ == "__main__":
    main()
