# DeltaCAT + Lance Experiments

This directory contains the benchmark scripts that produced the results in
[`../experiment.md`](../experiment.md). They exercise the Lance integration
that was backported to this fork from `deltacat-dev`.

## Setup

```bash
# Create venv inheriting cluster's ray/pyarrow/lance
uv venv --system-site-packages --no-managed-python .venv
source .venv/bin/activate

# Install this fork (editable, no deps)
uv pip install --no-deps -e .

# Missing runtime deps
uv pip install daft tenacity
```

`runtime_env.yaml` for `ray job submit`:

```yaml
env_vars:
  PATH: "/abs/path/to/.venv/bin/:$PATH"
  AWS_ACCESS_KEY_ID: "..."
  AWS_SECRET_ACCESS_KEY: "..."
  AWS_DEFAULT_REGION: "us-east-1"

py_executable: /abs/path/to/.venv/bin/python
working_dir: /abs/path/to/experiments/
excludes: [.venv, .git, __pycache__, output]
```

Run any script with:

```bash
RAY_ADDRESS="http://127.0.0.1:8265" ray job submit \
  --runtime-env runtime_env.yaml -- python NN_script_name.py
```

## Scripts

| # | Script | Purpose |
|---|---|---|
| 01 | `01_write_local.py` | **Thin-client write** — 344 Ray tasks each write an independent Lance file, then register manifests with DeltaCAT (`data=None, manifest=...` path). Demonstrates the no-lock parallel write pattern. |
| 02 | `02_read_local.py` | Three read patterns: eager (PyArrow full load), lazy (Lance `take()` for O(1) random access), distributed (Ray workers each read a subset). |
| 03 | `03_column_append.py` | **Column-append** via DeltaCAT `ADD` mode — compute a new column from existing data, write as a new Delta (MOR). |
| 04 | `04_benchmark.py` | Local FS benchmark: thin-client vs single writer, eager/lazy reads, column projection, distributed reads. |
| 05 | `05_benchmark_s3.py` | Same patterns but reading from S3 (SwiftStack). |
| 06 | `06_compaction_and_scaling.py` | Compact 344 small Lance files → 1 merged file on S3, then compare read perf. Also tests worker scaling on the compacted dataset. |
| 07 | `07_shard_scaling.py` | **Shard × worker matrix** on the small dataset (7.15 GB): split into [1, 2, 5, 10, 20, 50] shards, read with [1, 2, 5, 10, 20, 50] workers. Validates that single-shard reads can't scale across workers. |
| 08 | `08_lancedb_vs_deltacat_write.py` | LanceDB-style (all writers append to one Lance dataset, sharing manifest) vs DeltaCAT-style (each writer writes an independent dataset). Quantifies manifest lock contention. |
| 09 | `09_large_dataset_shard_scaling.py` | Convert `feb_08_500hr_lerobot_updated_fixed` parquet → Lance (54M rows, 35 GB), create local shards. (Upload to S3 done separately via parallel `s5cmd`.) |
| 10 | `10_read_scaling_s3.py` | Shard × worker matrix on the 35 GB dataset. Confirms `~5 shards + 5-50 workers` is the sweet spot. |
| 11 | `11_mar6_convert_and_shard.py` | Same conversion + sharding for `mar_6_umi_lerobot` (54M rows, 18.5 GB tabular, smaller per-row size). |
| 12 | `12_mar6_read_scaling.py` | Shard × worker matrix on the mar6 dataset. Reveals the row-size effect on S3 throughput ceiling (3.9 GB/s vs 2.3 GB/s vs 0.5 GB/s). |

## Workflow

The scripts are numbered to be run in order, but most can be run independently
once the merged Lance dataset exists. Typical flow:

```
01 (write local)  →  02 (read local)  →  03 (column append)  →  04 (benchmark local)
                                                                        ↓
                                            s5cmd cp output/* s3://...  ↓
                                                                        ↓
05 (benchmark S3)  ←  06 (compaction)  ←──────────────────────────────  ┘
        ↓
07 (shard scaling, small)
        ↓
09 → 10 (large dataset feb_08)
        ↓
11 → 12 (large dataset mar6)
```

`08` is independent — it focuses on write-side contention, not read scaling.

## S3 Upload

Inside Ray jobs, **don't call `mscsync`** — it submits its own Ray jobs and
nests poorly. Use `s5cmd` directly, or run `mscsync` from a shell outside the
Ray job:

```bash
# 6 parallel s5cmd uploads (one per shard config)
for n in 001 005 010 020 050 100; do
  s5cmd --credentials-file ~/.gear/aws_credentials \
    --endpoint-url https://pdx.s8k.io \
    cp --concurrency 50 "output/shards/n${n}/*" \
    "s3://bucket/path/n${n}/" &
done
wait
```

## Key gotchas

- **Lance forbids `.` in top-level column names** — rename
  `observation.state.xxx` → `observation__state__xxx` before writing.
- **`lance.write_dataset` with append mode** is required for incremental
  writes that don't fit in memory (e.g., 35 GB merged dataset).
- **Direct `lance.write_dataset` to S3** is ~25 MB/s (single-threaded). Always
  write locally first, then upload via parallel `s5cmd`.
- **Each S3 Lance dataset has ~3-8 s of fixed overhead** for connection
  setup + manifest read + fragment metadata parsing. Many small shards is
  pathological for low-parallelism reads.

See [`../experiment.md`](../experiment.md) for full benchmark numbers and the
finding that **optimal shard count matches reader parallelism, not data size**.
