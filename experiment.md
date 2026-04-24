# DeltaCAT + Lance 实验记录

**日期**: 2026-03-29
**集群**: 33-node Ray cluster (2800 CPUs, 280 GPUs), Python 3.12, pyarrow 23, lance 3.0.1, ray 2.54
**S3**: SwiftStack pdx.s8k.io (bucket: GearHome)
**代码**: `src/01_write_local.py` ~ `src/10_read_scaling_s3.py`

---

## 1. 环境搭建

- 基于 open-source `deltacat` 2.0.0.post9，从 `deltacat-dev` (branch 2.0) 移植了 Lance 集成层：
  - `ContentType.LANCE`, `DatasetType.LANCE` → `types/media.py`
  - `lance_utils.py` → `utils/lance_utils.py` (build_manifest_entries, write/read/open lance helpers)
  - `stage_delta_from_manifest()` → `storage/main/impl.py`
  - manifest-only write path → `catalog/main/impl.py`
  - Ray 2.54 兼容性修复 → `io/datasource/deltacat_datasource.py`
- Fork: `deltacat/` (editable install, `uv pip install --no-deps -e ./deltacat`)

---

## 2. 小数据集实验 (feb2_small_batch)

**数据**: 344 episodes, 7.15 GB (mp4 + json metadata)

### 2.1 写入 Benchmark (本地文件系统)

| 模式 | 时间 | 吞吐 | Speedup |
|---|---|---|---|
| **Thin-Client (344 Ray tasks, 独立 Lance 文件)** | **12.5s** | **572 MB/s** | **3.1x** |
| Direct lance.write_dataset (单进程) | 38.8s | 184 MB/s | 1.0x |

**结论**: Thin-client 模式（每个 writer 写独立 Lance 目录）完美并行，验证了 DeltaCAT 笔记的核心设计。

### 2.2 写入 Worker Scaling (本地文件系统)

| Workers | Time | MB/s | Speedup |
|---|---|---|---|
| 1 | 39.5s | 181 | 1.00x |
| 2 | 18.1s | 395 | 2.18x |
| 5 | 7.4s | 963 | 5.32x |
| 10 | 3.8s | 1,879 | 10.38x |
| 20 | 2.1s | 3,423 | 18.91x |
| 50 | 1.8s | 4,054 | 22.40x |

**结论**: 写入 scaling 近乎线性至 20 workers (18.9x)，之后因 344 episodes 不够分而收敛。

### 2.3 读取 Benchmark (本地 vs S3)

| 模式 | 本地 | S3 |
|---|---|---|
| Eager full (7.15 GB, all cols) | 24.1s | 33.3s |
| Eager metadata (3 cols, 跳过 video) | 3.6s | 1.5s* |
| Lazy take(10 random) | 0.34s | 7.9s |
| Lazy column projection (2 cols) | 0.007s | 1.6s |

*S3 metadata 看似更快是因为读的是 merged 单文件，本地读的是 344 个小文件。

### 2.4 Compaction 效果 (S3)

| 读取模式 | Compacted (1 文件) | Uncompacted (344 文件) | 差距 |
|---|---|---|---|
| Eager metadata (3 cols) | **1.40s** | **473.09s** | **338x** |
| Lazy take(10) | **10.54s** | **45.36s** | **4.3x** |

**结论**: Compaction 是 S3 读性能的决定性因素。不做 compaction，344 个小文件在 S3 上几乎不可用。

### 2.5 Shard x Worker 读取 Scaling (S3, 7.15 GB)

**Full read (all columns + video)**:

| Shards \ Workers | 1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| **1** (7.15 GB) | 34.5s (1.0x) | 36.0s | 26.6s (1.3x) | — | — | — |
| **2** (3.6 GB) | 38.1s | 20.0s (1.7x) | 22.2s | 18.8s | — | — |
| **5** (1.4 GB) | 58.7s | 38.8s | 18.9s (1.8x) | 20.1s | 18.6s | — |
| **10** (715 MB) | 122s | 55.1s | 28.4s | 20.0s (1.7x) | **17.0s (2.0x)** | 18.9s |
| **20** (358 MB) | 182s | 87.7s | 48.9s | 27.2s | 17.6s | **14.6s (2.4x)** |
| **50** (143 MB) | 415s | 182s | 89.8s | 48.9s | 25.4s | 22.6s (1.5x) |

**观察**: 数据太小 (7.15 GB)，S3 连接开销主导。Scaling 效率低（最高 2.4x）。

---

## 3. 大数据集实验 (feb_08_500hr_lerobot_updated_fixed)

**数据**: 54M rows, 35 GB (LeRobot format, 129 parquets → Lance)
- 转换: parquet → merged Lance 本地, 635s (55 MB/s)
- 列重命名: `observation.state.xxx` → `observation__state__xxx` (Lance 不允许 `.` 在顶层列名)

### 3.1 Full Read Scaling (S3, 35 GB, 54M rows)

| Shards \ Workers | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| **1** (35 GB) | 194s (1.0x) | 163s (1.2x) | 173s (1.1x) | — | — | — | — |
| **5** (7 GB) | 108s (1.8x) | 54s (3.6x) | **19s (10.2x)** | 23s | 30s | — | — |
| **10** (3.5 GB) | 135s | 56s | 24s | 21s (9.2x) | 22s | **15s (12.8x)** | — |
| **20** (1.75 GB) | 215s | 108s | 53s | 26s | 21s (9.4x) | 20s | 23s |
| **50** (700 MB) | 534s | 254s | 109s | 53s | 33s | 23s | **18s (10.6x)** |
| **100** (350 MB) | 797s | 412s | 182s | 98s | 53s | 26s | **18s (10.6x)** |

### 3.2 Metadata-only Read Scaling (S3, 6 columns)

| Shards \ Workers | 1 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|
| **1** | 13.5s (1.0x) | 12.3s (1.1x) | — | — | — | — |
| **5** | 27.0s | **6.0s (2.3x)** | 6.4s | 5.7s | — | — |
| **10** | 61.7s | 14.3s | 10.0s (1.4x) | 11.3s | 9.4s | — |
| **50** | 234s | 54.4s | 28.4s | 23.1s | **8.4s (1.6x)** | 9.0s |
| **100** | 391s | 77.5s | 46.2s | 22.6s | 11.1s | **6.7s (2.0x)** |

### 3.3 分析

**最优配置**: 5 shards + 5 workers = 19.1s (10.2x speedup) — 最佳性价比。

**绝对最快**: 10 shards + 50 workers = 15.2s (12.8x) — 仅比 5+5 快 20%，但用了 10x 资源。

**S3 带宽天花板**: ~2.3 GB/s 聚合吞吐 (≈15-18s 读完 35GB)。超过此上限加再多 worker 也无效。

**Shard 过多的惩罚**: 100 shards + 1 worker = 797s，比 1 shard + 1 worker (194s) 慢 4x。每个 shard 需要独立的 S3 连接、manifest 读取、fragment 元数据解析 — 每 shard 固定开销 ~3-8s。

---

## 4. 第三数据集验证 (mar_6_umi_lerobot)

**数据**: 54M rows, 18.5 GB tabular (12 cols: state vectors), 360 parquets, 1440 MP4 videos
**转换**: parquet → Lance 1993s (9 MB/s, append mode 逐文件写入)

### 4.1 Full Read Scaling (S3, 18.5 GB, 54M rows)

| Shards \ Workers | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| **1** (18.5 GB) | 74s (1.0x) | 48s (1.5x) | 50s (1.5x) | — | — | — | — |
| **5** (3.7 GB) | 38s (2.0x) | 15s (5.0x) | 7.5s (9.9x) | **4.8s (15.6x)** | 4.6s (16.1x) | — | — |
| **10** (1.85 GB) | 47s | 16s | 7.4s (10.1x) | 6.9s (10.8x) | 6.1s (12.2x) | **5.3s (14.0x)** | — |
| **20** (925 MB) | 81s | 37s | 16s | 10s (7.8x) | 14s | 7.3s (10.2x) | 8.9s |
| **50** (370 MB) | 172s | 83s | 34s | 14s | 14s | 14s | **5.4s (13.8x)** |
| **100** (185 MB) | 284s | 89s | 36s | 18s | 10s | 7.8s | **6.0s (12.4x)** |

**最优**: 5 shards + 10-20 workers = **4.6-4.8s** (15-16x speedup, ~3.9 GB/s throughput)

### 5.2 Metadata-only Read (S3, 4 columns)

| Shards \ Workers | 1 | 5 | 10 | 50 | 100 |
|---|---|---|---|---|---|
| **1** | 4.4s (1.0x) | 3.5s (1.3x) | — | — | — |
| **5** | 7.0s | **1.5s (2.9x)** | 1.9s | — | — |
| **10** | 14.6s | 2.8s | 1.9s (2.4x) | 2.1s | — |
| **100** | 99s | 21.9s | 12.9s | 5.4s | **3.0s (1.5x)** |

### 4.3 跨数据集对比

| 指标 | feb2_small | feb_08_500hr | mar_6_umi |
|---|---|---|---|
| 数据量 | 7.15 GB | 35 GB | 18.5 GB |
| 行数 | 344 | 54M | 54M |
| 行大小 | ~20 MB (含 video) | ~650 B | ~340 B |
| 最优 shards x workers | 20+50 (2.4x) | 5+5 (10.2x) | 5+10 (15.6x) |
| 最快时间 | 14.6s | 15.2s | 4.6s |
| 峰值吞吐 | 488 MB/s | 2.3 GB/s | **3.9 GB/s** |
| S3 带宽天花板 | ~500 MB/s | ~2.3 GB/s | ~3.9 GB/s |

**关键发现**:
- **行越小，S3 吞吐越高** — mar6 (340B/行) 达 3.9 GB/s, feb_08 (650B/行) 达 2.3 GB/s, feb2 (20MB/行, 含 video blob) 仅 488 MB/s
- **最优点一致: 5 shards + 5-20 workers** — 三个数据集都在这个范围达到最优
- **100 shards 始终比 5 shards 慢** — 除非 worker 数也到 100

---

## 5. 对 DeltaCAT 笔记观点的验证

### 5.1 "Thin-Client 写入最快" ✅ 完全验证

每个 writer 写独立 Lance 目录，零锁竞争。写入 scaling 近乎线性 (22.4x with 50 workers)。

### 5.2 "MOR 写快读慢，Compaction 后读快" ✅ 完全验证

Compacted 单文件 vs 344 个小文件：S3 metadata 读差 338x。

### 5.3 "Hash Buckets = ceil(bytes / 512 MB)" ⚠️ 需修正

笔记的公式:
```
Hash Buckets = ceil(on_disk_bytes / 512 MiB)
```
对 35GB 数据 → 70 shards，对 20TB 数据 → 40,000 shards。

**实验结论**: 此公式产生的 shard 数**过多**。最优 shard 数应匹配 **reader 并行度**，而非数据大小。

| 策略 | 35GB 实测 | 推论 (20TB) |
|---|---|---|
| 笔记公式: ceil(bytes/512MB) | 70 shards → 单 worker 534s，50 workers 23s | 40K shards → 需 40K workers |
| **最优: shards ≈ workers** | **5 shards + 5 workers = 19s** | **100 shards + 100 workers** |
| shard >> workers | 100 shards + 5 workers = 182s (慢 9.6x) | 严重浪费 |

**修正建议**:

```
Compacted_files = max(min_shards, num_training_workers)
Shard_size = total_bytes / Compacted_files
```

而不是:
```
Compacted_files = ceil(total_bytes / 512 MiB)   ← 笔记原公式
```

### 5.4 为什么笔记的公式可能在其原始场景成立

笔记的公式可能针对的是 **compaction 内部的 hash bucket 数**，而非最终输出文件数。Compaction V2 的 BSP pipeline 在 Phase 1 按 hash(PK) mod N 分桶，Phase 2 每桶独立 merge。N 需要足够大以避免单个 bucket OOM（`MIN_DELTA_BYTES_IN_BATCH = 5 GB`）。这是一个**计算资源约束**（内存），而非**读取性能最优点**。

最终输出文件数可以在 Phase 2 merge 后进一步合并（`MAX_RECORDS_PER_COMPACTED_FILE = 4M`），使 shard 数匹配训练并行度而非 hash bucket 数。

---

## 6. 工具链笔记

| 工具 | 用途 | 注意事项 |
|---|---|---|
| `uv venv --system-site-packages` | 创建 .venv 继承系统 ray/lance/pyarrow | 所有安装用 `uv pip` |
| `ray job submit --runtime-env` | 提交 Ray job | runtime_env.yaml 指定 py_executable 到 .venv |
| `s5cmd --credentials-file ... cp` | S3 上传/下载 | `--concurrency 50` 并行上传 |
| `mscsync.sh` | MSC + Ray 并行上传 | 不能从 Ray job 内调用（嵌套冲突）；当前 MSC 版本不支持 `force_overwrite` |
| `lance.write_dataset(mode="append")` | 增量写入 Lance | 避免 OOM：逐文件 append 而非全量 concat |
| Lance 列名 | 不允许 `.` 在顶层列名 | `observation.state.xxx` → `observation__state__xxx` |

---

## 7. 文件清单

```
PlayRayDeltaCAT/
  .venv/                          # uv venv --system-site-packages
  runtime_env.yaml                # Ray job submit config
  deltacat/                       # Fork with Lance integration
  experiment.md                   # 本文件
  src/
    01_write_local.py             # Thin-client write (344 parallel Ray tasks)
    02_read_local.py              # 3 read patterns (eager/lazy/distributed)
    03_column_append.py           # ADD mode column append
    04_benchmark.py               # Local read/write benchmark
    05_benchmark_s3.py            # S3 read benchmark
    06_compaction_and_scaling.py  # Compaction + worker scaling
    07_shard_scaling.py           # Shard x worker matrix (小数据集)
    08_lancedb_vs_deltacat_write.py # Write contention test (未完成)
    09_large_dataset_shard_scaling.py # 大数据集 shard creation + upload
    10_read_scaling_s3.py         # feb_08 S3 read scaling
    11_mar6_convert_and_shard.py  # mar6 parquet→Lance + local shards
    12_mar6_read_scaling.py       # mar6 S3 read scaling
  output/
    data/                         # 小数据集 Lance files
    catalog/                      # DeltaCAT catalog
    large/                        # 大数据集 merged + shards
```
