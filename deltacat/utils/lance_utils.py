"""
Lance format utilities for DeltaCAT.

Provides read, write, and manifest construction helpers for Lance-backed
DeltaCAT tables.  Lance datasets are directories (not single files); each
directory becomes exactly one ManifestEntry.

See deltacat/docs/lance/README.md for the integration design.
"""
from __future__ import annotations

import logging
import posixpath
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pyarrow as pa
import pyarrow.fs as pafs

import lance

from deltacat import logs
from deltacat.storage.model.manifest import (
    EntryType,
    ManifestEntry,
    ManifestEntryList,
    ManifestMeta,
)
from deltacat.types.media import ContentEncoding, ContentType
from deltacat.utils.filesystem import (
    FilesystemType,
    absolute_path_to_relative,
)

logger = logs.configure_deltacat_logger(logging.getLogger(__name__))

# Arrow schema metadata key for DeltaCAT schema ID embedding.
_DELTACAT_SCHEMA_ID_KEY = b"DELTACAT:schema_id"

# Default rows per Lance data file.  Lance's own default is 1M, and
# Parquet's compactor default is 4M.  For Lance multimodal data (PackDS),
# row sizes vary from 5 KB (state-only) to 500 KB (multi-camera video),
# so we use a lower default to keep files within compactor memory budgets.
#
# At 100K rows: ~500 MB for 5KB rows, ~50 GB for 500KB rows.
# Tables with very large rows should set RECORDS_PER_COMPACTED_FILE
# in table properties to a value appropriate for their row size.
DEFAULT_MAX_ROWS_PER_FILE = 100_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Mapping from PyArrow type to alignment requirement (bytes) used by
# arrow-rs ScalarBuffer<T> (std::mem::align_of::<T>()).  Types not listed
# here require only 1-byte alignment.
#
# Background:
#   - Root issue: https://github.com/apache/arrow-rs/issues/7136
#   - Original report: https://github.com/apache/arrow-rs/issues/6471
#   - Rejected upstream fix: https://github.com/apache/arrow-rs/pull/7137
#     (closed without merging — maintainer wanted opt-in, author abandoned)
#   - Lance 3.0.0 uses arrow-rs v57.x which does NOT auto-align FFI buffers.
_TYPE_ALIGNMENT: Dict[int, int] = {
    pa.int16().id: 2,
    pa.uint16().id: 2,
    pa.float16().id: 2,
    pa.int32().id: 4,
    pa.uint32().id: 4,
    pa.float32().id: 4,
    pa.date32().id: 4,
    pa.time32("ms").id: 4,
    pa.int64().id: 8,
    pa.uint64().id: 8,
    pa.float64().id: 8,
    pa.date64().id: 8,
    pa.time64("us").id: 8,
    pa.duration("us").id: 8,
    # decimal128 → Rust i128 → 16-byte alignment
    pa.decimal128(1).id: 16,
    # decimal256 → Rust i256 → 16-byte alignment (conservative)
    pa.decimal256(1).id: 16,
}
# All timestamp types share the same id regardless of unit/tz.
_TYPE_ALIGNMENT[pa.timestamp("us").id] = 8


def _alignment_for_type(arrow_type: pa.DataType) -> int:
    """Return the buffer alignment (bytes) that arrow-rs requires for *arrow_type*."""
    return _TYPE_ALIGNMENT.get(arrow_type.id, 1)


def _column_needs_realignment(chunked: pa.ChunkedArray) -> bool:
    """Return True if any buffer in *chunked* is not aligned for its type."""
    required = _alignment_for_type(chunked.type)
    if required <= 1:
        return False
    for chunk in chunked.chunks:
        for buf in chunk.buffers():
            if buf is not None and buf.address % required != 0:
                return True
        # Recurse into nested types (struct fields, list values)
        if pa.types.is_struct(chunked.type):
            for i in range(chunked.type.num_fields):
                child = pa.chunked_array([chunk.field(i) for chunk in chunked.chunks])
                if _column_needs_realignment(child):
                    return True
        elif pa.types.is_list(chunked.type) or pa.types.is_large_list(chunked.type):
            child = pa.chunked_array([chunk.values for chunk in chunked.chunks])
            if _column_needs_realignment(child):
                return True
    return False


def _realign_column(chunked: pa.ChunkedArray, column_name: str) -> pa.ChunkedArray:
    """Return a copy of *chunked* with all buffers freshly allocated.

    Uses ``pa.concat_arrays`` on each chunk which allocates new contiguous
    buffers via PyArrow's default allocator (jemalloc, 64-byte aligned).
    Falls back to ``to_pylist()`` round-trip for complex types where concat
    doesn't produce aligned buffers.
    """
    # For simple types, concat_arrays copies into a new contiguous buffer.
    try:
        result = pa.chunked_array(
            [pa.concat_arrays([chunk]) for chunk in chunked.chunks]
        )
        logger.info(
            "Realigned column %r (type=%s) via concat_arrays (fast path)",
            column_name,
            chunked.type,
        )
        return result
    except Exception:
        # Fallback: Python round-trip guarantees new allocations.
        values = chunked.to_pylist()
        logger.info(
            "Realigned column %r (type=%s) via to_pylist round-trip "
            "(slow path — concat_arrays insufficient for this type)",
            column_name,
            chunked.type,
        )
        return pa.chunked_array([pa.array(values, type=chunked.type)])


def _ensure_buffer_alignment(table: pa.Table) -> pa.Table:
    """Ensure all Arrow buffers meet the alignment required by Lance (arrow-rs).

    Lance's Rust FFI layer (via arrow-rs ``ScalarBuffer<T>``) panics when
    buffer pointers are not aligned to ``std::mem::align_of::<T>()``.
    Tables that have been through schema coercion, type casting, or cross-type
    conversions may have buffers with non-standard alignment.

    This function checks each column and only reallocates the ones that need
    it, avoiding the O(n) cost of ``from_pydict()`` for columns that are
    already aligned (the common case).

    Performance (10-col table with int/float/string/decimal/list types)::

        Rows  | from_pydict (old) | targeted (new) | speedup
        ------+-------------------+----------------+--------
          100 |           0.5 ms  |       0.07 ms  |     7x
         1 K  |           2.5 ms  |       0.05 ms  |    47x
        10 K  |            25 ms  |       0.05 ms  |   470x
       100 K  |           324 ms  |       0.05 ms  | 6,500x
         1 M  |         3,414 ms  |       0.05 ms  |65,000x

    When no columns need realignment, cost is ~50 µs (pointer checks only).

    See: https://github.com/apache/arrow-rs/issues/7136
    """
    misaligned = [
        name
        for name, col in zip(table.column_names, table.columns)
        if _column_needs_realignment(col)
    ]

    if not misaligned:
        logger.debug(
            "All %d columns are aligned — no realignment needed",
            table.num_columns,
        )
        return table

    logger.info(
        "Found %d/%d misaligned columns for Lance FFI: %s",
        len(misaligned),
        table.num_columns,
        misaligned,
    )
    new_columns = []
    for name, col in zip(table.column_names, table.columns):
        if name in misaligned:
            new_columns.append(_realign_column(col, name))
        else:
            new_columns.append(col)
    return pa.table(dict(zip(table.column_names, new_columns)), schema=table.schema)


def _storage_options_from_filesystem(
    filesystem: Optional[pafs.FileSystem],
    data_root=None,
) -> Optional[Dict[str, str]]:
    """Build Lance ``storage_options`` from a PyArrow FileSystem.

    Lance uses a plain dict for cloud credentials rather than a PyArrow
    FileSystem object.  For local filesystems, returns ``None``.

    For S3, resolves credentials via boto3's default credential chain
    (``~/.aws/config``, ``~/.aws/credentials``, environment variables,
    IAM roles, etc.).  This ensures Lance has the same credentials that
    DeltaCAT's PyArrow filesystem uses, regardless of how the user
    configured them.

    When ``data_root`` is provided (multi-root catalogs), the DataRoot's
    ``storage_config`` is used to resolve the endpoint URL and profile
    for non-default S3-compatible backends (e.g., SwiftStack).

    Lance's object_store crate key names:
    https://lancedb.github.io/lance/object_store.html
    """
    storage_config = (data_root.storage_config if data_root is not None else None) or {}

    if filesystem is None and not storage_config:
        return None

    fs_type = FilesystemType.from_filesystem(filesystem) if filesystem else None

    if fs_type == FilesystemType.S3 or storage_config:
        opts: Dict[str, str] = {}
        if filesystem is not None:
            region = getattr(filesystem, "region", None)
            if region:
                opts["region"] = region

        profile = storage_config.get("profile")
        endpoint_url = storage_config.get("endpoint_url")
        if endpoint_url:
            opts["aws_endpoint"] = endpoint_url
            opts["allow_http"] = "true" if "http://" in endpoint_url else "false"

        # Resolve credentials via boto3 using the profile from DataRoot
        # (or the default chain when no profile is configured).
        try:
            import boto3

            session = (
                boto3.Session(profile_name=profile) if profile else boto3.Session()
            )
            credentials = session.get_credentials()
            if credentials:
                resolved = credentials.get_frozen_credentials()
                if resolved.access_key:
                    opts["aws_access_key_id"] = resolved.access_key
                if resolved.secret_key:
                    opts["aws_secret_access_key"] = resolved.secret_key
                if resolved.token:
                    opts["aws_session_token"] = resolved.token
        except Exception:
            logger.debug(
                "Could not resolve AWS credentials via boto3; "
                "Lance will use its default credential chain.",
                exc_info=True,
            )

        return opts or None

    # GCS, local, and others: rely on environment credentials.
    return None


def _resolve_path(
    path: str,
    catalog_root: Optional[str],
) -> str:
    """Resolve a catalog-root-relative path to an absolute path."""
    if catalog_root and not (path.startswith("/") or "://" in path):
        return posixpath.join(catalog_root, path)
    return path


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------


def get_lance_dataset_stats(
    lance_path: str,
    filesystem: Optional[pafs.FileSystem] = None,
    data_root=None,
) -> Dict[str, Any]:
    """Read metadata from a Lance dataset for manifest entry construction.

    Uses ``LanceDataset.stats()`` — which reads from the Lance protobuf
    manifest (atomically committed at write time) — rather than directory
    listing, avoiding S3 eventual-consistency issues.

    Returns dict with keys: ``record_count``, ``content_length``,
    ``schema_id`` (int or None), ``schema`` (pa.Schema).
    """
    storage_options = _storage_options_from_filesystem(filesystem, data_root=data_root)
    ds = lance.dataset(lance_path, storage_options=storage_options)

    record_count = ds.count_rows()

    # Compute total data size from fragment metadata.  This reads from the
    # Lance protobuf manifest (atomically committed) rather than directory
    # listing, so it is consistent even on S3 immediately after writing.
    content_length = sum(
        df.file_size_bytes for frag in ds.get_fragments() for df in frag.metadata.files
    )

    schema = ds.schema
    schema_id_bytes = (schema.metadata or {}).get(_DELTACAT_SCHEMA_ID_KEY)
    schema_id = int(schema_id_bytes) if schema_id_bytes is not None else None

    return {
        "record_count": record_count,
        "content_length": content_length or 0,
        "schema_id": schema_id,
        "schema": schema,
    }


# ---------------------------------------------------------------------------
# Sort-column statistics (for compactor copy-by-reference)
# ---------------------------------------------------------------------------


def compute_sort_column_stats(
    lance_path: str,
    sort_column: str,
    filesystem: Optional[pafs.FileSystem] = None,
    data_root=None,
) -> Dict[str, Dict[str, Any]]:
    """Compute min/max for *sort_column* from a Lance dataset.

    For Parquet, these stats come from file metadata (zero data read).
    For Lance, we scan only the sort column — Lance reads just the
    relevant column chunk files, so this is lightweight relative to
    a full table scan.
    """
    storage_options = _storage_options_from_filesystem(filesystem, data_root=data_root)
    ds = lance.dataset(lance_path, storage_options=storage_options)
    col = ds.to_table(columns=[sort_column])[sort_column].combine_chunks()
    if len(col) == 0:
        logger.debug("Empty dataset at %s; skipping sort column stats.", lance_path)
        return {}
    # Use .as_py() then convert to msgpack-serializable types.
    # datetime.date, datetime.datetime, and Decimal are not serializable
    # by msgpack, so we convert to ISO strings.
    min_val = col[0].as_py()
    max_val = col[-1].as_py()

    def _to_serializable(v):
        if hasattr(v, "isoformat"):
            return v.isoformat()
        if isinstance(v, Decimal):
            return str(v)
        return v

    return {
        sort_column: {
            "min": _to_serializable(min_val),
            "max": _to_serializable(max_val),
        }
    }


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def build_manifest_entries(
    lance_path: str,
    filesystem: Optional[pafs.FileSystem] = None,
    schema_id: Optional[int] = None,
    sort_scheme_id: Optional[str] = None,
    column_stats: Optional[Dict[str, Any]] = None,
    catalog_root: Optional[str] = None,
    entry_type: Optional[EntryType] = EntryType.DATA,
    entry_params: Optional[Any] = None,
    precomputed_stats: Optional[Dict[str, Any]] = None,
    source_content_length: Optional[int] = None,
) -> ManifestEntryList:
    """Build a ManifestEntryList for an existing Lance dataset directory.

    Used in the thin-client write pattern: the caller writes a Lance
    dataset and passes the resulting entries to
    ``dc.write_to_table(manifest=entries, content_type=ContentType.LANCE)``.

    Unlike ``ManifestEntry.from_path()``, this accepts a directory path
    and derives ``content_length`` from fragment metadata (S3-consistent).

    When called from ``write_lance_table``, pass ``precomputed_stats`` to
    avoid re-opening the dataset a second time.
    """
    stats = precomputed_stats or get_lance_dataset_stats(lance_path, filesystem)
    resolved_schema_id = schema_id if schema_id is not None else stats["schema_id"]

    url = lance_path
    if catalog_root:
        url = absolute_path_to_relative(catalog_root, lance_path)

    meta = ManifestMeta.of(
        record_count=stats["record_count"],
        content_length=stats["content_length"],
        content_type=ContentType.LANCE.value,
        content_encoding=ContentEncoding.IDENTITY.value,
        source_content_length=source_content_length,
        schema_id=resolved_schema_id,
        sort_scheme_id=sort_scheme_id,
        entry_type=entry_type,
        entry_params=entry_params,
    )
    if column_stats:
        meta["column_stats"] = column_stats

    entry = ManifestEntry.of(url=url, meta=meta, mandatory=True)
    return ManifestEntryList.of([entry])


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def read_lance_table(
    path: str,
    filesystem: Optional[pafs.FileSystem] = None,
    catalog_root: Optional[str] = None,
    include_columns: Optional[List[str]] = None,
    row_filter: Optional[Any] = None,
    data_root=None,
) -> pa.Table:
    """Read a Lance dataset directory into a PyArrow Table.

    Called from the DeltaCAT storage/read path when a manifest entry has
    ``content_type == ContentType.LANCE``.  Column projection and predicate
    pushdown are forwarded to Lance's scanner.
    """
    path = _resolve_path(path, catalog_root)
    storage_options = _storage_options_from_filesystem(filesystem, data_root=data_root)
    ds = lance.dataset(path, storage_options=storage_options)

    kwargs: Dict[str, Any] = {}
    if include_columns:
        kwargs["columns"] = include_columns
    if row_filter is not None:
        kwargs["filter"] = row_filter

    logger.debug(
        "Reading Lance dataset: path=%s columns=%s filter=%s",
        path,
        include_columns,
        row_filter,
    )
    return ds.to_table(**kwargs)


def open_lance_dataset(
    path: str,
    filesystem: Optional[pafs.FileSystem] = None,
    catalog_root: Optional[str] = None,
    data_root=None,
) -> "lance.LanceDataset":
    """Open a Lance dataset without loading data (lazy).

    Returns a ``lance.LanceDataset`` supporting:
    - ``ds.take(indices)`` — O(1) random access for training
    - ``ds.count_rows()`` — row count from manifest (no data read)
    - ``ds.to_table(columns=...)`` — selective materialization
    """
    path = _resolve_path(path, catalog_root)
    storage_options = _storage_options_from_filesystem(filesystem, data_root=data_root)
    return lance.dataset(path, storage_options=storage_options)


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def write_lance_table(
    table: pa.Table,
    base_path: str,
    filesystem: Optional[pafs.FileSystem] = None,
    catalog_root: Optional[str] = None,
    schema_id: Optional[int] = None,
    sort_scheme_id: Optional[str] = None,
    sort_column: Optional[str] = None,
    max_rows_per_fragment: int = DEFAULT_MAX_ROWS_PER_FILE,
    entry_type: Optional[EntryType] = EntryType.DATA,
    entry_params: Optional[Any] = None,
    data_root=None,
) -> ManifestEntryList:
    """Write a PyArrow Table as a new Lance dataset directory.

    Produces exactly **one** Lance dataset directory = **one** manifest entry,
    regardless of how many internal fragments are created.  This keeps
    ``RoundCompletionInfo.hb_index_to_entry_range`` stable across compaction
    rounds.

    ``max_rows_per_fragment`` controls the maximum rows per data file within
    the Lance directory.  Default is 100K — lower than Lance's native 1M or
    Parquet's 4M because multimodal data (PackDS) has much larger rows
    (5 KB to 500 KB each).  Tables with very large or very small rows should
    set ``RECORDS_PER_COMPACTED_FILE`` in table properties.

    The DeltaCAT schema ID is embedded into the Lance Arrow schema metadata
    so it survives round-trips through the Lance manifest.
    """
    dataset_name = f"{uuid4()}.lance"
    lance_path = posixpath.join(base_path, dataset_name)
    storage_options = _storage_options_from_filesystem(filesystem, data_root=data_root)

    # Embed DeltaCAT schema ID in Arrow schema metadata.
    if schema_id is not None:
        existing_meta = table.schema.metadata or {}
        new_meta = {
            **existing_meta,
            _DELTACAT_SCHEMA_ID_KEY: str(schema_id).encode(),
        }
        table = table.replace_schema_metadata(new_meta)

    logger.debug(
        "Writing Lance dataset: path=%s rows=%d max_rows_per_file=%d",
        lance_path,
        len(table),
        max_rows_per_fragment,
    )

    table = _ensure_buffer_alignment(table)

    # Capture in-memory size before writing.
    from deltacat.types.tables import get_table_size

    source_content_length = get_table_size(table)

    try:
        lance.write_dataset(
            table,
            lance_path,
            mode="create",
            max_rows_per_file=max_rows_per_fragment,
            storage_options=storage_options,
        )
    except Exception:
        # Clean up partial Lance directory before re-raising.  A partial
        # .lance directory left on disk could appear valid to subsequent
        # reads, unlike a truncated Parquet file which is obviously invalid.
        logger.warning(
            "Lance write failed; cleaning up partial directory: %s",
            lance_path,
            exc_info=True,
        )
        try:
            cleanup_fs = filesystem or pafs.LocalFileSystem()
            cleanup_fs.delete_dir(lance_path)
        except Exception:
            logger.debug(
                "Failed to clean up partial Lance directory: %s",
                lance_path,
                exc_info=True,
            )
        raise

    # Read dataset stats once — avoids re-opening the dataset in
    # build_manifest_entries.
    stats = get_lance_dataset_stats(lance_path, filesystem, data_root=data_root)

    # Compute sort-column statistics if a sort scheme is active.
    column_stats: Optional[Dict[str, Any]] = None
    if sort_column:
        try:
            column_stats = compute_sort_column_stats(
                lance_path,
                sort_column,
                filesystem,
                data_root=data_root,
            )
        except Exception:
            logger.warning(
                "Failed to compute sort-column stats for %s; "
                "copy-by-reference will be unavailable for this entry.",
                lance_path,
                exc_info=True,
            )

    return build_manifest_entries(
        lance_path=lance_path,
        filesystem=filesystem,
        schema_id=schema_id,
        sort_scheme_id=sort_scheme_id,
        column_stats=column_stats,
        catalog_root=catalog_root,
        precomputed_stats=stats,
        entry_type=entry_type,
        entry_params=entry_params,
        source_content_length=source_content_length,
    )


def write_lance_entries(
    table,
    data_dir_path: str,
    catalog_root: str,
    filesystem: Optional[pafs.FileSystem],
    max_records_per_entry: Optional[int],
    entry_params=None,
    entry_type=None,
    table_writer_kwargs: Optional[Dict[str, Any]] = None,
) -> "ManifestEntryList":
    """Write strategy for Lance content type.

    Converts the input to PyArrow (if needed), then delegates to
    :func:`write_lance_table`.  Called from the storage layer's
    ``_CONTENT_TYPE_WRITE_STRATEGY`` dispatch table.

    All Lance-specific logic (PyArrow conversion, kwarg mapping) lives
    here so that ``storage/main/impl.py`` stays format-agnostic.
    """
    from deltacat.types.tables import to_pyarrow

    table_writer_kwargs = table_writer_kwargs or {}
    schema_id = table_writer_kwargs.get("schema_id")
    sort_scheme_id = table_writer_kwargs.get("sort_scheme_id")
    sort_column = table_writer_kwargs.get("sort_column")
    data_root = table_writer_kwargs.get("data_root")

    # Convert all dataset types to PyArrow before writing.  Native Daft/Ray
    # write_lance can hang on evolved schemas due to a pylance 3.0 C data
    # interface bug in Ray worker tasks.
    pa_table = table if isinstance(table, pa.Table) else to_pyarrow(table)
    logger.info(
        "Writing Lance dataset: rows=%d path=%s",
        len(pa_table),
        data_dir_path,
    )
    lance_kwargs: Dict[str, Any] = {
        "base_path": data_dir_path,
        "filesystem": filesystem,
        "catalog_root": catalog_root,
        "schema_id": schema_id,
        "sort_scheme_id": sort_scheme_id,
        "sort_column": sort_column,
        "entry_type": entry_type,
        "entry_params": entry_params,
        "data_root": data_root,
    }
    if max_records_per_entry is not None:
        lance_kwargs["max_rows_per_fragment"] = max_records_per_entry
    return write_lance_table(pa_table, **lance_kwargs)


# ---------------------------------------------------------------------------
# Zero-copy registration of existing Lance datasets
# ---------------------------------------------------------------------------


def register_lance_dataset(
    lance_path: str,
    table: str,
    namespace: str,
    catalog_name: Optional[str] = None,
    partition_values: Optional[Dict[str, Any]] = None,
    filesystem: Optional[pafs.FileSystem] = None,
    schema_id: Optional[int] = None,
    **write_kwargs: Any,
) -> None:
    """Register an existing Lance dataset as a DeltaCAT-managed table.

    No data is moved or copied.  DeltaCAT wraps the existing directory
    by creating a catalog entry whose manifest points to the supplied
    path.  The Arrow schema (including any ``packds:`` metadata keys)
    is preserved verbatim.

    Recommended first step when migrating PackDS ``.packds/steps.lance``
    directories into the DeltaCAT catalog.
    """
    import deltacat as dc
    from deltacat.storage.model.manifest import Manifest
    from deltacat.types.tables import TableWriteMode

    stats = get_lance_dataset_stats(lance_path, filesystem)
    resolved_schema_id = (
        schema_id if schema_id is not None else (stats["schema_id"] or 0)
    )

    entries = build_manifest_entries(
        lance_path=lance_path,
        filesystem=filesystem,
        schema_id=resolved_schema_id,
    )

    logger.info(
        "Registering Lance dataset: path=%s table=%s/%s rows=%d",
        lance_path,
        namespace,
        table,
        stats["record_count"],
    )

    kwargs: Dict[str, Any] = {}
    if catalog_name:
        kwargs["catalog"] = catalog_name
    if partition_values:
        kwargs["partition_values"] = partition_values

    dc.write_to_table(
        data=None,
        table=table,
        namespace=namespace,
        manifest=Manifest.of(entries=entries),
        content_type=ContentType.LANCE,
        mode=TableWriteMode.AUTO,
        **kwargs,
        **write_kwargs,
    )
