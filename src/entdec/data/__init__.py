"""Data loading, cleaning, sampling and splitting (PROJECT_SPEC.md sections 3, 4)."""

from __future__ import annotations

from entdec.data.io import (
    ParquetBatchWriter,
    iter_csv_batches,
    iter_parquet_batches,
    read_parquet,
    write_parquet,
)
from entdec.data.sampling import (
    accumulate_counts,
    counts_to_series,
    select_items,
    select_users,
    user_stats_frame,
)
from entdec.data.splits import SPLIT_CATEGORIES, assign_splits, validate_no_leakage

__all__ = [
    "SPLIT_CATEGORIES",
    "ParquetBatchWriter",
    "accumulate_counts",
    "assign_splits",
    "counts_to_series",
    "iter_csv_batches",
    "iter_parquet_batches",
    "read_parquet",
    "select_items",
    "select_users",
    "user_stats_frame",
    "validate_no_leakage",
    "write_parquet",
]
