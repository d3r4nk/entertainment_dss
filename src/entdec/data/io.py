"""Low level IO helpers.

The raw files are large (``ratings.csv`` 647 MB, ``recommendations.csv`` 1.9 GB),
so everything is streamed in batches with only the needed columns selected. All
intermediates are stored as Parquet.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from entdec.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_BLOCK_SIZE = 1 << 24  # 16 MB read blocks
DEFAULT_PARQUET_BATCH = 250_000


def iter_csv_batches(
    path: str | Path,
    columns: Sequence[str] | None = None,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    column_types: dict[str, pa.DataType] | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream a CSV file as pandas chunks.

    Args:
        path: CSV file to read.
        columns: Columns to select (``None`` keeps everything).
        block_size: Bytes read per block; bigger blocks mean fewer, larger batches.
        column_types: Explicit pyarrow types, useful to avoid inference surprises.

    Yields:
        One DataFrame per CSV block, with only ``columns`` present.
    """
    path = Path(path)
    convert = pcsv.ConvertOptions(
        include_columns=list(columns) if columns else None,
        column_types=column_types or {},
        strings_can_be_null=True,
    )
    read_options = pcsv.ReadOptions(block_size=block_size, use_threads=True)
    with pcsv.open_csv(path, read_options=read_options, convert_options=convert) as reader:
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            if batch.num_rows:
                yield batch.to_pandas()


class ParquetBatchWriter:
    """Incrementally write pandas chunks to a single Parquet file."""

    def __init__(self, path: str | Path, compression: str = "zstd") -> None:
        self.path = Path(path)
        self.compression = compression
        self._writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        """Append a chunk, inferring the schema from the first one."""
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self._schema = table.schema
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, self._schema, compression=self.compression)
        else:
            assert self._schema is not None
            table = table.select(self._schema.names).cast(self._schema, safe=False)
        self._writer.write_table(table)
        self.rows += table.num_rows

    def close(self) -> None:
        """Flush and close the underlying writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> ParquetBatchWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def iter_parquet_batches(
    path: str | Path, columns: Sequence[str] | None = None, batch_size: int = DEFAULT_PARQUET_BATCH
) -> Iterator[pd.DataFrame]:
    """Stream a Parquet file in batches."""
    parquet_file = pq.ParquetFile(Path(path))
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=list(columns) if columns else None):
        yield batch.to_pandas()


def read_parquet(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read a Parquet file into memory."""
    return pq.read_table(Path(path), columns=list(columns) if columns else None).to_pandas()


def write_parquet(path: str | Path, frame: pd.DataFrame, compression: str = "zstd") -> Path:
    """Write a DataFrame to Parquet, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression=compression)
    logger.debug("wrote %s (%d rows)", path.name, len(frame))
    return path


def row_count(path: str | Path) -> int:
    """Number of rows in a Parquet file, without loading it."""
    return pq.ParquetFile(Path(path)).metadata.num_rows


__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_PARQUET_BATCH",
    "ParquetBatchWriter",
    "iter_csv_batches",
    "iter_parquet_batches",
    "read_parquet",
    "row_count",
    "write_parquet",
]
