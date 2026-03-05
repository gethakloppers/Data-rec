"""Multi-format file reader for recommender system datasets.

Reads raw dataset files into pandas DataFrames. Supports CSV, TSV, JSON,
JSONL (including Python-repr dicts), Parquet, GZip, ZIP, and TAR formats.

This module is purely I/O — it reads files and returns DataFrames without
any data modification or standardisation.
"""

from __future__ import annotations

import ast
import gzip
import json
import logging
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from recdata.exceptions import DatasetLoadError

logger = logging.getLogger(__name__)


def read_file(
    filepath: str | Path,
    format: str,
    encoding: str = "utf-8",
    separator: str = ",",
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a raw dataset file into a pandas DataFrame.

    Dispatches to the appropriate reader based on the format string.
    Does not modify or standardise the data in any way.

    Args:
        filepath: Path to the file to read.
        format: File format. One of: csv, tsv, json, jsonl, parquet, gz, zip, tar.
        encoding: Character encoding for text files. Defaults to 'utf-8'.
        separator: Column separator for CSV/TSV files. Defaults to ','.
        **kwargs: Additional keyword arguments passed to the underlying reader.
            For zip/tar: 'inner_filename' specifies which file to extract.

    Returns:
        A pandas DataFrame containing the raw data.

    Raises:
        DatasetLoadError: If the file cannot be read or the format is unsupported.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise DatasetLoadError(f"File not found: {filepath}")

    format_lower = format.lower().strip()

    readers = {
        "csv": _read_csv,
        "tsv": _read_tsv,
        "json": _read_json,
        "jsonl": _read_jsonl,
        "parquet": _read_parquet,
        "gz": _read_gzip,
        "zip": _read_zip,
        "tar": _read_tar,
    }

    if format_lower not in readers:
        raise DatasetLoadError(
            f"Unsupported format: '{format}'. "
            f"Supported formats: {sorted(readers.keys())}"
        )

    try:
        df = readers[format_lower](filepath, encoding=encoding, separator=separator, **kwargs)
    except DatasetLoadError:
        raise
    except Exception as e:
        raise DatasetLoadError(f"Failed to read '{filepath}' as {format}: {e}") from e

    # Log memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1e6
    logger.info(
        "Loaded %s: %d rows x %d cols (%.1f MB in memory)",
        filepath.name,
        len(df),
        len(df.columns),
        memory_mb,
    )

    return df


def _read_csv(
    filepath: Path,
    encoding: str = "utf-8",
    separator: str = ",",
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV file."""
    return pd.read_csv(
        filepath,
        encoding=encoding,
        sep=separator,
        low_memory=False,
        **kwargs,
    )


def _read_tsv(
    filepath: Path,
    encoding: str = "utf-8",
    separator: str = "\t",
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a TSV file (tab-separated)."""
    return pd.read_csv(
        filepath,
        encoding=encoding,
        sep="\t",
        low_memory=False,
        **kwargs,
    )


def _read_json(
    filepath: Path,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a JSON array file."""
    return pd.read_json(filepath, encoding=encoding, **kwargs)


def _read_jsonl(
    filepath: Path,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a JSONL (newline-delimited JSON) file.

    For each line, first tries json.loads(). If that fails, falls back to
    ast.literal_eval() to handle Python-repr dicts (common in older
    Steam/Amazon datasets).

    For files with > 500k rows, prints progress every 500k rows.
    """
    try:
        from tqdm import tqdm

        has_tqdm = True
    except ImportError:
        has_tqdm = False

    records: list[dict] = []
    parse_errors = 0
    fallback_count = 0

    # Count lines for progress bar (fast scan)
    total_lines = 0
    if has_tqdm:
        with open(filepath, "r", encoding=encoding) as f:
            for _ in f:
                total_lines += 1

    with open(filepath, "r", encoding=encoding) as f:
        iterator = tqdm(f, total=total_lines, desc=f"Reading {filepath.name}") if has_tqdm else f

        for line_num, line in enumerate(iterator, 1):
            line = line.strip()
            if not line:
                continue

            # Try json.loads first
            try:
                record = json.loads(line)
                records.append(record)
                continue
            except (json.JSONDecodeError, ValueError):
                pass

            # Fall back to ast.literal_eval for Python-repr dicts
            try:
                record = ast.literal_eval(line)
                if isinstance(record, dict):
                    records.append(record)
                    fallback_count += 1
                    continue
                else:
                    parse_errors += 1
                    if parse_errors <= 5:
                        logger.warning(
                            "Line %d: literal_eval produced %s, expected dict",
                            line_num,
                            type(record).__name__,
                        )
            except (ValueError, SyntaxError):
                parse_errors += 1
                if parse_errors <= 5:
                    logger.warning("Line %d: failed to parse as JSON or Python dict", line_num)

            # Progress logging for large files without tqdm
            if not has_tqdm and line_num % 500_000 == 0:
                logger.info("  ... read %dk rows", line_num // 1000)

    if parse_errors > 0:
        logger.warning("Total unparseable lines: %d (out of %d)", parse_errors, line_num)

    if fallback_count > 0:
        logger.info(
            "Used ast.literal_eval fallback for %d lines (Python-repr format)",
            fallback_count,
        )

    if not records:
        raise DatasetLoadError(f"No valid records found in JSONL file: {filepath}")

    return pd.DataFrame(records)


def _read_parquet(
    filepath: Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a Parquet file."""
    return pd.read_parquet(filepath, engine="pyarrow", **kwargs)


def _read_gzip(
    filepath: Path,
    encoding: str = "utf-8",
    separator: str = ",",
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a GZip-compressed file.

    Detects the inner format from the filename (e.g., 'reviews.csv.gz' -> CSV)
    and decompresses transparently.
    """
    # Detect inner format from filename
    stem = filepath.stem  # e.g., 'reviews.csv' from 'reviews.csv.gz'
    inner_suffix = Path(stem).suffix.lower()

    format_map = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".parquet": "parquet",
    }

    inner_format = format_map.get(inner_suffix)

    if inner_format == "csv":
        return pd.read_csv(filepath, encoding=encoding, sep=separator, compression="gzip", low_memory=False, **kwargs)
    elif inner_format == "tsv":
        return pd.read_csv(filepath, encoding=encoding, sep="\t", compression="gzip", low_memory=False, **kwargs)
    elif inner_format == "json":
        return pd.read_json(filepath, encoding=encoding, compression="gzip", **kwargs)
    elif inner_format == "jsonl":
        # Decompress to temp file and read as JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding=encoding) as tmp:
            with gzip.open(filepath, "rt", encoding=encoding) as gz:
                for line in gz:
                    tmp.write(line)
            tmp_path = Path(tmp.name)
        try:
            return _read_jsonl(tmp_path, encoding=encoding, **kwargs)
        finally:
            tmp_path.unlink(missing_ok=True)
    elif inner_format == "parquet":
        return pd.read_parquet(filepath, engine="pyarrow", **kwargs)
    else:
        # Default: try as gzipped CSV
        logger.warning(
            "Could not detect inner format from '%s', trying as gzipped CSV",
            filepath.name,
        )
        return pd.read_csv(filepath, encoding=encoding, sep=separator, compression="gzip", low_memory=False, **kwargs)


def _read_zip(
    filepath: Path,
    encoding: str = "utf-8",
    separator: str = ",",
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a file from a ZIP archive.

    If 'inner_filename' is provided in kwargs, extracts that specific file.
    Otherwise, reads the first file in the archive (or the first data file
    if multiple files are present).
    """
    inner_filename = kwargs.pop("inner_filename", None)

    with zipfile.ZipFile(filepath, "r") as zf:
        names = zf.namelist()
        if not names:
            raise DatasetLoadError(f"ZIP archive is empty: {filepath}")

        if inner_filename:
            if inner_filename not in names:
                raise DatasetLoadError(
                    f"File '{inner_filename}' not found in ZIP archive. "
                    f"Available files: {names}"
                )
            target = inner_filename
        else:
            # Filter out directories and hidden files
            data_files = [
                n for n in names
                if not n.endswith("/") and not n.startswith("__MACOSX") and not n.startswith(".")
            ]
            if not data_files:
                raise DatasetLoadError(f"No data files found in ZIP archive: {filepath}")
            target = data_files[0]
            if len(data_files) > 1:
                logger.warning(
                    "Multiple files in ZIP archive, reading first: '%s'. "
                    "Specify 'inner_filename' to choose a different file. Available: %s",
                    target,
                    data_files,
                )

        # Detect format from inner filename
        suffix = Path(target).suffix.lower()
        with zf.open(target) as f:
            if suffix in (".csv", ".tsv"):
                sep = "\t" if suffix == ".tsv" else separator
                return pd.read_csv(f, encoding=encoding, sep=sep, low_memory=False)
            elif suffix == ".json":
                return pd.read_json(f, encoding=encoding)
            elif suffix == ".parquet":
                return pd.read_parquet(f, engine="pyarrow")
            else:
                # Default to CSV
                logger.warning("Unknown format '%s' in ZIP, trying as CSV", suffix)
                return pd.read_csv(f, encoding=encoding, sep=separator, low_memory=False)


def _read_tar(
    filepath: Path,
    encoding: str = "utf-8",
    separator: str = ",",
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a file from a TAR archive (including .tar.gz, .tar.bz2).

    If 'inner_filename' is provided in kwargs, extracts that specific file.
    Otherwise, reads the first data file in the archive.
    """
    inner_filename = kwargs.pop("inner_filename", None)

    mode = "r:*"  # Auto-detect compression (gz, bz2, xz, or none)

    with tarfile.open(filepath, mode) as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        if not members:
            raise DatasetLoadError(f"TAR archive contains no files: {filepath}")

        if inner_filename:
            target_member = None
            for m in members:
                if m.name == inner_filename or m.name.endswith("/" + inner_filename):
                    target_member = m
                    break
            if target_member is None:
                available = [m.name for m in members]
                raise DatasetLoadError(
                    f"File '{inner_filename}' not found in TAR archive. "
                    f"Available files: {available}"
                )
        else:
            # Filter hidden/system files
            data_members = [
                m for m in members
                if not m.name.startswith(".") and not m.name.startswith("__")
            ]
            if not data_members:
                raise DatasetLoadError(f"No data files found in TAR archive: {filepath}")
            target_member = data_members[0]
            if len(data_members) > 1:
                logger.warning(
                    "Multiple files in TAR archive, reading first: '%s'. "
                    "Specify 'inner_filename' to choose a different file.",
                    target_member.name,
                )

        # Extract to temp directory and read
        suffix = Path(target_member.name).suffix.lower()
        f = tf.extractfile(target_member)
        if f is None:
            raise DatasetLoadError(
                f"Cannot extract '{target_member.name}' from TAR archive"
            )

        with f:
            if suffix in (".csv", ".tsv"):
                sep = "\t" if suffix == ".tsv" else separator
                return pd.read_csv(f, encoding=encoding, sep=sep, low_memory=False)
            elif suffix == ".json":
                # For JSONL (newline-delimited), we need to write to temp file
                # since _read_jsonl expects a file path
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=suffix, delete=False
                ) as tmp:
                    tmp.write(f.read())
                    tmp_path = Path(tmp.name)
                try:
                    # Try as JSON array first, fall back to JSONL
                    try:
                        return pd.read_json(tmp_path, encoding=encoding)
                    except ValueError:
                        return _read_jsonl(tmp_path, encoding=encoding)
                finally:
                    tmp_path.unlink(missing_ok=True)
            elif suffix == ".parquet":
                return pd.read_parquet(f, engine="pyarrow")
            else:
                logger.warning("Unknown format '%s' in TAR, trying as CSV", suffix)
                return pd.read_csv(f, encoding=encoding, sep=separator, low_memory=False)
