"""Multi-format file reader for recommender system datasets.

Reads raw dataset files into pandas DataFrames. Supports CSV, TSV, JSON,
JSONL (including Python-repr dicts), Parquet, GZip, ZIP, and TAR formats.

Format, encoding, and separator are all auto-detected from the file when not
explicitly provided — so configs only need to specify a filename.

This module is purely I/O — it reads files and returns DataFrames without
any data modification or standardisation.
"""

from __future__ import annotations

import ast
import csv
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

# Extension → format mapping for unambiguous cases
_EXT_TO_FORMAT: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
    ".gz": "gz",
    ".zip": "zip",
    ".tar": "tar",
    ".tgz": "tar",
}


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detection functions
# ─────────────────────────────────────────────────────────────────────────────


def detect_format(filepath: str | Path) -> str:
    """Detect the file format from the filename extension.

    For ``.json`` files, peeks at the first non-empty line to distinguish a
    JSON array (format ``'json'``) from newline-delimited JSON (``'jsonl'``).
    This correctly handles datasets like Steam whose files are named ``.json``
    but contain one Python-repr dict per line.

    Args:
        filepath: Path to the file.

    Returns:
        One of: ``'csv'``, ``'tsv'``, ``'json'``, ``'jsonl'``,
        ``'parquet'``, ``'gz'``, ``'zip'``, ``'tar'``.

    Raises:
        DatasetLoadError: If the format cannot be determined.
    """
    path = Path(filepath)

    # Handle compound extensions first (.tar.gz, .tar.bz2, .tar.xz)
    lower_name = path.name.lower()
    for compound in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if lower_name.endswith(compound):
            return "tar"

    suffix = path.suffix.lower()

    if suffix in _EXT_TO_FORMAT:
        return _EXT_TO_FORMAT[suffix]

    if suffix == ".json":
        # Peek at the first non-empty line to distinguish array vs JSONL
        return _peek_json_format(path)

    raise DatasetLoadError(
        f"Cannot determine file format from extension '{suffix}' for '{path.name}'. "
        f"Supported extensions: {sorted(set(_EXT_TO_FORMAT) | {'.json'})}"
    )


def detect_encoding(filepath: str | Path) -> str:
    """Detect the character encoding of a text file.

    Uses ``chardet`` or ``charset_normalizer`` if available; otherwise
    returns ``'utf-8'`` as a safe default (correct for the vast majority of
    modern dataset releases).

    Args:
        filepath: Path to the file (binary formats are skipped).

    Returns:
        A Python-compatible encoding string, e.g. ``'utf-8'``, ``'latin-1'``.
    """
    path = Path(filepath)

    # Binary formats don't have a text encoding
    if path.suffix.lower() in (".parquet", ".zip", ".tar", ".tgz", ".gz"):
        return "utf-8"

    # Try chardet / charset_normalizer
    detector = _get_charset_detector()
    if detector is not None:
        try:
            raw = path.read_bytes()[:65536]  # sample first 64 KB
            result = detector(raw)
            if result and result.get("encoding"):
                enc = result["encoding"]
                confidence = result.get("confidence", 0)
                logger.debug(
                    "Detected encoding for '%s': %s (confidence %.0f%%)",
                    path.name, enc, confidence * 100,
                )
                return enc
        except Exception:
            pass  # Fall through to default

    return "utf-8"


def detect_separator(filepath: str | Path, encoding: str = "utf-8") -> str:
    """Detect the column separator of a CSV/TSV file using :class:`csv.Sniffer`.

    Args:
        filepath: Path to the CSV/TSV file.
        encoding: File encoding to use when reading the sample.

    Returns:
        A single-character separator string. Falls back to ``','`` if
        detection fails.
    """
    path = Path(filepath)

    # TSV by extension → always tab
    if path.suffix.lower() == ".tsv":
        return "\t"

    try:
        with open(path, "r", encoding=encoding, errors="replace") as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
        return dialect.delimiter
    except csv.Error:
        logger.debug("csv.Sniffer could not detect separator for '%s'; using ','", path.name)
        return ","


# ─────────────────────────────────────────────────────────────────────────────
# Main read_file entry point
# ─────────────────────────────────────────────────────────────────────────────


def read_file(
    filepath: str | Path,
    format: str | None = None,
    encoding: str | None = None,
    separator: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a raw dataset file into a pandas DataFrame.

    All parameters except ``filepath`` are optional. When omitted, the format,
    encoding, and separator are auto-detected from the filename and file content.

    Args:
        filepath: Path to the file to read.
        format: File format. One of: ``csv``, ``tsv``, ``json``, ``jsonl``,
            ``parquet``, ``gz``, ``zip``, ``tar``. If ``None``, auto-detected
            from the filename (and content for ``.json`` files).
        encoding: Character encoding for text files. If ``None``, auto-detected
            (defaults to ``'utf-8'`` when detection is unavailable).
        separator: Column separator for CSV/TSV files. If ``None``, auto-detected
            via :func:`csv.Sniffer` (defaults to ``','``).
        **kwargs: Additional keyword arguments passed to the underlying reader.
            For zip/tar: ``inner_filename`` specifies which file to extract.

    Returns:
        A pandas DataFrame containing the raw data.

    Raises:
        DatasetLoadError: If the file cannot be read or the format cannot be
            determined.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise DatasetLoadError(f"File not found: {filepath}")

    # Fill in any missing parameters via auto-detection
    if format is None:
        format = detect_format(filepath)
        logger.debug("Auto-detected format for '%s': %s", filepath.name, format)

    if encoding is None:
        encoding = detect_encoding(filepath)
        logger.debug("Auto-detected encoding for '%s': %s", filepath.name, encoding)

    format_lower = format.lower().strip()

    if separator is None:
        separator = detect_separator(filepath, encoding) if format_lower in ("csv", "tsv") else ","
        if format_lower in ("csv", "tsv"):
            logger.debug("Auto-detected separator for '%s': %r", filepath.name, separator)

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

    memory_mb = df.memory_usage(deep=True).sum() / 1e6
    logger.info(
        "Loaded %s [%s]: %d rows × %d cols (%.1f MB in memory)",
        filepath.name,
        format_lower,
        len(df),
        len(df.columns),
        memory_mb,
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _peek_json_format(filepath: Path, encoding: str = "utf-8") -> str:
    """Determine whether a .json file is a JSON array or JSONL by peeking."""
    try:
        with open(filepath, "r", encoding=encoding, errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    return "json" if stripped[0] == "[" else "jsonl"
    except OSError:
        pass
    return "jsonl"  # safe default for most recsys datasets


def _get_charset_detector() -> Any | None:
    """Return a charset detection callable, or None if no library is available."""
    try:
        from charset_normalizer import from_bytes

        def detect(raw: bytes) -> dict:
            result = from_bytes(raw).best()
            return {"encoding": str(result.encoding), "confidence": 0.9} if result else {}

        return detect
    except ImportError:
        pass

    try:
        import chardet

        return chardet.detect
    except ImportError:
        pass

    return None


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

    Supports an optional ``nrows`` kwarg to read only the first N rows —
    used for dry-run schema sampling.
    """
    try:
        from tqdm import tqdm

        has_tqdm = True
    except ImportError:
        has_tqdm = False

    # Pop nrows before passing further (not a standard JSONL kwarg)
    nrows: int | None = kwargs.pop("nrows", None)

    records: list[dict] = []
    parse_errors = 0
    fallback_count = 0
    line_num = 0

    # Count lines for progress bar (fast scan) — skip if nrows is small
    total_lines = 0
    if has_tqdm and nrows is None:
        with open(filepath, "r", encoding=encoding) as f:
            for _ in f:
                total_lines += 1

    with open(filepath, "r", encoding=encoding) as f:
        if has_tqdm and nrows is None:
            iterator = tqdm(f, total=total_lines, desc=f"Reading {filepath.name}")
        else:
            iterator = f

        for line_num, line in enumerate(iterator, 1):
            # Stop early if nrows limit reached
            if nrows is not None and len(records) >= nrows:
                break

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
