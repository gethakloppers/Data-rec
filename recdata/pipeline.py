"""RecData pipeline entry point.

Provides the high-level :func:`load_dataset` function and a CLI for running
the standardisation pipeline against a raw dataset directory.

CLI usage::

    python -m recdata.pipeline \\
        --config recdata/configs/steam.yaml \\
        --raw /path/to/raw/data \\
        --output /path/to/output \\
        [--dry-run]

``--dry-run`` prints the planned transformations without loading the full
dataset or writing any output files. It reads only the first 1,000 rows of each
file to infer column names and dtypes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from recdata.exceptions import ColumnNotFoundError, ConfigValidationError, DatasetLoadError
from recdata.loaders.base_loader import load_config, normalize_file_def
from recdata.loaders.file_reader import read_file
from recdata.processing.standardiser import describe_standardisation_plan, standardise_df

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(
    config: dict[str, Any] | str | Path,
    raw_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load and standardise a dataset from raw files using a YAML config.

    This is the main entry point for the RecData library. It:
    1. Loads and validates the config (if a path is given).
    2. Reads each declared file from ``raw_path`` into a DataFrame.
    3. Standardises each DataFrame (column renaming, type casting, null removal,
       column dropping).
    4. Returns a dict of standardised DataFrames keyed by role.

    The output is *not* written to disk by this function — use
    :mod:`recdata.exporters.exporter` for that.

    Args:
        config: Either a validated config dict or a path to a YAML config file.
        raw_path: Directory containing the raw dataset files declared in the config.
        output_path: (Unused in Phase 2 — reserved for Phase 4.) Directory where
            processed outputs will be written.

    Returns:
        A dict with keys ``'interactions'``, ``'items'``, and/or ``'users'``
        (only for roles that have files declared in the config). Each value is
        a standardised :class:`pandas.DataFrame`.

    Raises:
        FileNotFoundError: If a declared data file does not exist in ``raw_path``.
        DatasetLoadError: If a file cannot be read.
        ColumnNotFoundError: If required columns (user_id, item_id) are not found.
        ConfigValidationError: If the config is invalid.
    """
    # Accept config as dict or path
    if not isinstance(config, dict):
        config = load_config(config)

    raw_path = Path(raw_path)
    dataset_name = config["dataset_name"]
    files_config = config.get("files", {})

    logger.info("=== Loading dataset '%s' from %s ===", dataset_name, raw_path)

    result: dict[str, pd.DataFrame] = {}
    all_warnings: list[str] = []

    for role in ("interactions", "items", "users"):
        file_def = normalize_file_def(files_config.get(role))
        if file_def is None:
            logger.info("[%s] No file declared — skipping.", role)
            continue

        filepath = raw_path / file_def["filename"]
        if not filepath.exists():
            raise FileNotFoundError(
                f"[{role}] File not found: {filepath}\n"
                f"Expected in raw_path: {raw_path}"
            )

        logger.info("[%s] Reading file: %s", role, filepath.name)
        df_raw = read_file(
            filepath=filepath,
            format=file_def.get("format"),        # None → auto-detect
            encoding=file_def.get("encoding"),    # None → auto-detect
            separator=file_def.get("separator"),  # None → auto-detect
        )

        logger.info("[%s] Standardising %d rows × %d cols.", role, len(df_raw), len(df_raw.columns))
        df_std, warnings = standardise_df(df_raw, config, role)
        all_warnings.extend(warnings)

        result[role] = df_std

        if warnings:
            for w in warnings:
                logger.warning(w)

    logger.info(
        "=== Dataset '%s' loaded. Roles: %s ===",
        dataset_name,
        list(result.keys()),
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run printing
# ─────────────────────────────────────────────────────────────────────────────


def _print_dry_run_plan(
    role: str,
    filename: str,
    plan: dict[str, Any],
) -> None:
    """Pretty-print a dry-run plan for one file role."""
    sep = "─" * 60

    print(f"\n{'═' * 60}")
    print(f"  FILE ROLE : {role.upper()}")
    print(f"  FILE      : {filename}")
    print(f"{'═' * 60}")

    # Column renames
    if plan["renames"]:
        print(f"\n  {sep}")
        print("  COLUMN RENAMES")
        print(f"  {sep}")
        for entry in plan["renames"]:
            print(f"    {entry['from']!r:30s} → {entry['to']!r}")
    else:
        print("\n  No column renames needed (key columns already use standard names).")

    # Missing optionals
    if plan["missing_optional"]:
        print(f"\n  {sep}")
        print("  OPTIONAL COLUMNS NOT FOUND (will be absent in output)")
        print(f"  {sep}")
        for col in plan["missing_optional"]:
            print(f"    • {col}")

    # Missing required — fatal
    if plan["missing_required"]:
        print(f"\n  {sep}")
        print("  ⚠ REQUIRED COLUMNS NOT FOUND (pipeline will FAIL)")
        print(f"  {sep}")
        for col in plan["missing_required"]:
            print(f"    ✗ {col}")

    # Type casts
    if plan["type_casts"]:
        print(f"\n  {sep}")
        print("  TYPE CASTS")
        print(f"  {sep}")
        for cast in plan["type_casts"]:
            print(
                f"    {cast['column']:25s}  {cast['from_dtype']:15s} → {cast['to_dtype']}"
            )

    # Null ID rows
    if plan["null_id_rows"] > 0:
        print(f"\n  ⚠ {plan['null_id_rows']:,} rows would be dropped (null user_id or item_id).")

    # Drop columns
    if plan["drop_columns"]:
        print(f"\n  {sep}")
        print("  COLUMNS TO DROP")
        print(f"  {sep}")
        for col in plan["drop_columns"]:
            print(f"    - {col}")

    # Output columns
    print(f"\n  {sep}")
    print(f"  OUTPUT COLUMNS ({len(plan['output_columns'])} total)")
    print(f"  {sep}")
    for col in plan["output_columns"]:
        print(f"    {col}")

    # Warnings
    if plan["warnings"]:
        print(f"\n  {sep}")
        print("  WARNINGS")
        print(f"  {sep}")
        for w in plan["warnings"]:
            print(f"    ! {w}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="recdata.pipeline",
        description=(
            "RecData dataset standardisation pipeline.\n\n"
            "Loads raw dataset files and standardises them according to a YAML config."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the dataset YAML config file (e.g. recdata/configs/steam.yaml).",
    )
    parser.add_argument(
        "--raw",
        required=True,
        metavar="DIR",
        help="Directory containing the raw dataset files declared in the config.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help=(
            "Directory where processed outputs will be written. "
            "Not used in --dry-run mode. (Phase 4)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the planned transformations without loading the full dataset "
            "or writing any files. Reads only the first 1,000 rows of each file."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def _run_dry_run(config: dict[str, Any], raw_path: Path) -> None:
    """Execute dry-run mode: show the transformation plan without full loading."""
    dataset_name = config["dataset_name"]
    files_config = config.get("files", {})

    print(f"\n{'▓' * 60}")
    print(f"  DRY RUN — {dataset_name.upper()}")
    print(f"  Raw path: {raw_path}")
    print(f"{'▓' * 60}")
    print("  (Reading first 1,000 rows per file to infer schema.)")

    any_fatal = False

    for role in ("interactions", "items", "users"):
        file_def = normalize_file_def(files_config.get(role))
        if file_def is None:
            print(f"\n  [Skipping '{role}' — no file declared in config]")
            continue

        filepath = raw_path / file_def["filename"]
        if not filepath.exists():
            print(f"\n  ✗ [{role}] File not found: {filepath}")
            any_fatal = True
            continue

        # Read a small sample to get column names + dtypes
        try:
            df_sample = read_file(
                filepath=filepath,
                format=file_def.get("format"),
                encoding=file_def.get("encoding"),
                separator=file_def.get("separator"),
                nrows=1000,
            )
        except Exception as exc:
            print(f"\n  ✗ [{role}] Could not read file: {exc}")
            any_fatal = True
            continue

        plan = describe_standardisation_plan(df_sample, config, role)
        _print_dry_run_plan(role, file_def["filename"], plan)

        if plan["missing_required"]:
            any_fatal = True

    print(f"\n{'▓' * 60}")
    if any_fatal:
        print("  ✗ DRY RUN COMPLETE — pipeline would FAIL (see warnings above)")
    else:
        print("  ✓ DRY RUN COMPLETE — pipeline would succeed")
    print(f"{'▓' * 60}\n")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Load and validate config
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ConfigValidationError) as exc:
        print(f"✗ Config error: {exc}", file=sys.stderr)
        return 1

    raw_path = Path(args.raw)
    if not raw_path.is_dir():
        print(f"✗ Raw data directory not found: {raw_path}", file=sys.stderr)
        return 1

    if args.dry_run:
        _run_dry_run(config, raw_path)
        return 0

    # Full pipeline run (Phase 2 partial — no exporter yet)
    if args.output is None:
        print(
            "✗ --output is required for a full pipeline run (use --dry-run to preview).",
            file=sys.stderr,
        )
        return 1

    try:
        result = load_dataset(config=config, raw_path=raw_path, output_path=args.output)
    except (FileNotFoundError, DatasetLoadError, ColumnNotFoundError) as exc:
        print(f"✗ Pipeline failed: {exc}", file=sys.stderr)
        return 1

    # Summary
    print(f"\n{'─' * 60}")
    print(f"  Dataset '{config['dataset_name']}' loaded successfully.")
    for role, df in result.items():
        print(f"  {role:15s}: {len(df):>10,} rows × {len(df.columns)} cols")
    print(f"{'─' * 60}")
    print("  Note: Use --output with the exporter (Phase 4) to write Parquet files.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
