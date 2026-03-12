"""RecData pipeline entry point.

Provides the high-level :func:`process_dataset` function and a CLI for running
the full standardisation → quality → profile → export pipeline against a raw
dataset directory.

CLI usage::

    python -m recdata.pipeline \\
        --config recdata/configs/xwines.yaml \\
        --raw /path/to/raw/data \\
        --output /path/to/output \\
        [--dry-run] [--quality-only]

``--dry-run`` prints the planned transformations without loading the full
dataset or writing any output files. It reads only the first 1,000 rows of each
file to infer column names and dtypes.

``--quality-only`` runs standardisation + quality report but skips profiling and
export.  Useful for a quick diagnostic pass before committing to the full run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from recdata.exceptions import ColumnNotFoundError, ConfigValidationError, DatasetLoadError
from recdata.exporters.exporter import export_dataset
from recdata.loaders.base_loader import load_config, normalize_file_def
from recdata.loaders.file_reader import read_file
from recdata.processing.standardiser import describe_standardisation_plan, standardise_df
from recdata.profiler.dataset_profiler import build_id_mappings, profile_dataset
from recdata.profiler.quality_report import quality_report

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Raw file loading
# ─────────────────────────────────────────────────────────────────────────────


def load_raw_files(
    config: dict[str, Any],
    raw_path: str | Path,
) -> dict[str, pd.DataFrame]:
    """Read all declared raw files into DataFrames.

    Handles both direct files and archive-wrapped files (e.g. ZIP).  The
    ``files`` section of the config may contain an ``archive`` key alongside
    ``filename``; when present the archive is read and ``filename`` is used as
    the inner file to extract.

    Args:
        config: Validated config dict.
        raw_path: Directory containing the raw dataset files.

    Returns:
        Dict mapping role names (``'interactions'``, ``'items'``, ``'users'``)
        to raw DataFrames.

    Raises:
        FileNotFoundError: If a declared data file does not exist.
        DatasetLoadError: If a file cannot be read.
    """
    raw_path = Path(raw_path)
    files_config = config.get("files", {})
    dfs: dict[str, pd.DataFrame] = {}

    for role in ("interactions", "items", "users"):
        file_def = normalize_file_def(files_config.get(role))
        if file_def is None:
            logger.info("[%s] No file declared — skipping.", role)
            continue

        archive = file_def.get("archive")
        if archive:
            # Read from archive (e.g. ZIP), using filename as inner target
            filepath = raw_path / archive
            if not filepath.exists():
                raise FileNotFoundError(
                    f"[{role}] Archive not found: {filepath}\n"
                    f"Expected in raw_path: {raw_path}"
                )
            logger.info("[%s] Reading from archive: %s → %s", role, archive, file_def["filename"])
            df = read_file(
                filepath=filepath,
                format="zip",
                encoding=file_def.get("encoding"),
                separator=file_def.get("separator"),
                inner_filename=file_def["filename"],
            )
        else:
            filepath = raw_path / file_def["filename"]
            if not filepath.exists():
                raise FileNotFoundError(
                    f"[{role}] File not found: {filepath}\n"
                    f"Expected in raw_path: {raw_path}"
                )
            logger.info("[%s] Reading file: %s", role, filepath.name)
            df = read_file(
                filepath=filepath,
                format=file_def.get("format"),
                encoding=file_def.get("encoding"),
                separator=file_def.get("separator"),
            )

        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        logger.info("[%s] Loaded %d rows × %d cols (%.1f MB)", role, len(df), len(df.columns), mem_mb)
        dfs[role] = df

    return dfs


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(
    config: dict[str, Any] | str | Path,
    raw_path: str | Path,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load and standardise a dataset from raw files using a YAML config.

    This runs steps 1–2 of the pipeline (loading + standardisation) and returns
    the standardised DataFrames along with any processing warnings.

    Args:
        config: Either a validated config dict or a path to a YAML config file.
        raw_path: Directory containing the raw dataset files declared in the
            config.

    Returns:
        A tuple of ``(dfs, warnings)`` where *dfs* maps role names to
        standardised DataFrames, and *warnings* is a list of warning strings
        generated during standardisation.

    Raises:
        FileNotFoundError: If a declared data file does not exist in ``raw_path``.
        DatasetLoadError: If a file cannot be read.
        ColumnNotFoundError: If required columns (user_id, item_id) are not found.
        ConfigValidationError: If the config is invalid.
    """
    if not isinstance(config, dict):
        config = load_config(config)

    dataset_name = config["dataset_name"]
    logger.info("=== Loading dataset '%s' ===", dataset_name)

    # Step 1: Read raw files
    raw_dfs = load_raw_files(config, raw_path)

    # Step 2: Standardise each role
    std_dfs: dict[str, pd.DataFrame] = {}
    all_warnings: list[str] = []

    for role, df_raw in raw_dfs.items():
        logger.info("[%s] Standardising %d rows × %d cols.", role, len(df_raw), len(df_raw.columns))
        df_std, warnings = standardise_df(df_raw, config, role)
        std_dfs[role] = df_std
        all_warnings.extend(warnings)

        if warnings:
            for w in warnings:
                logger.warning(w)

    logger.info(
        "=== Dataset '%s' loaded and standardised. Roles: %s ===",
        dataset_name,
        list(std_dfs.keys()),
    )
    return std_dfs, all_warnings


def process_dataset(
    config: dict[str, Any] | str | Path,
    raw_path: str | Path,
    output_path: str | Path,
    *,
    quality_only: bool = False,
) -> Path:
    """Run the full pipeline: load → standardise → quality → profile → export.

    This is the main entry point for producing a complete dataset output
    directory with Parquet files, ID mappings, profile JSON, quality report,
    and a markdown summary.

    Args:
        config: Either a validated config dict or a path to a YAML config file.
        raw_path: Directory containing the raw dataset files.
        output_path: Root output directory. A subdirectory named after the
            dataset will be created inside it.
        quality_only: If ``True``, run only standardisation + quality report
            and print a summary to stdout. Skips profiling and export.

    Returns:
        Path to the created dataset output directory.

    Raises:
        FileNotFoundError: If a declared data file does not exist.
        DatasetLoadError: If a file cannot be read.
        ColumnNotFoundError: If required columns are not found.
        ConfigValidationError: If the config is invalid.
    """
    if not isinstance(config, dict):
        config = load_config(config)

    dataset_name = config["dataset_name"]

    # Steps 1-2: Load and standardise
    std_dfs, warnings = load_dataset(config, raw_path)

    # Step 3: Quality report
    logger.info("Running quality report...")
    qr = quality_report(std_dfs, config)
    logger.info("Quality report complete — sections: %s", list(qr.keys()))

    if quality_only:
        _print_quality_summary(dataset_name, std_dfs, qr)
        # Still write the quality report to disk
        output_dir = Path(output_path) / dataset_name / "profile"
        output_dir.mkdir(parents=True, exist_ok=True)
        import json
        qr_path = output_dir / "quality_report.json"
        with open(qr_path, "w", encoding="utf-8") as f:
            json.dump(qr, f, indent=2, default=str, ensure_ascii=False)
        logger.info("Wrote quality report to %s", qr_path)
        return output_dir.parent

    # Step 4: Profile
    logger.info("Profiling dataset...")
    profile = profile_dataset(std_dfs, config, warnings, qr)
    logger.info("Profile complete: %s", dataset_name)

    # Step 5: ID mappings
    logger.info("Building ID mappings...")
    id_maps = build_id_mappings(std_dfs)
    logger.info(
        "ID mappings: %d users, %d items",
        len(id_maps.get("user_map", {})),
        len(id_maps.get("item_map", {})),
    )

    # Step 6: Export
    logger.info("Exporting dataset...")
    result_path = export_dataset(std_dfs, profile, id_maps, qr, output_path)

    _print_export_summary(dataset_name, std_dfs, profile, result_path)
    return result_path


# ─────────────────────────────────────────────────────────────────────────────
# Summary printing
# ─────────────────────────────────────────────────────────────────────────────


def _print_quality_summary(
    name: str,
    dfs: dict[str, pd.DataFrame],
    qr: dict[str, Any],
) -> None:
    """Print a compact quality report summary to stdout."""
    sep = "─" * 60
    print(f"\n{'═' * 60}")
    print(f"  QUALITY REPORT — {name.upper()}")
    print(f"{'═' * 60}")

    # DataFrames summary
    for role, df in dfs.items():
        print(f"\n  {role.upper()}: {len(df):,} rows × {len(df.columns)} cols")

    # Type audit
    type_audit = qr.get("type_audit", {})
    mismatches = []
    for role_data in type_audit.values():
        if isinstance(role_data, list):
            for entry in role_data:
                if entry.get("match") == "mismatch":
                    mismatches.append(entry)
    if mismatches:
        print(f"\n  {sep}")
        print(f"  TYPE MISMATCHES ({len(mismatches)})")
        print(f"  {sep}")
        for m in mismatches:
            print(f"    {m['column']:25s}  declared={m.get('declared_type', '?'):10s}  actual={m.get('actual_dtype', '?')}")
    else:
        print(f"\n  ✓ No type mismatches found")

    # Mixed types
    mixed = qr.get("mixed_types", {})
    mixed_cols = []
    for role_data in mixed.values():
        if isinstance(role_data, list):
            mixed_cols.extend(role_data)
    if mixed_cols:
        print(f"\n  {sep}")
        print(f"  MIXED-TYPE COLUMNS ({len(mixed_cols)})")
        print(f"  {sep}")
        for m in mixed_cols:
            dom = m.get("dominant_type", "?")
            pct = m.get("dominant_pct", 0)
            print(f"    {m['column']:25s}  dominant={dom} ({pct:.1f}%)  types={list(m.get('type_counts', {}).keys())}")

    # Nulls
    null_data = qr.get("null_analysis", {})
    high_nulls = []
    for role_data in null_data.values():
        if isinstance(role_data, list):
            for entry in role_data:
                if entry.get("null_pct", 0) > 10:
                    high_nulls.append(entry)
    if high_nulls:
        print(f"\n  {sep}")
        print(f"  HIGH NULL COLUMNS (> 10%)")
        print(f"  {sep}")
        for h in high_nulls:
            print(f"    {h['column']:25s}  {h['null_pct']:.1f}% null ({h.get('null_count', 0):,} rows)")

    # Duplicates
    dup_data = qr.get("duplicate_analysis", {})
    for role, info in dup_data.items():
        if isinstance(info, dict):
            dup_count = info.get("duplicate_count", 0)
            if dup_count > 0:
                print(f"\n  ⚠ {role}: {dup_count:,} duplicate rows")

    # ID coverage
    coverage = qr.get("id_coverage", {})
    if coverage:
        print(f"\n  {sep}")
        print(f"  ID COVERAGE")
        print(f"  {sep}")
        for key, info in coverage.items():
            if isinstance(info, dict):
                total = info.get("total", 0)
                covered = info.get("covered", 0)
                pct = info.get("coverage_pct", 0)
                print(f"    {key:35s}  {covered:>10,} / {total:>10,} ({pct:.1f}%)")

    print(f"\n{'═' * 60}\n")


def _print_export_summary(
    name: str,
    dfs: dict[str, pd.DataFrame],
    profile: dict[str, Any],
    output_path: Path,
) -> None:
    """Print a compact export summary to stdout."""
    counts = profile.get("counts", {})

    print(f"\n{'═' * 60}")
    print(f"  EXPORT COMPLETE — {name.upper()}")
    print(f"{'═' * 60}")
    print(f"  Output: {output_path}")
    print()

    for role, df in dfs.items():
        print(f"  {role:15s}: {len(df):>12,} rows × {len(df.columns)} cols")

    print()
    print(f"  Users:         {counts.get('n_users', 0):>12,}")
    print(f"  Items:         {counts.get('n_items', 0):>12,}")
    print(f"  Interactions:  {counts.get('n_interactions', 0):>12,}")
    print(f"  Sparsity:      {counts.get('sparsity', 0):>12.6f}")

    inter = profile.get("interactions", {})
    if inter.get("has_rating"):
        rs = inter.get("rating_scale", {})
        print(f"  Rating range:  {rs.get('min', '?')} – {rs.get('max', '?')} (mean {rs.get('mean', 0):.2f})")
    if inter.get("has_timestamp"):
        ts = inter.get("timestamp_range", {})
        print(f"  Time span:     {str(ts.get('earliest', ''))[:10]} to {str(ts.get('latest', ''))[:10]}")

    print(f"\n{'═' * 60}\n")


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

        # Resolve the actual file to read (may be an archive)
        archive = file_def.get("archive")
        if archive:
            filepath = raw_path / archive
            read_kwargs: dict[str, Any] = {
                "format": "zip",
                "inner_filename": file_def["filename"],
            }
        else:
            filepath = raw_path / file_def["filename"]
            read_kwargs = {"format": file_def.get("format")}

        if not filepath.exists():
            print(f"\n  ✗ [{role}] File not found: {filepath}")
            any_fatal = True
            continue

        # Read a small sample to get column names + dtypes
        try:
            df_sample = read_file(
                filepath=filepath,
                encoding=file_def.get("encoding"),
                separator=file_def.get("separator"),
                nrows=1000,
                **read_kwargs,
            )
        except Exception as exc:
            print(f"\n  ✗ [{role}] Could not read file: {exc}")
            any_fatal = True
            continue

        plan = describe_standardisation_plan(df_sample, config, role)
        _print_dry_run_plan(role, file_def.get("archive", file_def["filename"]), plan)

        if plan["missing_required"]:
            any_fatal = True

    print(f"\n{'▓' * 60}")
    if any_fatal:
        print("  ✗ DRY RUN COMPLETE — pipeline would FAIL (see warnings above)")
    else:
        print("  ✓ DRY RUN COMPLETE — pipeline would succeed")
    print(f"{'▓' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="recdata.pipeline",
        description=(
            "RecData dataset standardisation pipeline.\n\n"
            "Loads raw dataset files, standardises them, generates a quality\n"
            "report and profile, then exports Parquet files + documentation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the dataset YAML config file (e.g. recdata/configs/xwines.yaml).",
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
        help="Directory where processed outputs will be written.",
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
        "--quality-only",
        action="store_true",
        help=(
            "Run standardisation + quality report only. Skips profiling and "
            "Parquet export. Useful for quick diagnostic checks."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


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

    # Dry-run mode
    if args.dry_run:
        _run_dry_run(config, raw_path)
        return 0

    # Full or quality-only run requires --output
    if args.output is None:
        print(
            "✗ --output is required for a pipeline run (use --dry-run to preview).",
            file=sys.stderr,
        )
        return 1

    try:
        result = process_dataset(
            config=config,
            raw_path=raw_path,
            output_path=args.output,
            quality_only=args.quality_only,
        )
    except (FileNotFoundError, DatasetLoadError, ColumnNotFoundError) as exc:
        print(f"✗ Pipeline failed: {exc}", file=sys.stderr)
        return 1

    if args.quality_only:
        print(f"Quality report saved to: {result / 'profile' / 'quality_report.json'}")
    else:
        print(f"Dataset exported to: {result}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
