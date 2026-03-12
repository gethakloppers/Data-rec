"""Standardisation pipeline for raw recommender system DataFrames.

Given a raw DataFrame and a dataset YAML config, this module produces a clean
DataFrame with predictable column names and correct types.

Key design decisions:
- Column names are standardised for key roles only (user_id, item_id, timestamp,
  rating). All other columns keep their original (lowercased, spaces→underscores)
  names.
- No ``__type`` suffixes are baked into column names. Feature type information
  lives in the profile metadata only.
- Original user_id / item_id values are always cast to str — they are tokens,
  never numeric indices.
- Columns typed ``datetime`` in the feature declarations are cast to
  ``datetime64[ns]`` in *every* role, not just interactions.
- No data is filtered or deduplicated. Null IDs are removed, but nothing else.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from recdata.exceptions import ColumnNotFoundError
from recdata.loaders.base_loader import get_feature_map

logger = logging.getLogger(__name__)

# Standard column names for key roles
STANDARD_USER_COL = "user_id"
STANDARD_ITEM_COL = "item_id"
STANDARD_TIMESTAMP_COL = "timestamp"
STANDARD_RATING_COL = "rating"


def standardise_df(
    df: pd.DataFrame,
    config: dict[str, Any],
    df_role: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Standardise a raw DataFrame using the provided dataset config.

    Applies the following transformations in order:
    1. Lowercase all column names and replace spaces with underscores.
    2. Rename key columns (user_id, item_id, timestamp, rating) using the
       schema mapping in the config (first match wins, case-insensitive).
    3. Remove rows where user_id or item_id is null.
    4. Cast user_id and item_id to str (object dtype).
    4b. Cast all token-typed columns to str (categorical values must be strings).
    5. Cast all datetime-typed columns to datetime64[ns] (all roles).
    6. Cast rating column to float32 (interactions only, if present).
    7. Exclude columns declared as 'exclude' in the feature declarations.

    Args:
        df: The raw DataFrame to standardise. Not modified in place — a copy
            is returned.
        config: A validated config dictionary loaded by
            :func:`recdata.loaders.base_loader.load_config`.
        df_role: One of ``'interactions'``, ``'items'``, or ``'users'``.
            Controls which feature declarations are used and whether
            timestamp/rating casting is applied.

    Returns:
        A tuple of ``(standardised_df, warnings)`` where ``warnings`` is a list
        of string messages describing non-fatal issues encountered.

    Raises:
        ColumnNotFoundError: If user_id or item_id candidates are not found in
            the DataFrame. All missing required columns are listed at once.
        ValueError: If ``df_role`` is not one of the accepted values.
    """
    
    valid_roles = {"interactions", "items", "users"}
    if df_role not in valid_roles:
        raise ValueError(f"df_role must be one of {sorted(valid_roles)}, got '{df_role}'")

    warnings: list[str] = []
    df = df.copy()

    # ── Step 1: Lowercase all column names + replace spaces with underscores ─
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    logger.debug("[%s] Lowercased %d column names.", df_role, len(df.columns))

    # ── Step 2: Rename key columns via schema mapping ────────────────────────
    df, rename_warnings = _rename_key_columns(df, config, df_role)
    warnings.extend(rename_warnings)

    # ── Step 3: Remove null IDs ───────────────────────────────────────────────
    df, null_warnings = _remove_null_ids(df, df_role)
    warnings.extend(null_warnings)

    # ── Step 4: Cast IDs to string ────────────────────────────────────────────
    for col in [STANDARD_USER_COL, STANDARD_ITEM_COL]:
        if col in df.columns:
            df[col] = df[col].astype(str)
            logger.debug("[%s] Cast '%s' to str.", df_role, col)

    # ── Step 4b: Cast all object-typed columns to string ──────────────────────
    #   Columns declared as 'object' (categorical) must be stored as str, even
    #   when the raw data has them as integers (e.g. regionid, wineryid).
    feature_map = get_feature_map(config, df_role)
    object_cols = [
        col_name for col_name, feat_type in feature_map.items()
        if feat_type == "object" and col_name in df.columns
    ]
    for col in object_cols:
        if df[col].dtype != object:
            df[col] = df[col].astype(str)
            logger.debug("[%s] Cast object column '%s' to str.", df_role, col)

    # ── Step 4c: Cast all bool-typed columns to boolean ──────────────────────
    bool_cols = [
        col_name for col_name, feat_type in feature_map.items()
        if feat_type == "bool" and col_name in df.columns
    ]
    for col in bool_cols:
        if df[col].dtype != bool:
            # Map common boolean-like values to True/False
            bool_map = {
                "true": True, "false": False,
                "yes": True, "no": False,
                "t": True, "f": False,
                "y": True, "n": False,
                "1": True, "0": False,
                "on": True, "off": False,
            }
            df[col] = (
                df[col].astype(str).str.lower().str.strip()
                .map(bool_map).astype("boolean")
            )
            logger.debug("[%s] Cast bool column '%s' to boolean.", df_role, col)

    # ── Step 5: Cast ALL datetime-typed columns to datetime64[ns] ───────────
    #   Applies to every role, not just interactions. Uses the feature map to
    #   find columns declared as 'datetime', plus the schema timestamp column
    #   if it was matched.
    datetime_cols: set[str] = set()
    for col_name, feat_type in feature_map.items():
        if feat_type == "datetime" and col_name in df.columns:
            datetime_cols.add(col_name)
    # Also include the standard timestamp column if present
    if STANDARD_TIMESTAMP_COL in df.columns:
        datetime_cols.add(STANDARD_TIMESTAMP_COL)

    for dt_col in sorted(datetime_cols):
        df, ts_warnings = _cast_timestamp_column(df, dt_col, df_role)
        warnings.extend(ts_warnings)

    # ── Step 5b: Cast rating to float32 (interactions only) ──────────────────
    if df_role == "interactions":
        if STANDARD_RATING_COL in df.columns:
            df = _cast_rating(df)

    # ── Step 7: Remove columns declared as 'exclude' (or legacy 'drop') ─────
    df, exclude_warnings = _exclude_declared_columns(df, config, df_role)
    warnings.extend(exclude_warnings)

    n_rows, n_cols = df.shape
    logger.info(
        "[%s] Standardisation complete: %d rows × %d cols.", df_role, n_rows, n_cols
    )

    return df, warnings


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _rename_key_columns(
    df: pd.DataFrame,
    config: dict[str, Any],
    df_role: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Rename key columns using the schema mapping from the config.

    The schema defines candidate names for each key role. The first candidate
    that matches an actual column name (case-insensitively) wins.

    Required columns per role:
    - interactions: both ``user_id`` and ``item_id`` are required.
    - items: only ``item_id`` is required (``user_id`` is not applicable).
    - users: only ``user_id`` is required (``item_id`` is not applicable).

    Collision handling: if a candidate matches but the standard target name
    already exists as a different column (e.g., Steam has both ``username`` and
    ``user_id``), the existing conflicting column is dropped before renaming.

    Args:
        df: DataFrame with lowercased column names.
        config: Validated config dictionary.
        df_role: 'interactions', 'items', or 'users'.

    Returns:
        Tuple of (df_with_renamed_cols, warnings).

    Raises:
        ColumnNotFoundError: If required columns for the given role are missing.
    """
    warnings: list[str] = []
    schema = config.get("schema", {})
    available_cols = list(df.columns)

    # Determine which standard columns to look for, based on role
    role_candidates: dict[str, list[str]] = {}

    if df_role in ("interactions", "users"):
        role_candidates[STANDARD_USER_COL] = [
            c.lower() for c in schema.get("user_identifier", [])
        ]
    if df_role in ("interactions", "items"):
        role_candidates[STANDARD_ITEM_COL] = [
            c.lower() for c in schema.get("item_identifier", [])
        ]
    if df_role == "interactions":
        role_candidates[STANDARD_TIMESTAMP_COL] = [
            c.lower() for c in schema.get("timestamp", [])
        ]
        role_candidates[STANDARD_RATING_COL] = [
            c.lower() for c in schema.get("rating", [])
        ]

    # Required columns for each role
    required_per_role: dict[str, set[str]] = {
        "interactions": {STANDARD_USER_COL, STANDARD_ITEM_COL},
        "items": {STANDARD_ITEM_COL},
        "users": {STANDARD_USER_COL},
    }
    required_cols = required_per_role.get(df_role, set())

    rename_map: dict[str, str] = {}
    cols_to_drop_before_rename: list[str] = []
    missing_required: list[str] = []

    for standard_name, candidates in role_candidates.items():
        is_required = standard_name in required_cols
        matched = _find_first_match(candidates, available_cols)

        if matched is None:
            if is_required:
                missing_required.append(standard_name)
                logger.warning(
                    "[%s] Required column '%s' not found. Candidates tried: %s. "
                    "Available: %s",
                    df_role, standard_name, candidates, available_cols,
                )
            else:
                msg = (
                    f"[{df_role}] Optional column '{standard_name}' not found "
                    f"(candidates: {candidates}). It will be absent in output."
                )
                warnings.append(msg)
                logger.info(msg)
        elif matched != standard_name:
            # Collision: winning candidate will be renamed to standard_name,
            # but standard_name may already exist as a different column.
            if standard_name in available_cols and standard_name != matched:
                msg = (
                    f"[{df_role}] Column '{standard_name}' already exists but "
                    f"'{matched}' was chosen as the '{standard_name}' identifier "
                    f"(first-match rule). Dropping the existing '{standard_name}' "
                    f"column to avoid a duplicate."
                )
                warnings.append(msg)
                logger.warning(msg)
                cols_to_drop_before_rename.append(standard_name)

            rename_map[matched] = standard_name
            logger.info(
                "[%s] Renaming '%s' → '%s'.", df_role, matched, standard_name
            )
        # else: matched == standard_name, already has the right name — no action

    if missing_required:
        raise ColumnNotFoundError(
            missing_columns=missing_required,
            available_columns=available_cols,
            message=(
                f"[{df_role}] Required columns not found: {missing_required}\n"
                f"Candidates searched: "
                + ", ".join(
                    f"'{r}' → {role_candidates[r]}" for r in missing_required
                )
                + f"\nAvailable columns: {available_cols}"
            ),
        )

    # Drop any conflicting columns before renaming
    if cols_to_drop_before_rename:
        df = df.drop(columns=cols_to_drop_before_rename)

    if rename_map:
        df = df.rename(columns=rename_map)

    return df, warnings


def _find_first_match(candidates: list[str], available: list[str]) -> str | None:
    """Return the first candidate that exists in available (case-insensitive).

    Args:
        candidates: Ordered list of candidate column names (already lowercased).
        available: List of actual column names in the DataFrame (already lowercased).

    Returns:
        The matched column name from ``available``, or ``None`` if no match.
    """
    available_set = set(available)
    for cand in candidates:
        if cand in available_set:
            return cand
    return None


def _remove_null_ids(
    df: pd.DataFrame,
    df_role: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop rows where user_id or item_id is null.

    Args:
        df: DataFrame (post-rename, so key columns have standard names).
        df_role: Used for warning messages.

    Returns:
        Tuple of (cleaned_df, warnings).
    """
    warnings: list[str] = []
    original_len = len(df)

    cols_to_check = [
        c for c in [STANDARD_USER_COL, STANDARD_ITEM_COL] if c in df.columns
    ]

    if cols_to_check:
        null_mask = df[cols_to_check].isnull().any(axis=1)
        n_null = null_mask.sum()

        if n_null > 0:
            df = df[~null_mask].reset_index(drop=True)
            msg = (
                f"[{df_role}] Dropped {n_null:,} rows with null IDs "
                f"({n_null / original_len:.1%} of {original_len:,} rows). "
                f"Columns checked: {cols_to_check}."
            )
            warnings.append(msg)
            logger.info(msg)

    return df, warnings


def _cast_timestamp_column(
    df: pd.DataFrame,
    col: str,
    df_role: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Cast a column to datetime64[ns].

    If the column is numeric, the unit is inferred from the median value using
    a log10 heuristic:
        - median < 1e11  → seconds
        - median < 1e14  → milliseconds
        - median < 1e17  → microseconds
        - otherwise      → nanoseconds

    Args:
        df: DataFrame containing the target column.
        col: Name of the column to cast.
        df_role: Role label for log messages.

    Returns:
        Tuple of (df_with_cast_column, warnings).
    """
    warnings: list[str] = []

    series = df[col]

    # Already datetime — nothing to do
    if pd.api.types.is_datetime64_any_dtype(series):
        logger.debug("Timestamp column is already datetime; skipping cast.")
        return df, warnings

    # Attempt numeric cast
    numeric_series = pd.to_numeric(series, errors="coerce")
    n_numeric = numeric_series.notna().sum()
    n_total = len(series)

    if n_numeric / max(n_total, 1) >= 0.5:
        # Treat as numeric timestamp
        median_val = float(numeric_series.dropna().median())
        if median_val == 0:
            unit = "s"
        else:
            mag = math.log10(abs(median_val))
            if mag < 11:
                unit = "s"
            elif mag < 14:
                unit = "ms"
            elif mag < 17:
                unit = "us"
            else:
                unit = "ns"

        msg = f"[{df_role}] Detected numeric datetime unit for '{col}': '{unit}' (median={median_val:.0f})."
        logger.info(msg)
        warnings.append(msg)

        df[col] = pd.to_datetime(numeric_series, unit=unit, errors="coerce")
    else:
        # String timestamp — let pandas infer the format
        df[col] = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")

    n_failed = df[col].isna().sum()
    if n_failed > 0:
        msg = (
            f"[{df_role}] {n_failed:,} values in '{col}' could not be parsed "
            f"as datetime and were set to NaT ({n_failed / n_total:.1%} of rows)."
        )
        warnings.append(msg)
        logger.warning(msg)

    return df, warnings


def _cast_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Cast the rating column to float32.

    Args:
        df: DataFrame containing a ``rating`` column.

    Returns:
        DataFrame with rating cast to float32. Values that cannot be converted
        are coerced to NaN.
    """
    df[STANDARD_RATING_COL] = (
        pd.to_numeric(df[STANDARD_RATING_COL], errors="coerce")
        .astype("float32")
    )
    logger.debug("[interactions] Cast 'rating' to float32.")
    return df


def _exclude_declared_columns(
    df: pd.DataFrame,
    config: dict[str, Any],
    df_role: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove columns declared as 'exclude' (or legacy 'drop') in the config.

    Args:
        df: The (post-rename, post-cast) DataFrame.
        config: Validated config dictionary.
        df_role: 'interactions', 'items', or 'users'.

    Returns:
        Tuple of (df_without_excluded_cols, warnings).
    """
    warnings: list[str] = []
    feature_map = get_feature_map(config, df_role)

    # Collect columns marked for exclusion (accept both 'exclude' and legacy 'drop')
    exclude_cols = [
        col for col, ftype in feature_map.items()
        if ftype in ("exclude", "drop")
    ]

    if not exclude_cols:
        return df, warnings

    # Only exclude columns that actually exist (some may be absent in this dataset)
    cols_present = [c for c in exclude_cols if c in df.columns]
    cols_absent = [c for c in exclude_cols if c not in df.columns]

    if cols_absent:
        msg = (
            f"[{df_role}] Declared 'exclude' columns not found (will be ignored): "
            f"{cols_absent}."
        )
        warnings.append(msg)
        logger.debug(msg)

    if cols_present:
        df = df.drop(columns=cols_present)
        logger.info(
            "[%s] Excluded %d declared columns: %s.", df_role, len(cols_present), cols_present
        )

    return df, warnings


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run inspection helper
# ─────────────────────────────────────────────────────────────────────────────


def describe_standardisation_plan(
    df: pd.DataFrame,
    config: dict[str, Any],
    df_role: str,
) -> dict[str, Any]:
    """Compute what standardisation *would* do, without modifying the DataFrame.

    This is the backbone of ``--dry-run`` mode. It returns a structured dict
    describing the planned transformations, which the CLI can print.

    Args:
        df: The raw DataFrame (columns will be lowercased for analysis).
        config: Validated config dictionary.
        df_role: One of 'interactions', 'items', or 'users'.

    Returns:
        A dict with keys:
        - ``renames``: list of ``{"from": raw_col, "to": standard_col}`` dicts.
        - ``missing_optional``: list of standard column names not found.
        - ``missing_required``: list of required standard column names not found.
        - ``null_id_rows``: int — number of rows that would be dropped.
        - ``type_casts``: list of ``{"column": name, "from_dtype": ..., "to_dtype": ...}``.
        - ``drop_columns``: list of column names that would be dropped.
        - ``output_columns``: list of column names after all transforms.
        - ``warnings``: list of warning strings.
    """
    plan: dict[str, Any] = {
        "renames": [],
        "missing_optional": [],
        "missing_required": [],
        "null_id_rows": 0,
        "type_casts": [],
        "drop_columns": [],
        "output_columns": [],
        "warnings": [],
    }

    # Work on lowercased column names only (spaces → underscores)
    lowered_cols = [str(c).lower().replace(" ", "_") for c in df.columns]
    col_dtype_map = {str(c).lower().replace(" ", "_"): str(df[c].dtype) for c in df.columns}

    schema = config.get("schema", {})

    # Build role-specific candidate sets (mirrors _rename_key_columns logic)
    role_candidates: dict[str, list[str]] = {}
    if df_role in ("interactions", "users"):
        role_candidates[STANDARD_USER_COL] = [
            c.lower() for c in schema.get("user_identifier", [])
        ]
    if df_role in ("interactions", "items"):
        role_candidates[STANDARD_ITEM_COL] = [
            c.lower() for c in schema.get("item_identifier", [])
        ]
    if df_role == "interactions":
        role_candidates[STANDARD_TIMESTAMP_COL] = [
            c.lower() for c in schema.get("timestamp", [])
        ]
        role_candidates[STANDARD_RATING_COL] = [
            c.lower() for c in schema.get("rating", [])
        ]

    required_per_role: dict[str, set[str]] = {
        "interactions": {STANDARD_USER_COL, STANDARD_ITEM_COL},
        "items": {STANDARD_ITEM_COL},
        "users": {STANDARD_USER_COL},
    }
    required_cols = required_per_role.get(df_role, set())

    # Simulate rename step
    renamed_cols = list(lowered_cols)
    for standard_name, candidates in role_candidates.items():
        matched = _find_first_match(candidates, lowered_cols)
        is_required = standard_name in required_cols

        if matched is None:
            if is_required:
                plan["missing_required"].append(standard_name)
                plan["warnings"].append(
                    f"Required column '{standard_name}' not found. "
                    f"Candidates: {candidates}"
                )
            else:
                plan["missing_optional"].append(standard_name)
        elif matched != standard_name:
            # Handle collision: if target name already exists, it will be dropped
            if standard_name in renamed_cols and standard_name != matched:
                plan["warnings"].append(
                    f"Collision: '{standard_name}' already exists; will be dropped "
                    f"in favour of '{matched}' (first-match rule)."
                )
                renamed_cols.remove(standard_name)
            plan["renames"].append({"from": matched, "to": standard_name})
            idx = renamed_cols.index(matched)
            renamed_cols[idx] = standard_name

    # Simulate null ID removal (count only — don't actually filter)
    id_cols_in_df = [c for c in [STANDARD_USER_COL, STANDARD_ITEM_COL]
                     if c in renamed_cols]
    # Map back to original df columns for null check
    orig_to_renamed = {str(c).lower(): str(c).lower() for c in df.columns}
    for entry in plan["renames"]:
        orig_to_renamed[entry["from"]] = entry["to"]

    for std_col in id_cols_in_df:
        orig_col = next(
            (old for old, new in orig_to_renamed.items() if new == std_col),
            None,
        )
        if orig_col and orig_col in col_dtype_map:
            # Get the actual original column in df
            actual_col = next(
                (c for c in df.columns if str(c).lower() == orig_col), None
            )
            if actual_col is not None:
                plan["null_id_rows"] += int(df[actual_col].isna().sum())

    # Simulate type casts
    for std_col in [STANDARD_USER_COL, STANDARD_ITEM_COL]:
        if std_col in renamed_cols:
            orig_col = next(
                (e["from"] for e in plan["renames"] if e["to"] == std_col),
                std_col,
            )
            orig_dtype = col_dtype_map.get(orig_col, "unknown")
            if orig_dtype != "object":
                plan["type_casts"].append(
                    {"column": std_col, "from_dtype": orig_dtype, "to_dtype": "str (object)"}
                )

    if df_role == "interactions":
        if STANDARD_TIMESTAMP_COL in renamed_cols:
            orig_col = next(
                (e["from"] for e in plan["renames"] if e["to"] == STANDARD_TIMESTAMP_COL),
                STANDARD_TIMESTAMP_COL,
            )
            orig_dtype = col_dtype_map.get(orig_col, "unknown")
            plan["type_casts"].append(
                {"column": STANDARD_TIMESTAMP_COL, "from_dtype": orig_dtype,
                 "to_dtype": "datetime64[ns]"}
            )
        if STANDARD_RATING_COL in renamed_cols:
            orig_col = next(
                (e["from"] for e in plan["renames"] if e["to"] == STANDARD_RATING_COL),
                STANDARD_RATING_COL,
            )
            orig_dtype = col_dtype_map.get(orig_col, "unknown")
            plan["type_casts"].append(
                {"column": STANDARD_RATING_COL, "from_dtype": orig_dtype,
                 "to_dtype": "float32"}
            )

    # Simulate exclude step
    feature_map = get_feature_map(config, df_role)
    exclude_cols_declared = [
        col for col, ftype in feature_map.items()
        if ftype in ("exclude", "drop")
    ]
    exclude_cols_present = [c for c in exclude_cols_declared if c in renamed_cols]
    plan["drop_columns"] = exclude_cols_present  # key kept for CLI compat

    # Compute final output columns
    output_cols = [c for c in renamed_cols if c not in exclude_cols_present]
    plan["output_columns"] = output_cols

    return plan
