"""Quality report for standardised recommender system DataFrames.

Produces a diagnostic dict that flags data issues after auto-standardisation:
type mismatches, list-valued columns, nulls, duplicates, ID coverage, and
per-column statistics.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from recdata.loaders.base_loader import get_feature_map

logger = logging.getLogger(__name__)


def quality_report(
    dfs: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate a quality report for standardised DataFrames.

    Args:
        dfs: Mapping of role to DataFrame, e.g.
             ``{"interactions": df_inter, "items": df_items}``.
        config: The validated dataset config dict.

    Returns:
        A JSON-serialisable dict with sections: ``type_audit``,
        ``list_detection``, ``null_analysis``, ``duplicate_analysis``,
        ``id_coverage``, ``column_statistics``.
    """
    report: dict[str, Any] = {
        "dataset_name": config.get("dataset_name", "unknown"),
        "type_audit": {},
        "mixed_types": {},
        "list_detection": {},
        "null_analysis": {},
        "duplicate_analysis": {},
        "id_coverage": {},
        "column_statistics": {},
    }

    for role, df in dfs.items():
        feature_map = get_feature_map(config, role)
        report["type_audit"][role] = _type_audit(df, feature_map)
        report["mixed_types"][role] = _mixed_type_detection(df)
        report["list_detection"][role] = _list_detection(df)
        report["null_analysis"][role] = _null_analysis(df)
        report["column_statistics"][role] = _column_statistics(df)

    report["duplicate_analysis"] = _duplicate_analysis(dfs)
    report["id_coverage"] = _id_coverage(dfs)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────────────


def _type_audit(
    df: pd.DataFrame, feature_map: dict[str, str]
) -> list[dict[str, Any]]:
    """Compare declared types against actual pandas dtypes.

    Returns a list of per-column audit entries.
    """
    results: list[dict[str, Any]] = []

    for col in df.columns:
        declared = feature_map.get(col, "")
        actual_dtype = str(df[col].dtype)

        mismatch = False
        note = ""

        if declared == "float":
            if df[col].dtype.kind not in ("f", "i", "u"):
                mismatch = True
                note = f"Declared float but actual dtype is {actual_dtype}"
        elif declared == "datetime":
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                mismatch = True
                note = f"Declared datetime but actual dtype is {actual_dtype}"
        elif declared == "object":
            if df[col].dtype != object and not pd.api.types.is_string_dtype(df[col]):
                mismatch = True
                note = f"Declared object but actual dtype is {actual_dtype}"

        results.append({
            "column": col,
            "declared_type": declared or "(undeclared)",
            "actual_dtype": actual_dtype,
            "mismatch": mismatch,
            "note": note,
        })

    return results


def _mixed_type_detection(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect columns with mixed value types.

    Inspects ``object``-dtype columns by sampling values and classifying each
    into its native Python type (int, float, bool, str, etc.).  A column is
    flagged as mixed when more than one value type is present.

    For each mixed column the report includes:
    - A breakdown of value types and their percentages.
    - The **dominant type** (the type that appears most often).
    - Example values for each type detected.

    This helps users identify columns where manual cleanup or explicit type
    casting may be needed (e.g., a column that is mostly integers but has a
    few string entries like ``"N/A"``).
    """
    results: list[dict[str, Any]] = []

    for col in df.columns:
        # Only object columns can have mixed types; numeric/datetime are uniform
        if df[col].dtype != object:
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        # Sample up to 1000 values for performance on large DataFrames
        sample = series.sample(n=min(1000, len(series)), random_state=42)

        type_counts: dict[str, int] = {}
        type_examples: dict[str, list[str]] = {}

        for val in sample:
            vtype = _classify_value(val)
            type_counts[vtype] = type_counts.get(vtype, 0) + 1
            if vtype not in type_examples:
                type_examples[vtype] = []
            if len(type_examples[vtype]) < 3:
                type_examples[vtype].append(str(val)[:80])

        # Only report columns with genuinely mixed types
        # (ignore "string" as the sole type — that's expected for object cols)
        if len(type_counts) <= 1:
            continue

        n_sampled = len(sample)
        dominant_type = max(type_counts, key=type_counts.get)  # type: ignore[arg-type]

        breakdown: list[dict[str, Any]] = []
        for vtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            breakdown.append({
                "type": vtype,
                "count": count,
                "pct": round(count / n_sampled * 100, 1),
                "examples": type_examples[vtype],
            })

        results.append({
            "column": col,
            "n_sampled": n_sampled,
            "n_types": len(type_counts),
            "dominant_type": dominant_type,
            "dominant_pct": round(type_counts[dominant_type] / n_sampled * 100, 1),
            "breakdown": breakdown,
        })

    return results


def _classify_value(val: Any) -> str:
    """Classify a single value into its native type category.

    Attempts to parse the string representation into numeric types before
    falling back to ``"string"``.  This mirrors what pandas does internally
    when it encounters mixed types in a CSV column.

    Categories returned: ``"integer"``, ``"float"``, ``"boolean"``,
    ``"datetime"``, ``"list"``, ``"string"``, or ``"other"``.
    """
    # Already typed (e.g., from JSONL loading)
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, int):
        return "integer"
    if isinstance(val, float):
        return "float" if not np.isnan(val) else "null"
    if isinstance(val, (list, tuple)):
        return "list"
    if isinstance(val, dict):
        return "dict"

    # String-based classification
    s = str(val).strip()
    if not s:
        return "empty"

    # Boolean-like
    if s.lower() in ("true", "false"):
        return "boolean"

    # Integer
    try:
        int(s)
        return "integer"
    except (ValueError, OverflowError):
        pass

    # Float
    try:
        float(s)
        return "float"
    except (ValueError, OverflowError):
        pass

    # List literal
    if s.startswith("[") and s.endswith("]"):
        return "list"

    return "string"


def _list_detection(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Scan object columns for list-valued entries.

    Detects Python list literals, JSON arrays, and delimited strings.
    """
    results: list[dict[str, Any]] = []

    for col in df.columns:
        if df[col].dtype != object:
            continue

        sample = df[col].dropna().head(200).astype(str)
        if sample.empty:
            continue

        info: dict[str, Any] = {
            "column": col,
            "is_list_column": False,
            "format": None,
            "delimiter": None,
            "sample": None,
        }

        # Check for Python list repr: "['a', 'b']"
        bracket_match = sample.str.match(r"^\s*\[.*\]\s*$")
        if bracket_match.mean() > 0.3:
            # Try to parse a sample value
            first_match = sample[bracket_match].iloc[0] if bracket_match.any() else None
            if first_match:
                parsed = _try_parse_list(first_match)
                if parsed is not None and len(parsed) > 1:
                    info["is_list_column"] = True
                    info["format"] = "bracket_list"
                    info["sample"] = str(parsed[:5])

        # Check for delimiter-separated values
        if not info["is_list_column"]:
            for sep in ["|", ";", ","]:
                has_sep = sample.str.contains(re.escape(sep), regex=True)
                if has_sep.mean() > 0.5:
                    # Verify it's not just commas in numbers
                    if sep == ",":
                        numeric_comma = sample.str.match(r"^\d+,\d+$")
                        if numeric_comma.mean() > 0.3:
                            continue
                    info["is_list_column"] = True
                    info["format"] = "delimited"
                    info["delimiter"] = sep
                    first_val = sample.iloc[0]
                    info["sample"] = first_val[:80] if len(first_val) > 80 else first_val
                    break

        if info["is_list_column"]:
            results.append(info)

    return results


def _null_analysis(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-column null counts and percentages."""
    n_rows = len(df)
    results: list[dict[str, Any]] = []

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = round(null_count / max(n_rows, 1) * 100, 2)
        results.append({
            "column": col,
            "null_count": null_count,
            "null_pct": null_pct,
            "high_null": null_pct > 50,
        })

    return results


def _duplicate_analysis(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Count duplicates in interactions (by user_id+item_id) and IDs in items/users."""
    result: dict[str, Any] = {}

    if "interactions" in dfs:
        inter = dfs["interactions"]
        if "user_id" in inter.columns and "item_id" in inter.columns:
            n_total = len(inter)
            n_unique_pairs = inter[["user_id", "item_id"]].drop_duplicates().shape[0]
            n_dup = n_total - n_unique_pairs
            result["interaction_duplicates"] = {
                "total_rows": n_total,
                "unique_user_item_pairs": n_unique_pairs,
                "duplicate_rows": n_dup,
                "duplicate_pct": round(n_dup / max(n_total, 1) * 100, 2),
            }

    for role in ("items", "users"):
        if role not in dfs:
            continue
        df = dfs[role]
        id_col = "item_id" if role == "items" else "user_id"
        if id_col in df.columns:
            n_total = len(df)
            n_unique = df[id_col].nunique()
            n_dup = n_total - n_unique
            result[f"{role}_id_duplicates"] = {
                "total_rows": n_total,
                "unique_ids": n_unique,
                "duplicate_rows": n_dup,
                "duplicate_pct": round(n_dup / max(n_total, 1) * 100, 2),
            }

    return result


def _id_coverage(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Check ID overlap between interactions and metadata tables."""
    result: dict[str, Any] = {}

    inter = dfs.get("interactions")
    if inter is None:
        return result

    # Item coverage
    if "item_id" in inter.columns:
        inter_items = set(inter["item_id"].unique())
        if "items" in dfs and "item_id" in dfs["items"].columns:
            meta_items = set(dfs["items"]["item_id"].unique())
            in_both = inter_items & meta_items
            in_inter_only = inter_items - meta_items
            in_meta_only = meta_items - inter_items
            result["items"] = {
                "in_interactions": len(inter_items),
                "in_metadata": len(meta_items),
                "in_both": len(in_both),
                "in_interactions_only": len(in_inter_only),
                "in_metadata_only": len(in_meta_only),
                "coverage_pct": round(
                    len(in_both) / max(len(inter_items), 1) * 100, 2
                ),
            }

    # User coverage
    if "user_id" in inter.columns:
        inter_users = set(inter["user_id"].unique())
        if "users" in dfs and "user_id" in dfs["users"].columns:
            meta_users = set(dfs["users"]["user_id"].unique())
            in_both = inter_users & meta_users
            in_inter_only = inter_users - meta_users
            in_meta_only = meta_users - inter_users
            result["users"] = {
                "in_interactions": len(inter_users),
                "in_metadata": len(meta_users),
                "in_both": len(in_both),
                "in_interactions_only": len(in_inter_only),
                "in_metadata_only": len(in_meta_only),
                "coverage_pct": round(
                    len(in_both) / max(len(inter_users), 1) * 100, 2
                ),
            }

    return result


def _column_statistics(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-column summary statistics."""
    results: list[dict[str, Any]] = []
    n_rows = len(df)

    for col in df.columns:
        stats: dict[str, Any] = {
            "column": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(int(df[col].isna().sum()) / max(n_rows, 1) * 100, 2),
            "n_unique": int(df[col].nunique()),
        }

        series = df[col].dropna()

        if series.dtype.kind in ("f", "i", "u"):
            # Numeric
            stats["min"] = _safe_scalar(series.min())
            stats["max"] = _safe_scalar(series.max())
            stats["mean"] = _safe_scalar(series.mean())
            stats["median"] = _safe_scalar(series.median())
            stats["std"] = _safe_scalar(series.std())
        elif pd.api.types.is_datetime64_any_dtype(series):
            # Datetime
            stats["min"] = str(series.min())
            stats["max"] = str(series.max())
        elif series.dtype == object:
            # Object/string — top values and sample
            top5 = series.value_counts().head(5)
            stats["top_values"] = {str(k): int(v) for k, v in top5.items()}
            sample_vals = series.head(3).tolist()
            stats["sample_values"] = [
                str(v)[:100] for v in sample_vals
            ]

        results.append(stats)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def summarise_mixed_types(report: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Extract a compact summary of mixed-type columns from a quality report.

    Returns a dict keyed by role, mapping column names to a human-readable
    note suitable for inclusion in a YAML config or markdown report.

    Example output::

        {
            "interactions": {
                "vintage": "mixed types: 98.3% integer, 1.7% string (e.g. 'N.V.')"
            }
        }
    """
    summary: dict[str, dict[str, str]] = {}
    mixed_section = report.get("mixed_types", {})

    for role, entries in mixed_section.items():
        if not entries:
            continue
        role_notes: dict[str, str] = {}
        for entry in entries:
            col = entry["column"]
            parts = []
            for b in entry["breakdown"]:
                example = b["examples"][0] if b["examples"] else "?"
                parts.append(f"{b['pct']}% {b['type']} (e.g. '{example}')")
            role_notes[col] = f"mixed types: {', '.join(parts)}"
        if role_notes:
            summary[role] = role_notes

    return summary


def _try_parse_list(val: str) -> list | None:
    """Try to parse a string as a Python list literal or JSON array."""
    val = val.strip()
    if not val.startswith("["):
        return None
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return None


def _safe_scalar(val: Any) -> float | None:
    """Convert a numpy scalar to a JSON-safe Python float."""
    if val is None:
        return None
    v = float(val)
    if np.isnan(v) or np.isinf(v):
        return None
    return round(v, 4)
