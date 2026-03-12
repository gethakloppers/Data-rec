"""Dataset profiler: produces a comprehensive profile dict for the web interface.

The profile dict is the **single source of truth** for the web UI — everything
the dataset detail page displays comes from this dict, serialised as
``dataset_profile.json``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from recdata.loaders.base_loader import get_feature_map

logger = logging.getLogger(__name__)


def profile_dataset(
    dfs: dict[str, pd.DataFrame],
    config: dict[str, Any],
    warnings: list[str] | None = None,
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a comprehensive profile dict for a standardised dataset.

    Args:
        dfs: Mapping of role to standardised DataFrame.
             Expected keys: ``"interactions"`` (required), ``"items"`` and
             ``"users"`` (optional).
        config: The validated dataset config dict.
        warnings: Processing warnings collected during standardisation.
        quality: Optional quality report dict (from ``quality_report()``).

    Returns:
        A JSON-serialisable profile dict with all sections populated.
    """
    if warnings is None:
        warnings = []

    inter = dfs.get("interactions")
    items = dfs.get("items")
    users = dfs.get("users")

    profile: dict[str, Any] = {}

    # ── Identity (from config) ────────────────────────────────────────────
    profile["dataset_name"] = config.get("dataset_name", "unknown")
    profile["domain"] = config.get("domain", "")
    profile["version"] = str(config.get("version", ""))
    profile["source_url"] = config.get("source_url", "")
    profile["description"] = (config.get("description", "") or "").strip()
    profile["citation"] = (config.get("citation", "") or "").strip()
    profile["processed_at"] = datetime.now(timezone.utc).isoformat()

    # ── Counts ────────────────────────────────────────────────────────────
    profile["counts"] = _compute_counts(inter, items, users)

    # ── Interaction characteristics ───────────────────────────────────────
    profile["interactions"] = _interaction_characteristics(inter)

    # ── Distribution statistics ───────────────────────────────────────────
    profile["distributions"] = _distribution_stats(inter)

    # ── Data taxonomy ─────────────────────────────────────────────────────
    profile["taxonomy"] = _build_taxonomy(dfs, config)

    # ── Stakeholder support ───────────────────────────────────────────────
    profile["stakeholder_support"] = _build_stakeholder_support(config)

    # ── Column inventory ──────────────────────────────────────────────────
    profile["columns"] = {}
    for role, df in dfs.items():
        feature_map = get_feature_map(config, role)
        profile["columns"][role] = _column_inventory(df, feature_map, role)

    # ── Raw file inventory (from config) ──────────────────────────────────
    profile["raw_files"] = _raw_file_inventory(dfs, config)

    # ── Quality report (optional) ─────────────────────────────────────────
    if quality:
        profile["quality"] = quality

    # ── Warnings ──────────────────────────────────────────────────────────
    profile["warnings"] = list(warnings)

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────────────


def _compute_counts(
    inter: pd.DataFrame | None,
    items: pd.DataFrame | None,
    users: pd.DataFrame | None,
) -> dict[str, Any]:
    """Compute basic dataset counts."""
    counts: dict[str, Any] = {
        "n_users": 0,
        "n_items": 0,
        "n_interactions": 0,
        "n_users_with_metadata": 0,
        "n_items_with_metadata": 0,
        "sparsity": 0.0,
        "density": 0.0,
    }

    if inter is not None:
        counts["n_interactions"] = len(inter)

        # Unique users: union of interactions + user metadata
        inter_users = set(inter["user_id"].unique()) if "user_id" in inter.columns else set()
        meta_users = set(users["user_id"].unique()) if users is not None and "user_id" in users.columns else set()
        all_users = inter_users | meta_users
        counts["n_users"] = len(all_users)

        # Unique items: union of interactions + item metadata
        inter_items = set(inter["item_id"].unique()) if "item_id" in inter.columns else set()
        meta_items = set(items["item_id"].unique()) if items is not None and "item_id" in items.columns else set()
        all_items = inter_items | meta_items
        counts["n_items"] = len(all_items)

    if users is not None:
        counts["n_users_with_metadata"] = len(users)
    if items is not None:
        counts["n_items_with_metadata"] = len(items)

    # Sparsity
    n_u = counts["n_users"]
    n_i = counts["n_items"]
    n_inter = counts["n_interactions"]
    if n_u > 0 and n_i > 0:
        density = n_inter / (n_u * n_i)
        counts["density"] = round(density, 8)
        counts["sparsity"] = round(1.0 - density, 8)

    return counts


def _interaction_characteristics(inter: pd.DataFrame | None) -> dict[str, Any]:
    """Characterise the interaction data."""
    chars: dict[str, Any] = {
        "type": "unknown",
        "has_timestamp": False,
        "has_rating": False,
        "rating_scale": None,
        "timestamp_range": None,
    }

    if inter is None:
        return chars

    # Rating
    if "rating" in inter.columns:
        chars["has_rating"] = True
        chars["type"] = "explicit"
        rating = inter["rating"].dropna()
        if len(rating) > 0:
            chars["rating_scale"] = {
                "min": _safe_scalar(rating.min()),
                "max": _safe_scalar(rating.max()),
                "mean": _safe_scalar(rating.mean()),
                "median": _safe_scalar(rating.median()),
                "std": _safe_scalar(rating.std()),
            }
    else:
        chars["type"] = "implicit"

    # Timestamp
    if "timestamp" in inter.columns:
        chars["has_timestamp"] = True
        ts = inter["timestamp"].dropna()
        if len(ts) > 0 and pd.api.types.is_datetime64_any_dtype(ts):
            chars["timestamp_range"] = {
                "earliest": str(ts.min()),
                "latest": str(ts.max()),
            }

    return chars


def _distribution_stats(inter: pd.DataFrame | None) -> dict[str, Any]:
    """Compute user and item interaction count distributions."""
    stats: dict[str, Any] = {
        "user_interaction_counts": None,
        "item_interaction_counts": None,
        "long_tail_ratio": None,
        "power_user_ratio": None,
    }

    if inter is None:
        return stats

    if "user_id" in inter.columns:
        user_counts = inter["user_id"].value_counts()
        stats["user_interaction_counts"] = _describe_series(user_counts)
        # Power user ratio: fraction of users with > 50 interactions
        n_power = (user_counts > 50).sum()
        stats["power_user_ratio"] = round(n_power / max(len(user_counts), 1), 4)

    if "item_id" in inter.columns:
        item_counts = inter["item_id"].value_counts()
        stats["item_interaction_counts"] = _describe_series(item_counts)
        # Long tail ratio: fraction of items with < 5 interactions
        n_tail = (item_counts < 5).sum()
        stats["long_tail_ratio"] = round(n_tail / max(len(item_counts), 1), 4)

    return stats


def _describe_series(s: pd.Series) -> dict[str, Any]:
    """Compute summary stats (min, max, mean, median, std, percentiles)."""
    return {
        "min": _safe_scalar(s.min()),
        "max": _safe_scalar(s.max()),
        "mean": _safe_scalar(s.mean()),
        "median": _safe_scalar(s.median()),
        "std": _safe_scalar(s.std()),
        "p25": _safe_scalar(s.quantile(0.25)),
        "p75": _safe_scalar(s.quantile(0.75)),
        "p95": _safe_scalar(s.quantile(0.95)),
    }


def _build_taxonomy(
    dfs: dict[str, pd.DataFrame], config: dict[str, Any]
) -> dict[str, Any]:
    """Build taxonomy audit from config and auto-detection.

    If the config has an explicit ``taxonomy`` section, use it directly.
    Otherwise, infer from column names in the processed DataFrames.
    """
    # Prefer explicit taxonomy from config
    config_taxonomy = config.get("taxonomy")
    if config_taxonomy and isinstance(config_taxonomy, dict):
        # Convert to the profile format (presence booleans)
        return _taxonomy_from_config(config_taxonomy, dfs)

    # Auto-detect from column names
    return _taxonomy_auto_detect(dfs, config)


def _taxonomy_from_config(
    taxonomy_config: dict[str, Any],
    dfs: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Convert explicit taxonomy config to a presence-based audit."""
    result: dict[str, Any] = {}

    # Users
    user_tax = taxonomy_config.get("users") or {}
    result["users"] = {
        "identifiers": bool(user_tax.get("user_id")),
        "demographics": bool(user_tax.get("demographics")),
        "additional_attributes": bool(user_tax.get("additional_attributes")),
    }

    # Items
    item_tax = taxonomy_config.get("items") or {}
    result["items"] = {
        "identifiers": bool(item_tax.get("item_id")),
        "descriptive_features": bool(item_tax.get("descriptive_features")),
        "content_features": bool(item_tax.get("content_features")),
        "provider_information": bool(item_tax.get("provider_upstream_info")),
    }

    # Interactions
    inter_tax = taxonomy_config.get("interactions") or {}
    result["interactions"] = {
        "explicit_feedback": bool(inter_tax.get("explicit_feedback")),
        "implicit_feedback": bool(inter_tax.get("implicit_feedback")),
        "timestamp": bool(inter_tax.get("timestamp")),
        "contextual_features": bool(inter_tax.get("interaction_context")),
        "session_data": bool(inter_tax.get("session_data")),
    }

    # Secondary
    result["secondary"] = {
        "system_data": False,
        "feedback_and_control": False,
        "external_knowledge": _has_columns_matching(
            dfs, ["social", "friend", "follow", "trust", "graph", "kg_"]
        ),
    }

    return result


def _taxonomy_auto_detect(
    dfs: dict[str, pd.DataFrame], config: dict[str, Any]
) -> dict[str, Any]:
    """Auto-detect taxonomy from column names in processed DataFrames."""
    all_cols = set()
    for df in dfs.values():
        all_cols.update(c.lower() for c in df.columns)

    result: dict[str, Any] = {
        "users": {
            "identifiers": "user_id" in all_cols,
            "demographics": _has_columns_matching(
                dfs, ["age", "gender", "occupation", "location", "country"]
            ),
            "additional_attributes": False,
        },
        "items": {
            "identifiers": "item_id" in all_cols,
            "descriptive_features": True,  # most datasets have some
            "content_features": _has_columns_with_type(config, "string") or _has_columns_with_type(config, "text"),
            "provider_information": _has_columns_matching(
                dfs,
                ["developer", "publisher", "author", "seller", "brand",
                 "manufacturer", "creator", "artist", "winery"],
            ),
        },
        "interactions": {
            "explicit_feedback": "rating" in all_cols,
            "implicit_feedback": False,
            "timestamp": "timestamp" in all_cols,
            "contextual_features": False,
            "session_data": _has_columns_matching(dfs, ["session"]),
        },
        "secondary": {
            "system_data": False,
            "feedback_and_control": False,
            "external_knowledge": _has_columns_matching(
                dfs, ["social", "friend", "follow", "trust", "graph", "kg_"]
            ),
        },
    }

    return result


def _build_stakeholder_support(config: dict[str, Any]) -> dict[str, Any]:
    """Build stakeholder support section from config."""
    support: dict[str, Any] = {}
    roles_config = config.get("stakeholder_roles", {}) or {}

    stakeholder_names = {
        "consumer": "Consumer",
        "system": "System",
        "provider": "Provider",
        "upstream": "Upstream",
        "downstream": "Downstream",
        "third_party": "Third-party",
    }

    for key, label in stakeholder_names.items():
        raw = roles_config.get(key)
        if raw is None:
            supported = False
            basis = "not declared in config"
        elif isinstance(raw, bool):
            supported = raw
            basis = "declared in config" if raw else "not supported per config"
        elif isinstance(raw, dict):
            supported = bool(raw.get("supported", False))
            features = raw.get("features", [])
            id_cols = raw.get("id", [])
            if supported and (features or id_cols):
                parts = []
                if id_cols:
                    parts.append(f"id: {', '.join(id_cols)}")
                if features:
                    parts.append(f"features: {', '.join(features)}")
                basis = "; ".join(parts)
            elif supported:
                basis = "enabled in config"
            else:
                basis = "not supported per config"
        else:
            supported = False
            basis = "unrecognised format in config"

        support[key] = {"supported": supported, "basis": basis}

    return support


def _column_inventory(
    df: pd.DataFrame,
    feature_map: dict[str, str],
    role: str,
) -> list[dict[str, Any]]:
    """Build a per-column inventory for the profile."""
    n_rows = len(df)
    inventory: list[dict[str, Any]] = []

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        n_unique = int(df[col].nunique())

        # Determine feature type
        if col in ("user_id", "item_id"):
            feature_type = "id"
        elif col == "timestamp":
            feature_type = "timestamp"
        elif col == "rating":
            feature_type = "rating"
        else:
            feature_type = feature_map.get(col, "unknown")

        entry: dict[str, Any] = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": null_count,
            "null_pct": round(null_count / max(n_rows, 1) * 100, 2),
            "feature_type": feature_type,
            "n_unique": n_unique,
        }

        # Sample values for object columns
        series = df[col].dropna()
        if series.dtype == object and len(series) > 0:
            samples = series.head(5).tolist()
            entry["sample_values"] = [str(v)[:80] for v in samples]

        # Detect if column contains list values
        if series.dtype == object and len(series) > 0:
            first_vals = series.head(20).astype(str)
            bracket_match = first_vals.str.match(r"^\s*\[.*\]\s*$")
            entry["is_list_column"] = bool(bracket_match.mean() > 0.3)
        else:
            entry["is_list_column"] = False

        inventory.append(entry)

    return inventory


def _raw_file_inventory(
    dfs: dict[str, pd.DataFrame], config: dict[str, Any]
) -> dict[str, Any]:
    """Build raw file metadata from config and loaded DataFrames."""
    files_config = config.get("files", {}) or {}
    inventory: dict[str, Any] = {}

    for role in ("interactions", "items", "users"):
        file_def = files_config.get(role)
        if file_def is None:
            inventory[role] = None
            continue

        filename = file_def if isinstance(file_def, str) else file_def.get("filename", "")
        fmt = file_def.get("format", "") if isinstance(file_def, dict) else ""
        df = dfs.get(role)

        entry: dict[str, Any] = {"filename": filename}
        if fmt:
            entry["format"] = fmt
        if df is not None:
            entry["n_rows"] = len(df)
            entry["n_cols"] = len(df.columns)

        inventory[role] = entry

    return inventory


def build_id_mappings(dfs: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    """Build contiguous integer ID mappings from string IDs.

    Collects unique user and item IDs from the union of interactions +
    metadata tables, then maps each to a contiguous integer starting from 1
    (0 reserved as padding index).

    The DataFrames are NOT modified — mappings are returned as separate dicts
    to be saved as JSON artifacts.

    Args:
        dfs: Mapping of role to standardised DataFrame.

    Returns:
        A dict with keys ``"user_map"`` and ``"item_map"``, each mapping
        original string IDs to integer indices.
    """
    inter = dfs.get("interactions")
    items = dfs.get("items")
    users = dfs.get("users")

    # Collect all user IDs
    user_ids: set[str] = set()
    if inter is not None and "user_id" in inter.columns:
        user_ids.update(inter["user_id"].unique())
    if users is not None and "user_id" in users.columns:
        user_ids.update(users["user_id"].unique())

    # Collect all item IDs
    item_ids: set[str] = set()
    if inter is not None and "item_id" in inter.columns:
        item_ids.update(inter["item_id"].unique())
    if items is not None and "item_id" in items.columns:
        item_ids.update(items["item_id"].unique())

    # Build contiguous mappings (sorted for reproducibility)
    user_map = {uid: idx + 1 for idx, uid in enumerate(sorted(user_ids))}
    item_map = {iid: idx + 1 for idx, iid in enumerate(sorted(item_ids))}

    logger.info(
        "Built ID mappings: %d users → [1, %d], %d items → [1, %d]",
        len(user_map), len(user_map), len(item_map), len(item_map),
    )

    return {"user_map": user_map, "item_map": item_map}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _has_columns_matching(
    dfs: dict[str, pd.DataFrame], patterns: list[str]
) -> bool:
    """Check if any column in any DataFrame contains one of the given patterns."""
    for df in dfs.values():
        for col in df.columns:
            col_lower = col.lower()
            for pattern in patterns:
                if pattern in col_lower:
                    return True
    return False


def _has_columns_with_type(config: dict[str, Any], dtype: str) -> bool:
    """Check if any feature declaration in the config uses the given type."""
    for key in ("item_features", "interaction_features", "user_features"):
        features = config.get(key)
        if features and isinstance(features, dict):
            cols = features.get(dtype, [])
            if cols:
                return True
    return False


def _safe_scalar(val: Any) -> float | None:
    """Convert a numpy scalar to a JSON-safe Python float."""
    if val is None:
        return None
    v = float(val)
    if np.isnan(v) or np.isinf(v):
        return None
    return round(v, 4)
