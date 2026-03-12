"""Export standardised datasets to Parquet, JSON profile, and markdown report.

Output structure::

    {output_path}/{dataset_name}/
    +-- processed/
    |   +-- interactions.parquet
    |   +-- items.parquet
    |   +-- users.parquet
    +-- mappings/
    |   +-- user_map.json
    |   +-- item_map.json
    +-- profile/
        +-- dataset_profile.json
        +-- quality_report.json
        +-- dataset_report.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def export_dataset(
    dfs: dict[str, pd.DataFrame],
    profile: dict[str, Any],
    id_mappings: dict[str, dict[str, int]],
    quality: dict[str, Any] | None,
    output_dir: str | Path,
) -> Path:
    """Export all dataset artifacts to disk.

    Args:
        dfs: Mapping of role to standardised DataFrame.
        profile: The full profile dict from ``profile_dataset()``.
        id_mappings: The user/item ID mappings from ``build_id_mappings()``.
        quality: Optional quality report dict.
        output_dir: Root output directory. A subdirectory named after the
            dataset will be created.

    Returns:
        Path to the created dataset output directory.
    """
    dataset_name = profile.get("dataset_name", "unknown")
    base_dir = Path(output_dir) / dataset_name
    processed_dir = base_dir / "processed"
    mappings_dir = base_dir / "mappings"
    profile_dir = base_dir / "profile"

    # Create directories
    for d in (processed_dir, mappings_dir, profile_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Parquet files ─────────────────────────────────────────────────────
    for role, df in dfs.items():
        dest = processed_dir / f"{role}.parquet"
        _save_parquet(df, dest)

    # ── ID mappings ───────────────────────────────────────────────────────
    _save_json(id_mappings.get("user_map", {}), mappings_dir / "user_map.json")
    _save_json(id_mappings.get("item_map", {}), mappings_dir / "item_map.json")

    # ── Profile JSON ──────────────────────────────────────────────────────
    _save_json(profile, profile_dir / "dataset_profile.json")

    # ── Quality report JSON ───────────────────────────────────────────────
    if quality:
        _save_json(quality, profile_dir / "quality_report.json")

    # ── Markdown report ───────────────────────────────────────────────────
    md = generate_markdown_report(profile)
    md_path = profile_dir / "dataset_report.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info("Wrote markdown report: %s", md_path)

    logger.info("Export complete: %s", base_dir)
    return base_dir


# ─────────────────────────────────────────────────────────────────────────────
# Parquet helpers
# ─────────────────────────────────────────────────────────────────────────────


def _save_parquet(df: pd.DataFrame, dest: Path) -> None:
    """Save a DataFrame to Parquet with proper type handling.

    - Object columns are cast to ``pd.StringDtype()`` for clean Arrow storage.
    - Snappy compression (pyarrow default).
    """
    df_out = df.copy()

    for col in df_out.columns:
        if df_out[col].dtype == object:
            df_out[col] = df_out[col].astype(pd.StringDtype())

    df_out.to_parquet(dest, engine="pyarrow", index=False)
    size_mb = round(dest.stat().st_size / (1024 * 1024), 1)
    logger.info("Saved %s (%d rows, %.1f MB)", dest.name, len(df_out), size_mb)


def _save_json(data: Any, dest: Path) -> None:
    """Save data to a JSON file with readable formatting."""
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    logger.info("Saved %s", dest.name)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────


def generate_markdown_report(profile: dict[str, Any]) -> str:
    """Generate a clean markdown report from the profile dict.

    Sections:
    1. Header + metadata
    2. Summary stats table
    3. Data taxonomy table
    4. Stakeholder support table
    5. Distribution statistics
    6. Column inventory per file
    7. Processing warnings
    """
    lines: list[str] = []

    # ── 1. Header ─────────────────────────────────────────────────────────
    name = profile.get("dataset_name", "Unknown")
    lines.append(f"# {name}")
    lines.append("")

    domain = profile.get("domain", "")
    version = profile.get("version", "")
    url = profile.get("source_url", "")
    processed = profile.get("processed_at", "")[:10]

    meta_parts = []
    if domain:
        meta_parts.append(f"**Domain:** {domain}")
    if version:
        meta_parts.append(f"**Version:** {version}")
    if url:
        meta_parts.append(f"**Source:** [{url}]({url})")
    if processed:
        meta_parts.append(f"**Processed:** {processed}")
    if meta_parts:
        lines.append(" | ".join(meta_parts))
        lines.append("")

    desc = profile.get("description", "")
    if desc:
        lines.append(desc)
        lines.append("")

    # ── 2. Summary stats ──────────────────────────────────────────────────
    lines.append("## Summary")
    lines.append("")
    counts = profile.get("counts", {})
    inter_chars = profile.get("interactions", {})

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Users | {_fmt_number(counts.get('n_users', 0))} |")
    lines.append(f"| Items | {_fmt_number(counts.get('n_items', 0))} |")
    lines.append(f"| Interactions | {_fmt_number(counts.get('n_interactions', 0))} |")
    lines.append(f"| Sparsity | {counts.get('sparsity', 0):.6f} |")
    lines.append(f"| Density | {counts.get('density', 0):.8f} |")
    lines.append(f"| Has timestamp | {'Yes' if inter_chars.get('has_timestamp') else 'No'} |")
    lines.append(f"| Has rating | {'Yes' if inter_chars.get('has_rating') else 'No'} |")
    lines.append(f"| Feedback type | {inter_chars.get('type', 'unknown')} |")

    rating_scale = inter_chars.get("rating_scale")
    if rating_scale:
        lines.append(
            f"| Rating range | {rating_scale['min']} – {rating_scale['max']} "
            f"(mean {rating_scale['mean']:.2f}) |"
        )

    ts_range = inter_chars.get("timestamp_range")
    if ts_range:
        lines.append(f"| Time span | {ts_range['earliest'][:10]} to {ts_range['latest'][:10]} |")

    lines.append("")

    # ── 3. Taxonomy ───────────────────────────────────────────────────────
    lines.append("## Data Taxonomy")
    lines.append("")
    taxonomy = profile.get("taxonomy", {})
    lines.append("| Category | Element | Present |")
    lines.append("|----------|---------|---------|")

    _TAX_LABELS = {
        "users": {
            "identifiers": "Identifiers",
            "demographics": "Demographics",
            "additional_attributes": "Additional Attributes",
        },
        "items": {
            "identifiers": "Identifiers",
            "descriptive_features": "Descriptive Features",
            "content_features": "Content Features",
            "provider_information": "Provider Information",
        },
        "interactions": {
            "explicit_feedback": "Explicit Feedback",
            "implicit_feedback": "Implicit Feedback",
            "timestamp": "Timestamp",
            "contextual_features": "Contextual Features",
            "session_data": "Session Data",
        },
        "secondary": {
            "system_data": "System Data",
            "feedback_and_control": "Feedback & Control",
            "external_knowledge": "External Knowledge",
        },
    }

    for group, elements in _TAX_LABELS.items():
        group_data = taxonomy.get(group, {})
        group_label = group.replace("_", " ").title()
        for key, label in elements.items():
            present = group_data.get(key, False)
            mark = "Yes" if present else "No"
            lines.append(f"| {group_label} | {label} | {mark} |")

    lines.append("")

    # ── 4. Stakeholder support ────────────────────────────────────────────
    lines.append("## Stakeholder Support")
    lines.append("")
    support = profile.get("stakeholder_support", {})
    lines.append("| Role | Supported | Basis |")
    lines.append("|------|-----------|-------|")

    for role in ("consumer", "system", "provider", "upstream", "downstream", "third_party"):
        s = support.get(role, {})
        mark = "Yes" if s.get("supported") else "No"
        basis = s.get("basis", "")
        lines.append(f"| {role.replace('_', ' ').title()} | {mark} | {basis} |")

    lines.append("")

    # ── 5. Distribution statistics ────────────────────────────────────────
    lines.append("## Distribution Statistics")
    lines.append("")
    dists = profile.get("distributions", {})

    for label, key in [
        ("User interactions per user", "user_interaction_counts"),
        ("Item interactions per item", "item_interaction_counts"),
    ]:
        stats = dists.get(key)
        if stats:
            lines.append(f"**{label}:**")
            lines.append("")
            lines.append("| Stat | Value |")
            lines.append("|------|-------|")
            for sk in ("min", "max", "mean", "median", "std", "p25", "p75", "p95"):
                v = stats.get(sk)
                if v is not None:
                    lines.append(f"| {sk.upper()} | {_fmt_number(v)} |")
            lines.append("")

    lt = dists.get("long_tail_ratio")
    pu = dists.get("power_user_ratio")
    if lt is not None:
        lines.append(f"- **Long-tail ratio** (items with < 5 interactions): {lt:.2%}")
    if pu is not None:
        lines.append(f"- **Power-user ratio** (users with > 50 interactions): {pu:.2%}")
    lines.append("")

    # ── 6. Column inventory ───────────────────────────────────────────────
    lines.append("## Column Inventory")
    lines.append("")
    columns_section = profile.get("columns", {})

    for role in ("interactions", "items", "users"):
        cols = columns_section.get(role, [])
        if not cols:
            continue

        lines.append(f"### {role.title()}")
        lines.append("")
        lines.append("| Column | Type | Dtype | Null % | Unique |")
        lines.append("|--------|------|-------|--------|--------|")

        for c in cols:
            lines.append(
                f"| {c['name']} | {c.get('feature_type', '')} | "
                f"{c['dtype']} | {c['null_pct']:.1f}% | "
                f"{_fmt_number(c['n_unique'])} |"
            )

        lines.append("")

    # ── 7. Warnings ───────────────────────────────────────────────────────
    warns = profile.get("warnings", [])
    if warns:
        lines.append("## Processing Warnings")
        lines.append("")
        for w in warns:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


def _fmt_number(n: float | int) -> str:
    """Format a number with comma separators or K/M suffixes for display."""
    if isinstance(n, float):
        if n == int(n) and abs(n) < 1e15:
            n = int(n)
        else:
            return f"{n:,.2f}"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:,.1f}M"
    if abs(n) >= 10_000:
        return f"{n / 1_000:,.1f}K"
    return f"{n:,}"
