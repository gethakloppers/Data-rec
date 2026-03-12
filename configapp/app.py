"""ConfigApp: Web-based dataset configuration tool.

A Flask app that helps configure recommender system datasets interactively.
Scans a folder for data files, previews their contents, and lets the user
configure column types, schema roles, and stakeholder mappings. Outputs a
YAML config file compatible with the recdata pipeline.

Usage::

    python configapp/app.py --data-path /path/to/data --output-path /path/to/configs

Or with environment variables::

    DATA_PATH=/data OUTPUT_PATH=/configs flask --app configapp/app.py run --port 5001
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path

import yaml
from flask import Flask, jsonify, render_template, request

# Import from the recdata library (no duplication)
import sys

# Ensure recdata is importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recdata.loaders.file_reader import detect_format, read_file
from recdata.exceptions import DatasetLoadError

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Path configuration — set via CLI args or environment variables
app.config["DATA_PATH"] = os.environ.get("DATA_PATH", "")
app.config["OUTPUT_PATH"] = os.environ.get("OUTPUT_PATH", "")

# File extensions we consider as potential data files
_DATA_EXTENSIONS = {
    ".csv", ".tsv", ".dat", ".json", ".jsonl", ".ndjson",
    ".parquet", ".gz", ".zip", ".tar", ".tgz",
    ".tar.gz", ".tar.bz2", ".tar.xz",
}


def _is_data_file(path: Path) -> bool:
    """Check if a file has a recognised data extension."""
    name_lower = path.name.lower()
    # Check compound extensions first
    for ext in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if name_lower.endswith(ext):
            return True
    return path.suffix.lower() in _DATA_EXTENSIONS


def _file_size_mb(path: Path) -> float:
    """Get file size in MB, rounded to 1 decimal."""
    return round(path.stat().st_size / (1024 * 1024), 1)


def _list_archive_contents(filepath: Path, fmt: str) -> list[dict]:
    """List data files inside a tar/zip archive with format and size.

    Returns a list of dicts with keys: ``filename``, ``format``, ``size_mb``.
    """
    import tarfile as _tarfile
    import zipfile as _zipfile

    results: list[dict] = []

    if fmt == "tar":
        with _tarfile.open(filepath, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                inner = Path(member.name)
                # Skip hidden / macOS system files
                if inner.name.startswith(".") or inner.name.startswith("__"):
                    continue
                if not _is_data_file(inner):
                    continue
                try:
                    inner_fmt = detect_format(inner)
                except DatasetLoadError:
                    inner_fmt = "unknown"
                results.append({
                    "filename": inner.name,
                    "format": inner_fmt,
                    "size_mb": round(member.size / (1024 * 1024), 1),
                })

    elif fmt == "zip":
        with _zipfile.ZipFile(filepath, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                inner = Path(info.filename)
                if inner.name.startswith(".") or inner.name.startswith("__"):
                    continue
                if not _is_data_file(inner):
                    continue
                try:
                    inner_fmt = detect_format(inner)
                except DatasetLoadError:
                    inner_fmt = "unknown"
                results.append({
                    "filename": inner.name,
                    "format": inner_fmt,
                    "size_mb": round(info.file_size / (1024 * 1024), 1),
                })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    """Serve the main wizard page."""
    return render_template(
        "index.html",
        default_path=app.config["DATA_PATH"],
        output_path=app.config["OUTPUT_PATH"],
    )


@app.route("/api/scan-folder", methods=["POST"])
def scan_folder():
    """Scan a folder for data files and return their metadata.

    Request JSON::

        {"folder_path": "/path/to/dataset"}

    Response JSON::

        {"files": [{"filename": "reviews.json", "format": "jsonl", "size_mb": 312.4}, ...]}
    """
    data = request.get_json(silent=True) or {}
    folder_path = data.get("folder_path", "").strip()

    if not folder_path:
        return jsonify({"error": "folder_path is required"}), 400

    path = Path(folder_path)
    if not path.is_dir():
        return jsonify({"error": f"Not a directory: {folder_path}"}), 400

    files = []
    for f in sorted(path.iterdir()):
        if f.is_file() and not f.name.startswith(".") and _is_data_file(f):
            try:
                fmt = detect_format(f)
            except DatasetLoadError:
                fmt = "unknown"

            # Expand tar/zip archives into individual inner files
            if fmt in ("tar", "zip"):
                try:
                    inner_files = _list_archive_contents(f, fmt)
                    if inner_files:
                        for inner in inner_files:
                            files.append({
                                "filename": inner["filename"],
                                "format": inner["format"],
                                "size_mb": inner["size_mb"],
                                "archive": f.name,
                            })
                        continue  # skip appending the archive itself
                except Exception:
                    logger.warning("Failed to list contents of '%s', showing as archive", f.name)

            files.append({
                "filename": f.name,
                "format": fmt,
                "size_mb": _file_size_mb(f),
            })

    return jsonify({"files": files})


@app.route("/api/preview-file", methods=["POST"])
def preview_file():
    """Preview a file: read first rows and return column info + sample data.

    Request JSON::

        {"folder_path": "/path/to/dataset", "filename": "reviews.json"}

    Response JSON::

        {
          "columns": ["username", "product_id", ...],
          "dtypes": {"username": "object", ...},
          "rows": [["val1", "val2", ...], ...],
          "suggested_types": {"username": "token", "hours": "float", ...},
          "suggested_separators": {"genres": "|"},
          "n_rows": 10
        }
    """
    data = request.get_json(silent=True) or {}
    folder_path = data.get("folder_path", "").strip()
    filename = data.get("filename", "").strip()
    archive = data.get("archive", "").strip()

    if not folder_path or not filename:
        return jsonify({"error": "folder_path and filename are required"}), 400

    if archive:
        # File lives inside a tar/zip archive
        filepath = Path(folder_path) / archive
        if not filepath.exists():
            return jsonify({"error": f"Archive not found: {filepath}"}), 404
        try:
            df = read_file(filepath, nrows=500, inner_filename=filename)
        except Exception as exc:
            return jsonify({"error": f"Failed to read file: {exc}"}), 500
    else:
        filepath = Path(folder_path) / filename
        if not filepath.exists():
            return jsonify({"error": f"File not found: {filepath}"}), 404
        try:
            df = read_file(filepath, nrows=500)
        except Exception as exc:
            return jsonify({"error": f"Failed to read file: {exc}"}), 500

    # Normalise column names: lowercase, spaces → underscores
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Get preview rows: take first 10 rows, filling nulls with ""
    # (dropna removes too many rows when columns have sparse data)
    df_preview = df.head(10).fillna("")

    # Build dtype map
    dtypes = {col: str(df[col].dtype) for col in df.columns}

    # Auto-suggest column types
    suggested_types = {}
    suggested_separators = {}

    for col in df.columns:
        suggested_types[col] = _suggest_type(df, col, suggested_separators)

    # Convert preview rows to serialisable lists
    rows = []
    for _, row in df_preview.iterrows():
        rows.append([_to_json_safe(v) for v in row])

    return jsonify({
        "columns": list(df.columns),
        "dtypes": dtypes,
        "rows": rows,
        "suggested_types": suggested_types,
        "suggested_separators": suggested_separators,
        "n_rows": len(df_preview),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Auto-suggestion heuristics
# ─────────────────────────────────────────────────────────────────────────────


def _suggest_type(
    df, col: str, separators: dict[str, str]
) -> str:
    """Suggest a feature type for a column based on its data.

    Detection priority:
        1. Boolean (dtype or 2-value enum)
        2. Numeric → datetime (Unix timestamp) or float
        3. Already datetime64 → datetime
        4. String-based: boolean → datetime → url → token_seq → text → token

    Returns one of: ``'float'``, ``'object'``, ``'string'``,
    ``'datetime'``, ``'bool'``, ``'misc'``, or ``''`` (unset).
    """
    import pandas as pd

    series = df[col].dropna()
    if series.empty:
        return ""

    # ── 1. Boolean dtype ──
    if series.dtype == bool:
        return "bool"

    # ── 2. Numeric columns → datetime (timestamp) or float ──
    if series.dtype.kind in ("i", "u", "f"):
        if _looks_like_unix_timestamp(series, col):
            return "datetime"
        return "float"

    # ── 3. Already datetime64 ──
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # ── 4. String-based heuristics ──
    if series.dtype == object:
        sample = series.head(100)
        str_vals = sample.astype(str)
        col_lower = col.lower()

        # 4a. Boolean-like string values
        if _looks_like_boolean(series):
            return "bool"

        # 4b. Datetime by column name + parsing
        _datetime_name_hints = (
            "date", "time", "timestamp", "created", "updated",
            "_at", "_on", "release",
        )
        if any(hint in col_lower for hint in _datetime_name_hints):
            parsed = pd.to_datetime(str_vals, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.5:
                return "datetime"

        # 4c. URL pattern → string type
        if _looks_like_url(str_vals, col_lower):
            return "string"

        # 4d. Delimiter-separated list → object (with separator recorded)
        for sep in ["|", ";", ","]:
            has_sep = str_vals.str.contains(sep, regex=False)
            if has_sep.mean() > 0.5:
                if sep == "," and _looks_numeric_with_comma(str_vals):
                    continue
                separators[col] = sep
                return "object"

        # 4e. Long strings → string (free text)
        lengths = str_vals.str.len()
        median_len = lengths.median()
        if median_len > 50:
            return "string"

        # 4f. Low cardinality → object (categorical)
        nunique = series.nunique()
        ratio = nunique / len(series) if len(series) > 0 else 1.0
        if ratio < 0.5:
            return "object"

        # 4g. Try datetime parse without name hint (higher threshold)
        parsed = pd.to_datetime(str_vals, errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.7:
            return "datetime"

    return ""


def _looks_like_unix_timestamp(series, col_name: str) -> bool:
    """Check if a numeric series looks like Unix timestamps.

    Only triggers when the column name suggests temporal data AND
    the median value falls in the Unix timestamp range (~1973–2603).
    """
    col_lower = col_name.lower()
    time_keywords = ("time", "date", "timestamp", "created", "updated", "_at")
    if not any(kw in col_lower for kw in time_keywords):
        return False
    median = series.dropna().median()
    return 1e8 < median < 2e10


def _looks_like_boolean(series) -> bool:
    """Check if a string series contains only boolean-like values."""
    unique_lower = set(series.astype(str).str.lower().str.strip().unique())
    # Remove empty strings / NaN representations
    unique_lower.discard("")
    unique_lower.discard("nan")
    unique_lower.discard("none")
    if len(unique_lower) != 2:
        return False
    boolean_sets = [
        {"true", "false"}, {"yes", "no"}, {"t", "f"}, {"y", "n"},
        {"0", "1"}, {"on", "off"},
    ]
    return any(unique_lower == bs for bs in boolean_sets)


def _looks_like_url(str_vals, col_lower: str) -> bool:
    """Check if string values look like URLs."""
    url_match = str_vals.str.match(r"^https?://|^www\.", case=False)
    if url_match.mean() > 0.5:
        return True
    # Also by column name with a lower threshold
    url_name_hints = ("url", "link", "href", "website", "homepage")
    if any(hint in col_lower for hint in url_name_hints):
        return url_match.mean() > 0.2
    return False


def _looks_numeric_with_comma(str_vals) -> bool:
    """Check if values look like numbers with comma as decimal separator."""
    import re
    pattern = re.compile(r"^\d+,\d+$")
    matches = str_vals.apply(lambda x: bool(pattern.match(str(x))))
    return matches.mean() > 0.5


def _to_json_safe(val):
    """Convert a value to a JSON-serialisable form."""
    import numpy as np
    import math

    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (list, dict)):
        return val
    return str(val)


# ─────────────────────────────────────────────────────────────────────────────
# YAML export
# ─────────────────────────────────────────────────────────────────────────────


class _LiteralStr(str):
    """String rendered with literal block style (|) in YAML."""


class _FoldedStr(str):
    """String rendered with folded style (>) in YAML."""


class _FlowList(list):
    """List rendered with flow style [...] in YAML."""


class _CustomDumper(yaml.Dumper):
    """YAML dumper with custom representers for config output."""


def _literal_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


def _folded_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=">")


def _flow_list_representer(dumper, data):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    )


def _ordered_dict_representer(dumper, data):
    return dumper.represent_mapping(
        "tag:yaml.org,2002:map", data.items()
    )


_CustomDumper.add_representer(_LiteralStr, _literal_representer)
_CustomDumper.add_representer(_FoldedStr, _folded_representer)
_CustomDumper.add_representer(_FlowList, _flow_list_representer)
_CustomDumper.add_representer(OrderedDict, _ordered_dict_representer)


# Wizard type → YAML feature type mapping
_WIZARD_TO_YAML_TYPE: dict[str, str | None] = {
    "float": "float",
    "object": "object",
    "string": "string",
    "datetime": "datetime",
    "bool": "bool",
    "misc": "misc",
    # Legacy wizard types (kept for backward compat with in-flight sessions)
    "text": "string",
    "boolean": "object",
    "url": "string",
    "exclude": "exclude",
    "token": "object",
    "token_seq": "object",
}

# YAML feature type → Wizard type (reverse mapping for loading configs)
_YAML_TO_WIZARD_TYPE: dict[str, str] = {
    "object": "object",
    "float": "float",
    "string": "string",
    "datetime": "datetime",
    "bool": "bool",
    "misc": "misc",
    "exclude": "exclude",
    # Legacy YAML types
    "text": "string",
    "token": "object",
    "token_seq": "object",
    "drop": "exclude",
}

# Schema categories handled by the schema section (excluded from features)
_SCHEMA_SKIP_CATEGORIES = {"user_id", "item_id", "timestamp"}

# Taxonomy category order per file role (mirrors the JS TAXONOMY constant)
_TAXONOMY_CATS: dict[str, list[str]] = {
    "interactions": [
        "user_id", "item_id", "explicit_feedback", "implicit_feedback",
        "timestamp", "session_data", "interaction_context", "other", "exclude",
    ],
    "items": [
        "item_id", "descriptive_features", "content_features",
        "provider_upstream_info", "other", "exclude",
    ],
    "users": [
        "user_id", "demographics", "downstream_stakeholder_info",
        "additional_attributes", "other", "exclude",
    ],
    # "other" files can use any category — superset of all roles
    "other": [
        "user_id", "item_id", "descriptive_features", "content_features",
        "provider_upstream_info", "demographics", "downstream_stakeholder_info",
        "additional_attributes", "explicit_feedback", "implicit_feedback",
        "timestamp", "session_data", "interaction_context", "other", "exclude",
    ],
}

# Entity stakeholders have an id_column; signal stakeholders do not
_ENTITY_STAKEHOLDERS = {"consumer", "provider", "upstream", "downstream"}


@app.route("/api/export-config", methods=["POST"])
def export_config():
    """Build a YAML config from the wizard state and return it as text.

    Request JSON — the full wizard payload with keys:
        ``metadata``, ``files``, ``columnConfigs``, ``stakeholderConfig``.

    Response JSON::

        {"yaml": "dataset_name: steam\\n..."}
    """
    data = request.get_json(silent=True) or {}

    try:
        yaml_str = _build_yaml_config(data)
        return jsonify({"yaml": yaml_str})
    except Exception as exc:
        logger.exception("Failed to build config")
        return jsonify({"error": f"Failed to build config: {exc}"}), 500


@app.route("/api/save-config", methods=["POST"])
def save_config():
    """Save a YAML config to the configured output directory.

    Request JSON::

        {"yaml": "dataset_name: steam\\n...", "filename": "steam.yaml"}

    Response JSON::

        {"saved_to": "/path/to/configs/steam.yaml"}
    """
    data = request.get_json(silent=True) or {}
    yaml_str = data.get("yaml", "")
    filename = data.get("filename", "dataset.yaml")

    output_dir = app.config.get("OUTPUT_PATH", "")
    if not output_dir:
        return jsonify({"error": "No output path configured"}), 400

    if not yaml_str:
        return jsonify({"error": "No YAML content provided"}), 400

    try:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        dest = out_path / filename
        dest.write_text(yaml_str, encoding="utf-8")
        return jsonify({"saved_to": str(dest)})
    except Exception as exc:
        logger.exception("Failed to save config")
        return jsonify({"error": f"Failed to save: {exc}"}), 500


@app.route("/api/parse-config", methods=["POST"])
def parse_config():
    """Parse an existing YAML config and return wizard-compatible state.

    Request JSON::

        {"config_path": "/path/to/steam.yaml"}

    Response JSON::

        {
          "metadata": {...},
          "fileRoles": [...],
          "stakeholderConfig": {...},
          "pendingColumnConfigs": {...}
        }
    """
    data = request.get_json(silent=True) or {}
    config_path = data.get("config_path", "").strip()

    if not config_path:
        return jsonify({"error": "config_path is required"}), 400

    path = Path(config_path)
    if not path.is_file():
        return jsonify({"error": f"File not found: {config_path}"}), 404

    try:
        result = _parse_yaml_to_wizard_state(path)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Failed to parse config")
        return jsonify({"error": f"Failed to parse config: {exc}"}), 500


def _parse_yaml_to_wizard_state(config_path: Path) -> dict:
    """Parse a YAML config file and return wizard-compatible state.

    Reverse-maps the YAML structure back to the wizard's internal format,
    allowing an existing config to be loaded for editing.

    Returns:
        A dict with keys: ``metadata``, ``fileRoles``, ``stakeholderConfig``,
        ``pendingColumnConfigs``.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config or not isinstance(config, dict):
        raise ValueError("Invalid YAML config: expected a mapping")

    # ── Metadata ──────────────────────────────────────────────────────────
    metadata = {
        "datasetName": config.get("dataset_name", "") or "",
        "domain": config.get("domain", "") or "",
        "version": str(config.get("version", "") or ""),
        "sourceUrl": config.get("source_url", "") or "",
        "description": (config.get("description", "") or "").strip(),
        "citation": (config.get("citation", "") or "").strip(),
    }

    # ── File roles ────────────────────────────────────────────────────────
    file_roles: list[dict] = []
    files_section = config.get("files", {}) or {}
    for role in ("interactions", "items", "users", "other"):
        entry = files_section.get(role)
        if entry is None:
            continue
        if isinstance(entry, str):
            file_roles.append({"filename": entry, "role": role})
        elif isinstance(entry, dict):
            fr: dict = {"filename": entry.get("filename", ""), "role": role}
            if entry.get("archive"):
                fr["archive"] = entry["archive"]
            file_roles.append(fr)
        elif isinstance(entry, list):
            for item in entry:
                if isinstance(item, str):
                    file_roles.append({"filename": item, "role": role})
                elif isinstance(item, dict):
                    fr = {"filename": item.get("filename", ""), "role": role}
                    if item.get("archive"):
                        fr["archive"] = item["archive"]
                    file_roles.append(fr)

    # ── Stakeholder config ────────────────────────────────────────────────
    stakeholder_cfg: dict = {}
    raw_stakeholders = config.get("stakeholder_roles", {}) or {}
    for sk in ("consumer", "provider", "upstream", "downstream",
               "system", "third_party"):
        raw = raw_stakeholders.get(sk)
        if raw is None:
            raw = False
        if isinstance(raw, bool):
            # Flat format: just a boolean
            entry_dict: dict = {"enabled": raw}
            if sk in _ENTITY_STAKEHOLDERS:
                entry_dict["id_column"] = ""
            entry_dict["columns"] = []
        elif isinstance(raw, dict):
            # Nested format (wizard-generated or hand-written)
            enabled = bool(raw.get("supported", raw.get("enabled", False)))
            entry_dict = {"enabled": enabled}
            if sk in _ENTITY_STAKEHOLDERS:
                id_list = raw.get("id", [])
                entry_dict["id_column"] = (
                    id_list[0] if isinstance(id_list, list) and id_list else ""
                )
            entry_dict["columns"] = list(
                raw.get("features", raw.get("columns", [])) or []
            )
        else:
            entry_dict = {"enabled": False}
            if sk in _ENTITY_STAKEHOLDERS:
                entry_dict["id_column"] = ""
            entry_dict["columns"] = []
        stakeholder_cfg[sk] = entry_dict

    # ── Pending column configs ────────────────────────────────────────────
    # Built from schema + *_features + taxonomy sections.
    pending: dict[str, dict] = {}

    # 1. Parse schema section → column name to {type, schema}
    schema = config.get("schema", {}) or {}
    schema_columns: dict[str, dict] = {}
    _SCHEMA_FIELD_MAP = {
        "user_identifier": {"type": "object", "schema": "user_id"},
        "item_identifier": {"type": "object", "schema": "item_id"},
        "timestamp":       {"type": "datetime", "schema": "timestamp"},
        "rating":          {"type": "float", "schema": "explicit_feedback"},
    }
    for field, defaults in _SCHEMA_FIELD_MAP.items():
        for col in (schema.get(field) or []):
            schema_columns[col] = dict(defaults)

    # 2. Parse feature sections + taxonomy per role
    _ROLE_FEATURE_KEYS = {
        "items": "item_features",
        "interactions": "interaction_features",
        "users": "user_features",
        "other": "other_features",
    }

    for role in ("interactions", "items", "users", "other"):
        feature_key = _ROLE_FEATURE_KEYS[role]
        features_section = config.get(feature_key, {}) or {}
        taxonomy_section = (
            (config.get("taxonomy", {}) or {}).get(role, {}) or {}
        )

        role_file_roles = [fr for fr in file_roles if fr["role"] == role]
        if not role_file_roles:
            continue

        # Feature type map: col → wizard type
        feature_type_map: dict[str, str] = {}
        for yaml_type, cols in features_section.items():
            if not cols:
                continue
            wizard_type = _YAML_TO_WIZARD_TYPE.get(yaml_type, "")
            if wizard_type:
                for col in cols:
                    feature_type_map[col] = wizard_type

        # Taxonomy schema map: col → category
        taxonomy_map: dict[str, str] = {}
        for cat, cols in taxonomy_section.items():
            if not cols:
                continue
            for col in cols:
                taxonomy_map[col] = cat

        # Build pending config per file in this role
        for fr in role_file_roles:
            tab_key = fr["role"] + "__" + fr["filename"]
            col_configs: dict[str, dict] = {}

            # Add schema columns (user_id, item_id, timestamp, rating)
            for col, cfg in schema_columns.items():
                col_configs[col] = {
                    "type": cfg["type"],
                    "schema": cfg["schema"],
                }

            # Add feature columns
            for col, wizard_type in feature_type_map.items():
                if col not in col_configs:
                    # Handle old exclude-as-type: convert to schema "exclude"
                    schema_val = taxonomy_map.get(col, "")
                    if wizard_type == "exclude":
                        schema_val = "exclude"
                        wizard_type = ""
                    col_configs[col] = {
                        "type": wizard_type,
                        "schema": schema_val,
                    }

            # Add taxonomy-only columns (in taxonomy but not features/schema)
            for cat, cols in taxonomy_section.items():
                if not cols:
                    continue
                for col in cols:
                    if col not in col_configs:
                        col_configs[col] = {
                            "type": "",
                            "schema": cat,
                        }

            pending[tab_key] = col_configs

    return {
        "metadata": metadata,
        "fileRoles": file_roles,
        "stakeholderConfig": stakeholder_cfg,
        "pendingColumnConfigs": pending,
    }


def _build_yaml_config(data: dict) -> str:
    """Assemble a YAML config string from the wizard payload.

    The output format matches ``recdata/configs/*.yaml`` so it can be
    consumed directly by the recdata pipeline.
    """
    metadata = data.get("metadata", {})
    files_list = data.get("files", [])
    column_configs = data.get("columnConfigs", {})
    stakeholder_config = data.get("stakeholderConfig", {})

    config: OrderedDict = OrderedDict()

    # ── Identity ──────────────────────────────────────────────────────────
    config["dataset_name"] = metadata.get("datasetName", "") or "untitled"
    config["domain"] = metadata.get("domain", "") or "other"

    version = metadata.get("version", "")
    if version:
        config["version"] = str(version)

    source_url = metadata.get("sourceUrl", "")
    if source_url:
        config["source_url"] = source_url

    description = (metadata.get("description", "") or "").strip()
    if description:
        config["description"] = _FoldedStr(description + "\n")

    citation = (metadata.get("citation", "") or "").strip()
    if citation:
        config["citation"] = _LiteralStr(citation + "\n")

    # ── Stakeholder roles ─────────────────────────────────────────────────
    roles: OrderedDict = OrderedDict()
    for sk in ("consumer", "provider", "upstream", "downstream",
               "system", "third_party"):
        sk_cfg = stakeholder_config.get(sk, {})
        enabled = bool(sk_cfg.get("enabled", False))
        sk_entry: OrderedDict = OrderedDict()
        sk_entry["supported"] = enabled
        if enabled:
            if sk in _ENTITY_STAKEHOLDERS:
                id_col = (sk_cfg.get("id_column") or "").strip()
                sk_entry["id"] = _FlowList([id_col] if id_col else [])
            cols = sk_cfg.get("columns") or []
            sk_entry["features"] = _FlowList(list(cols))
        roles[sk] = sk_entry
    config["stakeholder_roles"] = roles

    # ── Files ─────────────────────────────────────────────────────────────
    role_files: dict[str, list[dict]] = {}
    for f in files_list:
        role = f.get("role", "")
        fname = f.get("filename", "")
        archive = f.get("archive", "")
        if role and fname:
            entry: dict[str, str] = {"filename": fname}
            if archive:
                entry["archive"] = archive
            role_files.setdefault(role, []).append(entry)

    files_dict: OrderedDict = OrderedDict()
    for role in ("interactions", "items", "users", "other"):
        entries = role_files.get(role, [])
        if not entries:
            if role != "other":
                files_dict[role] = None
            continue

        def _file_entry(e: dict):
            """Convert a file entry to its YAML representation."""
            if e.get("archive"):
                d = OrderedDict()
                d["filename"] = e["filename"]
                d["archive"] = e["archive"]
                return d
            return e["filename"]

        if len(entries) == 1:
            files_dict[role] = _file_entry(entries[0])
        else:
            files_dict[role] = [_file_entry(e) for e in entries]

    config["files"] = files_dict

    # ── Schema ────────────────────────────────────────────────────────────
    schema: OrderedDict = OrderedDict()
    schema["user_identifier"] = []
    schema["item_identifier"] = []
    schema["timestamp"] = []
    schema["rating"] = []

    for _tab_key, cols in column_configs.items():
        for col_name, cfg in cols.items():
            cat = cfg.get("schema", "")
            col_type = cfg.get("type", "")

            if cat == "user_id":
                if col_name not in schema["user_identifier"]:
                    schema["user_identifier"].append(col_name)
            elif cat == "item_id":
                if col_name not in schema["item_identifier"]:
                    schema["item_identifier"].append(col_name)
            elif cat == "timestamp":
                if col_name not in schema["timestamp"]:
                    schema["timestamp"].append(col_name)
            elif cat in ("explicit_feedback", "implicit_feedback"):
                # Numeric feedback columns are rating candidates
                if col_type == "float" and col_name not in schema["rating"]:
                    schema["rating"].append(col_name)

    # Wrap all lists in FlowList for [...] output style
    for key in schema:
        schema[key] = _FlowList(schema[key])
    config["schema"] = schema

    # ── Feature declarations ──────────────────────────────────────────────
    _ROLE_FEATURE_KEY = {
        "items": "item_features",
        "interactions": "interaction_features",
        "users": "user_features",
        "other": "other_features",
    }
    for role in ("items", "interactions", "users", "other"):
        feature_key = _ROLE_FEATURE_KEY[role]

        role_tabs = [
            tk for tk in column_configs if tk.startswith(role + "__")
        ]
        if not role_tabs:
            config[feature_key] = None
            continue

        features: OrderedDict = OrderedDict()
        for yaml_type in ("object", "float", "string", "datetime", "bool", "misc", "exclude"):
            features[yaml_type] = []

        for tab_key in role_tabs:
            for col_name, cfg in column_configs[tab_key].items():
                cat = cfg.get("schema", "")
                col_type = cfg.get("type", "")

                # Columns with schema "exclude" go into the exclude feature type
                if cat == "exclude":
                    if col_name not in features["exclude"]:
                        features["exclude"].append(col_name)
                    continue

                if not col_type:
                    continue

                yaml_type = _WIZARD_TO_YAML_TYPE.get(col_type)
                if yaml_type and col_name not in features.get(yaml_type, []):
                    features[yaml_type].append(col_name)

        # Wrap in FlowList
        for key in features:
            features[key] = _FlowList(features[key])
        config[feature_key] = features

    # ── Taxonomy classification ───────────────────────────────────────────
    # Groups every column in every file by its taxonomy schema category.
    # Downstream profiling can use this to display columns by category.
    taxonomy: OrderedDict = OrderedDict()
    for role in ("interactions", "items", "users", "other"):
        role_tabs = [tk for tk in column_configs if tk.startswith(role + "__")]
        if not role_tabs:
            taxonomy[role] = None
            continue

        cats = _TAXONOMY_CATS.get(role, [])
        role_taxonomy: OrderedDict = OrderedDict()
        for cat in cats:
            role_taxonomy[cat] = []

        for tab_key in role_tabs:
            for col_name, cfg in column_configs[tab_key].items():
                cat = (cfg.get("schema") or "").strip()
                if not cat:
                    cat = "other"
                if cat in role_taxonomy:
                    if col_name not in role_taxonomy[cat]:
                        role_taxonomy[cat].append(col_name)
                else:
                    # Unknown category → bucket into "other"
                    if col_name not in role_taxonomy.get("other", []):
                        role_taxonomy.setdefault("other", []).append(col_name)

        # Wrap in FlowList
        for cat in role_taxonomy:
            role_taxonomy[cat] = _FlowList(role_taxonomy[cat])
        taxonomy[role] = role_taxonomy

    config["taxonomy"] = taxonomy

    # ── Dump to YAML string ───────────────────────────────────────────────
    yaml_str = yaml.dump(
        config,
        Dumper=_CustomDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )

    # Post-process: insert blank lines before major sections
    _section_keys = {
        "stakeholder_roles:", "files:", "schema:",
        "item_features:", "interaction_features:", "user_features:",
        "other_features:", "taxonomy:",
    }
    lines = yaml_str.split("\n")
    result: list[str] = []
    for i, line in enumerate(lines):
        token = line.lstrip().split(" ")[0]
        if token in _section_keys and i > 0:
            result.append("")
        result.append(line)

    return "\n".join(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RecData ConfigApp")
    parser.add_argument(
        "--data-path",
        default=os.environ.get("DATA_PATH", ""),
        help="Path to the raw dataset folder (pre-fills the folder input)",
    )
    parser.add_argument(
        "--output-path",
        default=os.environ.get("OUTPUT_PATH", ""),
        help="Directory where generated YAML configs are saved",
    )
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    app.config["DATA_PATH"] = args.data_path
    app.config["OUTPUT_PATH"] = args.output_path
    app.run(debug=True, port=args.port)
