"""RecData web interface — dataset explorer.

A Flask web application that reads profile JSONs from the pipeline output
directory and presents them in a browseable dataset catalogue.

Usage::

    export RECDATA_OUTPUT_DIR=/path/to/pipeline/output
    cd webapp && flask run

Or::

    python webapp/app.py --output-dir /path/to/pipeline/output
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

from flask import Flask, Response, abort, jsonify, render_template, send_file

logger = logging.getLogger(__name__)

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR: Path | None = None


def _get_output_dir() -> Path:
    """Resolve the output directory from config, env, or CLI args."""
    global OUTPUT_DIR
    if OUTPUT_DIR is not None:
        return OUTPUT_DIR

    d = app.config.get("OUTPUT_DIR") or os.environ.get("RECDATA_OUTPUT_DIR")
    if d:
        OUTPUT_DIR = Path(d)
    else:
        # Default: look for an 'output' directory alongside the project root
        OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

    if not OUTPUT_DIR.is_dir():
        logger.warning("Output directory does not exist: %s", OUTPUT_DIR)

    return OUTPUT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────


def _discover_datasets() -> list[dict[str, Any]]:
    """Scan the output directory for datasets with profile JSONs.

    Returns a list of summary dicts (one per dataset), sorted by name.
    """
    out = _get_output_dir()
    if not out.is_dir():
        return []

    datasets = []
    for child in sorted(out.iterdir()):
        profile_path = child / "profile" / "dataset_profile.json"
        if profile_path.is_file():
            try:
                with open(profile_path, encoding="utf-8") as f:
                    profile = json.load(f)
                datasets.append(_make_summary(profile, child))
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping %s: %s", child.name, exc)

    return datasets


def _load_profile(name: str) -> dict[str, Any] | None:
    """Load the full profile dict for a named dataset."""
    out = _get_output_dir()
    profile_path = out / name / "profile" / "dataset_profile.json"
    if not profile_path.is_file():
        return None

    with open(profile_path, encoding="utf-8") as f:
        return json.load(f)


def _load_quality(name: str) -> dict[str, Any] | None:
    """Load the quality report dict for a named dataset."""
    out = _get_output_dir()
    quality_path = out / name / "profile" / "quality_report.json"
    if not quality_path.is_file():
        return None

    with open(quality_path, encoding="utf-8") as f:
        return json.load(f)


def _make_summary(profile: dict[str, Any], dataset_dir: Path) -> dict[str, Any]:
    """Extract a compact summary from a full profile for the listing page."""
    counts = profile.get("counts", {})
    interactions = profile.get("interactions", {})
    support = profile.get("stakeholder_support", {})

    # Calculate total size of processed files
    processed_dir = dataset_dir / "processed"
    total_size = 0
    file_list = []
    if processed_dir.is_dir():
        for f in processed_dir.iterdir():
            if f.suffix == ".parquet":
                size = f.stat().st_size
                total_size += size
                file_list.append({"name": f.name, "size_mb": round(size / (1024 * 1024), 1)})

    return {
        "name": profile.get("dataset_name", dataset_dir.name),
        "domain": profile.get("domain", ""),
        "version": profile.get("version", ""),
        "description": profile.get("description", ""),
        "n_users": counts.get("n_users", 0),
        "n_items": counts.get("n_items", 0),
        "n_interactions": counts.get("n_interactions", 0),
        "sparsity": counts.get("sparsity", 0),
        "density": counts.get("density", 0),
        "feedback_type": interactions.get("type", "unknown"),
        "has_timestamp": interactions.get("has_timestamp", False),
        "has_rating": interactions.get("has_rating", False),
        "stakeholders": {
            role: info.get("supported", False)
            for role, info in support.items()
        },
        "total_size_mb": round(total_size / (1024 * 1024), 1),
        "parquet_files": file_list,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Template helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_number(n: float | int) -> str:
    """Format a number with K/M suffixes for display."""
    if n is None:
        return "—"
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


def _domain_label(domain: str) -> str:
    """Convert a snake_case domain to a readable label."""
    return domain.replace("_", " ").title() if domain else "Unknown"


# Register as Jinja2 filters
app.jinja_env.filters["fmt_number"] = _fmt_number
app.jinja_env.filters["domain_label"] = _domain_label


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Pages
# ─────────────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    """Dataset listing page."""
    datasets = _discover_datasets()
    return render_template("index.html", datasets=datasets)


@app.route("/dataset/<name>")
def dataset_detail(name: str):
    """Dataset detail page."""
    profile = _load_profile(name)
    if profile is None:
        abort(404, description=f"Dataset '{name}' not found")

    quality = _load_quality(name)
    summary = None

    # Build summary for template
    out = _get_output_dir()
    dataset_dir = out / name
    if dataset_dir.is_dir():
        summary = _make_summary(profile, dataset_dir)

    return render_template(
        "dataset.html",
        profile=profile,
        quality=quality,
        summary=summary,
        name=name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes — API
# ─────────────────────────────────────────────────────────────────────────────


@app.route("/api/datasets")
def api_datasets():
    """JSON list of all available datasets (minimal info)."""
    datasets = _discover_datasets()
    return jsonify(datasets)


@app.route("/api/dataset/<name>")
def api_dataset(name: str):
    """Full profile JSON for one dataset."""
    profile = _load_profile(name)
    if profile is None:
        abort(404, description=f"Dataset '{name}' not found")
    return jsonify(profile)


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Download
# ─────────────────────────────────────────────────────────────────────────────


@app.route("/download/<name>")
def download_dataset(name: str):
    """Stream a ZIP of the processed Parquet files for a dataset."""
    out = _get_output_dir()
    processed_dir = out / name / "processed"

    if not processed_dir.is_dir():
        abort(404, description=f"No processed files found for dataset '{name}'")

    parquet_files = list(processed_dir.glob("*.parquet"))
    if not parquet_files:
        abort(404, description=f"No Parquet files found for dataset '{name}'")

    # Create ZIP in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for pf in parquet_files:
            zf.write(pf, arcname=f"{name}/{pf.name}")

        # Include the profile JSON as well
        profile_path = out / name / "profile" / "dataset_profile.json"
        if profile_path.is_file():
            zf.write(profile_path, arcname=f"{name}/dataset_profile.json")

        # Include the markdown report
        md_path = out / name / "profile" / "dataset_report.md"
        if md_path.is_file():
            zf.write(md_path, arcname=f"{name}/dataset_report.md")

    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{name}_processed.zip",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", error=str(e)), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", error="Internal server error"), 500


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RecData web interface")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("RECDATA_OUTPUT_DIR", "output"),
        help="Path to the pipeline output directory (default: RECDATA_OUTPUT_DIR env or ./output)",
    )
    parser.add_argument("--port", type=int, default=5001, help="Port to run on (default: 5001)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    if not OUTPUT_DIR.is_dir():
        print(f"Warning: Output directory does not exist: {OUTPUT_DIR}", file=sys.stderr)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    app.run(host=args.host, port=args.port, debug=args.debug)
