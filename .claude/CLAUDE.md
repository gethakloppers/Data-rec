# RecData — Multistakeholder Recommender System Dataset Library

## Project Overview

A **standalone Python library with a companion web interface** for loading, standardising, profiling, and exploring recommender system datasets. Built for a **multistakeholder recommender systems research project**, the library handles diverse raw dataset formats and surfaces stakeholder-relevant metadata (provider info, economic signals, social structures) in both data outputs and the UI.

**Two distinct parts:**
1. **`recdata/`** — Python backend library for loading, standardising, profiling, and exporting datasets
2. **`webapp/`** — Flask web application that reads profiler outputs and presents a browser-based dataset explorer
3. **`configapp/`** — Flask web wizard for interactively building YAML config files

**No RecBole-specific outputs.** No atomic file conversion, no `field:type` column renaming. Clean Parquet outputs that a separate downstream step can later convert for RecBole.

**No filtering.** The library loads, standardises, profiles, and exports. No k-core, no minimum interaction counts, no deduplication.

---

## Project Structure

```
project/
├── recdata/                        # Core Python library
│   ├── __init__.py
│   ├── exceptions.py               # Custom exceptions
│   ├── pipeline.py                 # CLI entry point: process_dataset()
│   ├── configs/                    # One YAML config per dataset
│   │   ├── steam_2018.yaml
│   │   ├── movielens_1m.yaml
│   │   ├── xwines.yaml
│   │   └── hummus.yaml
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── base_loader.py          # Config loading, validation, feature maps
│   │   └── file_reader.py          # Multi-format file reader
│   ├── processing/
│   │   ├── __init__.py
│   │   └── standardiser.py         # 7-step standardisation pipeline
│   ├── profiler/
│   │   ├── __init__.py
│   │   ├── quality_report.py       # Diagnostics: type audit, mixed types, nulls, etc.
│   │   └── dataset_profiler.py     # Profile dict + ID mappings
│   └── exporters/
│       ├── __init__.py
│       └── exporter.py             # Parquet + JSON + markdown export
│
├── webapp/                         # Web interface (dataset explorer)
│   ├── app.py                      # Flask application
│   ├── static/css/style.css
│   └── templates/
│       ├── base.html
│       ├── index.html              # Dataset catalogue
│       ├── dataset.html            # Dataset detail page
│       └── error.html
│
├── configapp/                      # Config wizard (YAML builder)
│   ├── app.py
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/app.js
│   └── templates/index.html
│
└── notebooks/                      # Jupyter notebooks for testing
    └── standardise_dataset.ipynb
```

---

## Type System

The YAML config uses **6 canonical feature types**:

| Type | Purpose |
|------|---------|
| `object` | Categorical values — strings, IDs, codes, lists |
| `float` | Numeric values — prices, scores, counts |
| `text` | Free text — reviews, descriptions, titles |
| `datetime` | Date/time values |
| `misc` | Boolean or miscellaneous values |
| `exclude` | Columns to drop from output |

**Backward compatibility:** Legacy type names are normalised automatically:
- `token` → `object`
- `token_seq` → `object`
- `drop` → `exclude`

This is handled by `_TYPE_ALIASES` in `base_loader.py`.

---

## Pipeline Flow

The full pipeline (orchestrated by `recdata/pipeline.py`) runs these steps:

```
1. Load config (YAML)
2. Read raw files (CSV, JSONL, Parquet, ZIP, GZ, etc.)
3. Standardise each role (interactions, items, users)
4. Run quality report (type audit, mixed types, nulls, duplicates, ID coverage)
5. Profile dataset (counts, distributions, taxonomy, stakeholder support)
6. Build ID mappings (contiguous integers from 1)
7. Export (Parquet files, JSON mappings, profile JSON, quality report, markdown)
```

### CLI Usage

```bash
# Full pipeline
python -m recdata.pipeline \
    --config recdata/configs/xwines.yaml \
    --raw /path/to/raw/data \
    --output /path/to/output

# Dry run (inspect plan without processing)
python -m recdata.pipeline --config ... --raw ... --dry-run

# Quality report only (skip profiling/export)
python -m recdata.pipeline --config ... --raw ... --output ... --quality-only
```

### Output Structure

```
{output_path}/{dataset_name}/
├── processed/
│   ├── interactions.parquet
│   ├── items.parquet
│   └── users.parquet
├── mappings/
│   ├── user_map.json
│   └── item_map.json
└── profile/
    ├── dataset_profile.json
    ├── quality_report.json
    └── dataset_report.md
```

---

## Key Modules

### `recdata/loaders/base_loader.py`
- `load_config(path)` — Load and validate YAML config
- `normalize_file_def(file_def)` — Handle string/dict/None file definitions
- `get_feature_map(config, df_role)` — Build flat column-to-type mapping with legacy type normalization
- `_validate_config(config)` — Validate required keys, type names

### `recdata/loaders/file_reader.py`
- `read_file(filepath, format, encoding, separator, **kwargs)` — Multi-format reader
- Supports: CSV, TSV, JSON, JSONL, Parquet, GZip, ZIP, TAR, DAT
- Auto-detects format from file extension when `format=None`
- JSONL reader falls back to `ast.literal_eval` for Python-repr dicts

### `recdata/processing/standardiser.py`
- `standardise_df(df, config, df_role)` — 7-step standardisation:
  1. Lowercase column names
  2. Rename key columns (user_id, item_id, timestamp, rating)
  3. Remove null IDs
  4. Cast IDs to string + cast object-typed columns to string
  5. Cast timestamp (auto-detect unit via log10)
  6. Cast rating to float32
  7. Exclude declared columns
- `describe_standardisation_plan(df, config, role)` — Dry-run plan without modifying data
- Returns `(df, warnings)` tuple

### `recdata/profiler/quality_report.py`
- `quality_report(dfs, config)` — Diagnostic report with sections:
  - `type_audit`: declared vs actual dtype comparison
  - `mixed_types`: columns with multiple value types (e.g., 98% integer, 2% string)
  - `list_detection`: object columns containing list-like values
  - `null_analysis`: per-column null counts and percentages
  - `duplicate_analysis`: duplicate rows and IDs
  - `id_coverage`: cross-referencing between interactions and metadata
  - `column_statistics`: descriptive stats per column
- `summarise_mixed_types(report)` — Compact notes for YAML annotation

### `recdata/profiler/dataset_profiler.py`
- `profile_dataset(dfs, config, warnings, quality)` — Full profile dict:
  - Identity (name, domain, version, URL, citation)
  - Counts (users, items, interactions, sparsity, density)
  - Interaction characteristics (type, rating scale, timestamp range)
  - Distributions (user/item interaction counts, long-tail ratio, power-user ratio)
  - Taxonomy (auto-detected from column names + config)
  - Stakeholder support (from config, with basis text)
  - Column inventory (per file: name, dtype, null%, unique, feature_type)
- `build_id_mappings(dfs)` — Contiguous integer mappings (from 1), DataFrames NOT modified

### `recdata/exporters/exporter.py`
- `export_dataset(dfs, profile, id_mappings, quality, output_dir)` — Full export
- `generate_markdown_report(profile)` — Structured markdown with 7 sections
- Parquet: pyarrow engine, Snappy compression, object→StringDtype conversion

### `recdata/pipeline.py`
- `load_raw_files(config, raw_path)` — Read all declared files (handles archives)
- `load_dataset(config, raw_path)` — Load + standardise, returns `(dfs, warnings)`
- `process_dataset(config, raw_path, output_path)` — Full pipeline
- CLI with `--dry-run`, `--quality-only`, `--verbose` flags

---

## Web Interface (`webapp/`)

Flask app that reads `dataset_profile.json` files from the pipeline output directory.

### Running

```bash
# Via CLI
python webapp/app.py --output-dir /path/to/output --port 5001

# Via environment variable
export RECDATA_OUTPUT_DIR=/path/to/output
cd webapp && flask run
```

### Routes
- `GET /` — Dataset catalogue (card grid with key stats, stakeholder badges)
- `GET /dataset/<name>` — Detail page (overview, stakeholders, taxonomy, columns, distributions, download)
- `GET /api/datasets` — JSON summary list
- `GET /api/dataset/<name>` — Full profile JSON
- `GET /download/<name>` — ZIP of processed Parquet files + profile

### Frontend Stack
- Jinja2 templates (server-rendered)
- Alpine.js for tab switching (Column Explorer)
- Chart.js (CDN) for distribution bar charts
- Custom CSS with Source Serif 4 + Inter + JetBrains Mono fonts

---

## Config Wizard (`configapp/`)

Interactive web wizard for building YAML config files.

### Running

```bash
python configapp/app.py --port 5002
```

### Features
- Upload raw dataset files (CSV, JSON, JSONL, Parquet)
- Auto-detect column types via sampling
- Visual column configuration per file role
- Schema mapping (user_id, item_id, timestamp, rating candidates)
- Stakeholder role declaration
- Live YAML preview and download
- Import existing YAML configs for editing

---

## YAML Config Schema

```yaml
dataset_name: xwines
domain: ecommerce_and_retail
source_url: https://github.com/rogerioxavier/X-Wines
citation: |
  @Article{...}

stakeholder_roles:
  consumer:
    supported: true
    id: [userid]
    features: []
  provider:
    supported: true
    id: [wineryid]
    features: [wineryname]
  system:
    supported: true
    features: [regionid, regionname, country, code]
  upstream:
    supported: false
  downstream:
    supported: false
  third_party:
    supported: false

files:
  interactions:
    filename: XWines_Full_21M_ratings.csv
    archive: XWines_Full_21M_ratings.zip   # optional: read from archive
  items:
    filename: XWines_Full_100K_wines.csv
    archive: XWines_Full_100K_wines.zip
  users: null

schema:
  user_identifier: [userid]
  item_identifier: [wineid]
  timestamp: [date]
  rating: [rating]

interaction_features:
  object: [vintage]
  exclude: [ratingid]

item_features:
  object: [type, elaborate, grapes, harmonize, body, acidity, code, regionid, wineryid, vintages]
  float: [abv]
  text: [winename, country, regionname, wineryname, website]
```

---

## Implementation Constraints

- **ID handling**: `user_id` and `item_id` are always strings (object dtype). ID mappings are separate JSON artifacts.
- **Storage**: Parquet with Snappy compression (pyarrow default)
- **Memory**: Print memory usage after loading each file. Chunked JSONL reader for large files.
- **Error handling**: `DatasetLoadError`, `ConfigValidationError`, `ColumnNotFoundError` — never silently swallow exceptions
- **Code quality**: Full type hints (Python 3.10+), Google-style docstrings, pure functions in `processing/`
- **Dependencies**: pandas, pyarrow, pyyaml, numpy, flask

---

## Python Environment

```bash
# Conda environment with required packages
/opt/anaconda3/envs/msquared/bin/python
```

---

## Current Datasets

| Dataset | Config | Domain | Stakeholders |
|---------|--------|--------|-------------|
| XWines | `xwines.yaml` | E-commerce | C, S, P |
| Steam 2018 | `steam_2018.yaml` | Media & Entertainment | C, S, P |
| MovieLens 1M | `movielens_1m.yaml` | Media & Entertainment | C |
| Hummus | `hummus.yaml` | E-commerce | C |
