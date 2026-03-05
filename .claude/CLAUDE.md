# Claude Code Prompt: RecSys Dataset Library + Web Interface

## CRITICAL: Step-by-Step Implementation

**Do NOT implement everything at once. Follow the phases below strictly, in order. After completing each phase, stop and confirm with me before proceeding to the next.**

Each phase should be a working, runnable state of the project. Think of it as iterative development where each phase delivers something usable.

---

## Project Overview

Build a **standalone Python library with a companion web interface** for loading, standardising, profiling, and exploring recommender system datasets. This is for a **multistakeholder recommender systems research project**, so the library must handle diverse raw dataset formats and surface stakeholder-relevant metadata (provider info, economic signals, social structures) in both the data outputs and the UI.

The project has two distinct parts:
1. **`recdata/`** — Python backend library for loading, standardising, profiling, and exporting datasets
2. **`webapp/`** — A Flask (or FastAPI) web application that reads the profiler outputs and presents them in a browser-based dataset explorer

There are **no RecBole-specific outputs** in this version. No atomic file conversion, no `field:type` column renaming. The goal is clean, well-documented, standardised Parquet outputs that a separate downstream step can later convert for RecBole. There are also **no filtering options** — this library loads, standardises, profiles, and exports. Nothing more.

---

## Project Structure

```
project/
├── recdata/                        # Core Python library
│   ├── __init__.py
│   ├── configs/                    # One YAML config per dataset
│   │   ├── steam.yaml
│   │   ├── goodreads.yaml
│   │   ├── amazon2023.yaml
│   │   ├── yelp.yaml
│   │   └── movielens32m.yaml
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── base_loader.py          # Abstract base class
│   │   ├── file_reader.py          # Multi-format file reader
│   │   └── dataset_loaders.py      # Concrete loader classes (one per dataset)
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── standardiser.py         # Column renaming + type casting
│   │   └── feature_processor.py    # Feature map processing
│   ├── profiler/
│   │   ├── __init__.py
│   │   └── dataset_profiler.py     # Statistics + taxonomy audit + stakeholder support
│   ├── exporters/
│   │   ├── __init__.py
│   │   └── exporter.py             # Saves Parquet + profile JSON + markdown report
│   └── pipeline.py                 # Entry point: load_dataset(config, raw_path, output_path)
│
├── webapp/                         # Web interface
│   ├── app.py                      # Flask/FastAPI application
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── main.js
│   └── templates/
│       ├── base.html
│       ├── index.html              # Dataset listing page
│       └── dataset.html            # Dataset detail page
│
├── requirements.txt
└── README.md
```

---

## Phase 1: File Reader + Config System

**Goal:** Be able to point the library at a raw dataset folder and read any file into a pandas DataFrame, driven entirely by a YAML config.

### 1a. YAML Config Schema

Each dataset gets a single YAML file in `recdata/configs/`. The config fully declares how to load and standardise the dataset — no hardcoding in Python classes. Structure:

```yaml
dataset_name: steam
domain: media_and_entertainment
business_model: transactional_gaming_platform
version: "2018"
source_url: "https://github.com/kang205/SASRec"
description: >
  Steam game reviews and metadata. Contains 7.8M reviews, 2.5M users,
  and 32K games with developer/publisher metadata and pricing.

# Stakeholder roles supported (declared explicitly for documentation purposes)
stakeholder_roles:
  consumer: true
  system: true        # price data present → monetary reward evaluable
  provider: true      # developer/publisher metadata present
  upstream: false
  downstream: false
  third_party: false

# Raw file declarations
files:
  interactions:
    filename: "steam_reviews.csv"
    format: csv                     # csv | tsv | json | jsonl | parquet | gz | zip
    encoding: utf-8
    separator: ","
  items:
    filename: "steam_games.csv"
    format: csv
  users: null

# Schema: map possible raw column names to standard roles (first match wins)
schema:
  user_identifier: ["username", "user_id", "userid"]
  item_identifier: ["product_id", "app_id", "appid", "game_id"]
  timestamp: ["date", "timestamp", "time", "review_date"]
  rating: ["hours", "rating", "score"]

# Feature type declarations for each file
# Types: token (categorical), float (numeric), token_seq (list of categories),
#        text (free text), drop (exclude from output)
item_features:
  token: ["type", "publisher", "developer", "early_access"]
  float: ["price", "discount_price"]
  token_seq: ["genres", "tags", "specs", "platforms"]
  text: ["title", "app_name", "description"]
  drop: ["url", "reviews_url", "icon", "website"]

interaction_features:
  token: ["products", "funny", "helpful"]
  float: ["hours_played"]
  token_seq: []
  text: ["review"]
  drop: ["page", "page_order", "compensation"]

user_features: null
```

Write YAML configs for these 5 datasets. For datasets where provider fields are absent (e.g., MovieLens), **explicitly document this gap with a comment in the YAML** and set `provider: false` in `stakeholder_roles`.

**Datasets:**
1. **Steam** — `steam_reviews.json/csv` + `steam_games.json/csv`. Provider: `developer`, `publisher`. Economic: `price`, `discount_price`.
2. **Amazon2023** — JSONL reviews + JSONL metadata. Provider: `store`, `brand`, `manufacturer`. Economic: `price`.
3. **Goodreads** — JSONL reviews + JSONL book metadata. Provider: `author_id`, `publisher`.
4. **Yelp** — `yelp_academic_dataset_review.json` + business + user JSONs. Provider: business `name`. Social: `friends` in user file.
5. **MovieLens-32M** — `ratings.csv` + `movies.csv`. No provider fields — document this gap.

### 1b. File Reader (`loaders/file_reader.py`)

Single function `read_file(filepath, format, encoding='utf-8', separator=',', **kwargs) -> pd.DataFrame`.

Handle these formats:
- **CSV / TSV**: `pd.read_csv`. Auto-detect separator if `format='tsv'` or `separator='\t'`.
- **JSON array**: standard `pd.read_json`
- **JSONL (newline-delimited JSON)**: read line by line. For each line, first try `json.loads`; if that fails, fall back to `ast.literal_eval`. This handles Python-repr dicts from older Amazon/Steam datasets. For files > 500k rows, print progress every 500k rows.
- **Parquet**: `pd.read_parquet` with `pyarrow` engine
- **GZip**: detect inner format from inner filename (e.g., `reviews.csv.gz`), decompress transparently then read
- **ZIP**: extract and read the relevant file

Return a plain `pd.DataFrame`. Never modify the data — this is purely I/O.

**Custom exceptions** (put in `recdata/exceptions.py`):
- `DatasetLoadError`: file reading failed
- `ConfigValidationError`: missing or invalid config field
- `ColumnNotFoundError`: declared column not found in DataFrame (list ALL missing at once)

**Deliverable for Phase 1:** Running `python -c "from recdata.loaders.file_reader import read_file; df = read_file('data/steam_reviews.csv', 'csv'); print(df.shape)"` works. All 5 YAML configs are valid and loadable with `pyyaml`.

---

## Phase 2: Standardisation Pipeline

**Goal:** Given a raw DataFrame and a config, produce a clean standardised DataFrame with predictable column names and correct types.

### `processing/standardiser.py`

Function: `standardise_df(df, config, df_role) -> pd.DataFrame`

Where `df_role` is `'interactions'`, `'items'`, or `'users'`.

**Steps (in order):**

1. **Lowercase all column names** — do this first, unconditionally.

2. **Rename key columns** using the schema:
   - Find the first match from each candidate list in the config schema
   - Rename to standard names: `user_id`, `item_id`, `timestamp`, `rating`
   - If no match found for a required field (`user_id`, `item_id`), raise `ColumnNotFoundError`
   - Log a warning (append to a `warnings` list) if `timestamp` or `rating` not found

3. **Remove null IDs**: Drop rows where `user_id` or `item_id` is null. Log count dropped.

4. **Cast IDs to string**: `user_id` and `item_id` must be `str` (object dtype). IDs are tokens, never numbers — even if they look like integers. This prevents silent issues when IDs are things like Amazon ASINs.

5. **Cast timestamp** (interactions only, if present):
   - If numeric: detect unit using `log10(median)` → `< 11 → 's'`, `< 14 → 'ms'`, `< 17 → 'us'`, else `'ns'`
   - Convert to `datetime64[ns]`
   - Log detected unit

6. **Cast rating to float32** (interactions only, if present):
   - Use `pd.to_numeric(..., errors='coerce')` then `.astype('float32')`

### `processing/feature_processor.py`

Function: `process_features(df, feature_map_config) -> pd.DataFrame`

Where `feature_map_config` is the dict from the YAML (e.g., `item_features`).

- Parse the config into a flat `{col_name: type}` dict
- Validate all declared columns exist in df (case-insensitive). Raise `ColumnNotFoundError` with ALL missing columns listed.
- Drop columns marked `drop`
- Add a `_type` suffix to column names for documentation purposes: rename `genres` → `genres__token_seq`, `price` → `price__float`, etc. Use **double underscore** (`__`) as separator instead of colon — colons cause issues in Parquet column names and most downstream tools. The colon convention is RecBole-specific and we are not targeting RecBole here.
- For `token_seq` columns: parse string representations of lists (handle `"['a', 'b']"` strings that come from CSV), ensure every element within each list is a string.
- Leave `text` columns as plain strings.

**Deliverable for Phase 2:** A script `python -m recdata.pipeline --config recdata/configs/steam.yaml --raw /data/steam --output /processed --dry-run` prints what columns would be renamed, what types assigned, what rows dropped — without writing any files.

---

## Phase 3: ID Mapping + Profiler

**Goal:** Produce clean integer-indexed DataFrames and a rich profile dict that becomes the backbone of the web interface.

### ID Mapper (`processing/id_mapper.py`)

- `create_id_mappings(inter_df, item_df, user_df, config) -> (user_map, item_map)`:
  - Build maps from the **union** of IDs across interactions + metadata
  - Contiguous integers starting from 1 (0 reserved as padding)
  - Return as plain Python dicts: `{original_str_id: int_index}`

- `apply_id_mappings(inter_df, item_df, user_df, user_map, item_map) -> (inter, items, users)`:
  - Add `user_index` (Int64) and `item_index` (Int64) columns
  - Drop the original `user_id` / `item_id` string columns (keep them only in the mappings JSON)
  - Warn if any IDs in interactions are not found in the map

- `align_metadata(inter_df, item_df, user_df) -> (item_df, user_df, missing_items, missing_users)`:
  - For every `item_index` / `user_index` in interactions, ensure a row exists in the metadata table
  - Add NaN-filled rows for any missing indices
  - Return lists of indices that had no metadata

### Dataset Profiler (`profiler/dataset_profiler.py`)

Function: `profile_dataset(inter_df, item_df, user_df, config, warnings) -> dict`

Returns a fully structured dict that is serialised to `dataset_profile.json`. This dict is the **single source of truth** for the web interface — everything the UI shows comes from here.

Structure of the profile dict:

```python
{
  # --- Identity ---
  "dataset_name": "steam",
  "domain": "media_and_entertainment",
  "business_model": "transactional_gaming_platform",
  "version": "2018",
  "source_url": "...",
  "description": "...",
  "processed_at": "2025-01-15T10:30:00",

  # --- Basic Counts ---
  "counts": {
    "n_users": 2567538,
    "n_items": 32135,
    "n_interactions": 7793069,
    "n_users_with_metadata": 0,
    "n_items_with_metadata": 32135,
    "sparsity": 0.99991,
    "density": 0.00009
  },

  # --- Interaction Characteristics ---
  "interactions": {
    "type": "explicit",              # "explicit" | "implicit"
    "has_timestamp": true,
    "has_rating": true,
    "rating_scale": {"min": 0.0, "max": 100.0, "mean": 72.4, "median": 80.0, "std": 22.1},
    "timestamp_range": {"earliest": "2010-01-01", "latest": "2018-09-30"},
    "has_context_features": true,
    "has_session_data": false
  },

  # --- Distribution Statistics ---
  "distributions": {
    "user_interaction_counts": {
      "min": 1, "max": 4804, "mean": 3.04, "median": 1.0, "std": 8.2,
      "p25": 1.0, "p75": 3.0, "p95": 10.0
    },
    "item_interaction_counts": {
      "min": 1, "max": 409789, "mean": 242.5, "median": 45.0, "std": 1820.1,
      "p25": 8.0, "p75": 162.0, "p95": 1120.0
    },
    "long_tail_ratio": 0.43,         # fraction of items with < 5 interactions
    "power_user_ratio": 0.02         # fraction of users with > 50 interactions
  },

  # --- Data Taxonomy Audit (maps to your paper's Table 1) ---
  "taxonomy": {
    "users": {
      "identifiers": true,
      "demographics": false,
      "additional_attributes": false
    },
    "items": {
      "identifiers": true,
      "descriptive_features": true,
      "content_features": true,
      "provider_information": true    # developer/publisher columns present
    },
    "interactions": {
      "explicit_feedback": true,
      "implicit_feedback": false,
      "timestamp": true,
      "contextual_features": true,
      "session_data": false
    },
    "secondary": {
      "system_data": false,
      "feedback_and_control": false,
      "external_knowledge": false
    }
  },

  # --- Stakeholder Role Support (maps to your paper's Table 2) ---
  "stakeholder_support": {
    "consumer": {"supported": true, "basis": "user identifiers and interaction logs present"},
    "system": {"supported": true, "basis": "price data present (price__float in items)"},
    "provider": {"supported": true, "basis": "developer and publisher columns present"},
    "upstream": {"supported": false, "basis": "no external knowledge or upstream metadata"},
    "downstream": {"supported": false, "basis": "no social graph or user-to-user data"},
    "third_party": {"supported": false, "basis": "offline dataset; no regulatory/external signals"}
  },

  # --- Column Inventory (one entry per DataFrame) ---
  "columns": {
    "interactions": [
      {"name": "user_index", "dtype": "Int64", "null_count": 0, "null_pct": 0.0, "feature_type": "id"},
      {"name": "item_index", "dtype": "Int64", "null_count": 0, "null_pct": 0.0, "feature_type": "id"},
      {"name": "timestamp", "dtype": "datetime64[ns]", "null_count": 0, "null_pct": 0.0, "feature_type": "timestamp"},
      {"name": "hours_played__float", "dtype": "float32", "null_count": 12043, "null_pct": 0.15, "feature_type": "float"}
    ],
    "items": [...],
    "users": [...]
  },

  # --- Raw File Inventory ---
  "raw_files": {
    "interactions": {"filename": "steam_reviews.csv", "format": "csv", "n_rows": 7793069, "n_cols": 12, "size_mb": 312.4},
    "items": {"filename": "steam_games.csv", "format": "csv", "n_rows": 32135, "n_cols": 16, "size_mb": 2.1},
    "users": null
  },

  # --- Processing Log ---
  "warnings": [
    "users: no user file provided for this dataset",
    "items: 0 items without interaction records",
    "2,567,538 users without metadata rows (no user file)"
  ]
}
```


**Taxonomy inference rules** (auto-detected from column names in processed DataFrames, do not rely solely on config):
- `demographics`: any column with `age`, `gender`, `occupation`, `location`, `country` in name
- `provider_information`: any column with `developer`, `publisher`, `author`, `seller`, `brand`, `manufacturer`, `creator`, `artist` in name
- `content_features`: any column with `__text` suffix
- `session_data`: any column with `session` in name
- `external_knowledge`: any column with `social`, `friend`, `follow`, `trust`, `graph`, `kg_` in name


**Deliverable for Phase 3:** Running the pipeline on Steam data produces `dataset_profile.json` with all the fields above correctly populated.

---

## Phase 4: Exporter + Markdown Report

**Goal:** Save everything to disk in a clean, portable output structure.

### Output Structure

```
{output_path}/
└── {dataset_name}/
    ├── processed/
    │   ├── interactions.parquet
    │   ├── items.parquet
    │   └── users.parquet           # may be empty/null for datasets without user files
    ├── mappings/
    │   ├── user_map.json           # {"original_id": integer_index, ...}
    │   └── item_map.json
    └── profile/
        ├── dataset_profile.json    # full profile dict from Phase 3
        └── dataset_report.md       # human-readable markdown summary
```

### Parquet Export

- Use `pyarrow` engine
- For `object` dtype columns: cast to `pd.StringDtype()` before saving
- For list/sequence columns (`token_seq`): store as `object` (list of strings) — pyarrow handles this natively
- Do NOT include RecBole-style `field:type` headers — that is a downstream concern

### Markdown Report (`dataset_report.md`)

Generate a clean, readable `.md` file from the profile dict. Include:

1. **Header**: dataset name, domain, version, source URL, processing date
2. **Summary table**: n_users, n_items, n_interactions, sparsity, has_timestamp, has_rating
3. **Data Taxonomy table**: rows are taxonomy elements, column is ✓ / ✗
4. **Stakeholder Support table**: rows are stakeholder roles, columns are Supported (✓/✗) and Basis
5. **Distribution statistics**: formatted as a small table for user counts and item counts
6. **Column inventory**: one section per DataFrame (interactions / items / users) with a table of column name, dtype, null%, feature type
7. **Processing warnings**: bulleted list

**Deliverable for Phase 4:** Complete end-to-end pipeline run produces all output files. The markdown report renders correctly on GitHub.

---

## Phase 5: Web Interface

**Goal:** A clean, local web application that reads the profile JSONs from the output directory and presents them in a dataset explorer UI.

**Important design decisions:**
- The web app is **read-only** — it only reads from the `profile/dataset_profile.json` files. It never triggers processing or downloads from the internet.
- The "Download" button packages and downloads the **already-processed Parquet files** from the `processed/` directory as a `.zip` — it does not re-run the pipeline.
- The web app discovers available datasets by scanning the output directory for `*/profile/dataset_profile.json` files.

### Backend (`webapp/app.py`)

Use **Flask** (simpler for this use case than FastAPI since there's no async requirement).

Routes:
- `GET /` → dataset listing page
- `GET /dataset/<name>` → dataset detail page
- `GET /api/datasets` → JSON list of all available datasets (minimal info: name, domain, n_users, n_items, n_interactions, stakeholder_support summary)
- `GET /api/dataset/<name>` → full profile JSON for one dataset
- `GET /download/<name>` → streams a `.zip` of the `processed/` Parquet files for that dataset

Config: Flask app reads an `OUTPUT_DIR` environment variable (or a config file) to know where to look for processed datasets.

### Frontend Design

The UI should feel like a **research data catalogue** — clean, academic but not boring, data-forward. Think a well-designed academic tool, not a corporate dashboard.

**Aesthetic direction**: refined editorial minimalism with clear typographic hierarchy. Dark or light is your call — commit to one fully. Use a distinctive serif or semi-serif display font for headings and a clean mono or humanist sans for data/labels. The colour palette should be restrained — two or three colours maximum, used deliberately.

**Page 1: Dataset Listing (`/`)**

A catalogue view. For each available dataset, show a card containing:
- Dataset name (large, prominent)
- Domain badge (e.g., "Media & Entertainment")
- Three key stats inline: n_users / n_items / n_interactions (formatted with K/M suffixes)
- Sparsity as a thin horizontal fill bar
- Stakeholder role badges: small coloured pills for each supported role (C / S / P / U / D) — filled if supported, outlined if not
- "View Details" button

Cards should be scannable at a glance. A researcher should be able to immediately see which datasets support which stakeholder roles without clicking into anything.

**Page 2: Dataset Detail (`/dataset/<name>`)**

A single dataset page. Use a tabbed or sectioned layout with these sections:

**Section A — Overview**
- Full dataset name, domain, business model, version, source URL (clickable)
- Description paragraph
- Key stats: n_users, n_items, n_interactions, sparsity — displayed as large stat cards, not a table

**Section B — Stakeholder Representation**
This section maps directly to Table 2 of your paper. Show:
- A visual matrix or card grid with the 6 stakeholder roles (Consumer, System, Provider, Upstream, Downstream, Third-party)
- For each: supported (✓) or not (✗), with the `basis` text shown as a subtitle
- Make it visually clear — not just a text table. Use icons or visual indicators.

**Section C — Data Taxonomy**
This section maps directly to Table 1 of your paper. Show:
- Grouped by taxonomy category (Users / Items / Interactions / Secondary)
- Each element as a row: name + presence indicator
- This should look like a structured audit, not a random list

**Section D — Column Explorer**
Three sub-tabs: Interactions | Items | Users

For each, show a table with columns:
- Column name
- Feature type (id / timestamp / float / token / token_seq / text)
- Data type (dtype)
- Null %
- A thin null % bar visualisation

Columns should be colour-coded by feature type with a consistent legend.

**Section E — Distribution Statistics**
Two small charts side by side:
- User interaction count distribution (log-scale histogram or summary bar chart)
- Item interaction count distribution (log-scale histogram or summary bar chart)

Use plain JavaScript with Canvas or SVG — no external chart library unless you use Chart.js from cdnjs. Keep it lightweight.

**Section F — Download**
A prominent download button:
- Shows what will be downloaded: "3 Parquet files (interactions.parquet, items.parquet, users.parquet)"
- Shows approximate total size
- On click: triggers `GET /download/<name>` which streams the zip
- Clear note: "Downloads the standardised, processed dataset. Raw data is not included."

### Implementation Notes for the Web Interface

- Use **Jinja2 templates** (Flask built-in) for server-side rendering. Avoid a full SPA framework — this is a local research tool, not a deployed product.
- Keep JavaScript minimal. The page should be mostly server-rendered with a small amount of JS for the column explorer tabs and the charts.
- All data for the detail page comes from a single `fetch('/api/dataset/<name>')` call on page load, populated into the already-rendered template structure.
- The download endpoint should use `flask.send_file` with `as_attachment=True` on a dynamically created zip in memory (`io.BytesIO`).
- Error handling: if a dataset's profile JSON is malformed or missing, show a clear error state — not a 500 page.

**Deliverable for Phase 5:** Running `flask run` from the `webapp/` directory and navigating to `http://localhost:5000` shows the dataset catalogue. Clicking a dataset shows the full detail view. The download button produces a valid zip containing the Parquet files.

---

## Phase 6: Tests + README

**Goal:** Make the project usable by someone other than you.

### Tests (`tests/`)

Write pytest tests for:

- `tests/test_file_reader.py`: test each format (CSV, JSONL, Parquet, GZip) using small synthetic fixtures in `tests/fixtures/`
- `tests/test_standardiser.py`: test column renaming, type casting, null ID removal — use synthetic DataFrames
- `tests/test_feature_processor.py`: test feature map parsing, column renaming, drop, token_seq parsing
- `tests/test_profiler.py`: test that profile output contains all required top-level keys and that taxonomy inference rules work correctly
- `tests/test_pipeline.py`: end-to-end test with a tiny synthetic dataset (20 rows) verifying output files are created

### README (`README.md`)

Sections:
1. **What this is** — one paragraph
2. **Installation** — `pip install -r requirements.txt`
3. **Adding a new dataset** — step-by-step: create a YAML config, run the pipeline, check the output
4. **Running the web interface** — how to start the Flask app, what `OUTPUT_DIR` to set
5. **Output structure** — describe the folder layout
6. **Understanding the profile** — brief explanation of taxonomy and stakeholder fields, with reference to the Dagstuhl Seminar taxonomy

---

## Implementation Constraints (apply throughout all phases)

**No RecBole-specific outputs**: Do not generate `.inter`, `.item`, `.user` atomic files. Do not use `field:type` colon notation in column names (use `__type` double underscore instead). RecBole conversion is a separate downstream step.

**No filtering**: The library does not implement k-core, minimum interaction counts, deduplication, or any other filtering. It loads, standardises, and profiles. That's it.

**Storage format**: Parquet with Snappy compression (pyarrow default). This is 5–10x smaller than CSV/JSON for typical recommender datasets and far faster to load for downstream use.

**ID handling**: `user_id` and `item_id` are always strings (object dtype) in the standardised DataFrames — even if they appear numeric. `user_index` and `item_index` (the mapped integer indices) are `Int64`.

**Memory efficiency**: 
- Never load a file twice
- Print memory usage (`df.memory_usage(deep=True).sum() / 1e6` MB) after loading each file
- For JSONL files > 500k rows, use a chunked reader with progress reporting

**Error handling**:
- `DatasetLoadError`: file not found or unreadable
- `ConfigValidationError`: missing/invalid config key
- `ColumnNotFoundError`: declared column not in DataFrame — list ALL missing columns at once
- Never silently swallow exceptions — always log or re-raise

**Code quality**:
- Full type hints (Python 3.10+)
- Google-style docstrings on all public functions
- Pure functions in `processing/` (no side effects, DataFrame in → DataFrame out)
- No global state

**Dependencies (minimal)**:
```
pandas>=2.0
pyarrow
pyyaml
numpy
flask
```
Optional: `tqdm` for progress bars.

---

## Reference Code

Draw from these existing implementations when writing the library:

**From the attached ETL module (`base_converter.py`, `dataset_converters.py`, `utils.py`):**
- Keep the `load → process → save → run` pipeline structure from `AbstractBaseConverter`
- Keep `process_feature_map` / `_parse_feature_map` logic — it is correct, just move the feature maps to YAML
- Keep `_safe_save_parquet` helper
- Keep `standardise_pipeline` structure
- **Fix**: `cast_ids_to_int` should become `cast_ids_to_string` — IDs are tokens

**From the attached pipeline module (`data_utils.py`, `data_pipelines.py`):**
- Port `create_id_mappings`, `apply_id_mappings`, `align_dataset_indices` directly — they are well-structured
- Port `cast_timestamp_to_datetime` with the log10 unit detection
- Port `select_final_inter_columns` but generalise based on config
- **Fix**: same ID casting issue — string, not integer, for OIDs

**From the attached notebook (`data_loading.ipynb`):**
- Port the `parse()` / JSONL reader pattern into `file_reader.py` as the fallback JSON parser for Python-repr dicts

---

## Summary of Phases

| Phase | Deliverable | Test it by... |
|-------|------------|---------------|
| 1 | Config system + file reader | Reading a raw CSV/JSONL into a DataFrame |
| 2 | Standardisation pipeline | `--dry-run` mode prints rename/type plan |
| 3 | ID mapping + profiler | `dataset_profile.json` produced correctly |
| 4 | Exporter + markdown report | All output files present, report renders on GitHub |
| 5 | Web interface | `flask run` → browseable dataset catalogue with download |
| 6 | Tests + README | `pytest` passes; a new teammate can onboard from README alone |

**Remember: complete and verify each phase before starting the next.**
