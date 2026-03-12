# Plan: Update ConfigApp Type System

## Summary
Overhaul the type system across configapp and recdata:
- **New types**: Numerical→`float`, Categorical→`object`, Text→`string`, Datetime→`datetime`, Boolean→`bool`, Misc→`misc`
- **Remove**: `url` type, `exclude` type (moves to schema), separator column
- **Add**: Auto-detected separator for list display, "Exclude" as a schema/taxonomy category

## Changes by File

### 1. `configapp/static/js/app.js`
- Update `TYPE_LABELS`: 6 types (float, object, string, datetime, bool, misc)
- Add `"exclude"` category to `TAXONOMY` for each role
- Remove `separator` from column config objects (`{ type, schema }` only)
- Update `getSamples()` to use auto-detected separators from preview data instead of `cfg.separator`
- Update `autoUpdateSchema()` — remove exclude-type guard
- Update `allAvailableColumns` — filter by `cfg.schema === "exclude"` instead of `cfg.type === "exclude"`
- Update `rowClass()` — new type mappings, exclude check via schema
- Update `_autoSchemaCategory()` — `"text"`→`"string"`, remove `"url"`, `"boolean"`→`"bool"`
- Update `switchTab()` — remove separator from config init

### 2. `configapp/templates/index.html`
- Replace type dropdown options with 6 new types
- Remove separator column (`<th>` and `<td>`)

### 3. `configapp/app.py`
- Update `_WIZARD_TO_YAML_TYPE` and `_YAML_TO_WIZARD_TYPE` with new type names + backward compat
- Update `_suggest_type()` return values: `"boolean"`→`"bool"`, `"text"`→`"string"`, `"url"`→`"string"`
- Update `_build_yaml_config()`: iterate new YAML types, handle exclude-as-schema
- Update `_parse_yaml_to_wizard_state()`: remove separator field, handle old exclude-as-type configs
- Add `"exclude"` to `_TAXONOMY_CATS` for each role

### 4. `recdata/loaders/base_loader.py`
- Add type aliases: `"text"`→`"string"`, `"boolean"`→`"bool"`
- Update `_validate_config` valid types set

### 5. `recdata/processing/standardiser.py`
- Add `bool` type handling in casting step

### 6. `recdata/profiler/quality_report.py` & `dataset_profiler.py`
- Update type checks for `"string"` (was `"text"`) and `"bool"` (was checked via `"object"`)

### 7. Existing YAML configs (optional — aliases handle backward compat)
- Can update `text:` → `string:` and add `bool:` sections for consistency
