"""Configuration loader and validator for dataset YAML configs.

Loads YAML config files and validates that all required fields are present
before the pipeline attempts to use them.

File entries in the ``files`` block can be written as either a plain filename
string (shorthand) or a full dict. In both cases, ``format``, ``encoding``,
and ``separator`` are optional — they are auto-detected from the file at
read time if not supplied.

Shorthand example::

    files:
      interactions: "steam_reviews.json"   # format/encoding auto-detected
      items: "steam_games.json"
      users: null

Full dict example (explicit overrides)::

    files:
      interactions:
        filename: "ratings.csv"
        separator: ";"              # override auto-detection

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from recdata.exceptions import ConfigValidationError

logger = logging.getLogger(__name__)

# Required top-level keys in every dataset config
REQUIRED_KEYS = ["dataset_name", "files", "schema"]

# Required keys within the 'schema' block
REQUIRED_SCHEMA_KEYS = ["user_identifier", "item_identifier"]

# Valid file format specifiers (used only when format is declared explicitly)
VALID_FORMATS = {"csv", "tsv", "json", "jsonl", "parquet", "gz", "zip", "tar"}


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a dataset YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        A validated config dictionary.

    Raises:
        ConfigValidationError: If required keys are missing or values are invalid.
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ConfigValidationError(f"Config file is empty: {config_path}")

    _validate_config(config, config_path)

    logger.info("Loaded config for dataset '%s' from %s", config["dataset_name"], config_path.name)

    return config


def normalize_file_def(file_def: str | dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalise a file definition to a full dict form.

    Accepts three forms:
    - ``None`` → ``None`` (no file for this role)
    - A plain string → ``{"filename": string}``
    - A dict → returned as-is (already in full form)

    ``format``, ``encoding``, and ``separator`` are intentionally left absent
    when not declared — the reader will auto-detect them from the actual file.

    Args:
        file_def: The raw value from ``config["files"][role]``.

    Returns:
        A normalised dict with at least a ``"filename"`` key, or ``None``.
    """
    if file_def is None:
        return None
    if isinstance(file_def, str):
        return {"filename": file_def}
    return dict(file_def)  # shallow copy to avoid mutating the config


def get_feature_map(config: dict[str, Any], df_role: str) -> dict[str, str]:
    """Build a flat column-to-type mapping from the config's feature declarations.

    Args:
        config: The validated config dictionary.
        df_role: One of 'interactions', 'items', 'users'.

    Returns:
        A dict mapping column names (lowercase) to their declared types.
        Example: ``{'publisher': 'token', 'price': 'float', 'genres': 'token_seq'}``
    """
    feature_key = {
        "interactions": "interaction_features",
        "items": "item_features",
        "users": "user_features",
    }.get(df_role)

    if feature_key is None or feature_key not in config or config[feature_key] is None:
        return {}

    feature_config = config[feature_key]
    flat_map: dict[str, str] = {}

    for feat_type, columns in feature_config.items():
        if isinstance(columns, list):
            for col in columns:
                flat_map[col.lower()] = feat_type

    return flat_map


def _validate_config(config: dict[str, Any], config_path: Path) -> None:
    """Validate that a config dict contains all required fields.

    Args:
        config: The parsed YAML config dictionary.
        config_path: Path to the config file (for error messages).

    Raises:
        ConfigValidationError: If validation fails. All errors are collected
            and reported at once.
    """
    errors: list[str] = []

    # Check required top-level keys
    for key in REQUIRED_KEYS:
        if key not in config:
            errors.append(f"Missing required top-level key: '{key}'")

    # Validate 'files' block
    if "files" in config:
        files = config["files"]
        if not isinstance(files, dict):
            errors.append("'files' must be a dict mapping roles to file definitions")
        else:
            if "interactions" not in files or files["interactions"] is None:
                errors.append("'files.interactions' must be defined (cannot be null)")

            for role, file_def in files.items():
                if file_def is None:
                    continue
                # Accept string shorthand or dict
                if isinstance(file_def, str):
                    if not file_def.strip():
                        errors.append(f"'files.{role}' filename string cannot be empty")
                    continue
                if not isinstance(file_def, dict):
                    errors.append(
                        f"'files.{role}' must be a filename string or a dict (got {type(file_def).__name__})"
                    )
                    continue
                if "filename" not in file_def:
                    errors.append(f"'files.{role}' missing required key 'filename'")
                # format is now optional — only validate it if explicitly provided
                if "format" in file_def and file_def["format"] not in VALID_FORMATS:
                    errors.append(
                        f"'files.{role}.format' has invalid value '{file_def['format']}'. "
                        f"Must be one of: {sorted(VALID_FORMATS)}"
                    )

    # Validate 'schema' block
    if "schema" in config:
        schema = config["schema"]
        if not isinstance(schema, dict):
            errors.append("'schema' must be a dict mapping roles to column name candidates")
        else:
            for key in REQUIRED_SCHEMA_KEYS:
                if key not in schema:
                    errors.append(f"Missing required schema key: '{key}'")
                elif not isinstance(schema[key], list) or len(schema[key]) == 0:
                    errors.append(
                        f"'schema.{key}' must be a non-empty list of candidate column names"
                    )

    # Validate feature declarations (optional, but must be well-formed if present)
    for feature_key in ["item_features", "interaction_features", "user_features"]:
        if feature_key in config and config[feature_key] is not None:
            features = config[feature_key]
            if not isinstance(features, dict):
                errors.append(f"'{feature_key}' must be a dict or null")
            else:
                valid_types = {"token", "float", "token_seq", "text", "drop"}
                for feat_type, columns in features.items():
                    if feat_type not in valid_types:
                        errors.append(
                            f"'{feature_key}' has invalid type '{feat_type}'. "
                            f"Must be one of: {sorted(valid_types)}"
                        )
                    if not isinstance(columns, list):
                        errors.append(
                            f"'{feature_key}.{feat_type}' must be a list of column names"
                        )

    # Validate stakeholder_roles (optional, must be well-formed if present)
    if "stakeholder_roles" in config and config["stakeholder_roles"] is not None:
        roles = config["stakeholder_roles"]
        valid_roles = {"consumer", "system", "provider", "upstream", "downstream", "third_party"}
        for role_name in roles:
            if role_name not in valid_roles:
                errors.append(
                    f"'stakeholder_roles' has unknown role '{role_name}'. "
                    f"Must be one of: {sorted(valid_roles)}"
                )

    if errors:
        error_msg = f"Config validation failed for '{config_path}':\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ConfigValidationError(error_msg)
