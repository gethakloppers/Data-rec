"""Custom exceptions for the RecData library."""


class DatasetLoadError(Exception):
    """Raised when a dataset file cannot be read or parsed.

    This covers file-not-found, encoding errors, format mismatches,
    and any other I/O failure during file reading.
    """

    pass


class ConfigValidationError(Exception):
    """Raised when a dataset YAML config is missing required fields or has invalid values.

    Examples:
        - Missing 'dataset_name' key
        - Missing 'files' block
        - Invalid format specifier
    """

    pass


class ColumnNotFoundError(Exception):
    """Raised when declared columns are not found in a DataFrame.

    Always lists ALL missing columns at once, rather than failing on the first one.
    This helps users fix all column mapping issues in a single pass.

    Attributes:
        missing_columns: List of column names that were not found.
        available_columns: List of column names that are available in the DataFrame.
    """

    def __init__(
        self,
        missing_columns: list[str],
        available_columns: list[str] | None = None,
        message: str | None = None,
    ):
        self.missing_columns = missing_columns
        self.available_columns = available_columns
        if message is None:
            msg = f"Columns not found in DataFrame: {missing_columns}"
            if available_columns is not None:
                msg += f"\nAvailable columns: {available_columns}"
            message = msg
        super().__init__(message)


class DownloadError(Exception):
    """Raised when a dataset download or archive extraction fails.

    This covers network errors, checksum mismatches, corrupt archives,
    and missing inner paths within archives.
    """

    pass
