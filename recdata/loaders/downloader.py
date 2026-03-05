"""Download and extraction support for raw dataset files.

Handles downloading datasets from URLs, verifying checksums, and
extracting archives (ZIP, TAR, TAR.GZ). Inspired by the DataRec
library's source/resource separation pattern.

Files are only downloaded if they are not already present locally.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any

from recdata.exceptions import DownloadError

logger = logging.getLogger(__name__)


def prepare_dataset(
    config: dict[str, Any],
    raw_path: str | Path,
) -> dict[str, Path]:
    """Ensure all raw dataset files are available locally.

    Checks if each declared file exists in raw_path. If a 'source' block
    is defined in the config and files are missing, downloads and extracts
    them automatically.

    Args:
        config: Validated dataset config dictionary.
        raw_path: Directory where raw files should be located.

    Returns:
        A dict mapping file roles (e.g., 'interactions', 'items') to
        their local file paths.

    Raises:
        DownloadError: If download or extraction fails.
    """
    raw_path = Path(raw_path)
    raw_path.mkdir(parents=True, exist_ok=True)

    files_config = config.get("files", {})
    source_config = config.get("source")

    file_paths: dict[str, Path] = {}
    missing_files: list[str] = []

    # Check which files exist locally
    for role, file_def in files_config.items():
        if file_def is None:
            continue
        filename = file_def["filename"]
        local_path = raw_path / filename
        if local_path.exists() and local_path.stat().st_size > 0:
            file_paths[role] = local_path
            logger.info("Found local file for '%s': %s", role, local_path)
        else:
            missing_files.append(role)
            logger.info("Missing local file for '%s': %s", role, local_path)

    # If all files are present, return immediately
    if not missing_files:
        logger.info("All files found locally — skipping download")
        return file_paths

    # If files are missing but no source config, report what's needed
    if source_config is None:
        missing_names = [files_config[role]["filename"] for role in missing_files]
        raise DownloadError(
            f"Missing files for dataset '{config.get('dataset_name', '?')}': "
            f"{missing_names}. No 'source' block in config — cannot download automatically. "
            f"Please place the files in: {raw_path}"
        )

    # Download and extract
    logger.info(
        "Downloading dataset '%s' (%d missing files)...",
        config.get("dataset_name", "?"),
        len(missing_files),
    )

    url = source_config.get("url")
    if not url:
        raise DownloadError("'source.url' is required for automatic download")

    archive_type = source_config.get("archive")
    checksum = source_config.get("checksum")
    checksum_algo = source_config.get("checksum_algorithm", "md5")
    inner_paths = source_config.get("inner_paths", {})

    if archive_type:
        # Download archive file
        archive_suffix = {
            "zip": ".zip",
            "tar": ".tar",
            "tar.gz": ".tar.gz",
            "tgz": ".tar.gz",
        }.get(archive_type, f".{archive_type}")

        archive_path = raw_path / f"_download{archive_suffix}"

        download_file(url, archive_path, checksum=checksum, checksum_algo=checksum_algo)

        # Extract needed files
        extract_archive(
            archive_path=archive_path,
            dest_path=raw_path,
            archive_type=archive_type,
            inner_paths=inner_paths if inner_paths else None,
        )

        # Clean up archive
        if archive_path.exists():
            archive_path.unlink()
            logger.info("Cleaned up archive: %s", archive_path.name)
    else:
        # Direct file download (no archive)
        # For each missing file, check if there's a specific URL or use the base URL
        for role in missing_files:
            filename = files_config[role]["filename"]
            dest = raw_path / filename
            # If source has per-file URLs, use them
            file_urls = source_config.get("file_urls", {})
            file_url = file_urls.get(role, url)
            download_file(file_url, dest, checksum=checksum, checksum_algo=checksum_algo)

    # Verify all files are now present
    for role in missing_files:
        filename = files_config[role]["filename"]
        local_path = raw_path / filename
        if local_path.exists() and local_path.stat().st_size > 0:
            file_paths[role] = local_path
            logger.info("Successfully prepared '%s': %s", role, local_path)
        else:
            raise DownloadError(
                f"File for '{role}' still missing after download/extraction: {local_path}"
            )

    return file_paths


def download_file(
    url: str,
    dest_path: str | Path,
    checksum: str | None = None,
    checksum_algo: str = "md5",
) -> Path:
    """Download a file from a URL with optional checksum verification.

    Args:
        url: URL to download from.
        dest_path: Local path to save the file.
        checksum: Expected checksum string (optional).
        checksum_algo: Hash algorithm for checksum ('md5', 'sha256', etc.).

    Returns:
        Path to the downloaded file.

    Raises:
        DownloadError: If download fails or checksum doesn't match.
    """
    try:
        import requests
    except ImportError:
        raise DownloadError(
            "The 'requests' package is required for downloading. "
            "Install it with: pip install requests"
        )

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading: %s", url)
    logger.info("Destination: %s", dest_path)

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise DownloadError(f"Download failed for {url}: {e}") from e

    # Get total size for progress reporting
    total_size = int(response.headers.get("content-length", 0))

    try:
        from tqdm import tqdm

        has_tqdm = True
    except ImportError:
        has_tqdm = False

    hasher = hashlib.new(checksum_algo) if checksum else None
    downloaded = 0

    with open(dest_path, "wb") as f:
        if has_tqdm and total_size > 0:
            progress = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=dest_path.name,
            )
        else:
            progress = None

        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            if hasher:
                hasher.update(chunk)
            downloaded += len(chunk)

            if progress:
                progress.update(len(chunk))
            elif downloaded % (50 * 1024 * 1024) == 0:  # Log every 50 MB
                if total_size > 0:
                    pct = 100 * downloaded / total_size
                    logger.info("  ... %.1f%% (%d MB)", pct, downloaded // (1024 * 1024))
                else:
                    logger.info("  ... %d MB downloaded", downloaded // (1024 * 1024))

        if progress:
            progress.close()

    logger.info("Downloaded %.1f MB", downloaded / (1024 * 1024))

    # Verify checksum
    if checksum and hasher:
        computed = hasher.hexdigest()
        if computed != checksum:
            dest_path.unlink(missing_ok=True)
            raise DownloadError(
                f"Checksum mismatch for {dest_path.name}: "
                f"expected {checksum}, got {computed} ({checksum_algo})"
            )
        logger.info("Checksum verified (%s): %s", checksum_algo, computed)

    return dest_path


def extract_archive(
    archive_path: str | Path,
    dest_path: str | Path,
    archive_type: str,
    inner_paths: dict[str, str] | None = None,
) -> list[Path]:
    """Extract files from an archive.

    Args:
        archive_path: Path to the archive file.
        dest_path: Directory to extract files to.
        archive_type: Type of archive ('zip', 'tar', 'tar.gz', 'tgz').
        inner_paths: Optional dict mapping roles to paths within the archive.
            If provided, only these files are extracted. The files are placed
            directly in dest_path using their basename (not the full archive path).

    Returns:
        List of paths to extracted files.

    Raises:
        DownloadError: If extraction fails.
    """
    archive_path = Path(archive_path)
    dest_path = Path(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    extracted: list[Path] = []

    try:
        if archive_type == "zip":
            extracted = _extract_zip(archive_path, dest_path, inner_paths)
        elif archive_type in ("tar", "tar.gz", "tgz"):
            extracted = _extract_tar(archive_path, dest_path, inner_paths)
        else:
            raise DownloadError(f"Unsupported archive type: '{archive_type}'")
    except (zipfile.BadZipFile, tarfile.TarError) as e:
        raise DownloadError(f"Failed to extract archive '{archive_path}': {e}") from e

    logger.info("Extracted %d files to %s", len(extracted), dest_path)
    return extracted


def _extract_zip(
    archive_path: Path,
    dest_path: Path,
    inner_paths: dict[str, str] | None = None,
) -> list[Path]:
    """Extract files from a ZIP archive."""
    extracted: list[Path] = []

    with zipfile.ZipFile(archive_path, "r") as zf:
        if inner_paths:
            # Extract only specified files
            all_names = zf.namelist()
            for role, inner_path in inner_paths.items():
                # Find the matching file in the archive
                match = None
                for name in all_names:
                    if name == inner_path or name.endswith("/" + inner_path):
                        match = name
                        break

                if match is None:
                    raise DownloadError(
                        f"Inner path '{inner_path}' (role: {role}) not found in ZIP archive. "
                        f"Available: {all_names}"
                    )

                # Extract to dest_path with just the basename
                basename = Path(inner_path).name
                target = dest_path / basename
                with zf.open(match) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(target)
                logger.info("  Extracted: %s -> %s", match, target.name)
        else:
            # Extract everything
            zf.extractall(dest_path)
            extracted = [dest_path / name for name in zf.namelist() if not name.endswith("/")]

    return extracted


def _extract_tar(
    archive_path: Path,
    dest_path: Path,
    inner_paths: dict[str, str] | None = None,
) -> list[Path]:
    """Extract files from a TAR archive (including .tar.gz, .tar.bz2)."""
    extracted: list[Path] = []

    with tarfile.open(archive_path, "r:*") as tf:
        if inner_paths:
            all_names = [m.name for m in tf.getmembers() if m.isfile()]
            for role, inner_path in inner_paths.items():
                match = None
                for name in all_names:
                    if name == inner_path or name.endswith("/" + inner_path):
                        match = name
                        break

                if match is None:
                    raise DownloadError(
                        f"Inner path '{inner_path}' (role: {role}) not found in TAR archive. "
                        f"Available: {all_names}"
                    )

                member = tf.getmember(match)
                basename = Path(inner_path).name
                target = dest_path / basename

                f = tf.extractfile(member)
                if f is None:
                    raise DownloadError(f"Cannot extract '{match}' from TAR archive")

                with f, open(target, "wb") as dst:
                    shutil.copyfileobj(f, dst)
                extracted.append(target)
                logger.info("  Extracted: %s -> %s", match, target.name)
        else:
            # Extract everything safely (avoid path traversal)
            for member in tf.getmembers():
                # Security: skip members with absolute paths or path traversal
                if member.name.startswith("/") or ".." in member.name:
                    logger.warning("Skipping suspicious archive member: %s", member.name)
                    continue
                tf.extract(member, dest_path, filter="data")
                if member.isfile():
                    extracted.append(dest_path / member.name)

    return extracted
