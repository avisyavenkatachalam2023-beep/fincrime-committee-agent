"""
data/download_data.py
---------------------
Downloads the SAML-D (Synthetic AML Dataset) from Kaggle.
Falls back to generating a fully synthetic dataset if the download fails.

Dataset: berkanoztas/synthetic-transaction-monitoring-dataset-aml
"""

import os
import sys
import zipfile
import shutil
import importlib.util
import io
from pathlib import Path

import requests
from dotenv import load_dotenv

# Force UTF-8 output on Windows to prevent cp1252 UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is on sys.path so sibling imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KAGGLE_DATASET_OWNER = "berkanoztas"
KAGGLE_DATASET_NAME = "synthetic-transaction-monitoring-dataset-aml"
KAGGLE_API_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/"
    f"{KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_NAME}"
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
SYNTHETIC_SCRIPT = PROJECT_ROOT / "data" / "synthetic" / "generate_synthetic.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_banner(msg: str) -> None:
    """Print a clearly visible status banner."""
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {msg}")
    print(f"{line}\n")


def _ensure_dir(path: Path) -> None:
    """Create directory (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def _kaggle_token() -> str:
    """
    Retrieve the Kaggle bearer token from the environment.

    Returns
    -------
    str
        The raw token string (KGAT_...).

    Raises
    ------
    EnvironmentError
        If KAGGLE_TOKEN is not set in the environment.
    """
    token = os.getenv("KAGGLE_TOKEN")
    if not token:
        raise EnvironmentError(
            "KAGGLE_TOKEN is not set. Add it to your .env file or environment."
        )
    return token


# ---------------------------------------------------------------------------
# Kaggle download
# ---------------------------------------------------------------------------

def download_from_kaggle() -> bool:
    """
    Attempt to download the SAML-D dataset from Kaggle using the bearer token.

    The downloaded ZIP archive is extracted into ``data/raw/``.

    Returns
    -------
    bool
        True if the download and extraction succeeded, False otherwise.
    """
    try:
        token = _kaggle_token()
    except EnvironmentError as exc:
        print(f"[WARN] Token error: {exc}")
        return False

    _ensure_dir(RAW_DIR)
    zip_path = RAW_DIR / f"{KAGGLE_DATASET_NAME}.zip"

    _print_banner("Attempting Kaggle dataset download...")
    print(f"  URL   : {KAGGLE_API_URL}")
    print(f"  Target: {zip_path}\n")

    headers = {"Authorization": f"Bearer {token}"}

    try:
        with requests.get(KAGGLE_API_URL, headers=headers, stream=True, timeout=120) as resp:
            resp.raise_for_status()

            total_bytes = int(resp.headers.get("Content-Length", 0))
            downloaded = 0

            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total_bytes:
                            pct = downloaded / total_bytes * 100
                            print(f"\r  Progress: {pct:5.1f}%  ({downloaded:,} / {total_bytes:,} bytes)", end="")

        print(f"\n  Download complete -> {zip_path}")

    except requests.exceptions.HTTPError as exc:
        print(f"\n[ERROR] HTTP {exc.response.status_code}: {exc.response.reason}")
        print(f"        Response body: {exc.response.text[:400]}")
        _cleanup_zip(zip_path)
        return False

    except requests.exceptions.ConnectionError as exc:
        print(f"\n[ERROR] Connection failed: {exc}")
        _cleanup_zip(zip_path)
        return False

    except requests.exceptions.Timeout:
        print("\n[ERROR] Request timed out after 120 s.")
        _cleanup_zip(zip_path)
        return False

    except Exception as exc:  # noqa: BLE001
        print(f"\n[ERROR] Unexpected error during download: {exc}")
        _cleanup_zip(zip_path)
        return False

    # --- Extract the ZIP ---
    try:
        _print_banner(f"Extracting {zip_path.name} -> {RAW_DIR}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()  # remove zip after extraction
        extracted = list(RAW_DIR.iterdir())
        print(f"  Extracted {len(extracted)} item(s):")
        for item in extracted:
            size_kb = item.stat().st_size / 1024 if item.is_file() else 0
            print(f"    {'[DIR] ' if item.is_dir() else ''}{item.name}" + (f"  ({size_kb:,.0f} KB)" if item.is_file() else ""))
        return True

    except zipfile.BadZipFile as exc:
        print(f"[ERROR] The downloaded file is not a valid ZIP: {exc}")
        _cleanup_zip(zip_path)
        return False

    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Extraction failed: {exc}")
        _cleanup_zip(zip_path)
        return False


def _cleanup_zip(path: Path) -> None:
    """Remove a partially-downloaded or corrupt ZIP file."""
    if path.exists():
        path.unlink()
        print(f"  Cleaned up: {path.name}")


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def run_synthetic_generator() -> bool:
    """
    Import and execute ``data/synthetic/generate_synthetic.py`` to produce a
    fully synthetic AML dataset.

    Returns
    -------
    bool
        True if the generator ran without raising an exception, False otherwise.
    """
    _print_banner("Falling back to synthetic data generation...")
    print(f"  Script: {SYNTHETIC_SCRIPT}\n")

    if not SYNTHETIC_SCRIPT.exists():
        print(f"[ERROR] Synthetic generator not found at {SYNTHETIC_SCRIPT}")
        return False

    try:
        spec = importlib.util.spec_from_file_location("generate_synthetic", SYNTHETIC_SCRIPT)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        # Call the public entry-point if it exists, otherwise the module-level
        # code already ran during exec_module.
        if hasattr(module, "main"):
            module.main()

        _print_banner("Synthetic data generation completed successfully.")
        return True

    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Synthetic generator raised an exception: {exc}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the data acquisition pipeline:

    1. Try to download from Kaggle.
    2. On any failure, fall back to generating synthetic data locally.
    3. Exit with code 0 on success, 1 on complete failure.
    """
    _print_banner("AML Dataset Acquisition — Financial Crime Committee Agent")

    success = download_from_kaggle()

    if not success:
        print("[INFO] Kaggle download unavailable. Switching to synthetic generation.\n")
        success = run_synthetic_generator()

    if success:
        _print_banner("Data acquisition pipeline COMPLETE [OK]")
        sys.exit(0)
    else:
        _print_banner("Data acquisition pipeline FAILED -- check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
