"""
data/load_data.py
-----------------
Central data-loading API for the AML Financial Crime Committee Agent.

Priority: raw/ (Kaggle download) → synthetic/ (generated fallback).

Public API
----------
    load_transactions(data_dir='data') → pd.DataFrame
    load_customers(data_dir='data')    → pd.DataFrame
    load_jurisdictions(data_dir='data')→ pd.DataFrame

All three functions normalise column names to a canonical schema regardless
of whether the source is the Kaggle download or the synthetic dataset.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Canonical column schemas
# ---------------------------------------------------------------------------

# Transactions: exact column names expected downstream
TXN_CANONICAL_COLS = [
    "transaction_id",
    "timestamp",
    "sender_account",
    "receiver_account",
    "sender_bank",
    "receiver_bank",
    "sender_country",
    "receiver_country",
    "payment_type",
    "currency",
    "amount",
    "is_suspicious",
    "typology",
]

# Customer profiles
CUST_CANONICAL_COLS = [
    "customer_id",
    "account_id",
    "full_name",
    "occupation",
    "declared_income_band",
    "country_of_residence",
    "account_open_date",
    "is_pep",
    "is_high_risk_jurisdiction",
]

# High-risk jurisdictions
JURIS_CANONICAL_COLS = [
    "country_code",
    "country_name",
    "risk_level",
    "risk_reason",
]

# ---------------------------------------------------------------------------
# Column aliases: maps Kaggle column names → canonical names.
# Add more mappings here as the real dataset schema becomes known.
# ---------------------------------------------------------------------------
TXN_ALIASES: dict[str, str] = {
    # Possible Kaggle column names                   -> canonical
    "txn_id": "transaction_id",
    "id": "transaction_id",
    "trans_id": "transaction_id",
    "date": "timestamp",
    "datetime": "timestamp",
    # NOTE: 'time' maps to nothing — handled in post-processing for SAML-D
    "from_account": "sender_account",
    "to_account": "receiver_account",
    "source_account": "sender_account",
    "dest_account": "receiver_account",
    "originator": "sender_account",
    "beneficiary": "receiver_account",
    # SAML-D exact column names
    "sender_account": "sender_account",
    "receiver_account": "receiver_account",
    "sender_bank_location": "sender_country",
    "receiver_bank_location": "receiver_country",
    "payment_currency": "currency",
    "received_currency": "received_currency",   # keep as extra
    "is_laundering": "is_suspicious",
    "laundering_type": "typology",
    "amount": "amount",
    "payment_type": "payment_type",
    # generic alternatives
    "from_bank": "sender_bank",
    "to_bank": "receiver_bank",
    "from_country": "sender_country",
    "to_country": "receiver_country",
    "payment_method": "payment_type",
    "transaction_type": "payment_type",
    "ccy": "currency",
    "usd_amount": "amount",
    "trans_amount": "amount",
    "suspicious": "is_suspicious",
    "label": "is_suspicious",
    "fraud": "is_suspicious",
    "ml_type": "typology",
    "type": "typology",
}

CUST_ALIASES: dict[str, str] = {
    "name": "full_name",
    "cust_id": "customer_id",
    "acc_id": "account_id",
    "job": "occupation",
    "income": "declared_income_band",
    "country": "country_of_residence",
    "open_date": "account_open_date",
    "pep": "is_pep",
    "high_risk": "is_high_risk_jurisdiction",
}

JURIS_ALIASES: dict[str, str] = {
    "code": "country_code",
    "name": "country_name",
    "level": "risk_level",
    "reason": "risk_reason",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _postprocess_saml_d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle SAML-D specific column transformations after alias mapping:
    - Merge 'Date' + 'Time' columns into a single 'timestamp'
    - Ensure sender_bank / receiver_bank default to sender_country / receiver_country
    - Generate synthetic transaction_id if missing
    """
    # Merge Date + Time into timestamp
    # Note: 'Date' may have been aliased to 'timestamp' by _normalise_columns
    if 'timestamp' in df.columns and 'Time' in df.columns:
        df['timestamp'] = pd.to_datetime(
            df['timestamp'].astype(str) + ' ' + df['Time'].astype(str),
            errors='coerce'
        )
        df = df.drop(columns=['Time'], errors='ignore')
    elif 'timestamp' not in df.columns or df['timestamp'].isna().all():
        if 'Date' in df.columns and 'Time' in df.columns:
            df['timestamp'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                errors='coerce'
            )
            df = df.drop(columns=['Date', 'Time'], errors='ignore')
        elif 'Date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.drop(columns=['Date'], errors='ignore')

    # sender_bank defaults to sender_country for SAML-D (no separate bank column)
    if 'sender_bank' not in df.columns or df['sender_bank'].isna().all():
        if 'sender_country' in df.columns:
            df['sender_bank'] = df['sender_country']
    if 'receiver_bank' not in df.columns or df['receiver_bank'].isna().all():
        if 'receiver_country' in df.columns:
            df['receiver_bank'] = df['receiver_country']

    # Generate transaction_id if not present
    if 'transaction_id' not in df.columns or df['transaction_id'].isna().all():
        import uuid as _uuid
        df['transaction_id'] = [f"TXN_{i:07d}" for i in range(len(df))]

    # Normalise is_suspicious to bool (SAML-D uses 0/1 integers)
    if 'is_suspicious' in df.columns:
        df['is_suspicious'] = df['is_suspicious'].astype(bool)

    # Normalise typology: map SAML-D laundering_type values to uppercase
    if 'typology' in df.columns:
        def _map_typology(val):
            if pd.isna(val):
                return 'NORMAL'
            v = str(val).upper()
            if 'STRUCT' in v or 'CASH' in v:
                return 'STRUCTURING'
            if 'SMURF' in v or 'SCATTER' in v:
                return 'SMURFING'
            if 'LAYER' in v or 'CYCLE' in v or 'FAN' in v:
                return 'LAYERING'
            if 'NORMAL' in v:
                return 'NORMAL'
            return v
        df['typology'] = df['typology'].apply(_map_typology)

    # Drop Time column if still hanging around
    df = df.drop(columns=['Time'], errors='ignore')
    return df

def _resolve_data_dir(data_dir: str | Path) -> Path:
    """
    Resolve the data directory to an absolute path.

    If the provided path is relative, it is resolved relative to the
    project root (one level above this file's parent directory).

    Parameters
    ----------
    data_dir:
        Relative or absolute path to the data directory.

    Returns
    -------
    Path
        Absolute path to the data directory.
    """
    p = Path(data_dir)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p.resolve()


def _normalise_columns(df: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    """
    Rename DataFrame columns using an alias map (case-insensitive).

    Parameters
    ----------
    df:
        Source DataFrame whose columns may use non-canonical names.
    aliases:
        Mapping of ``{source_name: canonical_name}``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns renamed to canonical names where matches found.
    """
    lower_map = {k.lower(): v for k, v in aliases.items()}
    rename = {col: lower_map[col.lower()] for col in df.columns if col.lower() in lower_map}
    if rename:
        df = df.rename(columns=rename)
    return df


def _add_missing_cols(df: pd.DataFrame, canonical: list[str]) -> pd.DataFrame:
    """
    Add any missing canonical columns as NaN columns so downstream code
    can always reference every expected field.

    Parameters
    ----------
    df:
        DataFrame potentially missing some canonical columns.
    canonical:
        List of column names that must exist in the output.

    Returns
    -------
    pd.DataFrame
        DataFrame guaranteed to have all canonical columns (extras preserved).
    """
    for col in canonical:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _coerce_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply type coercions to the transactions DataFrame so dtypes are
    consistent regardless of data source.

    Parameters
    ----------
    df:
        Raw transactions DataFrame after column normalisation.

    Returns
    -------
    pd.DataFrame
        Transactions DataFrame with canonical dtypes applied.
    """
    # Timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)

    # Amount → float
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Boolean flags
    for bool_col in ("is_suspicious",):
        if bool_col in df.columns:
            col = df[bool_col]
            if col.dtype == object or col.dtype == str:
                df[bool_col] = col.map(
                    lambda v: True if str(v).strip().lower() in ("1", "true", "yes") else False
                )
            else:
                df[bool_col] = col.astype(bool, errors="ignore")

    # String columns: strip whitespace
    for str_col in (
        "transaction_id", "sender_account", "receiver_account",
        "sender_bank", "receiver_bank", "sender_country", "receiver_country",
        "payment_type", "currency", "typology",
    ):
        if str_col in df.columns and df[str_col].dtype == object:
            df[str_col] = df[str_col].str.strip()

    return df


def _coerce_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply type coercions to the customers DataFrame.

    Parameters
    ----------
    df:
        Raw customers DataFrame after column normalisation.

    Returns
    -------
    pd.DataFrame
        Customers DataFrame with canonical dtypes applied.
    """
    if "account_open_date" in df.columns:
        df["account_open_date"] = pd.to_datetime(df["account_open_date"], errors="coerce")

    for bool_col in ("is_pep", "is_high_risk_jurisdiction"):
        if bool_col in df.columns:
            col = df[bool_col]
            if col.dtype == object:
                df[bool_col] = col.map(
                    lambda v: True if str(v).strip().lower() in ("1", "true", "yes") else False
                )
            else:
                df[bool_col] = col.astype(bool, errors="ignore")

    return df


def _find_csv(directory: Path, candidates: list[str]) -> Optional[Path]:
    """
    Search a directory for a CSV file matching one of the candidate names
    (case-insensitive).  Returns the first match, or None.

    Parameters
    ----------
    directory:
        Directory to search.
    candidates:
        Ordered list of filenames to look for.

    Returns
    -------
    Path or None
        First found matching file path, or None if none found.
    """
    if not directory.exists():
        return None
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
    # Fallback: case-insensitive glob
    for name in candidates:
        matches = list(directory.glob(f"*{name.split('.')[-1]}"))
        for m in matches:
            if m.stem.lower() == name.split(".")[0].lower():
                return m
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_transactions(data_dir: str | Path = "data") -> pd.DataFrame:
    """
    Load the transactions dataset.

    Resolution order
    ----------------
    1. ``<data_dir>/raw/`` – Kaggle download (multiple candidate filenames).
    2. ``<data_dir>/synthetic/transactions.csv`` – generated synthetic data.

    Column normalisation and type coercions are applied automatically so
    all downstream code can rely on ``TXN_CANONICAL_COLS``.

    Parameters
    ----------
    data_dir:
        Root data directory (absolute or relative to project root).
        Defaults to ``'data'``.

    Returns
    -------
    pd.DataFrame
        Transactions DataFrame with canonical schema.

    Raises
    ------
    FileNotFoundError
        If no transactions file can be located in either location.
    """
    base = _resolve_data_dir(data_dir)

    # --- Try raw/ first (Kaggle) ---
    raw_candidates = [
        "SAML-D.csv",                  # actual SAML-D Kaggle dataset filename
        "transactions.csv",
        "HI-Small_Trans.csv",          # common SAML-D filename
        "HI-Large_Trans.csv",
        "LI-Small_Trans.csv",
        "LI-Large_Trans.csv",
        "transaction_monitoring.csv",
        "aml_transactions.csv",
    ]
    raw_path = _find_csv(base / "raw", raw_candidates)

    # --- Try synthetic/ fallback ---
    synth_path = _find_csv(base / "synthetic", ["transactions.csv"])

    path = raw_path or synth_path

    if path is None:
        raise FileNotFoundError(
            f"No transactions CSV found in {base / 'raw'} or {base / 'synthetic'}.\n"
            "Run `python data/download_data.py` to acquire or generate the dataset."
        )

    source = "Kaggle raw" if path == raw_path else "synthetic"
    print(f"[load_data] Loading transactions from {source}: {path}")

    df = pd.read_csv(path, low_memory=False)
    df = _normalise_columns(df, TXN_ALIASES)
    df = _postprocess_saml_d(df)
    df = _add_missing_cols(df, TXN_CANONICAL_COLS)
    df = _coerce_transactions(df)

    print(f"[load_data] Transactions loaded: {len(df):,} rows × {len(df.columns)} cols")
    return df


def load_customers(data_dir: str | Path = "data") -> pd.DataFrame:
    """
    Load the customer profiles dataset.

    Resolution order
    ----------------
    1. ``<data_dir>/raw/`` – Kaggle download.
    2. ``<data_dir>/synthetic/customers.csv`` – generated synthetic data.

    Parameters
    ----------
    data_dir:
        Root data directory (absolute or relative to project root).
        Defaults to ``'data'``.

    Returns
    -------
    pd.DataFrame
        Customers DataFrame with canonical schema.

    Raises
    ------
    FileNotFoundError
        If no customers file can be located.
    """
    base = _resolve_data_dir(data_dir)

    raw_candidates = [
        "customers.csv",
        "customer_profiles.csv",
        "accounts.csv",
        "account_info.csv",
        "HI-Small_Accounts.csv",
        "HI-Large_Accounts.csv",
    ]
    raw_path = _find_csv(base / "raw", raw_candidates)
    synth_path = _find_csv(base / "synthetic", ["customers.csv"])

    path = raw_path or synth_path

    if path is None:
        raise FileNotFoundError(
            f"No customers CSV found in {base / 'raw'} or {base / 'synthetic'}.\n"
            "Run `python data/download_data.py` to acquire or generate the dataset."
        )

    source = "Kaggle raw" if path == raw_path else "synthetic"
    print(f"[load_data] Loading customers from {source}: {path}")

    df = pd.read_csv(path, low_memory=False)
    df = _normalise_columns(df, CUST_ALIASES)
    df = _add_missing_cols(df, CUST_CANONICAL_COLS)
    df = _coerce_customers(df)

    print(f"[load_data] Customers loaded: {len(df):,} rows × {len(df.columns)} cols")
    return df


def load_jurisdictions(data_dir: str | Path = "data") -> pd.DataFrame:
    """
    Load the high-risk jurisdictions reference table.

    Resolution order
    ----------------
    1. ``<data_dir>/raw/`` – Kaggle download.
    2. ``<data_dir>/synthetic/high_risk_jurisdictions.csv``.

    Parameters
    ----------
    data_dir:
        Root data directory (absolute or relative to project root).
        Defaults to ``'data'``.

    Returns
    -------
    pd.DataFrame
        High-risk jurisdictions DataFrame with canonical schema.

    Raises
    ------
    FileNotFoundError
        If no jurisdictions file can be located.
    """
    base = _resolve_data_dir(data_dir)

    raw_candidates = [
        "high_risk_jurisdictions.csv",
        "jurisdictions.csv",
        "risk_countries.csv",
        "HI-Small_Countries.csv",
        "HI-Large_Countries.csv",
        "country_risk.csv",
    ]
    raw_path = _find_csv(base / "raw", raw_candidates)
    synth_path = _find_csv(base / "synthetic", ["high_risk_jurisdictions.csv"])

    path = raw_path or synth_path

    if path is None:
        raise FileNotFoundError(
            f"No jurisdictions CSV found in {base / 'raw'} or {base / 'synthetic'}.\n"
            "Run `python data/download_data.py` to acquire or generate the dataset."
        )

    source = "Kaggle raw" if path == raw_path else "synthetic"
    print(f"[load_data] Loading jurisdictions from {source}: {path}")

    df = pd.read_csv(path, low_memory=False)
    df = _normalise_columns(df, JURIS_ALIASES)
    df = _add_missing_cols(df, JURIS_CANONICAL_COLS)

    # Normalise risk_level to uppercase
    if "risk_level" in df.columns:
        df["risk_level"] = df["risk_level"].str.strip().str.upper()

    print(f"[load_data] Jurisdictions loaded: {len(df):,} rows × {len(df.columns)} cols")
    return df


# ---------------------------------------------------------------------------
# Quick sanity-check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    print("=" * 60)
    print("  load_data.py — sanity check")
    print("=" * 60)

    for loader, name in [
        (load_transactions, "Transactions"),
        (load_customers, "Customers"),
        (load_jurisdictions, "Jurisdictions"),
    ]:
        try:
            df = loader()
            print(f"\n{name}: {len(df):,} rows, columns: {list(df.columns)}")
            print(df.head(3).to_string())
        except FileNotFoundError as exc:
            print(f"\n[WARN] {name}: {exc}")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
