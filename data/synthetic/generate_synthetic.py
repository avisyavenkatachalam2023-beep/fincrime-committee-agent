"""
data/synthetic/generate_synthetic.py
-------------------------------------
Generates a fully synthetic AML dataset that mirrors the SAML-D schema.

Tables produced
---------------
* transactions.csv      – 10,000 rows of financial transactions
* customers.csv         – 500 customer/account profiles
* high_risk_jurisdictions.csv – ~20 high-risk countries
* generation_log.json   – all parameters used during generation

Typology breakdown
------------------
| Typology     | Share | Description                                            |
|--------------|-------|--------------------------------------------------------|
| NORMAL       |  80 % | Lognormal amounts, random geographies                  |
| STRUCTURING  |  15 % | Clusters ≤$10k per sender, within 7-day windows        |
| SMURFING     |   3 % | Multi-sender, single receiver, near-threshold amounts  |
| LAYERING     |   2 % | A→B→C→D chains, amounts decay 5-15 % each hop         |
| MUTATION     |  50 records injected on top of above, for Benford tests|

Run directly:
    python data/synthetic/generate_synthetic.py
"""

import json
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Force UTF-8 output on Windows (prevents cp1252 UnicodeEncodeError on special chars)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project-root bootstrap
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
SYNTHETIC_DIR = _THIS_DIR
OUTPUT_DIR = SYNTHETIC_DIR

# ---------------------------------------------------------------------------
# Global RNG seed (reproducible)
# ---------------------------------------------------------------------------
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# Domain value pools
# ---------------------------------------------------------------------------
BANKS = [
    "CitiBank", "HSBC", "Deutsche Bank", "Barclays", "BNP Paribas",
    "JPMorgan", "Wells Fargo", "UBS", "Credit Suisse", "Standard Chartered",
    "Santander", "ING", "UniCredit", "Nordea", "Rabobank",
    "Commerzbank", "RBS", "Lloyds", "Natwest", "Mizuho",
]

PAYMENT_TYPES = [
    "WIRE", "ACH", "SWIFT", "SEPA", "CHAPS",
    "RTGS", "INTERNAL", "CARD", "CRYPTO_BRIDGE", "CHEQUE",
]

CURRENCIES = [
    "USD", "EUR", "GBP", "CHF", "JPY",
    "AED", "SGD", "HKD", "CAD", "AUD",
]

NORMAL_COUNTRIES = [
    "US", "GB", "DE", "FR", "NL", "CA", "AU", "JP", "CH", "SE",
    "NO", "DK", "FI", "AT", "BE", "ES", "IT", "PT", "NZ", "KR",
]

HIGH_RISK_COUNTRIES = [
    "AF", "MM", "KP", "IR", "SY", "YE", "SO", "LY", "SD", "VE",
    "CU", "NI", "HT", "PK", "NG", "KH", "LA", "ZW", "BY", "UA",
]

ALL_COUNTRIES = NORMAL_COUNTRIES + HIGH_RISK_COUNTRIES

OCCUPATIONS = [
    "Software Engineer", "Doctor", "Accountant", "Lawyer", "Teacher",
    "Trader", "Real Estate Agent", "Consultant", "Business Owner", "Retired",
    "Civil Servant", "Pharmacist", "Architect", "Nurse", "Chef",
    "Import/Export Manager", "Freelancer", "Student", "Politician", "Unknown",
]

INCOME_BANDS = ["<25k", "25k-50k", "50k-100k", "100k-250k", "250k+"]

RISK_REASONS = {
    "AF": "Taliban governance, narcotics financing",
    "MM": "Military junta sanctions, jade & drug trade",
    "KP": "State-sponsored cybercrime, sanctions evasion",
    "IR": "OFAC sanctions, terrorism financing",
    "SY": "Civil war proceeds, sanctions",
    "YE": "Houthi financing, arms trafficking",
    "SO": "Piracy proceeds, al-Shabaab financing",
    "LY": "Militia financing, arms embargo",
    "SD": "Sanctions, gold smuggling",
    "VE": "Maduro regime, PDVSA sanctions evasion",
    "CU": "OFAC embargo, state asset transfers",
    "NI": "Ortega regime, money laundering",
    "HT": "Gang financing, sanctions (2024)",
    "PK": "FATF grey-list, hawala networks",
    "NG": "Oil bunkering, BEC fraud proceeds",
    "KH": "Pig-butchering scam hubs, casino ML",
    "LA": "Golden Triangle narcotics, casino ML",
    "ZW": "ZANU-PF sanctions, diamond smuggling",
    "BY": "Lukashenko sanctions, KGB asset movement",
    "UA": "Wartime capital flight (conflict zone)",
}

RISK_LEVELS = {
    "AF": "CRITICAL", "KP": "CRITICAL", "IR": "CRITICAL", "SY": "CRITICAL",
    "MM": "HIGH", "YE": "HIGH", "SO": "HIGH", "LY": "HIGH", "SD": "HIGH",
    "VE": "HIGH", "CU": "HIGH", "NI": "MEDIUM", "HT": "MEDIUM",
    "PK": "MEDIUM", "NG": "MEDIUM", "KH": "MEDIUM", "LA": "MEDIUM",
    "ZW": "MEDIUM", "BY": "HIGH", "UA": "MEDIUM",
}

COUNTRY_NAMES = {
    "AF": "Afghanistan", "MM": "Myanmar", "KP": "North Korea",
    "IR": "Iran", "SY": "Syria", "YE": "Yemen", "SO": "Somalia",
    "LY": "Libya", "SD": "Sudan", "VE": "Venezuela", "CU": "Cuba",
    "NI": "Nicaragua", "HT": "Haiti", "PK": "Pakistan", "NG": "Nigeria",
    "KH": "Cambodia", "LA": "Laos", "ZW": "Zimbabwe", "BY": "Belarus",
    "UA": "Ukraine",
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _uid() -> str:
    """Return a short unique transaction ID (TXN-<8 hex chars>)."""
    return f"TXN-{uuid.uuid4().hex[:8].upper()}"


def _account_id(prefix: str = "ACC") -> str:
    """Return a random account ID of the form ACC-XXXXXXXX."""
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


def _random_dt(start: datetime, end: datetime) -> datetime:
    """Return a uniformly random datetime between start and end."""
    delta = (end - start).total_seconds()
    offset = rng.integers(0, int(delta))
    return start + timedelta(seconds=int(offset))


def _random_bank() -> str:
    """Pick a random bank name from the pool."""
    return BANKS[rng.integers(0, len(BANKS))]


def _random_payment() -> str:
    """Pick a random payment type."""
    return PAYMENT_TYPES[rng.integers(0, len(PAYMENT_TYPES))]


def _random_currency() -> str:
    """Pick a random currency code, weighted towards USD/EUR/GBP."""
    weights = [0.35, 0.25, 0.10, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04]
    return rng.choice(CURRENCIES, p=weights)


def _fmt_dt(dt: datetime) -> str:
    """Format datetime as ISO-8601 string."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Generation parameters (also written to generation_log.json)
# ---------------------------------------------------------------------------
PARAMS = {
    "rng_seed": RNG_SEED,
    "total_transactions": 10_000,
    "normal_fraction": 0.80,
    "structuring_fraction": 0.15,
    "smurfing_fraction": 0.03,
    "layering_fraction": 0.02,
    "mutation_injection_count": 50,
    "normal_lognormal_mean": 8_000,
    "normal_lognormal_sigma": 1.5,
    "structuring_amount_min": 8_500,
    "structuring_amount_max": 9_999,
    "structuring_cluster_min_txns": 5,
    "structuring_cluster_max_txns": 15,
    "structuring_window_days": 7,
    "smurfing_amount_min": 8_500,
    "smurfing_amount_max": 9_999,
    "smurfing_senders_per_cluster": [3, 8],
    "layering_chain_length": 4,
    "layering_decay_min_pct": 0.05,
    "layering_decay_max_pct": 0.15,
    "layering_window_hours": 48,
    "mutation_center": 9_200,
    "mutation_sigma": 300,
    "date_range_start": "2022-01-01",
    "date_range_end": "2023-12-31",
    "num_customers": 500,
    "pep_fraction": 0.05,
    "high_risk_jurisdiction_fraction": 0.10,
}

DATE_START = datetime(2022, 1, 1)
DATE_END = datetime(2023, 12, 31)


# ---------------------------------------------------------------------------
# 1. Normal transactions (80 %)
# ---------------------------------------------------------------------------

def _generate_normal(n: int, sender_pool: list[str], receiver_pool: list[str]) -> list[dict]:
    """
    Generate n normal (benign) transactions.

    Amounts follow a lognormal distribution with mean≈8000 and sigma=1.5 in
    log-space, producing a realistic long-tail spend distribution.

    Parameters
    ----------
    n:
        Number of transactions to generate.
    sender_pool:
        Pre-allocated sender account IDs.
    receiver_pool:
        Pre-allocated receiver account IDs.

    Returns
    -------
    list[dict]
        List of transaction record dictionaries.
    """
    records = []
    # lognormal: mean in natural units ≈ exp(mu + sigma²/2)
    # We want median ≈ 8000, so mu = ln(8000)
    mu = np.log(PARAMS["normal_lognormal_mean"])
    sigma = PARAMS["normal_lognormal_sigma"]
    amounts = rng.lognormal(mu, sigma, n)

    for i in range(n):
        sender = sender_pool[rng.integers(0, len(sender_pool))]
        receiver = receiver_pool[rng.integers(0, len(receiver_pool))]
        s_country = NORMAL_COUNTRIES[rng.integers(0, len(NORMAL_COUNTRIES))]
        r_country = NORMAL_COUNTRIES[rng.integers(0, len(NORMAL_COUNTRIES))]
        dt = _random_dt(DATE_START, DATE_END)

        records.append({
            "transaction_id": _uid(),
            "timestamp": _fmt_dt(dt),
            "sender_account": sender,
            "receiver_account": receiver,
            "sender_bank": _random_bank(),
            "receiver_bank": _random_bank(),
            "sender_country": s_country,
            "receiver_country": r_country,
            "payment_type": _random_payment(),
            "currency": _random_currency(),
            "amount": round(float(amounts[i]), 2),
            "is_suspicious": False,
            "typology": "NORMAL",
        })
    return records


# ---------------------------------------------------------------------------
# 2. Structuring transactions (15 %)
# ---------------------------------------------------------------------------

def _generate_structuring(n: int, sender_pool: list[str], receiver_pool: list[str]) -> list[dict]:
    """
    Generate structuring transactions.

    Each structuring cluster consists of 5–15 transactions from a single
    sender to a fixed receiver, all with amounts in [$8,500, $9,999], spread
    across a 7-day window.  This mimics cash structuring designed to stay
    below the $10,000 CTR threshold.

    Parameters
    ----------
    n:
        Approximate total number of structuring records to produce.
    sender_pool:
        Pre-allocated sender account IDs.
    receiver_pool:
        Pre-allocated receiver account IDs.

    Returns
    -------
    list[dict]
        List of structuring transaction record dictionaries.
    """
    records = []
    while len(records) < n:
        cluster_size = int(rng.integers(
            PARAMS["structuring_cluster_min_txns"],
            PARAMS["structuring_cluster_max_txns"] + 1,
        ))
        sender = sender_pool[rng.integers(0, len(sender_pool))]
        receiver = receiver_pool[rng.integers(0, len(receiver_pool))]
        s_country = ALL_COUNTRIES[rng.integers(0, len(ALL_COUNTRIES))]
        r_country = ALL_COUNTRIES[rng.integers(0, len(ALL_COUNTRIES))]
        window_start = _random_dt(DATE_START, DATE_END - timedelta(days=7))

        for _ in range(cluster_size):
            dt = window_start + timedelta(hours=float(rng.uniform(0, 7 * 24)))
            amount = round(float(rng.uniform(
                PARAMS["structuring_amount_min"],
                PARAMS["structuring_amount_max"],
            )), 2)
            records.append({
                "transaction_id": _uid(),
                "timestamp": _fmt_dt(dt),
                "sender_account": sender,
                "receiver_account": receiver,
                "sender_bank": _random_bank(),
                "receiver_bank": _random_bank(),
                "sender_country": s_country,
                "receiver_country": r_country,
                "payment_type": _random_payment(),
                "currency": _random_currency(),
                "amount": amount,
                "is_suspicious": True,
                "typology": "STRUCTURING",
            })
    return records[:n]


# ---------------------------------------------------------------------------
# 3. Smurfing transactions (3 %)
# ---------------------------------------------------------------------------

def _generate_smurfing(n: int, sender_pool: list[str], receiver_pool: list[str]) -> list[dict]:
    """
    Generate smurfing transactions.

    Smurfing involves multiple sender accounts (3–8 per cluster) sending
    individually sub-threshold amounts to the same receiver within a 7-day
    window, collectively moving large sums while evading detection.

    Parameters
    ----------
    n:
        Approximate total number of smurfing records to produce.
    sender_pool:
        Pre-allocated sender account IDs.
    receiver_pool:
        Pre-allocated receiver account IDs.

    Returns
    -------
    list[dict]
        List of smurfing transaction record dictionaries.
    """
    records = []
    smin, smax = PARAMS["smurfing_senders_per_cluster"]

    while len(records) < n:
        n_senders = int(rng.integers(smin, smax + 1))
        senders = [sender_pool[rng.integers(0, len(sender_pool))] for _ in range(n_senders)]
        receiver = receiver_pool[rng.integers(0, len(receiver_pool))]
        r_country = HIGH_RISK_COUNTRIES[rng.integers(0, len(HIGH_RISK_COUNTRIES))]
        window_start = _random_dt(DATE_START, DATE_END - timedelta(days=7))

        for sender in senders:
            dt = window_start + timedelta(hours=float(rng.uniform(0, 7 * 24)))
            amount = round(float(rng.uniform(
                PARAMS["smurfing_amount_min"],
                PARAMS["smurfing_amount_max"],
            )), 2)
            s_country = ALL_COUNTRIES[rng.integers(0, len(ALL_COUNTRIES))]
            records.append({
                "transaction_id": _uid(),
                "timestamp": _fmt_dt(dt),
                "sender_account": sender,
                "receiver_account": receiver,
                "sender_bank": _random_bank(),
                "receiver_bank": _random_bank(),
                "sender_country": s_country,
                "receiver_country": r_country,
                "payment_type": _random_payment(),
                "currency": _random_currency(),
                "amount": amount,
                "is_suspicious": True,
                "typology": "SMURFING",
            })
    return records[:n]


# ---------------------------------------------------------------------------
# 4. Layering transactions (2 %)
# ---------------------------------------------------------------------------

def _generate_layering(n: int, account_pool: list[str]) -> list[dict]:
    """
    Generate layering transactions.

    Each cluster is a chain A → B → C → D within a 48-hour window.
    Each hop's amount is reduced by 5–15 % to simulate fee deductions and
    deliberate obfuscation.  Multiple such chains are generated until the
    target count is reached.

    Parameters
    ----------
    n:
        Approximate total number of layering records to produce.
    account_pool:
        Pool of account IDs used for chain nodes.

    Returns
    -------
    list[dict]
        List of layering transaction record dictionaries.
    """
    records = []
    chain_length = PARAMS["layering_chain_length"]  # 4 → 3 transactions per chain

    while len(records) < n:
        # Pick 4 distinct accounts for the chain
        chain_indices = rng.choice(len(account_pool), size=chain_length, replace=False)
        chain = [account_pool[i] for i in chain_indices]

        chain_start = _random_dt(DATE_START, DATE_END - timedelta(hours=48))
        initial_amount = round(float(rng.uniform(20_000, 500_000)), 2)

        current_amount = initial_amount
        current_time = chain_start

        for hop in range(chain_length - 1):
            decay = rng.uniform(
                PARAMS["layering_decay_min_pct"],
                PARAMS["layering_decay_max_pct"],
            )
            next_amount = round(current_amount * (1 - decay), 2)
            hop_hours = float(rng.uniform(1, 48 / (chain_length - 1)))
            hop_time = current_time + timedelta(hours=hop_hours)

            s_country = ALL_COUNTRIES[rng.integers(0, len(ALL_COUNTRIES))]
            r_country = ALL_COUNTRIES[rng.integers(0, len(ALL_COUNTRIES))]

            records.append({
                "transaction_id": _uid(),
                "timestamp": _fmt_dt(hop_time),
                "sender_account": chain[hop],
                "receiver_account": chain[hop + 1],
                "sender_bank": _random_bank(),
                "receiver_bank": _random_bank(),
                "sender_country": s_country,
                "receiver_country": r_country,
                "payment_type": _random_payment(),
                "currency": _random_currency(),
                "amount": next_amount,
                "is_suspicious": True,
                "typology": "LAYERING",
            })

            current_amount = next_amount
            current_time = hop_time

    return records[:n]


# ---------------------------------------------------------------------------
# 5. Mutation injection (50 records)
# ---------------------------------------------------------------------------

def _inject_mutations(records: list[dict]) -> list[dict]:
    """
    Inject 50 mutated structuring records to test Benford's Law detection
    on evolved patterns.

    Amounts are drawn from N(9200, 300²) and clipped to [8,000, 9,999].
    These records are labeled STRUCTURING/suspicious=True but their first-digit
    distribution subtly deviates from typical structuring patterns, providing
    a stress-test for the Benford analyser.

    Parameters
    ----------
    records:
        The existing transaction records list (modified in place and returned).

    Returns
    -------
    list[dict]
        The input list with 50 mutation records appended.
    """
    center = PARAMS["mutation_center"]
    sigma = PARAMS["mutation_sigma"]
    count = PARAMS["mutation_injection_count"]

    # Choose random senders/receivers from existing records
    existing_senders = list({r["sender_account"] for r in records})
    existing_receivers = list({r["receiver_account"] for r in records})

    for _ in range(count):
        raw_amount = float(rng.normal(center, sigma))
        amount = round(max(8_000.0, min(9_999.0, raw_amount)), 2)
        dt = _random_dt(DATE_START, DATE_END)
        s_country = ALL_COUNTRIES[rng.integers(0, len(ALL_COUNTRIES))]
        r_country = ALL_COUNTRIES[rng.integers(0, len(ALL_COUNTRIES))]

        records.append({
            "transaction_id": _uid(),
            "timestamp": _fmt_dt(dt),
            "sender_account": existing_senders[rng.integers(0, len(existing_senders))],
            "receiver_account": existing_receivers[rng.integers(0, len(existing_receivers))],
            "sender_bank": _random_bank(),
            "receiver_bank": _random_bank(),
            "sender_country": s_country,
            "receiver_country": r_country,
            "payment_type": _random_payment(),
            "currency": _random_currency(),
            "amount": amount,
            "is_suspicious": True,
            "typology": "STRUCTURING",  # labelled structuring but mutated
        })

    return records


# ---------------------------------------------------------------------------
# 6. Customer profiles (500 rows)
# ---------------------------------------------------------------------------

def _generate_customers(n: int, account_ids: list[str]) -> pd.DataFrame:
    """
    Generate synthetic customer profile records.

    Each customer is assigned a unique account_id drawn from the transaction
    account pool so that foreign-key joins are coherent.

    Parameters
    ----------
    n:
        Number of customer records to generate.
    account_ids:
        Pool of account IDs already used in transactions.

    Returns
    -------
    pd.DataFrame
        Customer profile table.
    """
    first_names = [
        "James", "Maria", "Chen", "Amara", "Yusuf", "Sofia", "Liam", "Anya",
        "Kwame", "Elena", "Mohammed", "Priya", "Lucas", "Fatima", "Noah",
        "Ingrid", "Omar", "Valentina", "Diego", "Aiko",
    ]
    last_names = [
        "Smith", "Garcia", "Wang", "Osei", "Hassan", "Mueller", "Brown",
        "Petrova", "Mensah", "Rossi", "Ali", "Sharma", "Nguyen", "Dubois",
        "Fernandez", "Lindqvist", "Ibrahim", "Santos", "Kim", "Tanaka",
    ]

    records = []
    used_accounts = set()
    all_accounts = list(set(account_ids))

    pep_fraction = PARAMS["pep_fraction"]
    hr_fraction = PARAMS["high_risk_jurisdiction_fraction"]

    for i in range(n):
        # Assign account — prefer one from the transaction pool, else generate new
        if i < len(all_accounts) and all_accounts[i] not in used_accounts:
            acc = all_accounts[i]
        else:
            acc = _account_id()
        used_accounts.add(acc)

        first = first_names[rng.integers(0, len(first_names))]
        last = last_names[rng.integers(0, len(last_names))]
        full_name = f"{first} {last}"

        is_pep = bool(rng.random() < pep_fraction)
        is_hr = bool(rng.random() < hr_fraction)
        country = (
            HIGH_RISK_COUNTRIES[rng.integers(0, len(HIGH_RISK_COUNTRIES))]
            if is_hr
            else NORMAL_COUNTRIES[rng.integers(0, len(NORMAL_COUNTRIES))]
        )

        open_year = int(rng.integers(2000, 2022))
        open_month = int(rng.integers(1, 13))
        open_day = int(rng.integers(1, 29))
        open_date = f"{open_year:04d}-{open_month:02d}-{open_day:02d}"

        records.append({
            "customer_id": f"CUS-{uuid.uuid4().hex[:8].upper()}",
            "account_id": acc,
            "full_name": full_name,
            "occupation": OCCUPATIONS[rng.integers(0, len(OCCUPATIONS))],
            "declared_income_band": INCOME_BANDS[rng.integers(0, len(INCOME_BANDS))],
            "country_of_residence": country,
            "account_open_date": open_date,
            "is_pep": is_pep,
            "is_high_risk_jurisdiction": is_hr,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 7. High-risk jurisdictions table
# ---------------------------------------------------------------------------

def _generate_jurisdictions() -> pd.DataFrame:
    """
    Build the high-risk jurisdictions reference table.

    Returns
    -------
    pd.DataFrame
        One row per high-risk country with code, name, risk level, and reason.
    """
    rows = []
    for code in HIGH_RISK_COUNTRIES:
        rows.append({
            "country_code": code,
            "country_name": COUNTRY_NAMES.get(code, code),
            "risk_level": RISK_LEVELS.get(code, "MEDIUM"),
            "risk_reason": RISK_REASONS.get(code, "Elevated ML/TF risk"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main generation entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate all synthetic data generation and save outputs.

    Generation steps
    ----------------
    1. Allocate account pools for senders and receivers.
    2. Generate normal, structuring, smurfing, and layering transactions.
    3. Inject 50 mutated structuring records for Benford stress-testing.
    4. Shuffle and save transactions.csv.
    5. Generate and save customers.csv.
    6. Generate and save high_risk_jurisdictions.csv.
    7. Write generation_log.json with all parameters.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = PARAMS["total_transactions"]
    n_normal = int(total * PARAMS["normal_fraction"])           # 8000
    n_struct = int(total * PARAMS["structuring_fraction"])      # 1500
    n_smurf = int(total * PARAMS["smurfing_fraction"])          # 300
    n_layer = total - n_normal - n_struct - n_smurf             # 200

    # --- Account pools ---
    n_senders = 800
    n_receivers = 400
    sender_pool = [_account_id("SND") for _ in range(n_senders)]
    receiver_pool = [_account_id("RCV") for _ in range(n_receivers)]
    all_accounts = sender_pool + receiver_pool

    print(f"[INFO] Generating {n_normal:,} NORMAL transactions …")
    normal_recs = _generate_normal(n_normal, sender_pool, receiver_pool)

    print(f"[INFO] Generating ~{n_struct:,} STRUCTURING transactions …")
    struct_recs = _generate_structuring(n_struct, sender_pool, receiver_pool)

    print(f"[INFO] Generating ~{n_smurf:,} SMURFING transactions …")
    smurf_recs = _generate_smurfing(n_smurf, sender_pool, receiver_pool)

    print(f"[INFO] Generating ~{n_layer:,} LAYERING transactions …")
    layer_recs = _generate_layering(n_layer, all_accounts)

    # Combine and inject mutations
    all_records = normal_recs + struct_recs + smurf_recs + layer_recs
    print(f"[INFO] Injecting {PARAMS['mutation_injection_count']} mutated structuring records …")
    all_records = _inject_mutations(all_records)

    # Shuffle for realism
    rng.shuffle(all_records)  # type: ignore[arg-type]

    txn_df = pd.DataFrame(all_records)
    txn_df["timestamp"] = pd.to_datetime(txn_df["timestamp"])
    txn_df = txn_df.sort_values("timestamp").reset_index(drop=True)

    txn_path = OUTPUT_DIR / "transactions.csv"
    txn_df.to_csv(txn_path, index=False)
    print(f"[OK]   Saved transactions → {txn_path}  ({len(txn_df):,} rows)")

    # --- Customers ---
    print(f"[INFO] Generating {PARAMS['num_customers']} customer profiles …")
    customers_df = _generate_customers(PARAMS["num_customers"], all_accounts)
    cust_path = OUTPUT_DIR / "customers.csv"
    customers_df.to_csv(cust_path, index=False)
    print(f"[OK]   Saved customers → {cust_path}  ({len(customers_df):,} rows)")

    # --- Jurisdictions ---
    print("[INFO] Building high-risk jurisdictions table …")
    juris_df = _generate_jurisdictions()
    juris_path = OUTPUT_DIR / "high_risk_jurisdictions.csv"
    juris_df.to_csv(juris_path, index=False)
    print(f"[OK]   Saved jurisdictions → {juris_path}  ({len(juris_df):,} rows)")

    # --- Generation log ---
    log = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "parameters": PARAMS,
        "actual_counts": {
            "normal": len(normal_recs),
            "structuring": len(struct_recs),
            "smurfing": len(smurf_recs),
            "layering": len(layer_recs),
            "mutations_injected": PARAMS["mutation_injection_count"],
            "total_transactions": len(txn_df),
            "customers": len(customers_df),
            "jurisdictions": len(juris_df),
        },
        "typology_counts": txn_df["typology"].value_counts().to_dict(),
        "suspicious_rate": float(txn_df["is_suspicious"].mean()),
        "amount_stats": {
            "min": float(txn_df["amount"].min()),
            "max": float(txn_df["amount"].max()),
            "mean": float(txn_df["amount"].mean()),
            "median": float(txn_df["amount"].median()),
            "std": float(txn_df["amount"].std()),
        },
    }
    log_path = OUTPUT_DIR / "generation_log.json"
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    print(f"[OK]   Saved generation log → {log_path}")
    print("\n[DONE] Synthetic dataset generation complete.")
    _print_summary(log)


def _print_summary(log: dict) -> None:
    """Print a human-readable summary of what was generated."""
    counts = log["actual_counts"]
    print("\n┌─────────────────────────────────────────────┐")
    print("│         SYNTHETIC DATASET SUMMARY           │")
    print("├─────────────────────────────────────────────┤")
    for typology, cnt in log["typology_counts"].items():
        pct = cnt / counts["total_transactions"] * 100
        print(f"│  {typology:<18}  {cnt:>6,} txns  ({pct:4.1f}%)   │")
    print(f"│  {'TOTAL':<18}  {counts['total_transactions']:>6,} txns           │")
    print(f"│  Suspicious rate: {log['suspicious_rate']:.1%}                   │")
    print("└─────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
