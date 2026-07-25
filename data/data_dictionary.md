# AML Financial Crime Committee Agent — Data Dictionary

> Last updated: 2026-07-25  
> Schema version: 1.0  
> Source priority: `data/raw/` (Kaggle SAML-D) → `data/synthetic/` (generated fallback)

---

## Table of Contents

1. [transactions.csv](#1-transactionscsv)
2. [customers.csv](#2-customerscsv)
3. [high_risk_jurisdictions.csv](#3-high_risk_jurisdictionscsv)
4. [generation_log.json](#4-generation_logjson)
5. [Typology Definitions](#5-typology-definitions)
6. [Synthetic Generation Logic](#6-synthetic-generation-logic)

---

## 1. `transactions.csv`

Primary table. Each row represents a single financial transaction.  
**Target size:** 10,000+ rows (synthetic) / variable (Kaggle).

| Column | Type | Nullable | Example Value | Description |
|---|---|---|---|---|
| `transaction_id` | `str` | No | `TXN-A3F2C19B` | Unique transaction identifier. Format: `TXN-<8 hex chars>` (synthetic) or source system ID (Kaggle). |
| `timestamp` | `datetime64[ns]` | No | `2022-08-14T09:33:21` | UTC datetime of the transaction. Parsed from ISO-8601 string on load. |
| `sender_account` | `str` | No | `SND-7F3A11DC` | Originating account ID. Format: `SND-<8 hex>` (synthetic). |
| `receiver_account` | `str` | No | `RCV-2B9E04FA` | Beneficiary account ID. Format: `RCV-<8 hex>` (synthetic). |
| `sender_bank` | `str` | No | `JPMorgan` | Name of the sending financial institution. Drawn from a pool of 20 global banks. |
| `receiver_bank` | `str` | No | `HSBC` | Name of the receiving financial institution. |
| `sender_country` | `str` (ISO 3166-1 alpha-2) | No | `US` | Country of the sending account. Normal txns use low-risk countries; suspicious txns may use high-risk ones. |
| `receiver_country` | `str` (ISO 3166-1 alpha-2) | No | `NG` | Country of the receiving account. |
| `payment_type` | `str` | No | `WIRE` | Payment rail. One of: `WIRE`, `ACH`, `SWIFT`, `SEPA`, `CHAPS`, `RTGS`, `INTERNAL`, `CARD`, `CRYPTO_BRIDGE`, `CHEQUE`. |
| `currency` | `str` (ISO 4217) | No | `USD` | Currency code. Weighted distribution: USD(35%), EUR(25%), GBP(10%), others. |
| `amount` | `float64` | No | `9823.47` | Transaction amount in the specified currency. Always positive. |
| `is_suspicious` | `bool` | No | `True` | Ground-truth label. `True` for STRUCTURING, SMURFING, and LAYERING typologies. |
| `typology` | `str` | No | `STRUCTURING` | Money laundering typology label. See [Typology Definitions](#5-typology-definitions). |

**Primary Key:** `transaction_id`  
**Foreign Keys:** `sender_account` / `receiver_account` → `customers.account_id`  
**Indexes recommended:** `timestamp`, `sender_account`, `receiver_account`, `is_suspicious`

---

## 2. `customers.csv`

Customer/account profile table. Each row is a unique customer-account pair.  
**Target size:** 500 rows.

| Column | Type | Nullable | Example Value | Description |
|---|---|---|---|---|
| `customer_id` | `str` | No | `CUS-F3A1B290` | Unique customer identifier. Format: `CUS-<8 hex>`. |
| `account_id` | `str` | No | `SND-7F3A11DC` | Account ID that links to `transactions.sender_account` or `transactions.receiver_account`. |
| `full_name` | `str` | No | `Maria Garcia` | Synthetic full name. Composed from first-name × last-name pools (20 × 20 = 400 combinations). |
| `occupation` | `str` | No | `Import/Export Manager` | Declared occupation. One of 20 categories including `Unknown`. |
| `declared_income_band` | `str` | No | `50k-100k` | Self-reported annual income band. One of: `<25k`, `25k-50k`, `50k-100k`, `100k-250k`, `250k+`. |
| `country_of_residence` | `str` (ISO 3166-1 alpha-2) | No | `NG` | Customer's declared country of residence. 10% of customers reside in high-risk jurisdictions. |
| `account_open_date` | `datetime64[ns]` | No | `2014-06-15` | Date account was opened. Synthetic: uniform random between 2000-01-01 and 2021-12-28. |
| `is_pep` | `bool` | No | `False` | Politically Exposed Person flag. Approximately 5% of synthetic customers are PEPs. |
| `is_high_risk_jurisdiction` | `bool` | No | `True` | Whether customer resides in a FATF-listed or sanctioned jurisdiction. Approximately 10% prevalence in synthetic data. |

**Primary Key:** `customer_id`  
**Unique Key:** `account_id`  
**Foreign Key:** `account_id` → `transactions.sender_account` / `transactions.receiver_account`

---

## 3. `high_risk_jurisdictions.csv`

Reference table of high-risk countries for AML screening.  
**Size:** 20 rows (fixed).

| Column | Type | Nullable | Example Value | Description |
|---|---|---|---|---|
| `country_code` | `str` (ISO 3166-1 alpha-2) | No | `KP` | Two-letter ISO country code. |
| `country_name` | `str` | No | `North Korea` | Full English country name. |
| `risk_level` | `str` | No | `CRITICAL` | Risk classification. One of: `CRITICAL`, `HIGH`, `MEDIUM`. |
| `risk_reason` | `str` | No | `State-sponsored cybercrime, sanctions evasion` | Brief description of why the jurisdiction is considered high-risk. |

**Primary Key:** `country_code`

### Risk Level Definitions

| Level | Meaning |
|---|---|
| `CRITICAL` | OFAC/UN sanctions; direct state-level ML/TF risk (AF, KP, IR, SY) |
| `HIGH` | FATF blacklist or severe ongoing conflict enabling capital flight (MM, YE, SO, VE, BY) |
| `MEDIUM` | FATF grey-list, elevated ML/TF typologies, or recent sanctions (PK, NG, NI, HT, LA, KH, ZW, UA) |

---

## 4. `generation_log.json`

Metadata file recording the exact parameters used to generate the synthetic dataset.  
Written once after each successful run of `generate_synthetic.py`.

```json
{
  "generated_at": "2026-07-25T09:00:00Z",
  "parameters": {
    "rng_seed": 42,
    "total_transactions": 10000,
    "normal_fraction": 0.80,
    ...
  },
  "actual_counts": {
    "normal": 8000,
    "structuring": 1500,
    "smurfing": 300,
    "layering": 200,
    "mutations_injected": 50,
    "total_transactions": 10050,
    "customers": 500,
    "jurisdictions": 20
  },
  "typology_counts": { "NORMAL": 8000, "STRUCTURING": 1550, ... },
  "suspicious_rate": 0.205,
  "amount_stats": { "min": ..., "max": ..., "mean": ..., "median": ..., "std": ... }
}
```

---

## 5. Typology Definitions

| Typology | `is_suspicious` | Real-world pattern | Detection signals |
|---|---|---|---|
| `NORMAL` | `False` | Legitimate business/consumer payments | Baseline behaviour |
| `STRUCTURING` | `True` | Breaking large cash deposits into sub-$10k amounts to evade CTR filing | Cluster of transactions per sender within a 7-day window, all $8,500–$9,999; Benford deviation |
| `SMURFING` | `True` | Multiple individuals ("smurfs") deposit cash to a single beneficiary | Many distinct senders → one receiver within a 7-day window, all amounts near $10k threshold |
| `LAYERING` | `True` | Moving funds through a chain of accounts to obscure origin | A→B→C→D chain within 48h; amounts decrease 5–15% per hop (fee simulation) |

---

## 6. Synthetic Generation Logic

### 6.1 Normal Transactions (80 %, ~8,000 rows)

```
amount ~ LogNormal(μ=ln(8000), σ=1.5)
         → median ≈ $8,000, mean ≈ $24,000 (long right tail)
sender_country  ∈ NORMAL_COUNTRIES (20 low-risk jurisdictions)
receiver_country ∈ NORMAL_COUNTRIES
timestamp ~ Uniform(2022-01-01, 2023-12-31)
```

### 6.2 Structuring Transactions (15 %, ~1,500 rows)

```
foreach cluster:
    cluster_size ~ Uniform(5, 15)         # transactions per cluster
    sender       = single account (fixed per cluster)
    receiver     = single account (fixed per cluster)
    window_start ~ Uniform(2022-01-01, 2023-12-24)
    
    foreach transaction in cluster:
        amount    ~ Uniform(8500, 9999)   # just below $10k CTR threshold
        timestamp = window_start + Uniform(0, 7*24) hours
        is_suspicious = True
        typology = "STRUCTURING"
```

### 6.3 Smurfing Transactions (3 %, ~300 rows)

```
foreach cluster:
    n_senders ~ Uniform(3, 8)             # distinct smurfs per receiver
    receiver  = single account (fixed per cluster)
    receiver_country ∈ HIGH_RISK_COUNTRIES
    window_start ~ Uniform(2022-01-01, 2023-12-24)
    
    foreach sender in cluster:
        amount    ~ Uniform(8500, 9999)
        timestamp = window_start + Uniform(0, 7*24) hours
        is_suspicious = True
        typology = "SMURFING"
```

### 6.4 Layering Transactions (2 %, ~200 rows)

```
foreach chain (A→B→C→D):
    accounts[0..3] = 4 distinct accounts from pool
    chain_start ~ Uniform(2022-01-01, 2023-12-30)
    initial_amount ~ Uniform(20000, 500000)
    
    foreach hop i in [0,1,2]:
        decay     ~ Uniform(0.05, 0.15)
        amount[i+1] = amount[i] * (1 - decay)
        timestamp  = chain_start + Uniform(1, 16) hours (accumulated)
        is_suspicious = True
        typology = "LAYERING"
```

### 6.5 Mutation Injection (50 records, appended)

```
Injected on top of the above to stress-test Benford's Law detection:
    amount ~ N(9200, 300²)   clipped to [8000, 9999]
    sender/receiver = random sample from existing accounts
    typology = "STRUCTURING"  (labelled as structuring)
    is_suspicious = True

Purpose: The Gaussian centering around $9,200 shifts the leading-digit
distribution away from the uniform $8,500–$9,999 structuring baseline,
creating a detectable anomaly in Benford analysis for evolved/adapted
structuring patterns.
```

### 6.6 Customer Profiles

```
n = 500 profiles
is_pep                ~ Bernoulli(0.05)     → ~25 PEPs
is_high_risk_jurisdiction ~ Bernoulli(0.10) → ~50 high-risk residents
account_open_date     ~ Uniform(2000-01-01, 2021-12-28)
occupation            ~ Uniform(OCCUPATIONS pool, 20 values)
declared_income_band  ~ Uniform(5 bands)
```

### 6.7 RNG Reproducibility

All randomness uses `numpy.random.default_rng(seed=42)`. Re-running
`generate_synthetic.py` with the same seed produces byte-identical output.

---

## Appendix: Column Name Aliases

The `load_data.py` loader automatically maps common Kaggle column names to
canonical schema names. Key mappings:

| Kaggle Name | Canonical Name |
|---|---|
| `txn_id`, `id`, `trans_id` | `transaction_id` |
| `date`, `datetime` | `timestamp` |
| `from_account`, `originator` | `sender_account` |
| `to_account`, `beneficiary` | `receiver_account` |
| `usd_amount`, `trans_amount` | `amount` |
| `suspicious`, `label`, `fraud` | `is_suspicious` |
| `laundering_type`, `ml_type`, `type` | `typology` |
