# The Financial Crime Committee — AI-Powered AML Detection Agent

## 1. Problem Statement
Financial institutions are mandated by regulators (FinCEN, FATF, local authorities) to run AML compliance programs. Traditional rule-based systems generate excessive false positives, overwhelming compliance teams and raising costs. Sophisticated laundering techniques — structuring, smurfing, layering — evade conventional detection. We are building an intelligent, autonomous agent that:
1. Performs automated EDA on transaction/customer data to understand baseline behavior
2. Detects anomalous transaction patterns indicative of laundering (structuring/smurfing/layering)
3. Applies anomaly detection (ML, rule-based, or hybrid)
4. Generates a risk score/flag per transaction or customer
5. Explains why a transaction is flagged
6. Recommends an escalation action (monitor / flag for review / report)

This is an agent-driven system that accepts natural-language instructions and autonomously orchestrates calls to internal tools to complete the task.

## 2. Architecture
```
                         ┌─────────────────────────────┐
                         │   USER (NL query, chat/API) │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │   1. QUERY UNDERSTANDING    │
                         │   (Intent/Entity/Filters)   │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │   2. DYNAMIC EXECUTION      │
                         │      PLANNER                │
                         └──────────────┬──────────────┘
                     ┌──────────────────┼───────────────────────┬───────────────┐
                     ▼                  ▼                       ▼               ▼
           ┌──────────────┐   ┌──────────────────┐   ┌────────────────────┐  ┌──────────────┐
           │ 3. EDA TOOL  │   │ 4. FEATURE ENG.  │   │ 5. ANOMALY DETECT. │  │ 6. NETWORK   │
           └──────┬───────┘   └────────┬─────────┘   └──────────┬─────────┘  └───────┬──────┘
                  └────────────────────┴──────────────────────────┴─────────────────────┘
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │ 7. RISK CLASSIFICATION TOOL   │
                                     └──────────────┬────────────────┘
                                                     ▼
                          ┌───────────────────────────────────────────────┐
                          │ 8. VIRTUAL FINANCIAL CRIME COMMITTEE          │
                          │    - Transaction Monitoring Analyst Agent     │
                          │    - KYC / UBO Analyst Agent                  │
                          │    - Sanctions / PEP Screening Agent          │
                          │    - Network Relationship Analyst Agent       │
                          │    - Chair Agent (synthesizes votes)          │
                          └───────────────────────┬───────────────────────┘
                                                  ▼
                          ┌───────────────────────────────────────────────┐
                          │ 9. EXPLANATION LAYER                          │
                          └──────────────────────┬────────────────────────┘
                                                 ▼
                          ┌───────────────────────────────────────────────┐
                          │ 10. OUTPUT FORMATTER                          │
                          │   - Execution Summary, Risk Memo, Minutes     │
                          └───────────────────────────────────────────────┘
```

## 3. Dataset Information
- **Primary Dataset**: SAML-D (Synthetic Anti-Money Laundering Dataset)
- **Source**: Kaggle — https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml
- **Author**: Oztas, B. (2023)
- **License**: CC0 Public Domain
- **Fields Used**: sender/receiver bank, country, payment type, currency, amount, timestamp, is_suspicious label, laundering_type.
- **Alternative/Supplement Note**: If more complex network-relationship structure is required in future builds, IBM AMLSim (GitHub) may be used as a graph-native synthetic dataset supplement.

**Synthetic Augmentation Required for this Build**:
To support the specialized agents in our Financial Crime Committee, we programmatically augmented the Kaggle dataset with the following synthetic tables (documented here per hackathon rules):

1. **Synthetic Customer Profile Table** *(Used by KYC/UBO Analyst Agent)*
   - **Fields**: declared occupation, declared income band, account open date, country of residence.
   - **Generation Logic**: Generated using simple, documented random logic (`numpy.random` distributions) to map synthetic occupations to income bands. *Note: All data is entirely synthetic; no real people or entities are represented or fabricated.*

2. **Synthetic High-Risk Jurisdiction List** *(Used by Sanctions/PEP Agent)*
   - **Fields**: country code, risk level, risk reason.
   - **Disclaimer**: *This is an illustrative/synthetic dataset created strictly for demonstration purposes and is NOT a real sanctions list.*

3. **Evolved Laundering Injection** *(Used to demonstrate Forensic tools)*
   - **Logic**: We optionally injected 50 deliberately mutated structuring patterns into the dataset. Instead of using flat $9,999 amounts, amounts are varied around thresholds with Gaussian noise (drawn from `N(9200, 300^2)` and clipped to `[8000, 9999]`) and irregular timing.
   - **Purpose**: This demonstrates how our forensic accounting tools (Benford's Law and threshold-proximity clustering) can catch "evolved" laundering patterns that naive fixed-threshold rules would completely miss.

**Baseline Synthetic Generation Logic** (Fallback if Kaggle is unreachable):
- 80% normal transactions: amounts ~lognormal(mean=8000, sigma=1.5).
- 15% structuring: clusters of 5-15 txns under $10,000 threshold within 7 days.
- 3% smurfing: multiple sender accounts, single receiver, near-threshold amounts.
- 2% layering: A→B→C→D chains within 48h, amounts decay 5-15% per hop.

## 4. Solution Approach
The system replicates a real bank's Financial Crime Committee: a forensic-accounting detection engine (Benford's Law + structuring-threshold clustering) feeds a panel of specialist agents (Transaction Monitoring, KYC/UBO, Sanctions/PEP, Network Analyst) who debate each case. A Chair Agent synthesizes their votes, outputting an actual Risk Memo and Committee Meeting Minutes.

## 5. Tech Stack
| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Data | pandas, numpy |
| ML | scikit-learn (IsolationForest) |
| Graph | networkx, python-louvain |
| Stats/Forensics | scipy (chi-square) |
| Visualization | matplotlib |
| UI | Streamlit |
| LLM | Google Gemini API (gemini-2.0-flash) |

## 6. Setup Instructions
```bash
git clone <repo>
cd aml-crime-committee-agent
pip install -r requirements.txt
cp .env.example .env  # add your GEMINI_API_KEY
python data/download_data.py
uvicorn app.main:app --reload
```

## 7. Usage Instructions
The system exposes a REST API via FastAPI. Once running locally, you can send queries to the endpoint.
You can view the interactive Swagger API documentation at `http://localhost:8000/docs`.

**Example `curl` request:**
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "Content-Type: application/json" \
     -d '{"query": "Find structuring patterns in the last 30 days"}'
```

**Example queries to try:**
1. Find structuring patterns in the last 30 days
2. Which customers made 10+ transactions under $10,000?
3. Is customer ID 4521 suspicious?
4. Analyse this dataset for suspicious activity
5. Who is coordinating transactions with customer 4521?
6. Why was transaction TXN_88213 flagged?

## 8. What Makes It Stand Out
1. **Forensic Accounting**: Real statistical analysis (Benford's Law, round-number clustering) over generic ML.
2. **Committee Debate**: A multi-agent deliberation framework that provides distinct compliance perspectives before a Chair consensus.
3. **Evolved-Pattern Detection**: Capable of detecting deliberate structuring variation that evades naive threshold rules.

## 9. Disclosures
Built with Google Gemini API (gemini-2.0-flash) for Query Understanding and Committee Agent reasoning. Agentic coding assistant used for code generation.

## 10. Risk Score Weights and Thresholds
- **Weights**: Benford 0.30, Threshold Clustering 0.25, ML Anomaly 0.25, Network Centrality 0.20
- **Thresholds**: LOW < 33, MEDIUM < 60, HIGH ≤ 100
