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
| UI | FastAPI + a single-page HTML/JS dashboard (`app/index.html`), served by FastAPI itself |
| LLM | Groq API — `llama-3.3-70b-versatile` for query understanding & committee reasoning; optional Groq vision model for the image-attachment feature |

## 6. Setup Instructions
```bash
git clone <repo>
cd aml-crime-committee-agent
pip install -r requirements.txt
cp .env.example .env  # add your GROQ_API_KEY (and optionally KAGGLE_TOKEN, PORT)
python data/download_data.py   # downloads SAML-D from Kaggle; falls back to synthetic data if unreachable
uvicorn app.main:app --reload --port 8000
# or: python app/main.py   (reads the port from .env's PORT, default 8000)
```
Then open `http://localhost:8000` (or whichever port you chose) in a browser — the dashboard and the API are served from the same origin, so no separate frontend server, and no frontend port configuration, is needed: `app/index.html` calls the API via `window.location.origin`, which always resolves to whatever host/port actually served the page. There is nothing to keep in sync between backend and frontend.

## 7. Usage Instructions
The system exposes a REST API via FastAPI, plus a browser dashboard at `/`.
You can view the interactive Swagger API documentation at `http://localhost:8000/docs`.

**Example `curl` request:**
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "Content-Type: application/json" \
     -d '{"query": "Find structuring patterns in the last 30 days"}'
```

**Attaching an image (e.g. a transaction screenshot or KYC document):**
```bash
curl -X POST "http://localhost:8000/api/v1/analyze-with-image" \
     -F "query=Is customer ID 4521 suspicious?" \
     -F "image=@/path/to/screenshot.png"
```
The dashboard exposes the same capability via the 📎 Attach button next to the query box.

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
4. **Multimodal input**: Analysts can attach a screenshot or scanned document alongside a query; a vision-capable model extracts relevant details (account numbers, amounts, red flags) and folds them into the same committee pipeline.
5. **Built to run on the real Kaggle dataset, not just a toy sample**: feature engineering and network analysis are vectorised (pandas groupby, sampled/unweighted betweenness centrality) so the pipeline stays responsive on tens of thousands of accounts rather than only the small synthetic fixture.

## 9. Disclosures
Built with the Groq API (`llama-3.3-70b-versatile`) for query understanding and committee-agent reasoning. Agentic coding assistant used for code generation.

## 10. Risk Score Weights and Thresholds
- **Weights**: Benford 0.30, Threshold Clustering 0.25, ML Anomaly 0.25, Network Centrality 0.20
- **Thresholds**: LOW < 33, MEDIUM < 60, HIGH ≤ 100
