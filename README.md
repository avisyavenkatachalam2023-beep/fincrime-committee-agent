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
- **SAML-D dataset**: Oztas, B. (2023). Synthetic Anti-Money Laundering Dataset (SAML-D). Kaggle. https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml
- **License**: CC0 Public Domain
- **Fields used**: All fields
- **Synthetic Augmentation**: Customer profiles, jurisdiction list, injected structuring patterns
- **Generation Logic**: 
  - 80% normal transactions: lognormal amount distribution
  - 15% structuring: clusters of 5-15 txns, $8,500-$9,999 in 7-day windows
  - 3% smurfing: multiple senders to same receiver in $8,500-$9,999 bands
  - 2% layering: chains A→B→C→D, 5-15% decay per hop
  - 50 Mutated records injected for Benford testing (N(9200, 300^2))

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
streamlit run app/streamlit_app.py
```

## 7. Usage Instructions
Launch the Streamlit app. Try out the 6 example queries on the left sidebar:
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
