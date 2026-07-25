# AML Financial Crime Committee Agent — System Architecture

> Version: 1.0 | Updated: 2026-07-25

---

## High-Level Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                  AML FINANCIAL CRIME COMMITTEE AGENT SYSTEM                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                          DATA ACQUISITION LAYER                              │
  │                                                                              │
  │   ┌─────────────────────┐          ┌──────────────────────────────────────┐  │
  │   │   Kaggle SAML-D API │          │  Synthetic Generator                 │  │
  │   │  (Bearer Token)     │ ──FAIL──▶│  generate_synthetic.py               │  │
  │   │  download_data.py   │          │  10,000 txns + profiles + jurisdics  │  │
  │   └──────────┬──────────┘          └────────────────┬─────────────────────┘  │
  │              │ SUCCESS                               │ FALLBACK               │
  │              ▼                                       ▼                        │
  │   ┌──────────────────────────────────────────────────────────────────────┐   │
  │   │                    data/raw/   OR   data/synthetic/                  │   │
  │   │   transactions.csv │ customers.csv │ high_risk_jurisdictions.csv     │   │
  │   └──────────────────────────────┬───────────────────────────────────────┘   │
  └─────────────────────────────────┼────────────────────────────────────────────┘
                                    │
                          load_data.py (schema normalisation)
                                    │
  ┌─────────────────────────────────▼────────────────────────────────────────────┐
  │                          DETECTION ENGINE LAYER                              │
  │                                                                              │
  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────────────┐   │
  │  │  Benford's Law  │  │  Graph Network  │  │  ML Classifier             │   │
  │  │  Analyser       │  │  Analyser       │  │  (IsolationForest +        │   │
  │  │                 │  │                 │  │   Random Forest)            │   │
  │  │ • First-digit   │  │ • NetworkX      │  │                            │   │
  │  │   distribution  │  │ • Louvain       │  │ • Feature engineering      │   │
  │  │ • χ² goodness   │  │   community     │  │ • Anomaly scoring          │   │
  │  │   of fit        │  │   detection     │  │ • Probability calibration  │   │
  │  │ • KL divergence │  │ • Centrality    │  │ • SHAP explanations        │   │
  │  │ • MAD score     │  │   metrics       │  │                            │   │
  │  │ • Z-score per   │  │ • Cycle/hub     │  │                            │   │
  │  │   digit         │  │   detection     │  │                            │   │
  │  └────────┬────────┘  └────────┬────────┘  └────────────┬───────────────┘   │
  │           │                    │                          │                   │
  │  ┌────────▼────────────────────▼──────────────────────────▼────────────────┐ │
  │  │                   Rule-Based Alert Engine                                │ │
  │  │   • Velocity rules (txn count per window)                                │ │
  │  │   • Threshold proximity rules ($8,500–$9,999 band)                       │ │
  │  │   • Geography rules (sender/receiver in high-risk jurisdictions)          │ │
  │  │   • PEP involvement rules                                                 │ │
  │  │   • Round-number detection                                                │ │
  │  └──────────────────────────────────────┬────────────────────────────────────┘ │
  └────────────────────────────────────────┼───────────────────────────────────────┘
                                           │
                                   Alert objects (SAR candidates)
                                           │
  ┌────────────────────────────────────────▼───────────────────────────────────────┐
  │                        COMMITTEE AGENT LAYER (Gemini)                          │
  │                                                                                 │
  │  ┌─────────────────────────────────────────────────────────────────────────┐   │
  │  │              Financial Crime Committee — Multi-Persona LLM              │   │
  │  │                                                                         │   │
  │  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │   │
  │  │  │  Investigator    │  │  Risk Officer    │  │  Compliance Officer  │  │   │
  │  │  │  Agent           │  │  Agent           │  │  Agent               │  │   │
  │  │  │                  │  │                  │  │                      │  │   │
  │  │  │ • Hypothesis     │  │ • Risk scoring   │  │ • Regulatory mapping │  │   │
  │  │  │   generation     │  │ • Exposure calc  │  │ • SAR narrative      │  │   │
  │  │  │ • Evidence       │  │ • Severity tier  │  │   drafting           │  │   │
  │  │  │   gathering      │  │   assignment     │  │ • FATF typology      │  │   │
  │  │  │ • Chain of       │  │ • Mitigation     │  │   classification     │  │   │
  │  │  │   custody        │  │   recommendations│  │                      │  │   │
  │  │  └──────────┬───────┘  └──────────┬───────┘  └──────────┬───────────┘  │   │
  │  │             │                      │                       │             │   │
  │  │             └──────────────────────▼───────────────────────┘             │   │
  │  │                          Committee Deliberation                          │   │
  │  │                    (structured debate → consensus verdict)               │   │
  │  └─────────────────────────────────────────────────────────────────────────┘   │
  │                                                                                 │
  │  Powered by: google-generativeai (Gemini 1.5 Pro / Flash)                      │
  └──────────────────────────────────────────────┬──────────────────────────────────┘
                                                 │
                                       Verdict + SAR draft
                                                 │
  ┌──────────────────────────────────────────────▼──────────────────────────────────┐
  │                            REPORTING & UI LAYER                                  │
  │                                                                                  │
  │  ┌───────────────────────────────┐    ┌──────────────────────────────────────┐  │
  │  │     Streamlit Dashboard       │    │     Report Generator                 │  │
  │  │                               │    │                                      │  │
  │  │  • Alert queue                │    │  • SAR narrative (Gemini)            │  │
  │  │  • Network graph (Plotly)     │    │  • PDF/HTML export                   │  │
  │  │  • Benford chart              │    │  • Audit trail log                   │  │
  │  │  • Committee debate viewer    │    │  • Regulatory filing metadata        │  │
  │  │  • Risk score heatmap         │    │                                      │  │
  │  │  • Case management sidebar    │    │                                      │  │
  │  └───────────────────────────────┘    └──────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
  data/
  ├── download_data.py          ← entry point (CLI)
  ├── load_data.py              ← imported by all detection modules
  ├── synthetic/
  │   └── generate_synthetic.py ← called by download_data.py as fallback
  └── data_dictionary.md

  detectors/
  ├── benford.py                ← uses load_data.load_transactions()
  ├── graph_analysis.py         ← uses load_data.load_transactions()
  ├── ml_detector.py            ← uses load_data.load_transactions()
  └── rule_engine.py            ← uses load_data + benford + graph scores

  agents/
  ├── committee.py              ← orchestrates persona agents
  ├── investigator.py           ← Gemini persona
  ├── risk_officer.py           ← Gemini persona
  └── compliance_officer.py     ← Gemini persona

  reporting/
  ├── sar_generator.py          ← uses committee output
  └── report_builder.py         ← PDF/HTML rendering

  app/
  └── streamlit_app.py          ← top-level UI, imports all above
```

---

## Data Flow Diagram

```
  [Raw/Synthetic CSVs]
         │
         ▼
  load_data.py ──────────────────────────────────────┐
         │                                            │
         ▼                                            ▼
  benford.py             graph_analysis.py       ml_detector.py
  (χ², KL, MAD)          (NetworkX, Louvain)    (IsolForest, RF)
         │                        │                   │
         └───────────────┬────────┘                   │
                         ▼                            │
                 rule_engine.py ◀─────────────────────┘
                 (alert scoring)
                         │
                         ▼ alerts[]
                 committee.py
                 ├── investigator.py  → Gemini API
                 ├── risk_officer.py  → Gemini API
                 └── compliance_officer.py → Gemini API
                         │
                         ▼ verdict + SAR draft
                 sar_generator.py
                         │
                         ▼
                 streamlit_app.py (Dashboard)
```

---

## Infrastructure Notes

| Component | Technology | Notes |
|---|---|---|
| Language | Python 3.11+ | Type hints throughout |
| LLM | Google Gemini 1.5 Pro / Flash | Via `google-generativeai` SDK |
| Graph | NetworkX + python-louvain | Louvain for community detection |
| ML | scikit-learn | IsolationForest + RandomForest + SHAP |
| Statistics | scipy | χ², KL divergence, z-scores |
| Data | pandas + numpy | DataFrames throughout |
| UI | Streamlit | Single-page dashboard |
| Visualisation | Plotly | Interactive charts and network graphs |
| Environment | python-dotenv | `.env` file for secrets |
| Data source | Kaggle API (Bearer token) | Fallback: synthetic generator |
