# AML Financial Crime Committee Agent — System Architecture

> Version: 2.0 | Updated: 2026-07-26
> This document describes the system as actually implemented in this repository.

---

## High-Level Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                  AML FINANCIAL CRIME COMMITTEE AGENT SYSTEM                      ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                          DATA ACQUISITION LAYER                              │
  │                                                                              │
  │   ┌─────────────────────┐          ┌──────────────────────────────────────┐ │
  │   │  Kaggle SAML-D API   │          │  Synthetic Generator                 │ │
  │   │  (Bearer Token)      │ ──FAIL──▶│  data/synthetic/generate_synthetic.py│ │
  │   │  data/download_data.py│          │  customers + jurisdictions           │ │
  │   └──────────┬───────────┘          └────────────────┬─────────────────────┘ │
  │              │ SUCCESS                                │ FALLBACK              │
  │              ▼                                        ▼                       │
  │   ┌──────────────────────────────────────────────────────────────────────┐   │
  │   │            data/raw/SAML-D.csv   OR   data/synthetic/*.csv           │   │
  │   │   9.5M transactions │ 500 synthetic customer profiles │ jurisdictions│   │
  │   └──────────────────────────────┬───────────────────────────────────────┘   │
  └─────────────────────────────────┼────────────────────────────────────────────┘
                                    │
                       data/load_data.py (column-alias normalisation,
                       type coercion — same canonical schema regardless
                       of whether the source is Kaggle or synthetic)
                                    │
  ┌─────────────────────────────────▼────────────────────────────────────────────┐
  │                       QUERY UNDERSTANDING + PLANNING                         │
  │                                                                              │
  │  src/query_understanding.py           src/planner.py                        │
  │  • Groq (llama-3.3-70b-versatile),     • Maps parsed intent → the subset    │
  │    JSON-mode, parses free text into      of tools actually needed for this  │
  │    {intent, filters, entities,           query (skips EDA/ML/network work   │
  │    target_pattern, requires_eda/ml}      the intent doesn't call for)       │
  │  • Regex/keyword fallback if Groq is                                       │
  │    unavailable or returns invalid JSON                                     │
  └─────────────────────────────────┬────────────────────────────────────────────┘
                                    │ execution plan
  ┌─────────────────────────────────▼────────────────────────────────────────────┐
  │                    DETECTION / FEATURE ENGINE LAYER (src/tools/)             │
  │                                                                              │
  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────────────┐   │
  │  │  Benford's Law  │  │  Threshold      │  │  ML Anomaly Detector       │   │
  │  │  (benford.py)   │  │  Clustering     │  │  (ml_model.py)             │   │
  │  │ • First-digit   │  │  • Sub-$10k band│  │  • IsolationForest over    │   │
  │  │   distribution  │  │    density      │  │    the feature matrix      │   │
  │  │ • χ² goodness   │  │  • Round-number │  │    from feature_engineer-  │   │
  │  │   of fit        │  │    detection    │  │    ing.py                  │   │
  │  │ • MAD score     │  │  • Spike score  │  │                            │   │
  │  └────────┬────────┘  └────────┬────────┘  └────────────┬───────────────┘   │
  │           │                    │                          │                  │
  │  ┌────────▼────────────────────▼──────────────────────────▼───────────────┐ │
  │  │        rule_engine.py — threshold counts, flagged-vs-unflagged compare  │ │
  │  └──────────────────────────────────────┬────────────────────────────────┘ │
  │                                          │                                  │
  │  ┌───────────────────────────────────────▼─────────────────────────────┐   │
  │  │  network_tool.py — directed weighted account graph (NetworkX)       │   │
  │  │  • Vectorised edge aggregation (groupby, not per-row loops)         │   │
  │  │  • Sampled, unweighted betweenness centrality (scales to 100k+      │   │
  │  │    nodes without exact all-pairs shortest paths)                    │   │
  │  │  • Louvain community detection, falls back to greedy modularity     │   │
  │  └───────────────────────────────────────────────────────────────────┘    │
  └──────────────────────────────────────────┬─────────────────────────────────┘
                                             │
                       risk_classification.py combines Benford + clustering
                       + ML + network signals into a weighted composite score
                                             │
  ┌──────────────────────────────────────────▼─────────────────────────────────┐
  │                    COMMITTEE AGENT LAYER (src/committee/)                  │
  │                                                                             │
  │  ┌─────────────────────────────────────────────────────────────────────┐   │
  │  │              Financial Crime Committee — Multi-Agent Vote            │   │
  │  │                                                                       │   │
  │  │  ┌───────────────┐ ┌──────────────┐ ┌───────────────┐ ┌───────────┐ │   │
  │  │  │ Transaction   │ │ KYC / UBO    │ │ Sanctions /   │ │ Network   │ │   │
  │  │  │ Monitoring    │ │ Agent        │ │ PEP Agent     │ │ Relation- │ │   │
  │  │  │ Agent         │ │              │ │               │ │ ship      │ │   │
  │  │  │               │ │ • Occupation/│ │ • PEP status  │ │ Agent     │ │   │
  │  │  │ • Structuring │ │   income vs. │ │ • High-risk   │ │           │ │   │
  │  │  │   / velocity  │ │   txn volume │ │   jurisdiction│ │ • Hub /   │ │   │
  │  │  │   signals     │ │   mismatch   │ │   exposure    │ │   coordi- │ │   │
  │  │  │               │ │              │ │               │ │   nator   │ │   │
  │  │  │               │ │              │ │               │ │   role    │ │   │
  │  │  └───────┬───────┘ └──────┬───────┘ └───────┬───────┘ └─────┬─────┘ │   │
  │  │          │                │                  │               │       │   │
  │  │          └────────────────┴──────────────────┴───────────────┘       │   │
  │  │                       Chair Agent (chair_agent.py)                   │   │
  │  │              synthesises votes → final decision + minutes            │   │
  │  └─────────────────────────────────────────────────────────────────────┘   │
  │                                                                              │
  │  Every specialist agent: Groq (llama-3.3-70b-versatile) for its vote +      │
  │  reasoning, with a deterministic rule-based fallback (risk-score            │
  │  thresholds) if the LLM call fails — the system never hard-fails just       │
  │  because the API is unavailable.                                           │
  └──────────────────────────────────────────────┬──────────────────────────────┘
                                                  │
                                        Votes + Chair verdict
                                                  │
  ┌───────────────────────────────────────────────▼─────────────────────────────┐
  │                         EXPLANATION + REPORTING LAYER                       │
  │                                                                              │
  │  src/tools/explanation.py           src/output_formatter.py                │
  │  • Plain-English narrative of        • Risk Memo (case reference,          │
  │    why an account was flagged,         executive summary, evidence)        │
  │    Groq-generated with a               • Committee meeting minutes         │
  │    template fallback                   • Execution-plan summary / raw JSON │
  └──────────────────────────────────────────────┬──────────────────────────────┘
                                                  │
  ┌───────────────────────────────────────────────▼─────────────────────────────┐
  │                              PRESENTATION LAYER                             │
  │                                                                              │
  │   app/main.py (FastAPI)                    app/index.html (dashboard)      │
  │   • /                 → serves the dashboard  • Query bar + 📎 image attach │
  │   • /health                                   • Execution-plan preview     │
  │   • /api/v1/plan      → dry-run the planner    • Risk banner, committee    │
  │   • /api/v1/analyze                            votes, risk memo, raw JSON  │
  │   • /api/v1/analyze-with-image                  tabs                       │
  │   • /api/v1/dataset-info                                                   │
  │   Same origin serves both the API and the UI — no separate frontend server.│
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
  data/
  ├── download_data.py          ← entry point (CLI): Kaggle → synthetic fallback
  ├── load_data.py               ← imported by app/main.py and src/orchestrator.py
  ├── synthetic/generate_synthetic.py
  └── data_dictionary.md

  src/
  ├── query_understanding.py     ← QueryUnderstandingTool (Groq + rule fallback)
  ├── planner.py                 ← DynamicExecutionPlanner
  ├── orchestrator.py            ← AMLOrchestrator: intent → handler dispatch
  ├── output_formatter.py        ← OutputFormatter (memo / minutes / summary)
  │
  ├── tools/
  │   ├── eda_tool.py
  │   ├── feature_engineering.py ← FeatureEngineeringTool (vectorised bulk path)
  │   ├── network_tool.py        ← NetworkAnalysisTool (NetworkX + Louvain)
  │   ├── risk_classification.py ← RiskClassifier (weighted composite score)
  │   ├── explanation.py         ← ExplanationTool (Groq + template fallback)
  │   └── anomaly_detection/
  │       ├── benford.py
  │       ├── threshold_clustering.py
  │       ├── ml_model.py        ← IsolationForest
  │       └── rule_engine.py
  │
  └── committee/
      ├── base_agent.py          ← BaseCommitteeAgent (Groq call + rule fallback)
      ├── transaction_monitoring_agent.py
      ├── kyc_ubo_agent.py
      ├── sanctions_pep_agent.py
      ├── network_relationship_agent.py
      └── chair_agent.py         ← ChairAgent.synthesize()

  app/
  ├── main.py                    ← FastAPI app; loads data once at startup,
  │                                 serves index.html + REST endpoints
  └── index.html                 ← single-page dashboard (vanilla JS, no build step)
```

---

## Data Flow Diagram

```
  [Kaggle SAML-D CSV / synthetic CSVs]
                 │
                 ▼
        data/load_data.py  (column normalisation, dtype coercion)
                 │
                 ▼
       src/query_understanding.py  (NL query → intent JSON)
                 │
                 ▼
          src/planner.py  (intent → execution plan)
                 │
                 ▼
        src/orchestrator.py  (dispatches to one of 8 intent handlers)
                 │
     ┌───────────┼─────────────────────────────┬───────────────────┐
     ▼           ▼                             ▼                   ▼
feature_    benford.py /              network_tool.py       rule_engine.py
engineering threshold_clustering.py   (graph, centrality,   (thresholds,
.py         ml_model.py               communities)          comparisons)
     │           │                             │                   │
     └───────────┴──────────────┬──────────────┴───────────────────┘
                                 ▼
                   risk_classification.py (composite score)
                                 │
                                 ▼
                    src/committee/*_agent.py (4 specialist votes)
                                 │
                                 ▼
                    chair_agent.py (synthesised verdict + minutes)
                                 │
                                 ▼
              explanation.py + output_formatter.py (memo, minutes, summary)
                                 │
                                 ▼
                    app/main.py → app/index.html (dashboard)
```

---

## Infrastructure Notes

| Component | Technology | Notes |
|---|---|---|
| Language | Python 3.11+ | Type hints throughout |
| LLM | Groq API — `llama-3.3-70b-versatile` | Query understanding + all committee agents; optional vision model for image attachments |
| Graph | NetworkX + python-louvain | Vectorised edge construction; sampled/unweighted betweenness for scale |
| ML | scikit-learn | IsolationForest |
| Statistics | scipy | χ², MAD |
| Data | pandas + numpy | Vectorised groupby aggregation for bulk feature computation |
| UI | FastAPI + static HTML/JS | Single origin serves both the API and the dashboard |
| Environment | python-dotenv | `.env` file for `GROQ_API_KEY` / `KAGGLE_TOKEN` |
| Data source | Kaggle API (Bearer token) | Fallback: synthetic generator (`data/synthetic/generate_synthetic.py`) |

## Scale Notes (Kaggle SAML-D, not the synthetic sample)

The Kaggle SAML-D dataset has ~9.5M rows and tens of thousands of unique
accounts, which is a different scale problem than the 500-customer synthetic
fixture the agents were originally validated against. Two design choices keep
the pipeline responsive at that scale:

1. **Feature engineering** (`feature_engineering.py`) computes per-customer
   features via `pandas.groupby` aggregation across the whole dataset in one
   pass, instead of looping over each customer and re-scanning the full
   DataFrame per customer (which is O(rows × customers) and does not finish
   in practical time once the customer count reaches the thousands).
2. **Network analysis** (`network_tool.py`) builds the transaction graph with
   vectorised `groupby` edge aggregation instead of row-by-row iteration, and
   computes betweenness centrality on a sampled, unweighted basis — coordination
   structure (hop distance) matters more for AML triage than dollar-weighted
   shortest paths, and it is an order of magnitude cheaper to compute.

`app/main.py` loads a configurable number of rows (`LOCAL_NROWS`, default
200,000) at startup for local development; set it to `None` to load the full
dataset when running on infrastructure sized for it.
