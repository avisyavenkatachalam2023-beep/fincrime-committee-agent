import os
import sys
import json
import math
import numpy as np
import pandas as pd

# --- Path setup MUST be first ---
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_APP_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.orchestrator import AMLOrchestrator
from src.query_understanding import QueryUnderstandingTool
from src.planner import DynamicExecutionPlanner

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Financial Crime Committee API",
    description="Autonomous Multi-Agent AML Detection & Escalation System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (JS, CSS etc.) from app/static/
_STATIC_DIR = os.path.join(_APP_DIR, "static")
os.makedirs(_STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# JSON serialisation helper — handles NaN / Timestamp etc.
# ---------------------------------------------------------------------------
def _safe_json(obj):
    """Recursively clean obj so it is JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.ndarray,)):
        return _safe_json(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language AML query")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the main UI."""
    index_path = os.path.join(_APP_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>UI not found — run the server from project root.</h1>")


@app.get("/health")
def health_check():
    """Health check for AWS Target Groups."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/v1/analyze")
def analyze_query(request: QueryRequest):
    """
    Run the AML Orchestrator on a natural-language query.
    Returns execution_summary, risk_memo, committee_minutes, explanation,
    plain_text_answer, charts, aggregation_table.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        orchestrator = AMLOrchestrator(data_dir=os.path.join(_PROJECT_ROOT, "data"))
        result = orchestrator.run(request.query)
        clean = _safe_json(result)
        return JSONResponse(content=clean)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{exc}\n\n{tb}")


@app.get("/api/v1/plan")
def get_plan(query: str):
    """
    Return only the execution plan for a query (fast, no data loading).
    Useful for previewing what the agent will do.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        qut = QueryUnderstandingTool()
        parsed = qut.parse(query)
        parsed["user_query"] = query
        planner = DynamicExecutionPlanner()
        summary = planner.plan(parsed)
        summary["user_query"] = query
        return _safe_json(summary)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/dataset-info")
def dataset_info():
    """Return basic stats about the loaded dataset."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "load_data", os.path.join(_PROJECT_ROOT, "data", "load_data.py")
        )
        ld = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ld)
        df = ld.load_transactions(os.path.join(_PROJECT_ROOT, "data"))
        suspicious = df["is_suspicious"].sum() if "is_suspicious" in df.columns else 0
        typologies = (
            df["typology"].value_counts().to_dict()
            if "typology" in df.columns else {}
        )
        return _safe_json({
            "total_transactions": len(df),
            "suspicious_count": int(suspicious),
            "suspicious_rate_pct": round(float(suspicious) / len(df) * 100, 2),
            "typology_breakdown": typologies,
            "date_range": {
                "min": str(df["timestamp"].min()) if "timestamp" in df.columns else "N/A",
                "max": str(df["timestamp"].max()) if "timestamp" in df.columns else "N/A",
            },
            "source": "Kaggle SAML-D" if "raw" in str(ld.__file__ or "") else "Synthetic",
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
