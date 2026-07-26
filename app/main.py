import os
import sys
import json
import math
import base64
import uuid
import mimetypes
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── Path setup ───────────────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_APP_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Load .env explicitly — don't rely on it being loaded as a side effect of
# importing src.query_understanding (which only happens lazily, inside a
# request handler, and only for the text-only /api/v1/analyze path).
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

logger = logging.getLogger("aml")

# ── Global cache (loaded once at startup) ────────────────────────────────────
_cache: dict = {
    "transactions": None,
    "customers": None,
    "jurisdictions": None,
    "dataset_info": None,
    "custom_dataset_name": None,
}

# ── How many rows to load (None = full Kaggle SAML-D dataset, ~9.5M rows) ────
# A smaller number (e.g. 200_000) is useful for fast local iteration; None
# loads the entire dataset — expect a longer startup (CSV read + graph/feature
# construction on ~9.5M rows) and correspondingly higher memory use.
LOCAL_NROWS = None


def _load_data_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "load_data", os.path.join(_PROJECT_ROOT, "data", "load_data.py")
    )
    ld = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ld)
    return ld


def _update_dataset_info(df: pd.DataFrame, source: str, note: str = "") -> None:
    """(Re)compute the summary stats shown in the UI's Dataset Stats panel and
    returned by /api/v1/dataset-info. Shared by startup loading, dataset
    upload, and dataset reset so all three paths report figures the same way.
    """
    susp = int(df["is_suspicious"].sum()) if "is_suspicious" in df.columns else 0
    typo = df["typology"].value_counts().to_dict() if "typology" in df.columns else {}
    _cache["dataset_info"] = {
        "total_transactions": len(df),
        "suspicious_count": susp,
        "suspicious_rate_pct": round(susp / max(len(df), 1) * 100, 2),
        "typology_breakdown": typo,
        "date_range": {
            "min": str(df["timestamp"].min()) if "timestamp" in df.columns else "N/A",
            "max": str(df["timestamp"].max()) if "timestamp" in df.columns else "N/A",
        },
        "source": source,
        "is_custom": bool(_cache.get("custom_dataset_name")),
        "note": note or f"Loaded {len(df):,} rows.",
    }


def _prime_cache():
    """Load the default dataset (Kaggle raw, falling back to synthetic) into
    memory. Called once at startup, and again by /api/v1/reset-dataset to
    undo an uploaded dataset."""
    global _cache
    print("[startup] Loading dataset into memory — this happens ONCE…")
    ld = _load_data_module()
    data_dir = os.path.join(_PROJECT_ROOT, "data")
    _cache["custom_dataset_name"] = None

    # Detect if Kaggle raw file exists
    raw_path = os.path.join(data_dir, "raw", "SAML-D.csv")
    if os.path.exists(raw_path) and LOCAL_NROWS:
        print(f"[startup] Kaggle SAML-D detected — loading first {LOCAL_NROWS:,} rows for local dev…")
        df = pd.read_csv(raw_path, nrows=LOCAL_NROWS, low_memory=False)
        # Apply same post-processing as load_data.py
        df = ld._normalise_columns(df, ld.TXN_ALIASES)
        df = ld._postprocess_saml_d(df)
        df = ld._add_missing_cols(df, ld.TXN_CANONICAL_COLS)
        df = ld._coerce_transactions(df)
        source = "Kaggle SAML-D (sampled)"
    else:
        df = ld.load_transactions(data_dir)
        source = "Kaggle SAML-D" if os.path.exists(raw_path) else "Synthetic"

    _cache["transactions"] = df
    _cache["customers"] = ld.load_customers(data_dir)
    _cache["jurisdictions"] = ld.load_jurisdictions(data_dir)

    _update_dataset_info(
        df, source,
        note=f"Loaded {len(df):,} rows. Set LOCAL_NROWS=None in main.py for full dataset.",
    )
    print(f"[startup] Done — {len(df):,} transactions loaded from {source}.")


# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _prime_cache()
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Financial Crime Committee API",
    description="Autonomous Multi-Agent AML Detection & Escalation System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = os.path.join(_APP_DIR, "static")
_UPLOADS_DIR = os.path.join(_STATIC_DIR, "uploads")
os.makedirs(_UPLOADS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# The anomaly-detection/EDA/network tools save their PNG charts to
# data/charts/ (see CHARTS_DIR in each tool module), which is a different
# directory from app/static/ above — without this mount, every chart_path
# returned by the orchestrator points at a filesystem path with no URL that
# actually serves it, so charts silently could never be viewed in the UI.
_CHARTS_DIR = os.path.join(_PROJECT_ROOT, "data", "charts")
os.makedirs(_CHARTS_DIR, exist_ok=True)
app.mount("/charts", StaticFiles(directory=_CHARTS_DIR), name="charts")

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB
MAX_DATASET_BYTES = 250 * 1024 * 1024  # 250 MB


def _chart_url(path: str) -> str:
    """Convert an on-disk chart path (e.g. 'data/charts/benford_123.png',
    possibly with Windows backslashes) into a URL servable via the /charts
    static mount above."""
    if not path:
        return ""
    basename = os.path.basename(str(path).replace("\\", "/"))
    return f"/charts/{basename}"


def _convert_chart_paths(result: dict) -> dict:
    """Rewrite every entry in result['charts'] from a filesystem path to a
    URL, in place, and drop entries whose chart was never generated."""
    charts = result.get("charts")
    if isinstance(charts, dict):
        result["charts"] = {k: _chart_url(v) for k, v in charts.items() if v}
    return result


def _describe_image_with_groq(image_bytes: bytes, mime_type: str, user_query: str) -> str:
    """Use a Groq vision-capable model to describe an uploaded image and pull
    out anything relevant to the AML investigation (account numbers, amounts,
    document type, visible red flags).

    Vision model access depends on the Groq account/plan; if the configured
    model isn't available, this degrades gracefully to a note explaining that
    the image was stored but not visually analysed — the text query still
    runs through the full pipeline either way.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Image received, but no GROQ_API_KEY is configured — visual analysis skipped."

    vision_model = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"
        response = client.chat.completions.create(
            model=vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "You are an AML compliance analyst. In under 120 words, describe what "
                        "this image shows (e.g. transaction screenshot, ID document, wire receipt, "
                        "chart) and extract any account numbers, amounts, names, dates, or red flags "
                        "relevant to a financial-crime investigation.\n\n"
                        f"Analyst's question: {user_query}"
                    )},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            temperature=0.2,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Vision analysis unavailable (model=%s): %s", vision_model, exc)
        return (
            "Image attached and stored, but automated visual analysis is unavailable right now "
            f"({exc.__class__.__name__}). The text query was still analysed in full below."
        )


# ── JSON sanitiser ────────────────────────────────────────────────────────────
def _safe(obj):
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return _safe(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


# ── Pydantic models ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language AML query")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    index_path = os.path.join(_APP_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        # no-store: this file changes across deploys/edits, and a stale cached
        # copy in the browser would keep calling old/removed API shapes long
        # after the server has moved on — surfacing as confusing "failed to
        # fetch" errors that have nothing to do with the actual backend.
        return HTMLResponse(content=f.read(), headers={"Cache-Control": "no-store"})


@app.get("/health")
def health_check():
    loaded = _cache["transactions"] is not None
    return {
        "status": "ok",
        "version": "1.0.0",
        "data_loaded": loaded,
        "rows": len(_cache["transactions"]) if loaded else 0,
    }


@app.get("/api/v1/dataset-info")
def dataset_info():
    if _cache["dataset_info"] is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet. Please wait.")
    return _safe(_cache["dataset_info"])


@app.post("/api/v1/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Replace the in-memory transactions dataset with a user-uploaded CSV.
    Every query after this call (across all users of this server, since the
    cache is process-global like the rest of the app) runs against the
    uploaded data until /api/v1/reset-dataset is called or the server
    restarts.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_DATASET_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large (max {MAX_DATASET_BYTES // (1024*1024)} MB).",
        )

    import io
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV has no rows.")

    ld = _load_data_module()
    df = ld._normalise_columns(df, ld.TXN_ALIASES)
    df = ld._postprocess_saml_d(df)
    df = ld._add_missing_cols(df, ld.TXN_CANONICAL_COLS)
    df = ld._coerce_transactions(df)

    # Every downstream tool assumes usable amount/timestamp columns — without
    # this check a badly-shaped CSV would silently normalise to all-NaN and
    # every query would just return empty results with no clear reason why.
    missing = []
    if "amount" not in df.columns or df["amount"].notna().sum() == 0:
        missing.append("amount")
    if "timestamp" not in df.columns or df["timestamp"].notna().sum() == 0:
        missing.append("timestamp (or Date/Time)")
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Uploaded CSV is missing usable column(s): {', '.join(missing)}. "
                f"Expected columns (or recognised aliases of): {', '.join(ld.TXN_CANONICAL_COLS)}"
            ),
        )

    _cache["transactions"] = df
    _cache["custom_dataset_name"] = file.filename
    _update_dataset_info(
        df, source=f"Uploaded: {file.filename}",
        note=f"Loaded {len(df):,} rows from an uploaded CSV. Customer/jurisdiction reference data unchanged.",
    )
    print(f"[upload] Replaced transactions dataset with '{file.filename}' — {len(df):,} rows.")
    return _safe(_cache["dataset_info"])


@app.post("/api/v1/reset-dataset")
def reset_dataset():
    """Discard an uploaded dataset and reload the default Kaggle/synthetic
    dataset from disk (same logic as startup)."""
    _prime_cache()
    return _safe(_cache["dataset_info"])


@app.get("/api/v1/plan")
def get_plan(query: str):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        from src.query_understanding import QueryUnderstandingTool
        from src.planner import DynamicExecutionPlanner
        qut = QueryUnderstandingTool()
        parsed = qut.parse(query)
        parsed["user_query"] = query
        planner = DynamicExecutionPlanner()
        summary = planner.plan(parsed)
        summary["user_query"] = query
        return _safe(summary)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _run_orchestrator(query: str) -> dict:
    """Build an AMLOrchestrator against the pre-loaded in-memory cache and run
    it against *query*. Shared by the text-only and image-attached endpoints.
    """
    from src.orchestrator import AMLOrchestrator
    orch = AMLOrchestrator(data_dir=os.path.join(_PROJECT_ROOT, "data"))
    orch._customers_df = _cache["customers"]
    orch._jurisdictions_df = _cache["jurisdictions"]

    # Re-point load_data at the in-memory cache instead of hitting disk again,
    # but still apply date-range filtering exactly as the un-patched
    # load_data() would — a no-op here would silently ignore every
    # "last 30 days" / "Q1 2024" style filter in the user's query.
    def _cached_load(filters=None):
        df = _cache["transactions"]
        if filters and filters.get("date_range"):
            df = orch._apply_date_filter(df, filters["date_range"])
        orch._transactions_df = df
    orch.load_data = _cached_load

    result = orch.run(query)
    return _convert_chart_paths(result)


@app.post("/api/v1/analyze")
def analyze_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if _cache["transactions"] is None:
        raise HTTPException(status_code=503, detail="Data still loading. Please wait a moment and retry.")
    try:
        result = _run_orchestrator(request.query)
        return JSONResponse(content=_safe(result))
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{exc}\n\n{tb}")


@app.post("/api/v1/analyze-with-image")
async def analyze_with_image(query: str = Form(...), image: UploadFile = File(...)):
    """Same analysis pipeline as /api/v1/analyze, but accepts an attached
    image (e.g. a screenshot of a transaction record, a scanned KYC document,
    or a wire receipt). The image is described by a vision-capable model and
    that description is appended as context to the query before running the
    normal committee pipeline — so the rest of the system doesn't need to
    know images exist at all.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if _cache["transactions"] is None:
        raise HTTPException(status_code=503, detail="Data still loading. Please wait a moment and retry.")
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {image.content_type}")

    image_bytes = await image.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image too large (max 8 MB).")

    ext = mimetypes.guess_extension(image.content_type) or ".png"
    filename = f"{uuid.uuid4().hex}{ext}"
    with open(os.path.join(_UPLOADS_DIR, filename), "wb") as f:
        f.write(image_bytes)

    image_analysis = _describe_image_with_groq(image_bytes, image.content_type, query)
    combined_query = f"{query}\n\n[Attached image analysis]: {image_analysis}"

    try:
        result = _run_orchestrator(combined_query)
        result["image_url"] = f"/static/uploads/{filename}"
        result["image_analysis"] = image_analysis
        return JSONResponse(content=_safe(result))
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{exc}\n\n{tb}")


if __name__ == "__main__":
    # `uvicorn app.main:app --port N` is the primary way to run this (see
    # README), but `python app/main.py` also works and reads the port from
    # .env's PORT so there's a single source of truth instead of a number
    # copy-pasted into a shell command that can drift from what's documented.
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
