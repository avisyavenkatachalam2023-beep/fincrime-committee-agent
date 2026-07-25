import os
import sys

# --- Path setup MUST be first, before any project imports ---
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_APP_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from src.orchestrator import AMLOrchestrator

st.set_page_config(
    page_title="The Financial Crime Committee",
    layout="wide",
    page_icon="🏦"
)

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, * { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a1628 100%); color: #c9d1d9; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #30363d;
}

.main-header {
    background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 28px;
}
.main-header h1 { color: #e6edf3; margin:0; font-size: 1.8rem; font-weight: 700; }
.main-header p  { color: #8b949e; margin: 6px 0 0 0; font-size: 0.95rem; }

.intent-badge {
    display: inline-block;
    background: #1f6feb;
    color: #fff;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}

.exec-step {
    background: #161b22;
    border-left: 3px solid #388bfd;
    padding: 7px 14px;
    margin: 4px 0;
    border-radius: 0 6px 6px 0;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #c9d1d9;
}

.skipped-step {
    background: #161b22;
    border-left: 3px solid #3d444d;
    padding: 7px 14px;
    margin: 4px 0;
    border-radius: 0 6px 6px 0;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #484f58;
    text-decoration: line-through;
}

.memo-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 24px;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    white-space: pre-wrap;
    color: #c9d1d9;
    max-height: 620px;
    overflow-y: auto;
    line-height: 1.6;
}

.minutes-box {
    background: #0d1117;
    border: 1px solid #388bfd;
    border-radius: 8px;
    padding: 24px;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    white-space: pre-wrap;
    color: #c9d1d9;
    max-height: 720px;
    overflow-y: auto;
    line-height: 1.6;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #1f6feb, #0f3460);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
    font-weight: 600;
    font-size: 0.9rem;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85; }

div[data-testid="stTextInput"] input {
    background: #161b22;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 8px;
}
</style>
''', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('''
<div class="main-header">
  <h1>🏦 The Financial Crime Committee</h1>
  <p>Autonomous Multi-Agent AML Detection &amp; Escalation System — Forensic Accounting + Committee Debate</p>
</div>
''', unsafe_allow_html=True)

# ── Sidebar: example queries ──────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "Find structuring patterns in the last 30 days",
    "Which customers made 10+ transactions under $10,000?",
    "Is customer ID 4521 suspicious?",
    "Analyse this dataset for suspicious activity",
    "Who is coordinating transactions with customer 4521?",
    "Why was transaction TXN_88213 flagged?",
    "What should compliance do about customer 4521?",
    "Compare average transaction size of flagged vs non-flagged customers",
]

if 'query_text' not in st.session_state:
    st.session_state.query_text = EXAMPLE_QUERIES[0]

with st.sidebar:
    st.markdown("### 📋 Example Queries")
    st.markdown("Click any query to populate the input box:")
    for q in EXAMPLE_QUERIES:
        if st.button(q, key=f"btn_{hash(q)}", use_container_width=True):
            st.session_state.query_text = q
            st.rerun()

# ── Query input ───────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        label="Enter your AML query:",
        value=st.session_state.query_text,
        placeholder="e.g. Find structuring patterns in the last 30 days",
        key="query_input",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("Execute", type="primary", use_container_width=True)

# ── Run ───────────────────────────────────────────────────────────────────────
if run_clicked and query.strip():
    with st.spinner("Agent orchestrating analysis — please wait..."):
        try:
            orchestrator = AMLOrchestrator(
                data_dir=os.path.join(_PROJECT_ROOT, 'data')
            )
            result = orchestrator.run(query)
        except Exception as exc:
            st.error(f"Orchestrator error: {exc}")
            st.stop()

    summary = result.get('execution_summary', {})

    # ── Tabs ─────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "📊 Execution Summary",
        "📋 Risk Memo",
        "🏛️ Committee Minutes",
        "📈 Charts",
        "🔍 Data",
    ])

    # ── Tab 1: Execution Summary ─────────────────────────────────────────────
    with t1:
        st.markdown("#### Dynamic Execution Plan")
        c1, c2, c3 = st.columns(3)
        c1.metric("Intent", summary.get('parsed_intent', 'N/A'))
        c2.metric("Target Pattern", summary.get('target_pattern', 'N/A'))
        entities = summary.get('detected_entities', [])
        c3.metric("Entities Detected", ", ".join(entities) if entities else "None")

        filters = summary.get('detected_filters', {})
        active_filters = {k: v for k, v in filters.items() if v}
        if active_filters:
            st.markdown("**Filters detected:**")
            st.json(active_filters)

        st.markdown("---")
        st.markdown("**Tools Executed (in order):**")
        for i, step in enumerate(summary.get('execution_plan', []), 1):
            st.markdown(
                f'<div class="exec-step"><b>{i}.</b> {step}</div>',
                unsafe_allow_html=True,
            )

        skipped = summary.get('tools_skipped', [])
        if skipped:
            st.markdown("**Tools Skipped:**")
            st.info(summary.get('reason_for_skips', ''))
            for step in skipped:
                st.markdown(
                    f'<div class="skipped-step">{step}</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("Raw JSON Summary"):
            st.code(result.get('execution_summary_json', '{}'), language='json')

    # ── Tab 2: Risk Memo ─────────────────────────────────────────────────────
    with t2:
        st.markdown("#### Risk Memorandum")
        content = result.get('risk_memo') or result.get('plain_text_answer', '')
        if content:
            st.markdown(
                f'<div class="memo-box">{content}</div>', unsafe_allow_html=True
            )
        else:
            st.info("No Risk Memo generated for this query intent.")
            agg = result.get('aggregation_table', '')
            if agg:
                st.markdown(agg)

    # ── Tab 3: Committee Minutes ─────────────────────────────────────────────
    with t3:
        st.markdown("#### Committee Meeting Minutes")
        minutes = result.get('committee_minutes', '')
        if minutes:
            st.markdown(
                f'<div class="minutes-box">{minutes}</div>', unsafe_allow_html=True
            )
        else:
            st.info(
                "Committee review was intentionally skipped for this query intent "
                "(e.g., pure aggregation or explain-only queries do not require a committee deliberation)."
            )

    # ── Tab 4: Charts ────────────────────────────────────────────────────────
    with t4:
        st.markdown("#### Supporting Analysis Charts")
        charts = result.get('charts', {})
        displayed = 0
        for name, path in charts.items():
            if path and os.path.exists(str(path)):
                st.image(str(path), caption=name.replace('_', ' ').title(), use_column_width=True)
                displayed += 1
        if displayed == 0:
            st.info("No charts were generated for this query intent.")

    # ── Tab 5: Raw data ──────────────────────────────────────────────────────
    with t5:
        st.markdown("#### Raw Aggregation / Data")
        agg = result.get('aggregation_table', '')
        if agg:
            st.markdown(agg)
        else:
            st.info("No tabular data output for this query.")

elif run_clicked and not query.strip():
    st.warning("Please enter a query before clicking Execute.")
