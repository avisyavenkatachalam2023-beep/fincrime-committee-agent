import streamlit as st
import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orchestrator import AMLOrchestrator

st.set_page_config(page_title="Financial Crime Committee", layout="wide", page_icon="🏦")

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a1628 100%); color: #c9d1d9;}
.main-header { 
    background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
}
.execution-plan-step {
    background: #161b22;
    border-left: 3px solid #388bfd;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 0 6px 6px 0;
    font-family: monospace;
    font-size: 13px;
    color: #c9d1d9;
}
.skipped-tool {
    background: #161b22;
    border-left: 3px solid #6e7681;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 0 6px 6px 0;
    font-family: monospace;
    font-size: 13px;
    color: #6e7681;
    text-decoration: line-through;
}
.memo-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 24px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    white-space: pre-wrap;
    color: #c9d1d9;
    max-height: 600px;
    overflow-y: auto;
}
.minutes-box {
    background: #0d1117;
    border: 1px solid #388bfd;
    border-radius: 8px;
    padding: 24px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    white-space: pre-wrap;
    color: #c9d1d9;
    max-height: 700px;
    overflow-y: auto;
}
</style>
''', unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🏦 The Financial Crime Committee</h1><p>Autonomous Agentic AML Detection & Escalation System</p></div>', unsafe_allow_html=True)

if 'query' not in st.session_state:
    st.session_state.query = "Find structuring patterns in the last 30 days"

def set_query(q):
    st.session_state.query = q

st.sidebar.title("Example Queries")
queries = [
    "Find structuring patterns in the last 30 days",
    "Which customers made 10+ transactions under $10,000?",
    "Is customer ID 4521 suspicious?",
    "Analyse this dataset for suspicious activity",
    "Who is coordinating transactions with customer 4521?",
    "Why was transaction TXN_88213 flagged?"
]
for q in queries:
    st.sidebar.button(q, on_click=set_query, args=(q,))

query = st.text_input("Enter your command:", key="query")

if st.button("Execute"):
    with st.spinner("Agent is orchestrating analysis..."):
        orchestrator = AMLOrchestrator()
        result = orchestrator.run(query)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Execution Summary", "📋 Risk Memo", "🏛️ Committee Minutes", "📈 Charts", "🔍 Raw Data"])
        
        with tab1:
            st.subheader("Dynamic Execution Plan")
            summary = result.get('execution_summary', {})
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Intent", summary.get('parsed_intent', 'N/A'))
            col2.metric("Target Pattern", summary.get('target_pattern', 'N/A'))
            col3.metric("Entities Detected", ", ".join(summary.get('detected_entities', [])) or "None")
            
            st.markdown("### Execution Plan")
            for i, step in enumerate(summary.get('execution_plan', [])):
                st.markdown(f'<div class="execution-plan-step">{i+1}. {step}</div>', unsafe_allow_html=True)
                
            st.markdown("### Skipped Tools")
            st.info(summary.get('reason_for_skips', 'No reason provided.'))
            for step in summary.get('tools_skipped', []):
                st.markdown(f'<div class="skipped-tool">{step}</div>', unsafe_allow_html=True)
                
            with st.expander("Raw JSON Summary"):
                st.json(result.get('execution_summary_json', '{}'))
                
        with tab2:
            st.subheader("Final Risk Memorandum")
            if result.get('risk_memo'):
                st.markdown(f'<div class="memo-box">{result["risk_memo"]}</div>', unsafe_allow_html=True)
            elif result.get('plain_text_answer'):
                st.markdown(f'<div class="memo-box">{result["plain_text_answer"]}</div>', unsafe_allow_html=True)
            else:
                st.write("No risk memo generated for this query intent.")
                
        with tab3:
            st.subheader("Committee Deliberations")
            if result.get('committee_minutes'):
                st.markdown(f'<div class="minutes-box">{result["committee_minutes"]}</div>', unsafe_allow_html=True)
            else:
                st.write("Committee review was intentionally skipped for this query intent.")
                
        with tab4:
            st.subheader("Supporting Analysis Charts")
            charts = result.get('charts', {})
            has_charts = False
            for cname, cpath in charts.items():
                if cpath and os.path.exists(cpath):
                    has_charts = True
                    st.image(cpath, caption=cname.replace('_', ' ').title())
            if not has_charts:
                st.write("No charts generated for this query.")
                
        with tab5:
            if result.get('aggregation_table'):
                st.markdown(result['aggregation_table'])
            else:
                st.write("No raw tabular data generated for this query.")
