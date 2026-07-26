import json
from datetime import date
import random

from src.committee.transaction_monitoring_agent import _HIGH_STRUCTURING_REGULARITY

class OutputFormatter:
    def format_execution_summary(self, execution_summary: dict) -> str:
        # Returns pretty-printed JSON string of the execution summary
        return json.dumps(execution_summary, indent=2)
    
    def format_risk_memo(self, case_file: dict, risk_classification: dict,
                         agent_votes: list, chair_result: dict, 
                         explanation: str, charts: dict = None) -> str:
        today = date.today().strftime("%Y-%m-%d")
        case_id = f"CASE-{date.today().year}-{random.randint(1000, 9999)}"
        
        red_flags = []
        benford_results = case_file.get('benford_results', {}) or {}
        if benford_results.get('insufficient_sample'):
            red_flags.append(f"- {benford_results.get('sample_warning', 'Insufficient transaction volume for Benford analysis.')} Risk scoring falls back to the threshold-clustering signal alone for this component.")
        elif benford_results.get('deviation_score', 0) > 0.3:
            red_flags.append(f"- Benford's Law deviation: {benford_results['deviation_score']:.2f}")
        # threshold_clustering.analyze_customer() has no top-level "spike_score"
        # or "sub_threshold_band_count" key (nested under sub_threshold/
        # round_numbers instead) — this condition previously never fired, and
        # would have raised a KeyError if it somehow had. composite_clustering_score
        # is the tool's own top-level 0-1 signal; sub_threshold_count (the actual
        # near-threshold transaction count) lives on features, not clustering_results.
        if case_file.get('clustering_results', {}).get('composite_clustering_score', 0) > 0.5:
            sub_threshold_count = case_file.get('features', {}).get('sub_threshold_count', 0)
            red_flags.append(f"- Sub-threshold clustering: {sub_threshold_count} transactions just under reporting threshold")
        # Same threshold the Transaction Monitoring Analyst's rule-based
        # fallback uses to trigger its "structuring_regularity" signal —
        # this is a genuinely computed feature (near-identical amounts and
        # intervals), but was previously invisible here even when it was
        # the actual signal driving the committee's vote.
        structuring_regularity = case_file.get('features', {}).get('structuring_regularity_score', 0)
        if structuring_regularity > _HIGH_STRUCTURING_REGULARITY:
            red_flags.append(f"- Structuring regularity: {structuring_regularity:.2f} (near-identical amounts/intervals across transactions)")
        if case_file.get('network_results', {}).get('is_hub', False):
            red_flags.append(f"- Network centrality: Identified as a hub in a transaction network")
        for flag in (chair_result.get('key_signals') or []):
            red_flags.append(f"- {flag}")
            
        if not red_flags:
            red_flags.append("- No significant quantitative red flags identified.")

        red_flags_text = "\n   ".join(red_flags)
        
        deliberation_text = ""
        for vote in agent_votes:
            deliberation_text += f"   - {vote['agent']}: {vote['vote']} — {vote['reasoning']}\n"
        deliberation_text += f"   - Chair Decision: {chair_result['final_decision']} — {chair_result['synthesis']}"
        if chair_result.get('tier_decision_note'):
            deliberation_text += f"\n   - Score/Decision Note: {chair_result['tier_decision_note']}"

        memo = f"""RISK MEMORANDUM
Case Reference: {case_id}
Prepared by: Financial Crime Committee (Autonomous Agent)
Date: {today}

1. EXECUTIVE SUMMARY
   {explanation}

2. BACKGROUND
   Customer/Transaction ID: {case_file.get('customer_id', 'N/A')}
   Triggering Query: {case_file.get('query', 'N/A')}

3. RED FLAGS IDENTIFIED
   {red_flags_text}

4. COMMITTEE DELIBERATION SUMMARY
{deliberation_text}

5. QUANTITATIVE RISK SCORE
   {risk_classification.get('composite_score', 0):.0f}/100 → {risk_classification.get('risk_tier', 'UNKNOWN')}

6. RECOMMENDATION
   {chair_result.get('escalation_action', chair_result.get('final_decision', 'MONITOR'))}

Sign-off: Autonomous AML Agent v1.0
"""
        return memo

    def format_aggregation_result(self, rule_results: dict, execution_summary: dict) -> str:
        res = "Aggregation Query Results:\n"
        res += f"Query: {execution_summary.get('user_query', '')}\n\n"
        if 'customer_id' in rule_results.columns:
            res += rule_results.to_markdown(index=False)
        else:
            res += json.dumps(rule_results, indent=2)
        return res

    def format_comparison_result(self, comparison_data: dict, execution_summary: dict) -> str:
        return f"Comparison Results:\n\n{json.dumps(comparison_data, indent=2)}"

    def format_network_result(self, network_results: dict, execution_summary: dict) -> str:
        return f"Network Query Results:\n\n{json.dumps(network_results, indent=2)}"

    def format_full_result(self, **kwargs) -> dict:
        return {
            'execution_summary_json': kwargs.get('execution_summary_json', ''),
            'execution_summary': kwargs.get('execution_summary', {}),
            # Single source of truth for the composite risk score/tier, read
            # by every panel (banner, Answer, Risk Memo, Committee). None
            # when this query type never computed a risk assessment (e.g.
            # aggregation_query, comparative_query) — the UI shows "N/A" for
            # that, consistent with risk_memo also being empty in that case.
            'risk_score': kwargs.get('risk_score'),
            'risk_tier': kwargs.get('risk_tier'),
            'risk_memo': kwargs.get('risk_memo', ''),
            'committee_minutes': kwargs.get('committee_minutes', ''),
            'explanation': kwargs.get('explanation', ''),
            'plain_text_answer': kwargs.get('plain_text_answer', ''),
            'charts': kwargs.get('charts', {}),
            'aggregation_table': kwargs.get('aggregation_table', ''),
            'agent_votes': kwargs.get('agent_votes', []),
            'chair_result': kwargs.get('chair_result', {}),
            'top_suspicious': kwargs.get('top_suspicious', []),
        }
