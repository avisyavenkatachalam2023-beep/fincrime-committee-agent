import json
from datetime import date
import random

from src.committee.transaction_monitoring_agent import _HIGH_STRUCTURING_REGULARITY
from src.committee.network_relationship_agent import _HIGH_BETWEENNESS_CENTRALITY


def build_red_flags(case_file: dict) -> list[str]:
    """Single source of truth for which signals count as a red flag for a
    case. Used identically by the Risk Memo's "Red Flags Identified"
    section and the Executive Summary narrative (explanation.py) — each
    previously computed its own independent list with different
    thresholds, so the two sections could (and did) disagree for the same
    customer.

    Returns a list of plain description strings (no bullet/prefix).
    """
    flags: list[str] = []
    features = case_file.get('features', {}) or {}
    benford = case_file.get('benford_results', {}) or {}
    clustering = case_file.get('clustering_results', {}) or {}
    ml = case_file.get('ml_results', {}) or {}
    network = case_file.get('network_results', {}) or {}

    if benford.get('insufficient_sample'):
        flags.append(
            f"{benford.get('sample_warning', 'Insufficient transaction volume for Benford analysis.')} "
            "Risk scoring falls back to the threshold-clustering signal alone for this component."
        )
    else:
        deviation_score = float(benford.get('deviation_score', 0) or 0)
        if deviation_score > 0.3:
            flags.append(
                f"Benford's Law deviation score of {deviation_score:.2f} indicates a non-conforming amount distribution."
            )

    composite_clustering = float(clustering.get('composite_clustering_score', 0) or 0)
    if composite_clustering > 0.5:
        sub_threshold_count = features.get('sub_threshold_count', 0)
        flags.append(
            f"Sub-threshold clustering score of {composite_clustering:.2f}: "
            f"{sub_threshold_count} transactions found just under the reporting threshold."
        )

    round_ratio = float(clustering.get('round_numbers', {}).get('round_number_ratio', 0) or 0)
    if round_ratio > 0.25:
        flags.append(f"Round-number transaction ratio of {round_ratio:.1%}, well above the ~5% baseline.")

    structuring_regularity = float(features.get('structuring_regularity_score', 0) or 0)
    if structuring_regularity > _HIGH_STRUCTURING_REGULARITY:
        flags.append(
            f"Structuring regularity score of {structuring_regularity:.2f}: "
            "near-identical amounts/intervals across transactions."
        )

    ml_score = float(ml.get('ml_anomaly_score', 0) or 0)
    if ml_score > 0.6:
        flags.append(f"ML Isolation Forest anomaly score of {ml_score:.3f}: statistical outlier in the feature space.")

    # Only assert a hub/network concern with a real absolute signal behind
    # it — network_tool.py's is_hub is a relative (top-decile) designation
    # that can be True even when betweenness_centrality is ~0 and there are
    # no connected flagged accounts, which previously produced a "hub"
    # red flag alongside betweenness_centrality=0.0000 and 0 connections.
    hub_score = float(network.get('hub_score', 0) or 0)
    is_hub = bool(network.get('is_hub', False))
    flagged_accounts_field = network.get('connected_flagged_accounts', 0)
    connected_flagged_count = (
        len(flagged_accounts_field) if isinstance(flagged_accounts_field, list)
        else int(flagged_accounts_field or 0)
    )
    if is_hub and (hub_score > _HIGH_BETWEENNESS_CENTRALITY or connected_flagged_count > 0):
        flags.append(f"Network betweenness centrality of {hub_score:.4f}: identified as a hub in the transaction network.")
    if connected_flagged_count > 0:
        flags.append(f"Directly connected to {connected_flagged_count} flagged account(s).")

    return flags


class OutputFormatter:
    def format_execution_summary(self, execution_summary: dict) -> str:
        # Returns pretty-printed JSON string of the execution summary
        return json.dumps(execution_summary, indent=2)
    
    def format_risk_memo(self, case_file: dict, risk_classification: dict,
                         agent_votes: list, chair_result: dict, 
                         explanation: str, charts: dict = None) -> str:
        today = date.today().strftime("%Y-%m-%d")
        # Reuse the Chair's case_reference (generated once per run) instead
        # of minting a second random ID here — previously the memo and the
        # Committee Minutes each generated their own, so the two almost
        # never matched for the same analysis.
        case_id = (chair_result or {}).get('case_reference') or f"CASE-{date.today().year}-{random.randint(1000, 9999)}"

        red_flags = [f"- {f}" for f in build_red_flags(case_file)]
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
