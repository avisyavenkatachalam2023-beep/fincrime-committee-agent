import json
from datetime import date
import random

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
        if case_file.get('benford_results', {}).get('deviation_score', 0) > 0.3:
            red_flags.append(f"- Benford's Law deviation: {case_file['benford_results']['deviation_score']:.2f}")
        if case_file.get('clustering_results', {}).get('spike_score', 0) > 0.5:
            red_flags.append(f"- Sub-threshold clustering: {case_file['clustering_results']['sub_threshold_band_count']} transactions just under reporting threshold")
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
