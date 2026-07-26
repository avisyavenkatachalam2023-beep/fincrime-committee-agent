"""
transaction_monitoring_agent.py
--------------------------------
Specialist agent focusing on transaction velocity, Benford's Law deviations,
round-number clustering, sub-threshold patterns, and rolling sum anomalies.
"""

import logging

try:
    from src.committee.base_agent import BaseCommitteeAgent
except ImportError:
    from .base_agent import BaseCommitteeAgent

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a Transaction Monitoring Analyst Agent at a Financial Crime Committee.
Your specialty: analyzing transaction velocity, volume patterns, Benford's Law deviations,
round-number clustering, and rolling sum anomalies.

You will receive a case file with transaction analytics. Your job is to vote on the risk level:
- MONITOR: Low concern, watch but no action needed
- REVIEW: Moderate concern, compliance team should review
- REPORT: High concern, file a Suspicious Activity Report (SAR)

Focus on: transaction velocity, sub-threshold transaction counts, Benford deviation score,
round number clustering, rolling 30-day sums vs declared income.

Respond ONLY with valid JSON:
{
  "vote": "MONITOR|REVIEW|REPORT",
  "reasoning": "2-4 sentence explanation citing specific numbers from the case file",
  "confidence": 0.0-1.0,
  "key_signals": ["list of signal names that drove your vote"]
}"""

# ---------------------------------------------------------------------------
# Thresholds used in the rule-based fallback
# ---------------------------------------------------------------------------
_HIGH_VELOCITY_THRESHOLD = 5.0          # txns / day
_HIGH_SUB_THRESHOLD_COUNT = 5           # structuring signals
_HIGH_BENFORD_DEVIATION = 0.15          # deviation score
_HIGH_ROUND_NUMBER_SCORE = 0.6          # fraction of round numbers
_HIGH_STRUCTURING_REGULARITY = 0.7      # regularity index
_HIGH_SPIKE_SCORE = 0.7                 # spike anomaly score


class TransactionMonitoringAgent(BaseCommitteeAgent):
    """
    Analyses transaction-level features and statistical signals (Benford's Law,
    clustering, velocity) to produce a MONITOR / REVIEW / REPORT vote.
    """

    def __init__(self) -> None:
        """Initialise the Transaction Monitoring Agent with its specialist prompt."""
        super().__init__(
            agent_name="Transaction Monitoring Analyst",
            system_prompt=_SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------
    # Override fallback with specialist heuristics
    # ------------------------------------------------------------------

    def _rule_based_fallback(self, case_file: dict) -> dict:
        """
        Specialist rule-based fallback that evaluates transaction features,
        Benford deviation, clustering scores, and velocity rather than relying
        solely on the generic ``risk_score`` threshold.

        Decision logic
        --------------
        High concern (REPORT) if ≥ 3 of the following conditions hold:
          • velocity > 5 txns/day
          • sub_threshold_count > 5
          • benford deviation_score > 0.15
          • round_number_score > 0.6
          • structuring_regularity_score > 0.7
          • spike_score > 0.7
          • pattern is 'structuring' or 'smurfing'

        Moderate concern (REVIEW) if 1–2 conditions hold.
        Low concern (MONITOR) if 0 conditions hold.

        Parameters
        ----------
        case_file : dict
            Structured case file.

        Returns
        -------
        dict
            ``{vote, reasoning, confidence, key_signals}``
        """
        features = case_file.get("features", {})
        benford = case_file.get("benford_results", {})
        clustering = case_file.get("clustering_results", {})
        pattern = case_file.get("pattern", "none").lower()

        velocity = float(features.get("velocity", 0.0))
        sub_threshold_count = int(features.get("sub_threshold_count", 0))
        structuring_regularity = float(features.get("structuring_regularity_score", 0.0))
        rolling_sum = float(features.get("rolling_sum_30d", 0.0))
        avg_amount = float(features.get("avg_amount", 0.0))

        benford_deviation = float(benford.get("deviation_score", 0.0))
        p_value = float(benford.get("p_value", 1.0))
        benford_insufficient = bool(benford.get("insufficient_sample"))

        # threshold_clustering.analyze_customer() nests these under
        # "round_numbers"/"composite_clustering_score", not top-level
        # "round_number_score"/"spike_score" — reading the wrong keys
        # silently defaulted both signals to 0.0 for every case.
        round_number_score = float(clustering.get("round_numbers", {}).get("round_number_ratio", 0.0))
        spike_score = float(clustering.get("composite_clustering_score", 0.0))

        # Below MIN_SAMPLE_SIZE, benford.py forces deviation_score=0.0 and
        # p_value=1.0 as neutral placeholders, NOT a real "conforms cleanly"
        # result — citing them as if computed previously produced text like
        # "Benford deviation of 0.000 (p=1.0000) indicates anomalous
        # activity", which is both uncomputed and, even taken at face value,
        # backwards (p=1.0 means indistinguishable from the expected
        # distribution, the opposite of anomalous).
        if benford_insufficient:
            benford_note = (
                f"Benford analysis was not computed (insufficient transaction "
                f"volume, N={benford.get('sample_size', '?')})."
            )
        elif benford_deviation > _HIGH_BENFORD_DEVIATION:
            benford_note = (
                f"Benford deviation score of {benford_deviation:.3f} (p={p_value:.4f}) "
                "shows the amount distribution deviates from the expected pattern."
            )
        else:
            benford_note = (
                f"Benford deviation score of {benford_deviation:.3f} (p={p_value:.4f}) "
                "is within the expected range, showing no distributional anomaly."
            )

        triggered_signals: list[str] = []

        if velocity > _HIGH_VELOCITY_THRESHOLD:
            triggered_signals.append(f"high_velocity ({velocity:.2f} txns/day)")
        if sub_threshold_count > _HIGH_SUB_THRESHOLD_COUNT:
            triggered_signals.append(
                f"elevated_sub_threshold_count ({sub_threshold_count})"
            )
        if benford_deviation > _HIGH_BENFORD_DEVIATION:
            triggered_signals.append(
                f"benford_deviation_high ({benford_deviation:.3f})"
            )
        if round_number_score > _HIGH_ROUND_NUMBER_SCORE:
            triggered_signals.append(
                f"round_number_clustering ({round_number_score:.2f})"
            )
        if structuring_regularity > _HIGH_STRUCTURING_REGULARITY:
            triggered_signals.append(
                f"structuring_regularity ({structuring_regularity:.2f})"
            )
        if spike_score > _HIGH_SPIKE_SCORE:
            triggered_signals.append(f"spike_anomaly ({spike_score:.2f})")
        if pattern in ("structuring", "smurfing"):
            triggered_signals.append(f"detected_pattern_{pattern}")

        signal_count = len(triggered_signals)

        if signal_count >= 3:
            vote = "REPORT"
            reasoning = (
                f"{signal_count} high-severity transaction signals detected: "
                f"{', '.join(triggered_signals[:3])}. "
                f"Rolling 30-day sum of USD {rolling_sum:,.0f} combined with "
                f"avg amount USD {avg_amount:,.0f} and pattern '{pattern}' "
                "strongly suggest intentional structuring behaviour warranting SAR filing."
            )
            confidence = min(0.95, 0.70 + signal_count * 0.04)
        elif signal_count >= 1:
            vote = "REVIEW"
            reasoning = (
                f"{signal_count} transaction signal(s) warrant compliance review: "
                f"{', '.join(triggered_signals)}. "
                f"{benford_note} Velocity is {velocity:.2f} txns/day. "
                "Not conclusively suspicious, but warrants a closer look."
            )
            confidence = 0.50 + signal_count * 0.07
        else:
            vote = "MONITOR"
            reasoning = (
                f"No transaction monitoring thresholds breached. "
                f"Velocity ({velocity:.2f} txns/day) and round-number score "
                f"({round_number_score:.2f}) are within acceptable ranges. {benford_note} "
                "Continue routine monitoring."
            )
            confidence = 0.75

        return {
            "vote": vote,
            "reasoning": reasoning,
            "confidence": round(min(1.0, confidence), 3),
            "key_signals": triggered_signals if triggered_signals else ["no_signals_triggered"],
        }
