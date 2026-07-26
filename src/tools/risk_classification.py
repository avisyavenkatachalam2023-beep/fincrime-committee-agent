"""
risk_classification.py
----------------------
Composite risk scorer and customer risk tier classifier for the AML Financial
Crime Committee Agent.

Combines signals from Benford's Law analysis, threshold clustering, ML anomaly
detection, and network centrality into a single interpretable risk tier.

Weight documentation (IMPORTANT — must be kept in sync with README):

    | Signal                | Weight |
    |-----------------------|--------|
    | Benford deviation     |  0.30  |
    | Threshold clustering  |  0.25  |
    | ML anomaly score      |  0.25  |
    | Network centrality    |  0.20  |

Tier thresholds (composite score 0–100):
    LOW    : 0  – 33
    MEDIUM : 33 – 60
    HIGH   : 60 – 100
"""

from __future__ import annotations

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class RiskClassifier:
    """Compute composite AML risk scores and assign tier classifications.

    All individual signal scores are expected in the range [0, 1].  The
    composite score is expressed on a 0–100 scale for readability.
    """

    # Documented signal weights — must match README
    WEIGHTS: dict[str, float] = {
        "benford": 0.30,
        "threshold_clustering": 0.25,
        "ml_anomaly": 0.25,
        "network_centrality": 0.20,
    }

    # Composite score (0–100) thresholds for tier assignment
    THRESHOLDS: dict[str, float] = {
        "low": 33.0,
        "medium": 60.0,
        "high": 100.0,
    }

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score(
        self,
        benford_score: float = 0.0,
        clustering_score: float = 0.0,
        ml_score: float = 0.0,
        network_score: float = 0.0,
        benford_reliable: bool = True,
    ) -> dict[str, Any]:
        """Compute a composite risk score from individual signal scores.

        Args:
            benford_score: Benford deviation score in [0, 1].
            clustering_score: Threshold clustering composite score in [0, 1].
            ml_score: ML Isolation Forest anomaly score in [0, 1].
            network_score: Network betweenness centrality score in [0, 1].
            benford_reliable: False when the entity had too few transactions
                for a statistically meaningful Benford analysis (see
                BenfordAnalyzer.MIN_SAMPLE_SIZE). In that case Benford's
                weight is redistributed entirely onto threshold_clustering
                instead of silently averaging in a meaningless near-zero
                score — the composite falls back to the round-number /
                sub-threshold clustering signal alone for the forensic-
                accounting component.

        Returns:
            Dictionary with keys:
            composite_score (float 0–100), risk_tier ('LOW'|'MEDIUM'|'HIGH'),
            score_components (dict of weighted contributions),
            interpretation (str).
        """
        # Clip all inputs to [0, 1]
        b = max(0.0, min(1.0, float(benford_score)))
        c = max(0.0, min(1.0, float(clustering_score)))
        m = max(0.0, min(1.0, float(ml_score)))
        n = max(0.0, min(1.0, float(network_score)))

        if benford_reliable:
            weights = self.WEIGHTS
        else:
            weights = dict(self.WEIGHTS)
            weights["threshold_clustering"] += weights["benford"]
            weights["benford"] = 0.0
            b = 0.0

        raw_composite = (
            weights["benford"] * b
            + weights["threshold_clustering"] * c
            + weights["ml_anomaly"] * m
            + weights["network_centrality"] * n
        )

        composite_score = round(float(raw_composite * 100), 2)
        risk_tier = self._assign_tier(composite_score)
        interpretation = self._interpret(composite_score, risk_tier, b, c, m, n)

        score_components = {
            "benford": {
                "raw_score": round(b, 4),
                "weight": weights["benford"],
                "contribution": round(weights["benford"] * b * 100, 2),
            },
            "threshold_clustering": {
                "raw_score": round(c, 4),
                "weight": weights["threshold_clustering"],
                "contribution": round(weights["threshold_clustering"] * c * 100, 2),
            },
            "ml_anomaly": {
                "raw_score": round(m, 4),
                "weight": weights["ml_anomaly"],
                "contribution": round(weights["ml_anomaly"] * m * 100, 2),
            },
            "network_centrality": {
                "raw_score": round(n, 4),
                "weight": weights["network_centrality"],
                "contribution": round(weights["network_centrality"] * n * 100, 2),
            },
        }

        return {
            "composite_score": composite_score,
            "risk_tier": risk_tier,
            "score_components": score_components,
            "interpretation": interpretation,
        }

    # ------------------------------------------------------------------
    # Full customer classification
    # ------------------------------------------------------------------

    def classify_customer(
        self,
        customer_id: str,
        feature_results: dict,
        benford_results: Optional[dict] = None,
        clustering_results: Optional[dict] = None,
        ml_results: Optional[dict] = None,
        network_results: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Classify a customer by combining all available analysis signals.

        Gracefully handles missing signal sources by treating them as a
        neutral score of 0.0, preserving the weighted scheme.

        Args:
            customer_id: Account / customer identifier.
            feature_results: Output of FeatureEngineeringTool for this customer
                (used as supplementary context).
            benford_results: Output of BenfordAnalyzer.analyze_customer().
            clustering_results: Output of ThresholdClusteringAnalyzer.analyze_customer().
            ml_results: Output of MLAnomalyDetector.predict_single().
            network_results: Output of NetworkAnalysisTool.analyze_entity().

        Returns:
            Dictionary with keys:
            customer_id, composite_score, risk_tier, score_components,
            interpretation, signals_present (list), raw_signal_scores (dict).
        """
        signals_present: list[str] = []
        raw_scores: dict[str, float] = {}

        # --- Benford ---
        b_score = 0.0
        benford_reliable = True
        if benford_results:
            benford_reliable = not benford_results.get("insufficient_sample", False)
            if benford_reliable:
                b_score = float(benford_results.get("deviation_score", 0.0))
                signals_present.append("benford")
        raw_scores["benford"] = b_score

        # --- Threshold clustering ---
        c_score = 0.0
        if clustering_results:
            c_score = float(clustering_results.get("composite_clustering_score", 0.0))
            signals_present.append("threshold_clustering")
        raw_scores["threshold_clustering"] = c_score

        # --- ML anomaly ---
        m_score = 0.0
        if ml_results:
            m_score = float(ml_results.get("ml_anomaly_score", 0.0))
            signals_present.append("ml_anomaly")
        raw_scores["ml_anomaly"] = m_score

        # --- Network centrality ---
        n_score = 0.0
        if network_results:
            hub_score = float(network_results.get("hub_score", 0.0))
            # Normalise betweenness centrality (typically 0–0.5 on most graphs)
            n_score = min(hub_score * 2.0, 1.0)
            signals_present.append("network_centrality")
        raw_scores["network_centrality"] = n_score

        scoring_result = self.score(
            benford_score=b_score,
            clustering_score=c_score,
            ml_score=m_score,
            network_score=n_score,
            benford_reliable=benford_reliable,
        )

        # Penalty: if no signals present, composite must be zero
        if not signals_present:
            scoring_result["composite_score"] = 0.0
            scoring_result["risk_tier"] = "LOW"
            scoring_result["interpretation"] = (
                f"No analysis signals available for customer {customer_id}. "
                "Cannot compute risk score."
            )

        return {
            "customer_id": customer_id,
            "composite_score": scoring_result["composite_score"],
            "risk_tier": scoring_result["risk_tier"],
            "score_components": scoring_result["score_components"],
            "interpretation": scoring_result["interpretation"],
            "signals_present": signals_present,
            "raw_signal_scores": raw_scores,
            "feature_summary": self._summarise_features(feature_results),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assign_tier(self, composite_score: float) -> str:
        """Map a composite score (0–100) to a risk tier string.

        Args:
            composite_score: Float in [0, 100].

        Returns:
            'LOW', 'MEDIUM', or 'HIGH'.
        """
        if composite_score < self.THRESHOLDS["low"]:
            return "LOW"
        elif composite_score < self.THRESHOLDS["medium"]:
            return "MEDIUM"
        else:
            return "HIGH"

    def _interpret(
        self,
        composite_score: float,
        risk_tier: str,
        b: float,
        c: float,
        m: float,
        n: float,
    ) -> str:
        """Generate a plain-English interpretation of the composite risk score.

        Args:
            composite_score: Composite score 0–100.
            risk_tier: 'LOW', 'MEDIUM', or 'HIGH'.
            b: Benford score.
            c: Clustering score.
            m: ML score.
            n: Network score.

        Returns:
            Interpretation string.
        """
        tier_descriptors = {
            "LOW": "The customer presents LOW risk.",
            "MEDIUM": "The customer presents MEDIUM risk and warrants enhanced due diligence.",
            "HIGH": "The customer presents HIGH risk — recommend immediate SAR review.",
        }
        base = tier_descriptors[risk_tier]

        dominant = max(
            [("Benford", b), ("Clustering", c), ("ML anomaly", m), ("Network", n)],
            key=lambda x: x[1],
        )
        driver = f" Primary risk driver: {dominant[0]} signal (score {dominant[1]:.4f})."

        return (
            f"Composite score: {composite_score:.1f}/100. "
            f"{base}{driver} "
            f"Weights applied — Benford: {self.WEIGHTS['benford']}, "
            f"Clustering: {self.WEIGHTS['threshold_clustering']}, "
            f"ML: {self.WEIGHTS['ml_anomaly']}, "
            f"Network: {self.WEIGHTS['network_centrality']}."
        )

    def _summarise_features(self, feature_results: dict) -> dict:
        """Extract a compact feature summary for inclusion in the output dict.

        Args:
            feature_results: Full feature dictionary for the customer.

        Returns:
            Compact dictionary with the most relevant feature values.
        """
        if not feature_results:
            return {}
        keys = [
            "transaction_count",
            "avg_amount",
            "velocity",
            "rolling_sum_30d",
            "sub_threshold_count",
            "structuring_regularity_score",
            "rapid_cash_out_ratio",
        ]
        return {k: feature_results.get(k) for k in keys if k in feature_results}
