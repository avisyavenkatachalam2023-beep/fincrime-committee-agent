"""
explanation.py
--------------
Natural-language explanation generator for the AML Financial Crime Committee
Agent.

Generates investigation-grade narrative explanations that tie the original
analyst query to specific detected signals and quantitative evidence.  Uses
Groq (llama-3.3-70b-versatile) when available, with a deterministic template-based fallback
for offline / API-unavailable scenarios.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq client initialisation (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from groq import Groq
    from dotenv import load_dotenv

    load_dotenv()
    _api_key = os.environ.get("GROQ_API_KEY", "")
    if _api_key:
        _LLM_AVAILABLE = True
    else:
        _LLM_AVAILABLE = False
        logger.warning("GROQ_API_KEY not found in environment — using template fallback.")
except ImportError:
    _LLM_AVAILABLE = False
    logger.warning("groq not installed — using template fallback.")


class ExplanationTool:
    """Generate human-readable, evidence-anchored AML investigation narratives.

    The tool always cites specific numeric values (e.g. Benford MAD, clustering
    spike score) rather than vague qualitative phrases.  This satisfies
    regulatory requirements for auditable SAR narratives.

    Behaviour:
    - If the LLM is available, a structured prompt is sent to the model and the
      LLM-generated explanation is returned.
    - If the LLM is unavailable or fails, a deterministic template fills in all
      available numeric signals.
    """

    # LLM model name
    LLM_MODEL = "llama-3.3-70b-versatile"

    # Minimum MAD score above which Benford deviation is flagged in text
    BENFORD_FLAG_THRESHOLD = 0.015

    def __init__(self) -> None:
        """Initialise the ExplanationTool and LLM client if available."""
        self._llm_available = _LLM_AVAILABLE
        if self._llm_available:
            try:
                self._client = Groq(api_key=_api_key)
                logger.info("ExplanationTool: LLM model '%s' initialised.", self.LLM_MODEL)
            except Exception as exc:
                logger.warning("LLM model init failed (%s); falling back to template.", exc)
                self._llm_available = False
                self._client = None
        else:
            self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        customer_id: str,
        query: str,
        pattern: str,
        risk_classification: dict,
        benford_results: Optional[dict] = None,
        clustering_results: Optional[dict] = None,
        ml_results: Optional[dict] = None,
        network_results: Optional[dict] = None,
        features: Optional[dict] = None,
    ) -> str:
        """Generate a complete investigation narrative for one customer.

        The narrative is always anchored to:
        1. The original analyst query.
        2. The detected AML typology pattern.
        3. Specific numeric signal values.
        4. The composite risk tier.

        Args:
            customer_id: Account identifier.
            query: Original analyst question (used to frame the response).
            pattern: Detected typology / pattern label
                (e.g. 'STRUCTURING', 'LAYERING').
            risk_classification: Output of RiskClassifier.classify_customer().
            benford_results: Output of BenfordAnalyzer.analyze_customer().
            clustering_results: Output of ThresholdClusteringAnalyzer.analyze_customer().
            ml_results: Output of MLAnomalyDetector.predict_single().
            network_results: Output of NetworkAnalysisTool.analyze_entity().
            features: Customer feature dictionary from FeatureEngineeringTool.

        Returns:
            Multi-sentence investigation narrative string.
        """
        context = self._build_context(
            customer_id=customer_id,
            query=query,
            pattern=pattern,
            risk_classification=risk_classification,
            benford_results=benford_results,
            clustering_results=clustering_results,
            ml_results=ml_results,
            network_results=network_results,
            features=features,
        )

        if self._llm_available and self._client is not None:
            try:
                return self._explain_with_llm(context)
            except Exception as exc:
                logger.warning(
                    "LLM explanation failed for %s (%s); using template.", customer_id, exc
                )

        return self._explain_with_template(customer_id, context["signals"])

    def retrieve_and_explain_flag(
        self, entity_id: str, existing_flags: dict
    ) -> str:
        """Explain a previously stored flag for a customer.

        Intended for the ``explain_flag`` intent where the agent retrieves
        an existing flag record and provides a narrative explanation.

        Args:
            entity_id: Account or customer identifier.
            existing_flags: Dictionary of flags, keyed by entity ID.  Each
                value should be a dict with at least ``pattern``, ``score``,
                and ``signals`` keys.

        Returns:
            Explanation string.  If no flag exists for the entity, returns a
            specific not-found message.
        """
        if entity_id not in existing_flags:
            return (
                f"No existing flag record found for entity '{entity_id}'. "
                "Run a full analysis first to generate signals."
            )

        flag = existing_flags[entity_id]
        pattern = flag.get("pattern", "UNKNOWN")
        score = flag.get("score", 0.0)
        signals = flag.get("signals", {})
        risk_tier = flag.get("risk_tier", "UNKNOWN")

        signal_lines = self._format_signals(signals)

        return (
            f"Existing flag for entity '{entity_id}': "
            f"Pattern detected — {pattern}. "
            f"Composite risk score: {score:.1f}/100 (Tier: {risk_tier}). "
            f"Contributing signals: {signal_lines}. "
            "Review the detailed signal outputs for full numeric evidence."
        )

    # ------------------------------------------------------------------
    # LLM-based explanation
    # ------------------------------------------------------------------

    def _explain_with_llm(self, context: dict) -> str:
        """Generate explanation using the Groq LLM.

        Constructs a detailed prompt containing all signal data and instructs
        the model to produce an investigation-grade narrative.

        Args:
            context: Pre-built context dictionary from _build_context().

        Returns:
            LLM-generated explanation string.
        """
        signals_json = json.dumps(context["signals"], indent=2)

        prompt = f"""You are a senior AML (Anti-Money Laundering) analyst writing an investigation narrative.

TASK: Write a concise, evidence-based investigation narrative for the following customer case.

ORIGINAL ANALYST QUERY: {context['query']}

CUSTOMER ID: {context['customer_id']}
DETECTED PATTERN: {context['pattern']}
COMPOSITE RISK SCORE: {context['risk_classification'].get('composite_score', 0.0):.1f}/100
RISK TIER: {context['risk_classification'].get('risk_tier', 'UNKNOWN')}

QUANTITATIVE SIGNALS:
{signals_json}

INSTRUCTIONS:
1. ALWAYS cite specific numeric values (e.g. "Benford MAD score of 0.023, exceeding the 0.015 threshold").
2. NEVER use vague phrases like "looks suspicious" or "seems unusual".
3. Tie each finding to the original query and the detected pattern.
4. Structure as: (a) Pattern summary, (b) Key evidence, (c) Risk assessment, (d) Recommended action.
5. Maximum 250 words.
6. Write in third person (e.g. "Account ACC-001 exhibits...").

Write the investigation narrative now:"""

        response = self._client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=self.LLM_MODEL,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Template-based explanation
    # ------------------------------------------------------------------

    def _explain_with_template(
        self, customer_id: str, signals: dict
    ) -> str:
        """Generate a deterministic template-based explanation from numeric signals.

        Template:
        'Customer {id} exhibits {N} red flags: [signal list with numbers]...'

        Args:
            customer_id: Account identifier.
            signals: Dictionary of signal values from _build_context().

        Returns:
            Formatted narrative string.
        """
        red_flags: list[str] = []

        # Benford
        benford_mad = signals.get("benford_mad_score")
        benford_dev = signals.get("benford_deviation_score")
        if benford_mad is not None and benford_mad > self.BENFORD_FLAG_THRESHOLD:
            red_flags.append(
                f"Benford MAD score of {benford_mad:.4f} (threshold: {self.BENFORD_FLAG_THRESHOLD}) "
                f"with deviation score {benford_dev:.4f} — digit distribution is non-conforming"
            )

        # Clustering
        spike = signals.get("clustering_spike_score")
        round_ratio = signals.get("round_number_ratio")
        if spike is not None and spike > 0.3:
            red_flags.append(
                f"Sub-threshold clustering spike score of {spike:.4f} — transactions cluster "
                "abnormally close to regulatory reporting thresholds"
            )
        if round_ratio is not None and round_ratio > 0.25:
            red_flags.append(
                f"Round-number transaction ratio of {round_ratio:.1%} — far above baseline (~5%)"
            )

        # ML
        ml_score = signals.get("ml_anomaly_score")
        if ml_score is not None and ml_score > 0.6:
            red_flags.append(
                f"ML Isolation Forest anomaly score of {ml_score:.4f} — customer is a "
                "statistical outlier in the feature space"
            )

        # Network
        hub_score = signals.get("network_hub_score")
        flagged_connections = signals.get("connected_flagged_accounts", 0)
        if hub_score is not None and hub_score > 0.05:
            red_flags.append(
                f"Network betweenness centrality of {hub_score:.4f} — elevated hub role "
                "in the transaction network"
            )
        if isinstance(flagged_connections, list) and len(flagged_connections) > 0:
            red_flags.append(
                f"Directly connected to {len(flagged_connections)} flagged account(s)"
            )

        # Features
        sub_thresh = signals.get("sub_threshold_count")
        if sub_thresh is not None and sub_thresh > 2:
            red_flags.append(
                f"{sub_thresh} transactions found in the 85–99.9% structuring band "
                "within the rolling 30-day window"
            )

        regularity = signals.get("structuring_regularity_score")
        if regularity is not None and regularity > 0.7:
            red_flags.append(
                f"Structuring regularity score of {regularity:.4f} — near-identical "
                "amounts and intervals consistent with mechanised splitting"
            )

        # Risk tier
        risk_tier = signals.get("risk_tier", "UNKNOWN")
        composite = signals.get("composite_score", 0.0)

        if not red_flags:
            return (
                f"Customer {customer_id} exhibits no significant AML red flags at this time. "
                f"Composite risk score: {composite:.1f}/100 (Tier: {risk_tier}). "
                "Continued monitoring is recommended."
            )

        flags_text = "; ".join(f"({i+1}) {f}" for i, f in enumerate(red_flags))
        pattern = signals.get("pattern", "UNKNOWN")

        return (
            f"Customer {customer_id} exhibits {len(red_flags)} AML red flag(s) "
            f"consistent with a {pattern} typology: {flags_text}. "
            f"Composite risk score: {composite:.1f}/100 (Tier: {risk_tier}). "
            f"Recommended action: "
            + (
                "File a Suspicious Activity Report (SAR) immediately."
                if risk_tier == "HIGH"
                else "Conduct enhanced due diligence and monitor closely."
                if risk_tier == "MEDIUM"
                else "Continue standard monitoring."
            )
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_context(
        self,
        customer_id: str,
        query: str,
        pattern: str,
        risk_classification: dict,
        benford_results: Optional[dict],
        clustering_results: Optional[dict],
        ml_results: Optional[dict],
        network_results: Optional[dict],
        features: Optional[dict],
    ) -> dict:
        """Assemble all signals into a flat context dictionary.

        Args:
            customer_id: Account identifier.
            query: Original analyst query.
            pattern: Detected typology.
            risk_classification: Risk classification output dict.
            benford_results: Benford analysis results.
            clustering_results: Clustering analysis results.
            ml_results: ML anomaly detection results.
            network_results: Network analysis results.
            features: Feature engineering output for the customer.

        Returns:
            Context dictionary for use in Groq prompt or template.
        """
        signals: dict[str, Any] = {
            "pattern": pattern,
            "risk_tier": risk_classification.get("risk_tier"),
            "composite_score": risk_classification.get("composite_score", 0.0),
        }

        if benford_results:
            signals["benford_mad_score"] = benford_results.get("mad_score")
            signals["benford_deviation_score"] = benford_results.get("deviation_score")
            signals["benford_p_value"] = benford_results.get("p_value")
            signals["benford_sample_size"] = benford_results.get("sample_size")

        if clustering_results:
            sub = clustering_results.get("sub_threshold", {})
            rnd = clustering_results.get("round_numbers", {})
            signals["clustering_spike_score"] = sub.get("spike_score")
            signals["clustering_composite_score"] = clustering_results.get("composite_clustering_score")
            signals["round_number_ratio"] = rnd.get("round_number_ratio")
            signals["common_round_amounts"] = rnd.get("common_round_amounts", [])

        if ml_results:
            signals["ml_anomaly_score"] = ml_results.get("ml_anomaly_score")
            signals["ml_is_outlier"] = ml_results.get("ml_is_outlier")

        if network_results:
            signals["network_hub_score"] = network_results.get("hub_score")
            signals["network_is_hub"] = network_results.get("is_hub")
            signals["connected_flagged_accounts"] = network_results.get(
                "connected_flagged_accounts", []
            )
            signals["community_id"] = network_results.get("community_id")

        if features:
            for key in [
                "transaction_count", "avg_amount", "velocity", "rolling_sum_30d",
                "sub_threshold_count", "structuring_regularity_score", "rapid_cash_out_ratio",
            ]:
                if key in features:
                    signals[key] = features[key]

        return {
            "customer_id": customer_id,
            "query": query,
            "pattern": pattern,
            "risk_classification": risk_classification,
            "signals": signals,
        }

    def _format_signals(self, signals: dict) -> str:
        """Format a signals dictionary into a compact readable string.

        Args:
            signals: Dictionary of signal key/value pairs.

        Returns:
            Comma-separated string of 'key=value' pairs.
        """
        parts = []
        for k, v in signals.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            elif v is not None:
                parts.append(f"{k}={v}")
        return ", ".join(parts) if parts else "no signals available"
