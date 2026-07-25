"""
sanctions_pep_agent.py
----------------------
Specialist agent for jurisdictional risk assessment and PEP screening.

IMPORTANT DISCLAIMER
--------------------
All country classifications used in this module are SYNTHETIC and for
illustrative/educational purposes only.  They do NOT reflect any real
sanctions list, OFAC SDN list, FATF grey/black list, or any other
regulatory instrument.  Do NOT use in production without integrating
a certified, up-to-date sanctions data provider.
"""

import logging

try:
    from src.committee.base_agent import BaseCommitteeAgent
except ImportError:
    from .base_agent import BaseCommitteeAgent

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a Sanctions and PEP (Politically Exposed Person) Screening Analyst Agent.
IMPORTANT NOTE: The jurisdiction list used is SYNTHETIC and illustrative only — not a real sanctions list.

Your specialty: assessing jurisdictional risk based on sender/receiver countries and counterparty profiles.

High-risk indicators:
- Transactions to/from high-risk jurisdictions (see case file)
- Customer is a PEP (enhanced due diligence required)
- Transaction chains that pass through multiple jurisdictions
- Counterparties in different risk-tier countries than the customer

Respond ONLY with valid JSON:
{
  "vote": "MONITOR|REVIEW|REPORT",
  "reasoning": "2-4 sentence explanation citing specific jurisdictional and PEP concerns",
  "confidence": 0.0-1.0,
  "key_signals": ["list of jurisdiction/PEP risk factors"]
}"""

# ---------------------------------------------------------------------------
# SYNTHETIC jurisdiction risk tiers — NOT a real sanctions list
# ---------------------------------------------------------------------------
_SYNTHETIC_HIGH_RISK_COUNTRIES: frozenset[str] = frozenset(
    {
        # These are purely illustrative synthetic entries
        "country_alpha", "country_beta", "country_gamma",
        "country_delta", "country_epsilon", "country_zeta",
    }
)

_SYNTHETIC_MEDIUM_RISK_COUNTRIES: frozenset[str] = frozenset(
    {
        "country_eta", "country_theta", "country_iota",
        "country_kappa", "country_lambda",
    }
)

# Patterns that, combined with PEP or high-risk jurisdiction, force REPORT
_CRITICAL_PATTERNS = {"structuring", "smurfing", "layering"}

# Thresholds for PEP + volume escalation
_PEP_HIGH_VOLUME_THRESHOLD_USD = 50_000   # 30-day rolling sum
_PEP_AVG_AMOUNT_THRESHOLD_USD = 10_000


class SanctionsPepAgent(BaseCommitteeAgent):
    """
    Assesses jurisdictional and PEP risk to produce a MONITOR / REVIEW / REPORT vote.

    Signals assessed
    ----------------
    * Customer resides in a synthetic high-risk or medium-risk jurisdiction
    * PEP flag (is_pep) combined with transaction volume or pattern
    * Pattern type (structuring / layering / smurfing) multiplied by jurisdiction risk
    * ML anomaly score as a corroborating indicator
    """

    def __init__(self) -> None:
        """Initialise the Sanctions / PEP Agent with its specialist system prompt."""
        super().__init__(
            agent_name="Sanctions/PEP Analyst",
            system_prompt=_SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------
    # Specialist rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based_fallback(self, case_file: dict) -> dict:
        """
        Evaluate sanctions and PEP risk using heuristic rules over the case file.

        Decision logic
        --------------
        Each triggered risk factor is scored.  Final vote:

        * REPORT if:
          - Customer is in a synthetic high-risk country AND is a PEP
          - OR pattern is critical AND (PEP OR high-risk jurisdiction)
          - OR ≥ 3 risk factors triggered
        * REVIEW if:
          - 1–2 risk factors triggered
        * MONITOR if:
          - No risk factors

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
        ml = case_file.get("ml_results", {})
        profile = case_file.get("customer_profile", {})
        pattern = case_file.get("pattern", "none").lower()

        is_pep: bool = bool(profile.get("is_pep", False))
        is_high_risk_jurisdiction: bool = bool(profile.get("is_high_risk_jurisdiction", False))
        country = str(profile.get("country_of_residence", "unknown")).lower().strip()

        rolling_sum_30d = float(features.get("rolling_sum_30d", 0.0))
        avg_amount = float(features.get("avg_amount", 0.0))
        anomaly_score = float(ml.get("anomaly_score", 0.0))
        is_outlier: bool = bool(ml.get("is_outlier", False))

        risk_factors: list[str] = []

        # --- 1. Jurisdiction tier classification ---
        is_synthetic_high_risk = country in _SYNTHETIC_HIGH_RISK_COUNTRIES
        is_synthetic_medium_risk = country in _SYNTHETIC_MEDIUM_RISK_COUNTRIES

        if is_high_risk_jurisdiction or is_synthetic_high_risk:
            risk_factors.append(
                f"high_risk_jurisdiction (country='{country}', "
                f"flagged={'case_file' if is_high_risk_jurisdiction else 'synthetic_list'})"
            )
        elif is_synthetic_medium_risk:
            risk_factors.append(f"medium_risk_jurisdiction (country='{country}')")

        # --- 2. PEP screening ---
        if is_pep:
            risk_factors.append("pep_status_confirmed")
            if rolling_sum_30d > _PEP_HIGH_VOLUME_THRESHOLD_USD:
                risk_factors.append(
                    f"pep_high_volume (30d_sum=USD {rolling_sum_30d:,.0f} "
                    f"> threshold USD {_PEP_HIGH_VOLUME_THRESHOLD_USD:,})"
                )
            if avg_amount > _PEP_AVG_AMOUNT_THRESHOLD_USD:
                risk_factors.append(
                    f"pep_large_avg_transaction (avg=USD {avg_amount:,.0f})"
                )

        # --- 3. Pattern-jurisdiction interaction ---
        if pattern in _CRITICAL_PATTERNS and (is_pep or is_high_risk_jurisdiction):
            risk_factors.append(
                f"critical_pattern_in_risky_context (pattern='{pattern}')"
            )

        # --- 4. ML anomaly corroboration ---
        if is_outlier and anomaly_score > 0.7:
            risk_factors.append(
                f"ml_anomaly_corroboration (anomaly_score={anomaly_score:.3f})"
            )

        # ------------------------------------------------------------------
        # Vote determination
        # ------------------------------------------------------------------
        force_report = (
            (is_pep and (is_high_risk_jurisdiction or is_synthetic_high_risk))
            or (pattern in _CRITICAL_PATTERNS and (is_pep or is_high_risk_jurisdiction))
        )

        factor_count = len(risk_factors)

        if force_report or factor_count >= 3:
            vote = "REPORT"
            reasoning = (
                f"{factor_count} sanctions/PEP risk factor(s) identified: "
                f"{'; '.join(risk_factors[:3])}. "
                f"{'PEP status combined with high-risk jurisdiction creates an immediate escalation obligation. ' if is_pep and is_high_risk_jurisdiction else ''}"
                f"Pattern '{pattern}' and 30-day flow of USD {rolling_sum_30d:,.0f} "
                "require SAR filing and enhanced due-diligence freeze."
            )
            confidence = min(0.96, 0.70 + factor_count * 0.07)
        elif factor_count >= 1:
            vote = "REVIEW"
            reasoning = (
                f"{factor_count} sanctions/PEP concern(s) flagged: "
                f"{'; '.join(risk_factors)}. "
                f"{'PEP status requires enhanced due diligence. ' if is_pep else ''}"
                f"Country of residence ('{profile.get('country_of_residence', 'N/A')}') "
                "warrants further investigation by the compliance team."
            )
            confidence = 0.50 + factor_count * 0.10
        else:
            vote = "MONITOR"
            reasoning = (
                f"No sanctions or PEP risk factors triggered. "
                f"Customer country ('{profile.get('country_of_residence', 'N/A')}') "
                "is not flagged on the synthetic jurisdiction list and PEP status is negative. "
                "Standard monitoring is sufficient."
            )
            confidence = 0.74

        return {
            "vote": vote,
            "reasoning": reasoning,
            "confidence": round(min(1.0, confidence), 3),
            "key_signals": risk_factors if risk_factors else ["no_sanctions_pep_flags"],
        }
