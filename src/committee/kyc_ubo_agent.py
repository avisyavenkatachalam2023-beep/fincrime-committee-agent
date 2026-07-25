"""
kyc_ubo_agent.py
----------------
Specialist agent that assesses profile-behaviour consistency:
occupation vs. transaction volumes, account age, PEP status,
and declared income vs. observed cash flows.
"""

import logging
from datetime import date, datetime

try:
    from src.committee.base_agent import BaseCommitteeAgent
except ImportError:
    from .base_agent import BaseCommitteeAgent

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a KYC (Know Your Customer) / UBO (Ultimate Beneficial Owner) Analyst Agent.
Your specialty: assessing whether a customer's transaction behavior is consistent with
their declared profile (occupation, income, account history).

Red flags you look for:
- Transaction volumes far exceeding declared income band
- Occupation inconsistent with large wire transfers (e.g., 'student' with $500k/year flow)
- Recently opened accounts with immediate high-volume activity
- PEP (Politically Exposed Person) status combined with suspicious patterns
- Account opened shortly before suspicious activity window

Respond ONLY with valid JSON:
{
  "vote": "MONITOR|REVIEW|REPORT",
  "reasoning": "2-4 sentence explanation citing specific profile mismatches",
  "confidence": 0.0-1.0,
  "key_signals": ["list of KYC red flags identified"]
}"""

# ---------------------------------------------------------------------------
# Income band → approximate annual USD equivalent (upper bound)
# ---------------------------------------------------------------------------
_INCOME_BAND_UPPER_USD: dict[str, float] = {
    "under_25k": 25_000,
    "25k_50k": 50_000,
    "50k_100k": 100_000,
    "100k_250k": 250_000,
    "250k_500k": 500_000,
    "500k_1m": 1_000_000,
    "over_1m": float("inf"),
    # Aliases / alternate spellings
    "low": 30_000,
    "medium": 75_000,
    "high": 300_000,
    "very_high": float("inf"),
    "unknown": 50_000,  # conservative default
}

# Occupations considered high-risk when paired with large flows
_HIGH_RISK_OCCUPATIONS = {
    "student",
    "unemployed",
    "retired",
    "homemaker",
    "part_time",
    "part-time",
    "volunteer",
    "intern",
}

# Minimum months an account should have been open before large activity
_MIN_ACCOUNT_AGE_MONTHS = 6

# Rolling 30d sum that, if exceeded above the income cap, raises a flag
_INCOME_MULTIPLIER_THRESHOLD = 2.0   # more than 2× annual income / 12 months


def _parse_account_open_date(date_str: str) -> date | None:
    """
    Parse an ISO-8601 or common date string into a :class:`datetime.date`.

    Parameters
    ----------
    date_str : str
        Date string such as ``'2023-04-15'`` or ``'2023/04/15'``.

    Returns
    -------
    datetime.date or None
        Parsed date, or ``None`` if parsing fails.
    """
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


class KycUboAgent(BaseCommitteeAgent):
    """
    Evaluates Know Your Customer / UBO profile-behaviour consistency.

    Signals assessed
    ----------------
    * Declared income band vs observed 30-day rolling sum
    * Occupation risk category vs transaction volume
    * Account age (recently opened accounts with immediate high activity)
    * PEP status combined with suspicious patterns
    * High-risk jurisdiction flag
    """

    def __init__(self) -> None:
        """Initialise the KYC/UBO Agent with its specialist system prompt."""
        super().__init__(
            agent_name="KYC/UBO Analyst",
            system_prompt=_SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------
    # Specialist rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based_fallback(self, case_file: dict) -> dict:
        """
        Evaluate KYC/UBO risk using profile-behaviour consistency checks.

        Decision logic
        --------------
        Each triggered red flag contributes to a severity score.  The final
        vote is determined by how many flags are triggered and their severity:

        * ≥ 3 flags  → REPORT
        * 1–2 flags  → REVIEW
        * 0 flags    → MONITOR

        PEP status or high-risk jurisdiction combined with a suspicious pattern
        automatically escalates to at least REVIEW.

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
        profile = case_file.get("customer_profile", {})
        pattern = case_file.get("pattern", "none").lower()

        rolling_sum_30d = float(features.get("rolling_sum_30d", 0.0))
        avg_amount = float(features.get("avg_amount", 0.0))
        transaction_count = int(features.get("transaction_count", 0))

        occupation = str(profile.get("occupation", "unknown")).lower().strip()
        income_band = str(profile.get("declared_income_band", "unknown")).lower().strip()
        account_open_date_str = str(profile.get("account_open_date", ""))
        is_pep: bool = bool(profile.get("is_pep", False))
        is_high_risk_jurisdiction: bool = bool(profile.get("is_high_risk_jurisdiction", False))

        triggered_flags: list[str] = []

        # --- 1. Income-volume mismatch ---
        annual_income_cap = _INCOME_BAND_UPPER_USD.get(income_band, 50_000)
        monthly_income_cap = annual_income_cap / 12.0
        if annual_income_cap < float("inf") and rolling_sum_30d > (
            monthly_income_cap * _INCOME_MULTIPLIER_THRESHOLD
        ):
            ratio = rolling_sum_30d / max(monthly_income_cap, 1.0)
            triggered_flags.append(
                f"income_volume_mismatch (30d sum USD {rolling_sum_30d:,.0f} "
                f"= {ratio:.1f}× monthly income cap)"
            )

        # --- 2. High-risk occupation ---
        if occupation in _HIGH_RISK_OCCUPATIONS and avg_amount > 5_000:
            triggered_flags.append(
                f"high_risk_occupation_with_large_amounts "
                f"(occupation='{occupation}', avg_amount=USD {avg_amount:,.0f})"
            )

        # --- 3. Recently opened account ---
        open_date = _parse_account_open_date(account_open_date_str)
        account_age_months = 0
        if open_date is not None:
            today = date.today()
            account_age_months = (today.year - open_date.year) * 12 + (
                today.month - open_date.month
            )
            if account_age_months < _MIN_ACCOUNT_AGE_MONTHS and transaction_count > 10:
                triggered_flags.append(
                    f"new_account_high_activity "
                    f"(age={account_age_months} months, txn_count={transaction_count})"
                )

        # --- 4. PEP status with suspicious pattern ---
        if is_pep:
            if pattern in ("structuring", "smurfing", "layering"):
                triggered_flags.append(
                    f"pep_with_suspicious_pattern (pattern='{pattern}')"
                )
            else:
                triggered_flags.append("pep_status_requires_edd")

        # --- 5. High-risk jurisdiction ---
        if is_high_risk_jurisdiction:
            triggered_flags.append("high_risk_jurisdiction_residence")

        # ------------------------------------------------------------------
        # Vote determination
        # ------------------------------------------------------------------
        flag_count = len(triggered_flags)

        # PEP + pattern or income mismatch alone can force minimum REVIEW
        pep_with_pattern = is_pep and pattern in ("structuring", "smurfing", "layering")

        if flag_count >= 3 or (flag_count >= 2 and pep_with_pattern):
            vote = "REPORT"
            reasoning = (
                f"{flag_count} KYC/UBO red flags identified: "
                f"{'; '.join(triggered_flags[:3])}. "
                f"The customer's occupation ('{occupation}') and declared income band "
                f"('{income_band}') are inconsistent with a 30-day flow of "
                f"USD {rolling_sum_30d:,.0f}. "
                "Immediate SAR filing and enhanced due diligence are recommended."
            )
            confidence = min(0.95, 0.68 + flag_count * 0.06)
        elif flag_count >= 1 or pep_with_pattern:
            vote = "REVIEW"
            reasoning = (
                f"{flag_count} KYC/UBO concern(s) identified: "
                f"{'; '.join(triggered_flags)}. "
                f"Profile (occupation='{occupation}', income='{income_band}') "
                f"{'includes PEP status requiring enhanced due diligence. ' if is_pep else ''}"
                "Manual compliance review is recommended before further action."
            )
            confidence = 0.52 + flag_count * 0.08
        else:
            vote = "MONITOR"
            reasoning = (
                f"Customer profile appears consistent with declared parameters. "
                f"Occupation ('{occupation}') and income band ('{income_band}') "
                f"are broadly aligned with observed 30-day sum of USD {rolling_sum_30d:,.0f}. "
                "No KYC/UBO escalation required at this time."
            )
            confidence = 0.72

        return {
            "vote": vote,
            "reasoning": reasoning,
            "confidence": round(min(1.0, confidence), 3),
            "key_signals": triggered_flags if triggered_flags else ["no_kyc_flags_triggered"],
        }
