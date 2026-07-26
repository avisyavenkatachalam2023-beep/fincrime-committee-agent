"""
base_agent.py
-------------
Abstract base class for all AML Financial Crime Committee specialist agents.
Handles Groq API initialization, prompt formatting, response parsing,
and rule-based fallback logic.
"""

import os
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vote ordering for comparison utilities used by subclasses
# ---------------------------------------------------------------------------
VOTE_LEVELS = {"MONITOR": 0, "REVIEW": 1, "REPORT": 2}
VOTE_THRESHOLDS = {
    "MONITOR": (0.0, 33.0),
    "REVIEW": (33.0, 60.0),
    "REPORT": (60.0, 100.0),
}

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

# ---------------------------------------------------------------------------
# Numeric-claim verification
# ---------------------------------------------------------------------------
# An LLM can narrate plausible-sounding statistics it was never actually
# given (e.g. citing a Benford p-value when Benford wasn't computed for lack
# of data). Every number an agent's generated text cites is checked against
# the real signal values it was handed; anything unmatched triggers a
# corrective retry, then a hard rejection (caller falls back to the
# deterministic rule-based path) rather than trusting an invented figure.

# Numbers that legitimately appear in ordinary reasoning prose without being
# drawn from the case file's computed signals: small integers (day-count
# windows like "30-day", "5 business days", list positions), the risk-tier
# cut points from the scoring scheme (see risk_classification.py), and
# common confidence-style fractions.
_ALLOWED_STRUCTURAL_NUMBERS = {float(n) for n in range(0, 32)} | {
    33.0, 60.0, 100.0, 0.1, 0.15, 0.25, 0.5, 0.6, 0.7, 0.75, 0.9, 0.95,
}


def _extract_numeric_claims(text: str) -> list[float]:
    """Extract every numeric literal a generated narrative cites."""
    out: list[float] = []
    for tok in re.findall(r"-?\d[\d,]*\.?\d*", text or ""):
        tok = tok.strip(".").replace(",", "")
        if not tok or tok == "-":
            continue
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def _flatten_numeric_values(obj) -> set[float]:
    """Collect every real numeric leaf value out of a nested dict/list
    (e.g. the full case file), for cross-checking narrative claims."""
    values: set[float] = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple, set)):
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            return
        elif isinstance(o, (int, float)):
            values.add(round(float(o), 6))

    walk(obj)
    return values


def find_unverifiable_number(
    narrative_text: str, case_file: dict, extra_real_numbers: Optional[set] = None
) -> Optional[float]:
    """Return the first number cited in *narrative_text* that does not
    approximately match any real value in *case_file*, or None if every
    cited number checks out.

    Used by both specialist agents and the Chair to catch a generated
    narrative citing a statistic that was never actually computed.

    Args:
        narrative_text: The generated text to check.
        case_file: Structured case file whose numeric leaf values are
            treated as ground truth.
        extra_real_numbers: Additional legitimate numbers not present as
            typed values in case_file — e.g. numbers embedded in specialist
            agents' free-text reasoning, which the Chair is entitled to
            quote even though they only exist as text, not JSON numbers.
    """
    real_values = _flatten_numeric_values(case_file) | (extra_real_numbers or set())
    for num in _extract_numeric_claims(narrative_text):
        if num in _ALLOWED_STRUCTURAL_NUMBERS:
            continue
        if any(abs(num - r) <= max(0.015, abs(r) * 0.02) for r in real_values):
            continue
        # Tolerate the narrative rounding a real value differently than the
        # source data does (e.g. a true 0.8854 cited as "0.89").
        if any(abs(round(num, 1) - round(r, 1)) < 1e-6 for r in real_values):
            continue
        return num
    return None


class BaseCommitteeAgent(ABC):
    """
    Abstract base class for AML Financial Crime Committee specialist agents.

    Subclasses must supply a unique ``agent_name`` and ``system_prompt`` that
    define the specialist's role and the signals they focus on.  The
    ``deliberate`` method drives the full workflow:
        1. Attempt to call the Groq API.
        2. Parse and validate the JSON response.
        3. Fall back to the rule-based heuristic if the API call fails or the
           response cannot be parsed.
    """

    def __init__(self, agent_name: str, system_prompt: str) -> None:
        """
        Initialise the agent and configure the Groq client.

        Parameters
        ----------
        agent_name : str
            Human-readable name shown in committee minutes.
        system_prompt : str
            The specialist system prompt injected before the case brief.
        """
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self._client = None
        self._model_name = None
        self._llm_available: bool = False
        self._init_llm()

    # ------------------------------------------------------------------
    # LLM initialisation
    # ------------------------------------------------------------------

    def _init_llm(self) -> None:
        """
        Load the Groq API key from the environment and configure the
        ``groq`` client.  Sets ``self._llm_available``
        to ``False`` if the key is missing or the model cannot be reached.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning(
                "[%s] GROQ_API_KEY not found — rule-based fallback will be used.",
                self.agent_name,
            )
            self._llm_available = False
            return

        try:
            self._client = Groq(api_key=api_key)
            self._model_name = PRIMARY_MODEL
            self._llm_available = True
            logger.info("[%s] Groq client ready using %s.", self.agent_name, self._model_name)
        except Exception as exc:
            logger.warning(
                "[%s] Could not initialise Groq client: %s",
                self.agent_name,
                exc,
            )
            self._client = None
            self._llm_available = False

        if not self._llm_available:
            logger.warning(
                "[%s] LLM unavailable — rule-based fallback active.",
                self.agent_name,
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def deliberate(self, case_file: dict) -> dict:
        """
        Main entry point.  Produce a vote dict for the supplied *case_file*.

        Attempts the Groq API first; falls back to ``_rule_based_fallback``
        on any failure.

        Parameters
        ----------
        case_file : dict
            Structured case file matching the committee schema.

        Returns
        -------
        dict
            ``{agent, vote, reasoning, confidence, key_signals}``
        """
        result: Optional[dict] = None

        if self._llm_available and self._client is not None:
            try:
                result = self._call_llm(case_file)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[%s] LLM call failed (%s); using fallback.", self.agent_name, exc
                )

        if result is None:
            result = self._rule_based_fallback(case_file)

        # Ensure the agent name is always stamped on the result
        result["agent"] = self.agent_name
        return result

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _call_llm(self, case_file: dict) -> dict:
        """
        Format *case_file* as a readable brief, call the Groq API,
        and parse the JSON response into the standard vote dict.

        Parameters
        ----------
        case_file : dict
            Structured case file.

        Returns
        -------
        dict
            Validated vote dict ``{vote, reasoning, confidence, key_signals}``.

        Raises
        ------
        ValueError
            If the response cannot be parsed or required keys are missing.
        RuntimeError
            If the Groq API call itself fails.
        """
        brief = self._format_case_brief(case_file)
        prompt = (
            "Analyse the following AML case file and respond with a JSON object "
            "containing your vote, reasoning, confidence, and key_signals.\n\n"
            "Only cite numeric values explicitly present in the case file below. "
            "Never invent, estimate, or approximate a statistic. If a section is "
            "marked N/A or NOT COMPUTED, say so explicitly rather than citing a "
            "number for it.\n\n"
            f"{brief}"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        last_bad_number: Optional[float] = None
        for attempt in range(2):
            try:
                response = self._client.chat.completions.create(
                    messages=messages,
                    model=self._model_name,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=1024,
                )
                raw_text = response.choices[0].message.content.strip()
            except Exception as exc:
                raise RuntimeError(f"Groq API error: {exc}") from exc

            # Strip markdown code fences if present
            clean_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            clean_text = re.sub(r"\s*```$", "", clean_text)

            try:
                parsed = json.loads(clean_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Non-JSON response from Groq: {clean_text[:300]}") from exc

            result = self._validate_and_normalise(parsed)

            claim_text = result["reasoning"] + " " + " ".join(result["key_signals"])
            bad_number = find_unverifiable_number(claim_text, case_file)
            if bad_number is None:
                return result

            last_bad_number = bad_number
            logger.warning(
                "[%s] LLM cited unverifiable number %s not present in the case "
                "file (attempt %d/2).", self.agent_name, bad_number, attempt + 1,
            )
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": (
                f"Your response cited the number {bad_number}, which does not "
                "appear anywhere in the case file above. Respond again with "
                "corrected JSON, citing only numbers explicitly present in the "
                "case file (or state a value is not computed/N/A instead of "
                "citing a number for it)."
            )})

        raise ValueError(
            f"LLM response still cited unverifiable number {last_bad_number} "
            "after a corrective retry; rejecting in favour of the rule-based fallback."
        )

    def _validate_and_normalise(self, parsed: dict) -> dict:
        """
        Validate that the LLM JSON response contains the required keys and
        that values are within acceptable ranges.  Normalises casing on the
        vote field.

        Parameters
        ----------
        parsed : dict
            Raw parsed JSON from Groq.

        Returns
        -------
        dict
            Cleaned and validated vote dict.

        Raises
        ------
        ValueError
            If required keys are absent or the vote value is invalid.
        """
        required_keys = {"vote", "reasoning", "confidence", "key_signals"}
        missing = required_keys - parsed.keys()
        if missing:
            raise ValueError(f"LLM response missing keys: {missing}")

        vote = str(parsed["vote"]).upper().strip()
        if vote not in VOTE_LEVELS:
            raise ValueError(f"Invalid vote value from LLM: '{vote}'")

        confidence = float(parsed["confidence"])
        confidence = max(0.0, min(1.0, confidence))

        key_signals = parsed["key_signals"]
        if not isinstance(key_signals, list):
            key_signals = [str(key_signals)]

        return {
            "vote": vote,
            "reasoning": str(parsed["reasoning"]).strip(),
            "confidence": confidence,
            "key_signals": [str(s) for s in key_signals],
        }

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based_fallback(self, case_file: dict) -> dict:
        """
        Produce a vote based solely on the ``risk_score`` field using fixed
        thresholds.  Used when the LLM API is unavailable.

        Thresholds
        ----------
        risk_score < 33  → MONITOR
        33 ≤ risk_score < 60 → REVIEW
        risk_score ≥ 60  → REPORT

        Parameters
        ----------
        case_file : dict
            Structured case file.

        Returns
        -------
        dict
            ``{vote, reasoning, confidence, key_signals}``
        """
        risk_score: float = float(case_file.get("risk_score", 0.0))
        pattern: str = case_file.get("pattern", "none")
        risk_tier: str = case_file.get("risk_tier", "LOW").upper()

        if risk_score < 33.0:
            vote = "MONITOR"
            reasoning = (
                f"Risk score of {risk_score:.1f} falls below the REVIEW threshold. "
                f"Current pattern is '{pattern}' with a {risk_tier} risk tier. "
                "No immediate escalation warranted; continue routine monitoring."
            )
            confidence = max(0.4, 1.0 - (risk_score / 33.0) * 0.4)
            key_signals = ["risk_score_low"]
        elif risk_score < 60.0:
            vote = "REVIEW"
            reasoning = (
                f"Risk score of {risk_score:.1f} falls in the REVIEW band (33–60). "
                f"Pattern '{pattern}' detected with a {risk_tier} tier designation. "
                "Compliance team should conduct a manual review of this account."
            )
            confidence = 0.55 + ((risk_score - 33.0) / 27.0) * 0.2
            key_signals = ["risk_score_moderate", f"pattern_{pattern}"]
        else:
            vote = "REPORT"
            reasoning = (
                f"Risk score of {risk_score:.1f} exceeds the REPORT threshold of 60. "
                f"Pattern '{pattern}' identified with a {risk_tier} risk tier. "
                "Immediate SAR filing is recommended per regulatory obligation."
            )
            confidence = 0.65 + min(0.3, (risk_score - 60.0) / 100.0)
            key_signals = ["risk_score_high", f"pattern_{pattern}", "threshold_breach"]

        return {
            "vote": vote,
            "reasoning": reasoning,
            "confidence": round(confidence, 3),
            "key_signals": key_signals,
        }

    # ------------------------------------------------------------------
    # Case brief formatter
    # ------------------------------------------------------------------

    def _format_case_brief(self, case_file: dict) -> str:
        """
        Render *case_file* as structured human-readable text suitable for
        injection into an LLM prompt.

        Parameters
        ----------
        case_file : dict
            Structured case file.

        Returns
        -------
        str
            Multi-line text brief.
        """
        features = case_file.get("features", {})
        benford = case_file.get("benford_results", {})
        clustering = case_file.get("clustering_results", {})
        ml = case_file.get("ml_results", {})
        network = case_file.get("network_results", {})
        profile = case_file.get("customer_profile", {})

        # Benford is only meaningful above BenfordAnalyzer.MIN_SAMPLE_SIZE —
        # below that, deviation_score/p_value are forced to 0.0/1.0 as
        # neutral placeholders (see benford.py), NOT real statistics. Showing
        # those numbers unlabelled here previously let both the LLM and the
        # rule-based fallback narrate a "clean"/"anomalous" Benford result
        # that was never actually computed. State the not-computed status
        # explicitly instead of a bare 0.0/1.0 the reader has to interpret.
        if benford.get("insufficient_sample"):
            benford_lines = [
                "--- BENFORD'S LAW ANALYSIS ---",
                f"  Status: NOT COMPUTED — {benford.get('sample_warning', 'insufficient transaction volume for a statistically meaningful result.')}",
            ]
        elif benford:
            benford_lines = [
                "--- BENFORD'S LAW ANALYSIS ---",
                f"  Deviation Score : {benford.get('deviation_score', 'N/A')}",
                f"  MAD Score       : {benford.get('mad_score', 'N/A')}",
                f"  Chi-Square      : {benford.get('chi_square', 'N/A')}",
                f"  P-Value         : {benford.get('p_value', 'N/A')}",
            ]
        else:
            benford_lines = [
                "--- BENFORD'S LAW ANALYSIS ---",
                "  Status: NOT COMPUTED for this query.",
            ]

        # threshold_clustering.analyze_customer() nests these under
        # sub_threshold.spike_score / round_numbers.round_number_ratio /
        # composite_clustering_score — there is no top-level
        # sub_threshold_band_count, round_number_score, or spike_score key.
        # Reading those non-existent keys previously fed "N/A" for all three
        # clustering fields into every agent's prompt, every time.
        clustering_lines = (
            [
                "--- CLUSTERING ANALYSIS ---",
                f"  Composite Clustering Score : {clustering.get('composite_clustering_score', 'N/A')}",
                f"  Sub-threshold Spike Score  : {clustering.get('sub_threshold', {}).get('spike_score', 'N/A')}",
                f"  Round Number Ratio         : {clustering.get('round_numbers', {}).get('round_number_ratio', 'N/A')}",
            ]
            if clustering
            else ["--- CLUSTERING ANALYSIS ---", "  Status: NOT COMPUTED for this query."]
        )

        brief_lines = [
            "=" * 60,
            "AML CASE FILE — COMMITTEE REVIEW",
            "=" * 60,
            f"Customer ID      : {case_file.get('customer_id', 'N/A')}",
            f"Investigator Note: {case_file.get('query', 'N/A')}",
            f"Risk Score       : {case_file.get('risk_score', 'N/A')} / 100",
            f"Risk Tier        : {case_file.get('risk_tier', 'N/A')}",
            f"Detected Pattern : {case_file.get('pattern', 'none')}",
            "",
            "--- TRANSACTION FEATURES ---",
            f"  Transaction Count           : {features.get('transaction_count', 'N/A')}",
            f"  Average Amount (USD)        : {features.get('avg_amount', 'N/A')}",
            f"  Velocity (txns/day)         : {features.get('velocity', 'N/A')}",
            f"  Sub-threshold Count         : {features.get('sub_threshold_count', 'N/A')}",
            f"  Structuring Regularity Score: {features.get('structuring_regularity_score', 'N/A')}",
            f"  Rolling Sum 30d (USD)       : {features.get('rolling_sum_30d', 'N/A')}",
            "",
            *benford_lines,
            "",
            *clustering_lines,
            "",
            "--- ML ANOMALY DETECTION ---",
            f"  Anomaly Score : {ml.get('ml_anomaly_score', 'N/A')}",
            f"  Is Outlier    : {ml.get('ml_is_outlier', 'N/A')}",
            "",
            "--- NETWORK / GRAPH ANALYSIS ---",
            f"  Betweenness Centrality       : {network.get('hub_score', 'N/A')}",
            f"  Is Hub                       : {network.get('is_hub', 'N/A')}",
            f"  Community Size               : {network.get('community_size', 'N/A')}",
            f"  Connected Flagged Accounts   : "
            f"{len(network['connected_flagged_accounts']) if isinstance(network.get('connected_flagged_accounts'), list) else network.get('connected_flagged_accounts', 'N/A')}",
            "",
            "--- CUSTOMER PROFILE ---",
            f"  Occupation              : {profile.get('occupation', 'N/A')}",
            f"  Declared Income Band    : {profile.get('declared_income_band', 'N/A')}",
            f"  Country of Residence    : {profile.get('country_of_residence', 'N/A')}",
            f"  Account Open Date       : {profile.get('account_open_date', 'N/A')}",
            f"  PEP Status              : {'YES' if profile.get('is_pep') else 'NO'}",
            f"  High-Risk Jurisdiction  : {'YES' if profile.get('is_high_risk_jurisdiction') else 'NO'}",
            "=" * 60,
        ]
        return "\n".join(brief_lines)
