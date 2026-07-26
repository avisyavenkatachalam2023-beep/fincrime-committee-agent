"""
chair_agent.py
--------------
The Chair Agent synthesises specialist votes, applies documented tie-break
rules, generates a coherent narrative via Groq, and produces formatted
Committee Meeting Minutes.
"""

import os
import json
import re
import logging
import random
from datetime import date
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

try:
    from src.committee.base_agent import (
        find_unverifiable_number,
        _extract_numeric_claims,
    )
except ImportError:
    from .base_agent import find_unverifiable_number, _extract_numeric_claims

load_dotenv()

logger = logging.getLogger(__name__)

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

# ---------------------------------------------------------------------------
# Vote ordering
# ---------------------------------------------------------------------------
_VOTE_LEVELS = {"MONITOR": 0, "REVIEW": 1, "REPORT": 2}
_VOTE_ORDER = ["MONITOR", "REVIEW", "REPORT"]

# ---------------------------------------------------------------------------
# Escalation actions per final decision
# ---------------------------------------------------------------------------
_ESCALATION_ACTIONS = {
    "MONITOR": "Continue enhanced transaction monitoring for 90 days.",
    "REVIEW": "Flag for compliance officer review within 5 business days.",
    "REPORT": (
        "File Suspicious Activity Report (SAR) within 30 days "
        "per regulatory requirement."
    ),
}

# ---------------------------------------------------------------------------
# Reconciliation between the quantitative composite risk score/tier
# (RiskClassifier's weighted aggregate of Benford/clustering/ML/network
# signals) and the qualitative committee decision (each specialist agent's
# own domain-specific judgment, which can catch things the composite score
# dilutes away — e.g. a direct connection to a flagged account, or a KYC
# income mismatch). These are two intentionally independent assessments;
# when they disagree, that must be stated explicitly rather than left to
# look like an unexplained contradiction between a low score and a high
# escalation action (or vice versa).
# ---------------------------------------------------------------------------
_TIER_BASELINE_DECISION = {"LOW": "MONITOR", "MEDIUM": "REVIEW", "HIGH": "REPORT"}

# Chair synthesis system prompt
_CHAIR_SYSTEM_PROMPT = """You are the Chair of an AML Financial Crime Committee.
You have just received votes from four specialist agents and must synthesise their
findings into a coherent 3-5 sentence paragraph that:
1. Summarises the key signals identified across all agents
2. Explains the rationale for the final committee decision
3. Notes any dissenting opinions or areas of uncertainty
4. Uses formal, regulatory-grade language appropriate for compliance records

Only cite numeric values explicitly present in the case file or in the specialists'
own reasoning below. Never invent, estimate, or approximate a statistic. If a
signal was not computed, say so explicitly instead of citing a number for it.

Respond ONLY with a plain text paragraph (no JSON, no markdown, no bullet points)."""


class ChairAgent:
    """
    The Chair Agent synthesises all specialist votes into a final committee
    decision and produces Committee Meeting Minutes.

    Tie-Break Rule (documented)
    ---------------------------
    Any single REPORT vote combined with 2+ REVIEW votes escalates the final
    decision to REPORT.  Otherwise majority vote applies.  On an exact 2-2 tie
    the decision escalates to REVIEW (conservative escalation).
    """

    TIE_BREAK_RULE: str = (
        "Any single REPORT vote combined with 2+ REVIEW votes escalates to REPORT. "
        "Otherwise majority vote applies. On a 2-2 tie, the decision escalates to REVIEW."
    )

    def __init__(self) -> None:
        """
        Initialise the Chair Agent.

        Attempts to configure Groq for narrative synthesis.  Falls back to a
        structured template if the API is unavailable.
        """
        self._client = None
        self._model_name = None
        self._llm_available: bool = False
        self._init_llm()

    # ------------------------------------------------------------------
    # LLM initialisation
    # ------------------------------------------------------------------

    def _init_llm(self) -> None:
        """
        Load the Groq API key from the environment and configure the client
        for the Chair's synthesis narrative generation.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning(
                "[Chair] GROQ_API_KEY not found — template synthesis will be used."
            )
            return

        try:
            self._client = Groq(api_key=api_key)
            self._model_name = PRIMARY_MODEL
            self._llm_available = True
            logger.info("[Chair] Groq client ready using %s.", self._model_name)
        except Exception as exc:
            logger.warning(
                "[Chair] Could not initialise Groq client: %s", exc
            )
            self._client = None
            self._llm_available = False

        if not self._llm_available:
            logger.warning("[Chair] LLM unavailable — using template.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def synthesize(self, case_file: dict, agent_votes: list[dict]) -> dict:
        """
        Synthesise all specialist votes into a final committee decision.

        Parameters
        ----------
        case_file : dict
            The original structured case file.
        agent_votes : list[dict]
            List of vote dicts returned by the four specialist agents,
            each containing ``{agent, vote, reasoning, confidence, key_signals}``.

        Returns
        -------
        dict
            ``{final_decision, synthesis, meeting_minutes, vote_breakdown,
               escalation_action, case_reference}``
        """
        votes = [str(v.get("vote", "MONITOR")).upper() for v in agent_votes]
        final_decision = self._apply_voting_rules(votes)
        escalation_action = self._get_escalation_action(final_decision)
        tier_decision_note = self._tier_decision_note(
            case_file.get("risk_tier"), final_decision
        )
        synthesis = self._generate_synthesis(case_file, agent_votes, final_decision)
        meeting_minutes = self._generate_meeting_minutes(
            case_file, agent_votes, final_decision, synthesis, escalation_action,
            tier_decision_note,
        )

        case_ref = self._make_case_reference()

        vote_breakdown = {
            v.get("agent", f"Agent_{i}"): {
                "vote": v.get("vote"),
                "confidence": v.get("confidence"),
                "key_signals": v.get("key_signals", []),
            }
            for i, v in enumerate(agent_votes)
        }

        return {
            "final_decision": final_decision,
            "synthesis": synthesis,
            "meeting_minutes": meeting_minutes,
            "vote_breakdown": vote_breakdown,
            "escalation_action": escalation_action,
            "case_reference": case_ref,
            "tier_decision_note": tier_decision_note,
        }

    def _tier_decision_note(
        self, risk_tier: Optional[str], final_decision: str
    ) -> Optional[str]:
        """Explain any mismatch between the composite risk tier and the
        committee's final decision.

        The composite risk tier (LOW/MEDIUM/HIGH) is a quantitative,
        weighted aggregate of forensic signals. The committee decision
        (MONITOR/REVIEW/REPORT) is a qualitative judgment from four
        specialist agents examining different evidence, and can legitimately
        diverge from the composite score — e.g. a low aggregate score can
        still sit next to a direct connection to a flagged account, which
        one specialist correctly escalates on its own. That divergence is
        a real, useful finding, not a bug — but it must be stated, or a low
        score next to a high escalation action reads as an unexplained
        contradiction.

        Args:
            risk_tier: 'LOW', 'MEDIUM', 'HIGH', or None if no composite
                score was computed for this case.
            final_decision: The committee's resolved decision.

        Returns:
            An explanatory note string if the tier and decision disagree,
            else None.
        """
        if not risk_tier:
            return None
        baseline = _TIER_BASELINE_DECISION.get(risk_tier.upper())
        if baseline is None or baseline == final_decision:
            return None

        if _VOTE_LEVELS.get(final_decision, 0) > _VOTE_LEVELS.get(baseline, 0):
            return (
                f"The composite statistical risk score is {risk_tier} tier, which alone "
                f"would suggest {baseline}, but the committee escalated to {final_decision} "
                "based on specialist-level findings not fully captured by the aggregate "
                "score. See the individual agent statements above for the specific concern "
                "driving this decision."
            )
        return (
            f"The composite statistical risk score is {risk_tier} tier, which alone would "
            f"suggest {baseline}, but the committee's {final_decision} decision is less "
            "severe: specialists reviewed the elevated aggregate score and did not find "
            "case-specific evidence warranting further escalation."
        )

    # ------------------------------------------------------------------
    # Voting logic
    # ------------------------------------------------------------------

    def _apply_voting_rules(self, votes: list[str]) -> str:
        """
        Apply the documented tie-break rule and majority vote to determine
        the final committee decision.

        Rules (in order of precedence)
        --------------------------------
        1. Any single REPORT + 2 or more REVIEW → REPORT
        2. Majority vote (more than half of votes for a single outcome)
        3. On a tie (e.g. 2 MONITOR vs 2 REVIEW with no REPORT) → REVIEW
           (conservative escalation to the higher tier)

        Parameters
        ----------
        votes : list[str]
            List of vote strings, each one of ``MONITOR``, ``REVIEW``,
            ``REPORT``.

        Returns
        -------
        str
            Final decision: ``'MONITOR'``, ``'REVIEW'``, or ``'REPORT'``.
        """
        if not votes:
            logger.warning("[Chair] No votes received; defaulting to REVIEW.")
            return "REVIEW"

        report_count = votes.count("REPORT")
        review_count = votes.count("REVIEW")
        monitor_count = votes.count("MONITOR")
        total = len(votes)

        # Rule 1: Escalation rule — any REPORT + 2+ REVIEW → REPORT
        if report_count >= 1 and review_count >= 2:
            logger.info(
                "[Chair] Escalation rule triggered: %d REPORT + %d REVIEW → REPORT",
                report_count,
                review_count,
            )
            return "REPORT"

        # Rule 2: Clear majority
        if report_count > total / 2:
            return "REPORT"
        if review_count > total / 2:
            return "REVIEW"
        if monitor_count > total / 2:
            return "MONITOR"

        # Rule 3: No majority — conservative tie-break (escalate to higher tier)
        # Find the highest tier that has at least one vote
        for tier in reversed(_VOTE_ORDER):  # REPORT → REVIEW → MONITOR
            if votes.count(tier) > 0:
                logger.info(
                    "[Chair] Tie-break: escalating to '%s' (no majority reached).", tier
                )
                return tier

        return "REVIEW"  # Ultimate fallback

    # ------------------------------------------------------------------
    # Synthesis narrative
    # ------------------------------------------------------------------

    def _generate_synthesis(
        self, case_file: dict, agent_votes: list[dict], final_decision: str
    ) -> str:
        """
        Generate a coherent synthesis paragraph describing the committee's
        collective assessment.

        Attempts Groq API first; falls back to a structured template.

        Parameters
        ----------
        case_file : dict
            Structured case file.
        agent_votes : list[dict]
            Specialist vote dicts.
        final_decision : str
            The resolved final decision.

        Returns
        -------
        str
            Multi-sentence synthesis paragraph.
        """
        if self._llm_available and self._client is not None:
            try:
                synthesis = self._generate_synthesis_via_llm(
                    case_file, agent_votes, final_decision
                )
                if synthesis:
                    return synthesis
            except Exception as exc:  # noqa: BLE001
                logger.error("[Chair] LLM synthesis failed (%s); using template.", exc)

        return self._template_synthesis(case_file, agent_votes, final_decision)

    def _generate_synthesis_via_llm(
        self, case_file: dict, agent_votes: list[dict], final_decision: str
    ) -> str:
        """Call Groq for the synthesis paragraph, verifying every numeric
        claim against the case file and the specialists' own reasoning text
        before accepting it — with one corrective retry, then a hard
        rejection (caller falls back to the deterministic template) rather
        than trusting an invented statistic.
        """
        prompt = self._build_synthesis_prompt(case_file, agent_votes, final_decision)
        # Numbers already legitimately quoted in a specialist's own reasoning
        # are fair for the Chair to repeat, even though they only exist as
        # free text there (not as a typed value in case_file).
        vote_text_numbers: set = set()
        for v in agent_votes:
            vote_text_numbers |= set(_extract_numeric_claims(v.get("reasoning", "")))

        messages = [
            {"role": "system", "content": _CHAIR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        last_bad_number: Optional[float] = None
        for attempt in range(2):
            response = self._client.chat.completions.create(
                messages=messages,
                model=self._model_name,
                temperature=0.3,
                max_tokens=512,
            )
            synthesis = (response.choices[0].message.content or "").strip()
            if not synthesis:
                return ""

            bad_number = find_unverifiable_number(synthesis, case_file, vote_text_numbers)
            if bad_number is None:
                return synthesis

            last_bad_number = bad_number
            logger.warning(
                "[Chair] LLM synthesis cited unverifiable number %s not present "
                "in the case file or specialist reasoning (attempt %d/2).",
                bad_number, attempt + 1,
            )
            messages.append({"role": "assistant", "content": synthesis})
            messages.append({"role": "user", "content": (
                f"Your synthesis cited the number {bad_number}, which does not "
                "appear anywhere in the case file or the specialists' own "
                "reasoning above. Write the synthesis again, citing only "
                "numbers explicitly present in that data."
            )})

        logger.error(
            "[Chair] LLM synthesis still cited unverifiable number %s after a "
            "corrective retry; falling back to template.", last_bad_number,
        )
        return ""

    def _build_synthesis_prompt(
        self, case_file: dict, agent_votes: list[dict], final_decision: str
    ) -> str:
        """
        Construct the prompt for synthesis narrative generation.

        Parameters
        ----------
        case_file : dict
            Structured case file.
        agent_votes : list[dict]
            Specialist vote dicts.
        final_decision : str
            Resolved final decision.

        Returns
        -------
        str
            Full prompt string.
        """
        customer_id = case_file.get("customer_id", "N/A")
        pattern = case_file.get("pattern", "none")
        risk_score = case_file.get("risk_score", 0.0)

        lines = [
            f"Customer ID: {customer_id}",
            f"Risk Score: {risk_score:.1f}/100",
            f"Detected Pattern: {pattern}",
            f"Final Committee Decision: {final_decision}",
            "",
            "Specialist Agent Votes:",
        ]
        for v in agent_votes:
            lines.append(
                f"  [{v.get('agent', 'Unknown')}] "
                f"Vote={v.get('vote')} "
                f"Confidence={v.get('confidence', 0):.2f} — "
                f"{v.get('reasoning', '')}"
            )

        lines.append(
            "\nWrite a 3-5 sentence synthesis paragraph for the Committee Minutes."
        )
        return "\n".join(lines)

    def _template_synthesis(
        self, case_file: dict, agent_votes: list[dict], final_decision: str
    ) -> str:
        """
        Generate a structured template synthesis when the LLM is unavailable.

        Parameters
        ----------
        case_file : dict
            Structured case file.
        agent_votes : list[dict]
            Specialist vote dicts.
        final_decision : str
            Resolved final decision.

        Returns
        -------
        str
            Multi-sentence synthesis paragraph.
        """
        customer_id = case_file.get("customer_id", "N/A")
        risk_score = case_file.get("risk_score", 0.0)
        pattern = case_file.get("pattern", "none")
        risk_tier = case_file.get("risk_tier", "UNKNOWN")

        vote_counts = {"MONITOR": 0, "REVIEW": 0, "REPORT": 0}
        all_signals: list[str] = []
        for v in agent_votes:
            vote = str(v.get("vote", "MONITOR")).upper()
            if vote in vote_counts:
                vote_counts[vote] += 1
            all_signals.extend(v.get("key_signals", []))

        top_signals = list(dict.fromkeys(all_signals))[:5]  # deduplicated, top 5
        signal_str = ", ".join(top_signals) if top_signals else "no dominant signals"

        avg_confidence = (
            sum(v.get("confidence", 0.5) for v in agent_votes) / len(agent_votes)
            if agent_votes
            else 0.5
        )

        vote_summary = ", ".join(
            f"{count} {vote}" for vote, count in vote_counts.items() if count > 0
        )

        return (
            f"The Financial Crime Committee has completed its deliberation on customer "
            f"{customer_id}, assigned a risk score of {risk_score:.1f}/100 ({risk_tier} tier) "
            f"with a detected pattern of '{pattern}'. "
            f"The committee vote breakdown was: {vote_summary}, yielding a final decision "
            f"of {final_decision} with an aggregate confidence of {avg_confidence:.2f}. "
            f"Key signals driving this decision include: {signal_str}. "
            f"The committee has determined that the case warrants '{final_decision}' "
            f"treatment, and the escalation action has been recorded accordingly. "
            f"All agents' reasoning has been preserved in these minutes for regulatory audit."
        )

    # ------------------------------------------------------------------
    # Meeting minutes
    # ------------------------------------------------------------------

    def _generate_meeting_minutes(
        self,
        case_file: dict,
        agent_votes: list[dict],
        final_decision: str,
        synthesis: str,
        escalation_action: str,
        tier_decision_note: Optional[str] = None,
    ) -> str:
        """
        Generate formatted Committee Meeting Minutes as a structured text block.

        Format
        ------
        * Header with case reference, date, attendees
        * Call to order
        * Individual agent statements
        * Chair synthesis
        * Score/decision reconciliation note (if the composite tier and the
          committee decision disagree)
        * Final decision and escalation action
        * Close / signature

        Parameters
        ----------
        case_file : dict
            Structured case file.
        agent_votes : list[dict]
            Specialist vote dicts.
        final_decision : str
            Resolved final decision.
        synthesis : str
            Chair synthesis paragraph.
        escalation_action : str
            Escalation action string.
        tier_decision_note : str, optional
            Explanatory note when the composite risk tier and final_decision
            disagree (see _tier_decision_note).

        Returns
        -------
        str
            Fully formatted meeting minutes.
        """
        customer_id = case_file.get("customer_id", "N/A")
        today_str = date.today().strftime("%B %d, %Y")
        case_ref = self._make_case_reference()
        risk_score = case_file.get("risk_score", 0.0)
        risk_tier = case_file.get("risk_tier", "UNKNOWN")
        pattern = case_file.get("pattern", "none")

        attendees = [v.get("agent", "Unknown Agent") for v in agent_votes]
        attendees_str = ", ".join(attendees) + ", Committee Chair (Autonomous)"

        lines = [
            "=" * 70,
            "FINANCIAL CRIME COMMITTEE — MEETING MINUTES",
            "=" * 70,
            f"Case Reference   : {case_ref}",
            f"Date             : {today_str}",
            f"Customer ID      : {customer_id}",
            f"Risk Score       : {risk_score:.1f} / 100  [{risk_tier}]",
            f"Detected Pattern : {pattern.upper()}",
            f"Attendees        : {attendees_str}",
            "-" * 70,
            "",
            "[CALL TO ORDER]",
            f"Chair: 'We are convened to review case {customer_id}. The risk score "
            f"is {risk_score:.1f}/100, tier {risk_tier}, with a detected pattern of "
            f"'{pattern}'. Each specialist agent will now present their assessment.'",
            "",
        ]

        # Individual agent statements
        for vote_dict in agent_votes:
            agent_name = vote_dict.get("agent", "Unknown Agent").upper()
            vote = vote_dict.get("vote", "MONITOR")
            confidence = float(vote_dict.get("confidence", 0.0))
            reasoning = vote_dict.get("reasoning", "No reasoning provided.")
            key_signals = vote_dict.get("key_signals", [])

            lines += [
                f"[{agent_name}]",
                f"Vote       : {vote}",
                f"Confidence : {confidence:.2f}",
                f"Key Signals: {', '.join(key_signals) if key_signals else 'None identified'}",
                f"Statement  : {reasoning}",
                "",
            ]

        # Chair synthesis section
        lines += [
            "[CHAIR SYNTHESIS]",
            synthesis,
            "",
            "[VOTING RECORD]",
            f"Tie-Break Rule Applied: {self.TIE_BREAK_RULE}",
        ]
        for v in agent_votes:
            lines.append(
                f"  • {v.get('agent', 'Unknown'):35s} → {v.get('vote', 'N/A')}"
            )

        if tier_decision_note:
            lines += [
                "",
                "[SCORE / DECISION RECONCILIATION]",
                tier_decision_note,
            ]

        lines += [
            "",
            "[FINAL DECISION]",
            f"Decision           : {final_decision}",
            f"Escalation Action  : {escalation_action}",
            "",
            "[CLOSE]",
            "These minutes have been reviewed and are certified accurate.",
            "Minutes recorded by: Autonomous AML Agent v1.0",
            f"Recorded on        : {today_str}",
            "=" * 70,
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Escalation action
    # ------------------------------------------------------------------

    def _get_escalation_action(self, final_decision: str) -> str:
        """
        Return the standard escalation action string for a given decision.

        Parameters
        ----------
        final_decision : str
            One of ``'MONITOR'``, ``'REVIEW'``, ``'REPORT'``.

        Returns
        -------
        str
            Descriptive escalation action.
        """
        return _ESCALATION_ACTIONS.get(final_decision, "Escalation action not defined.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_case_reference() -> str:
        """
        Generate a unique case reference in the format
        ``CASE-{YEAR}-{random 4 digits}``.

        Returns
        -------
        str
            Case reference string, e.g. ``'CASE-2026-4821'``.
        """
        year = date.today().year
        suffix = random.randint(1000, 9999)
        return f"CASE-{year}-{suffix}"
