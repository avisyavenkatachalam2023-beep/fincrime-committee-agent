"""
network_relationship_agent.py
------------------------------
Specialist agent for financial transaction graph analysis.
Identifies hub accounts, money-mule networks, ring structures,
and layering chains through graph topology metrics.
"""

import logging

try:
    from src.committee.base_agent import BaseCommitteeAgent
except ImportError:
    from .base_agent import BaseCommitteeAgent

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a Network Relationship Analyst Agent specializing in financial transaction graph analysis.
Your specialty: identifying hub accounts, money-mule networks, ring structures, and layering chains
through graph topology analysis.

Key metrics you analyze:
- Betweenness centrality (high = potential hub/coordinator)
- Community membership (flagged accounts in same community = network risk)
- Connected flagged accounts count
- Is this account a network hub?

A high betweenness_centrality (>0.1) with connected_flagged_accounts > 2 is a strong red flag.

Respond ONLY with valid JSON:
{
  "vote": "MONITOR|REVIEW|REPORT",
  "reasoning": "2-4 sentence explanation citing specific graph metrics",
  "confidence": 0.0-1.0,
  "key_signals": ["list of network signals"]
}"""

# ---------------------------------------------------------------------------
# Thresholds for network-based rule triggers
# ---------------------------------------------------------------------------
_HIGH_BETWEENNESS_CENTRALITY = 0.10       # above this → potential coordinator
_CRITICAL_BETWEENNESS_CENTRALITY = 0.25  # above this → strong hub indicator
_FLAGGED_ACCOUNTS_REVIEW = 1             # ≥ 1 connected flagged account → review
_FLAGGED_ACCOUNTS_REPORT = 3             # ≥ 3 connected flagged accounts → report
_SMALL_COMMUNITY_HIGH_RISK = 5           # small isolated community → suspicious
_LARGE_COMMUNITY_HUB_RISK = 50           # large community with hub status → layering risk


class NetworkRelationshipAgent(BaseCommitteeAgent):
    """
    Evaluates financial transaction graph topology signals to identify
    money-mule networks, hub coordinators, and layering structures.

    Signals assessed
    ----------------
    * Betweenness centrality magnitude
    * Hub account designation
    * Number of connected flagged accounts
    * Community size (isolated small ring or large coordinated network)
    * ML outlier status as a corroborating signal
    * Detected pattern type (layering is a primary network concern)
    """

    def __init__(self) -> None:
        """Initialise the Network Relationship Agent with its specialist system prompt."""
        super().__init__(
            agent_name="Network Relationship Analyst",
            system_prompt=_SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------
    # Specialist rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based_fallback(self, case_file: dict) -> dict:
        """
        Evaluate network / graph risk using topology heuristics.

        Decision logic
        --------------
        Red flags are scored individually.  Final vote:

        * REPORT if:
          - is_hub AND connected_flagged_accounts ≥ 3
          - OR betweenness_centrality > 0.25 AND connected_flagged_accounts ≥ 1
          - OR ≥ 3 network red flags triggered
        * REVIEW if:
          - 1–2 network red flags triggered
        * MONITOR if:
          - No red flags

        Parameters
        ----------
        case_file : dict
            Structured case file.

        Returns
        -------
        dict
            ``{vote, reasoning, confidence, key_signals}``
        """
        network = case_file.get("network_results", {})
        ml = case_file.get("ml_results", {})
        pattern = case_file.get("pattern", "none").lower()

        # network_tool.analyze_entity() exposes the betweenness score as
        # "hub_score" at the top level (it's also nested under
        # centrality_scores.betweenness_centrality) and
        # "connected_flagged_accounts" as the actual *list* of flagged
        # neighbour account IDs, not a pre-computed count — reading it as an
        # int directly crashes this rule-based fallback exactly when it's
        # needed most (when the LLM path has failed and we're relying on it).
        betweenness_centrality = float(network.get("hub_score", 0.0))
        is_hub: bool = bool(network.get("is_hub", False))
        community_size = int(network.get("community_size", 0))
        flagged_accounts_field = network.get("connected_flagged_accounts", 0)
        connected_flagged_accounts = (
            len(flagged_accounts_field) if isinstance(flagged_accounts_field, list)
            else int(flagged_accounts_field)
        )

        anomaly_score = float(ml.get("anomaly_score", 0.0))
        is_outlier: bool = bool(ml.get("is_outlier", False))

        network_flags: list[str] = []

        # --- 1. Hub account identification ---
        if is_hub:
            network_flags.append(
                f"hub_account_designation "
                f"(betweenness_centrality={betweenness_centrality:.4f})"
            )

        # --- 2. Betweenness centrality thresholds ---
        if betweenness_centrality > _CRITICAL_BETWEENNESS_CENTRALITY:
            network_flags.append(
                f"critical_betweenness_centrality ({betweenness_centrality:.4f} "
                f"> {_CRITICAL_BETWEENNESS_CENTRALITY})"
            )
        elif betweenness_centrality > _HIGH_BETWEENNESS_CENTRALITY:
            network_flags.append(
                f"elevated_betweenness_centrality ({betweenness_centrality:.4f} "
                f"> {_HIGH_BETWEENNESS_CENTRALITY})"
            )

        # --- 3. Connected flagged account count ---
        if connected_flagged_accounts >= _FLAGGED_ACCOUNTS_REPORT:
            network_flags.append(
                f"high_connected_flagged_accounts "
                f"(count={connected_flagged_accounts} ≥ {_FLAGGED_ACCOUNTS_REPORT})"
            )
        elif connected_flagged_accounts >= _FLAGGED_ACCOUNTS_REVIEW:
            network_flags.append(
                f"connected_flagged_accounts (count={connected_flagged_accounts})"
            )

        # --- 4. Suspicious community structure ---
        if 0 < community_size <= _SMALL_COMMUNITY_HIGH_RISK and is_hub:
            network_flags.append(
                f"small_isolated_hub_community (community_size={community_size})"
            )
        elif community_size > _LARGE_COMMUNITY_HUB_RISK and is_hub:
            network_flags.append(
                f"large_network_hub (community_size={community_size})"
            )

        # --- 5. Layering pattern in graph context ---
        if pattern == "layering":
            network_flags.append("layering_pattern_detected")

        # --- 6. ML anomaly corroboration ---
        if is_outlier and anomaly_score > 0.65:
            network_flags.append(
                f"ml_outlier_corroboration (anomaly_score={anomaly_score:.3f})"
            )

        # ------------------------------------------------------------------
        # Vote determination — key compound conditions
        # ------------------------------------------------------------------
        strong_hub_signal = (
            is_hub and connected_flagged_accounts >= _FLAGGED_ACCOUNTS_REPORT
        )
        critical_centrality_signal = (
            betweenness_centrality > _CRITICAL_BETWEENNESS_CENTRALITY
            and connected_flagged_accounts >= _FLAGGED_ACCOUNTS_REVIEW
        )

        flag_count = len(network_flags)

        if strong_hub_signal or critical_centrality_signal or flag_count >= 3:
            vote = "REPORT"
            reasoning = (
                f"{flag_count} network topology red flag(s) identified: "
                f"{'; '.join(network_flags[:3])}. "
                f"Betweenness centrality of {betweenness_centrality:.4f} "
                f"{'(hub-designated) ' if is_hub else ''}"
                f"with {connected_flagged_accounts} connected flagged account(s) "
                f"in a community of {community_size} node(s) is consistent with "
                "a hub coordinator or layering structure requiring SAR filing."
            )
            confidence = min(0.95, 0.68 + flag_count * 0.06)
        elif flag_count >= 1:
            vote = "REVIEW"
            reasoning = (
                f"{flag_count} network concern(s) flagged: {'; '.join(network_flags)}. "
                f"Betweenness centrality of {betweenness_centrality:.4f} "
                f"and {connected_flagged_accounts} connected flagged account(s) "
                "suggest potential involvement in a broader transaction network. "
                "Graph-level investigation by the compliance team is recommended."
            )
            confidence = 0.50 + flag_count * 0.09
        else:
            vote = "MONITOR"
            reasoning = (
                f"No network topology red flags detected. "
                f"Betweenness centrality ({betweenness_centrality:.4f}) is below "
                f"the hub threshold ({_HIGH_BETWEENNESS_CENTRALITY}), "
                f"and {connected_flagged_accounts} connected flagged account(s) "
                "does not indicate network involvement. Standard monitoring applies."
            )
            confidence = 0.73

        return {
            "vote": vote,
            "reasoning": reasoning,
            "confidence": round(min(1.0, confidence), 3),
            "key_signals": network_flags if network_flags else ["no_network_flags_triggered"],
        }
