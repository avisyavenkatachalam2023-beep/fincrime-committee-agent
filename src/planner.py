"""
src/planner.py
--------------
Dynamic Execution Planner for the AML Financial Crime Committee Agent.

Responsibilities:
  - Accept the structured parsed-query dict produced by QueryUnderstandingTool
  - Determine the ordered execution plan (list of tool-call strings)
  - Determine which tools are skipped and why
  - Return the complete Execution Summary dict (Section 8.1 schema)

Execution Summary schema:
{
  'user_query'        : str,
  'parsed_intent'     : str,
  'detected_filters'  : {'date_range': str, 'country': str,
                          'segment': str, 'transaction_type': str},
  'detected_entities' : list,
  'target_pattern'    : str,
  'requires_eda'      : bool,
  'requires_ml'       : bool,
  'execution_plan'    : list,   # ordered tool-call strings
  'tools_skipped'     : list,
  'reason_for_skips'  : str,
}
"""

from __future__ import annotations

import logging
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants – valid intents (mirrors query_understanding.py)
# ---------------------------------------------------------------------------

INTENTS: list[str] = [
    "pattern_detection",
    "aggregation_query",
    "single_entity_lookup",
    "explain_flag",
    "network_query",
    "comparative_query",
    "escalation_query",
    "broad_exploration",
]

PATTERNS: list[str] = ["structuring", "smurfing", "layering", "none"]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DynamicExecutionPlanner:
    """Translate a parsed query dict into a complete Execution Summary.

    The planner applies a deterministic intent-to-plan mapping.  Each intent
    produces an ordered list of tool-call strings (``execution_plan``), a list
    of skipped tools (``tools_skipped``), and a human-readable justification
    (``reason_for_skips``).

    Usage::

        from src.query_understanding import QueryUnderstandingTool
        from src.planner import DynamicExecutionPlanner

        qu = QueryUnderstandingTool()
        planner = DynamicExecutionPlanner()

        parsed = qu.parse("Find structuring patterns in the last 30 days")
        summary = planner.plan(parsed)
    """

    def plan(self, parsed_query: dict) -> dict:
        """Build and return the complete Execution Summary dict.

        Args:
            parsed_query: The structured dict returned by
                          ``QueryUnderstandingTool.parse()``.  Must contain at
                          minimum the keys ``intent``, ``filters``,
                          ``entities``, ``target_pattern``, ``requires_eda``,
                          and ``requires_ml``.

        Returns:
            Execution Summary dict conforming to Section 8.1 schema.

        Raises:
            ValueError: If ``parsed_query`` is missing mandatory keys.
        """
        # Validate input structure
        required_keys = {"intent", "filters", "entities", "target_pattern",
                         "requires_eda", "requires_ml"}
        missing = required_keys - set(parsed_query.keys())
        if missing:
            raise ValueError(
                f"parsed_query is missing required keys: {missing}.  "
                "Pass the dict returned by QueryUnderstandingTool.parse()."
            )

        intent: str = parsed_query.get("intent", "broad_exploration")
        filters: dict = parsed_query.get("filters", {}) or {}
        entities: list = parsed_query.get("entities", []) or []
        target_pattern: str = parsed_query.get("target_pattern", "none")
        requires_eda: bool = bool(parsed_query.get("requires_eda", False))
        requires_ml: bool = bool(parsed_query.get("requires_ml", False))

        # Normalise intent to a known value
        if intent not in INTENTS:
            logger.warning(
                "Unknown intent '%s' received; defaulting to broad_exploration.", intent
            )
            intent = "broad_exploration"

        logger.info(
            "Planning execution for intent='%s', pattern='%s', entities=%s",
            intent, target_pattern, entities,
        )

        execution_plan, tools_skipped, reason_for_skips = self._build_plan_for_intent(
            intent=intent,
            pattern=target_pattern,
            entities=entities,
            filters=filters,
        )

        # Retrieve original user query if present (injected by the orchestrator)
        user_query: str = parsed_query.get("user_query", "")

        return {
            "user_query":         user_query,
            "parsed_intent":      intent,
            "detected_filters":   {
                "date_range":       filters.get("date_range"),
                "country":          filters.get("country"),
                "segment":          filters.get("segment"),
                "transaction_type": filters.get("transaction_type"),
            },
            "detected_entities":  entities,
            "target_pattern":     target_pattern,
            "requires_eda":       requires_eda,
            "requires_ml":        requires_ml,
            "execution_plan":     execution_plan,
            "tools_skipped":      tools_skipped,
            "reason_for_skips":   reason_for_skips,
        }

    # ------------------------------------------------------------------
    # Private: intent-to-plan builder
    # ------------------------------------------------------------------

    def _build_plan_for_intent(
        self,
        intent: str,
        pattern: str,
        entities: list,
        filters: dict,
    ) -> tuple[list[str], list[str], str]:
        """Return (execution_plan, tools_skipped, reason_for_skips) for the intent.

        Each branch implements the EXACT mapping specified in Section 8.1 of
        the project specification.  Sub-branches exist for pattern_detection
        when a specific typology (structuring / smurfing / layering) is named.

        Args:
            intent:   One of the eight valid intent strings.
            pattern:  Detected AML pattern or 'none'.
            entities: List of detected entity ID strings.
            filters:  Filter dict with date_range, country, segment,
                      transaction_type keys.

        Returns:
            3-tuple of (execution_plan list, tools_skipped list, skip_reason str).
        """
        # Convenience shorthand
        date_range = filters.get("date_range") or "all"
        first_entity: str = entities[0] if entities else "unknown"

        # ------------------------------------------------------------------
        # 1. pattern_detection
        # ------------------------------------------------------------------
        if intent == "pattern_detection":
            return self._plan_pattern_detection(pattern, date_range, filters)

        # ------------------------------------------------------------------
        # 2. aggregation_query
        # ------------------------------------------------------------------
        if intent == "aggregation_query":
            execution_plan = [
                "load_filtered_data(all)",
                "rule_engine(threshold_aggregation)",
                "output_formatter(aggregation_results)",
            ]
            tools_skipped = [
                "full_eda",
                "feature_engineering",
                "ml_isolation_forest",
                "benford",
                "network_tool",
                "risk_classification",
                "committee",
                "explanation",
            ]
            skip_reason = (
                "Query is a pure aggregation/threshold rule query. "
                "ML anomaly detection is not required and is explicitly skipped. "
                "No committee review needed for a counting query."
            )
            return execution_plan, tools_skipped, skip_reason

        # ------------------------------------------------------------------
        # 3. single_entity_lookup
        # ------------------------------------------------------------------
        if intent == "single_entity_lookup":
            execution_plan = [
                f"load_entity_data(entity={first_entity})",
                "feature_engineering(all_features, entity_scoped)",
                "anomaly_detection(benford, threshold_clustering, ml_model, entity_scoped)",
                "risk_classification(entity_scoped)",
                "committee(all_agents, entity_scoped)",
                "explanation(entity_scoped)",
                "output_formatter(risk_memo)",
            ]
            tools_skipped = [
                "full_eda",
                "full_dataset_feature_engineering",
                "network_tool",
            ]
            skip_reason = (
                "Single entity lookup — full-dataset EDA and full-dataset feature engineering "
                "are skipped. All analysis is scoped to the requested entity only."
            )
            return execution_plan, tools_skipped, skip_reason

        # ------------------------------------------------------------------
        # 4. explain_flag
        # ------------------------------------------------------------------
        if intent == "explain_flag":
            execution_plan = [
                f"retrieve_existing_flags(entity={first_entity})",
                "explanation(retrieve_and_explain)",
                "output_formatter(explanation_only)",
            ]
            tools_skipped = [
                "full_eda",
                "feature_engineering",
                "anomaly_detection",
                "risk_classification",
                "network_tool",
                "committee",
            ]
            skip_reason = (
                "Explain-only query — detection already done. "
                "Retrieving existing flag and running explanation layer only."
            )
            return execution_plan, tools_skipped, skip_reason

        # ------------------------------------------------------------------
        # 5. network_query
        # ------------------------------------------------------------------
        if intent == "network_query":
            entity_scope = entities[0] if entities else "all"
            execution_plan = [
                f"load_filtered_data(entity_neighborhood={entity_scope})",
                "network_tool(centrality, community_detection)",
                "risk_classification(network_signals_only)",
                "output_formatter(network_results)",
            ]
            tools_skipped = [
                "full_eda",
                "feature_engineering",
                "benford",
                "threshold_clustering",
                "ml_isolation_forest",
                "committee",
            ]
            skip_reason = (
                "Network query — only graph analysis is needed. "
                "Benford/threshold/ML tools are not relevant for relationship analysis."
            )
            return execution_plan, tools_skipped, skip_reason

        # ------------------------------------------------------------------
        # 6. comparative_query
        # ------------------------------------------------------------------
        if intent == "comparative_query":
            execution_plan = [
                "load_filtered_data(all)",
                "rule_engine(comparative_aggregation)",
                "output_formatter(comparison_table)",
            ]
            tools_skipped = [
                "full_eda",
                "feature_engineering",
                "anomaly_detection",
                "network_tool",
                "risk_classification",
                "committee",
            ]
            skip_reason = (
                "Comparative query — pure statistics/aggregation. "
                "No anomaly detection or committee review needed."
            )
            return execution_plan, tools_skipped, skip_reason

        # ------------------------------------------------------------------
        # 7. escalation_query
        # ------------------------------------------------------------------
        if intent == "escalation_query":
            execution_plan = [
                f"retrieve_existing_risk_data(entity={first_entity})",
                "committee(all_agents, escalation_mode)",
                "output_formatter(risk_memo, escalation_recommendation)",
            ]
            tools_skipped = [
                "full_eda",
                "feature_engineering",
                "anomaly_detection",
                "network_tool",
            ]
            skip_reason = (
                "Escalation query — detection assumed already completed. "
                "Invoking Committee and Risk Memo generator only using existing scores."
            )
            return execution_plan, tools_skipped, skip_reason

        # ------------------------------------------------------------------
        # 8. broad_exploration  (also the default fallback)
        # ------------------------------------------------------------------
        execution_plan = [
            "load_filtered_data(all)",
            "eda_tool(full)",
            "feature_engineering(all_features)",
            "anomaly_detection(benford, threshold_clustering, ml_isolation_forest)",
            "network_tool(centrality, community_detection)",
            "risk_classification(all_signals)",
            "committee(all_agents)",
            "explanation(all_flagged)",
            "output_formatter(full_report)",
        ]
        tools_skipped: list[str] = []
        skip_reason = (
            "Broad exploration query — all tools are invoked for comprehensive "
            "dataset analysis."
        )
        return execution_plan, tools_skipped, skip_reason

    # ------------------------------------------------------------------
    # Private: pattern_detection sub-planner
    # ------------------------------------------------------------------

    def _plan_pattern_detection(
        self,
        pattern: str,
        date_range: str,
        filters: dict,
    ) -> tuple[list[str], list[str], str]:
        """Build the execution plan for pattern_detection intent.

        Three typology-specific sub-plans are supported (structuring,
        smurfing, layering).  When the pattern is 'none' or unrecognised
        the generic pattern-detection plan is returned which runs all
        anomaly-detection tools.

        Args:
            pattern:    Detected AML pattern string.
            date_range: Resolved date_range filter string.
            filters:    Full filter dict (for future extension).

        Returns:
            3-tuple of (execution_plan, tools_skipped, skip_reason).
        """
        if pattern == "structuring":
            execution_plan = [
                f"load_filtered_data(date_range={date_range})",
                "feature_engineering(structuring_features_only)",
                "anomaly_detection(benford, round_number_clustering)",
                "risk_classification",
                "committee(transaction_monitoring_only)",
                "explanation",
                "output_formatter",
            ]
            tools_skipped = [
                "full_eda",
                "ml_isolation_forest",
                "network_tool",
                "kyc_agent",
                "sanctions_agent",
            ]
            skip_reason = (
                "Query intent is targeted pattern detection with a specified typology "
                "(structuring); full EDA and unrelated committee agents are not required "
                "to answer this query."
            )
            return execution_plan, tools_skipped, skip_reason

        if pattern == "smurfing":
            execution_plan = [
                f"load_filtered_data(date_range={date_range})",
                "feature_engineering(smurfing_features_only)",
                "anomaly_detection(velocity_clustering, multi_account_grouping)",
                "risk_classification",
                "committee(transaction_monitoring_only)",
                "explanation",
                "output_formatter",
            ]
            tools_skipped = [
                "full_eda",
                "benford",
                "ml_isolation_forest",
                "network_tool",
                "kyc_agent",
                "sanctions_agent",
            ]
            skip_reason = (
                "Query intent is targeted pattern detection with a specified typology "
                "(smurfing); full EDA, Benford analysis, and unrelated committee agents "
                "are not required to answer this query."
            )
            return execution_plan, tools_skipped, skip_reason

        if pattern == "layering":
            execution_plan = [
                f"load_filtered_data(date_range={date_range})",
                "feature_engineering(layering_features_only)",
                "network_tool(transaction_chain_analysis)",
                "anomaly_detection(chain_complexity_scoring)",
                "risk_classification",
                "committee(transaction_monitoring_only)",
                "explanation",
                "output_formatter",
            ]
            tools_skipped = [
                "full_eda",
                "benford",
                "threshold_clustering",
                "ml_isolation_forest",
                "kyc_agent",
                "sanctions_agent",
            ]
            skip_reason = (
                "Query intent is targeted pattern detection with a specified typology "
                "(layering); network graph tools are used instead of Benford/threshold "
                "tools. Full EDA and unrelated committee agents are not required."
            )
            return execution_plan, tools_skipped, skip_reason

        # Generic pattern_detection (pattern == 'none' or unrecognised)
        execution_plan = [
            f"load_filtered_data(date_range={date_range})",
            "feature_engineering(all_features)",
            "anomaly_detection(benford, threshold_clustering, ml_isolation_forest)",
            "risk_classification",
            "committee(transaction_monitoring_only)",
            "explanation",
            "output_formatter",
        ]
        tools_skipped = [
            "full_eda",
            "network_tool",
            "kyc_agent",
            "sanctions_agent",
        ]
        skip_reason = (
            "Query intent is pattern detection but no specific typology was identified. "
            "Running full anomaly detection suite; full EDA and network tools skipped."
        )
        return execution_plan, tools_skipped, skip_reason
