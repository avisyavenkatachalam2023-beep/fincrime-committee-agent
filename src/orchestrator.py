"""
orchestrator.py
---------------
Top-level entry point for the AML Financial Crime Committee Agent.
Accepts a natural-language query, builds an execution plan, runs the
relevant tool subset, and returns a fully formatted result dict.
"""

import os
import sys
import json
import glob
import logging
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Ensure project root is always on sys.path regardless of how the module is invoked
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.query_understanding import QueryUnderstandingTool
from src.planner import DynamicExecutionPlanner
from src.tools.eda_tool import EDATool
from src.tools.feature_engineering import FeatureEngineeringTool
from src.tools.anomaly_detection.benford import BenfordAnalyzer
from src.tools.anomaly_detection.threshold_clustering import ThresholdClusteringAnalyzer
from src.tools.anomaly_detection.ml_model import MLAnomalyDetector
from src.tools.anomaly_detection.rule_engine import RuleEngine
from src.tools.network_tool import NetworkAnalysisTool
from src.tools.risk_classification import RiskClassifier
from src.tools.explanation import ExplanationTool

# Committee agents — use exact class names as defined in each module
from src.committee.transaction_monitoring_agent import TransactionMonitoringAgent
from src.committee.kyc_ubo_agent import KycUboAgent
from src.committee.sanctions_pep_agent import SanctionsPepAgent
from src.committee.network_relationship_agent import NetworkRelationshipAgent
from src.committee.chair_agent import ChairAgent

from src.output_formatter import OutputFormatter


class AMLOrchestrator:
    """Main orchestrator: query -> plan -> execute tools -> formatted output."""

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.qut = QueryUnderstandingTool()
        self.planner = DynamicExecutionPlanner()
        self.formatter = OutputFormatter()

        # Tool instances (lazy-initialised where possible)
        self.eda_tool = EDATool()
        self.fe_tool = FeatureEngineeringTool()
        self.benford = BenfordAnalyzer()
        self.clustering = ThresholdClusteringAnalyzer()
        self.ml_model = MLAnomalyDetector()
        self.rule_engine = RuleEngine()
        self.network_tool = NetworkAnalysisTool()
        self.risk_class = RiskClassifier()
        self.explainer = ExplanationTool()

        self._transactions_df = None
        self._customers_df = None
        self._jurisdictions_df = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, filters: dict = None):
        """Load transaction, customer, and jurisdiction data; apply filters."""
        # Resolve data_dir relative to project root
        data_dir = self.data_dir
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(_PROJECT_ROOT, data_dir)

        # Import load_data from the data package
        sys.path.insert(0, _PROJECT_ROOT)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "load_data", os.path.join(data_dir, "load_data.py")
        )
        ld = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ld)

        self._transactions_df = ld.load_transactions(data_dir)
        self._customers_df = ld.load_customers(data_dir)
        self._jurisdictions_df = ld.load_jurisdictions(data_dir)

        if filters and filters.get('date_range'):
            self._transactions_df = self._apply_date_filter(
                self._transactions_df, filters['date_range']
            )

    def _apply_date_filter(self, df: pd.DataFrame, date_range: str) -> pd.DataFrame:
        """Filter dataframe by date_range string."""
        df = df.copy()
        col = 'timestamp' if 'timestamp' in df.columns else None
        if col is None:
            return df
        df[col] = pd.to_datetime(df[col], errors='coerce')
        max_date = df[col].max()
        if pd.isna(max_date):
            return df
        mapping = {
            'last_7_days': 7,
            'last_30_days': 30,
            'last_90_days': 90,
            'last_365_days': 365,
        }
        if date_range in mapping:
            cutoff = max_date - timedelta(days=mapping[date_range])
            df = df[df[col] >= cutoff]
        elif date_range.startswith('Q') and '_' in date_range:
            parts = date_range.split('_')
            q, year = int(parts[0][1]), int(parts[1])
            q_start = pd.Timestamp(f'{year}-{(q-1)*3+1:02d}-01')
            q_end = q_start + pd.DateOffset(months=3)
            df = df[(df[col] >= q_start) & (df[col] < q_end)]
        return df

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, query: str) -> dict:
        """Parse query, plan, execute tools, return formatted result."""
        parsed_query = self.qut.parse(query)
        parsed_query['user_query'] = query
        execution_summary = self.planner.plan(parsed_query)
        execution_summary['user_query'] = query

        intent = parsed_query.get('intent', 'broad_exploration')
        dispatch = {
            'pattern_detection': self._run_pattern_detection,
            'aggregation_query': self._run_aggregation,
            'single_entity_lookup': self._run_single_entity_lookup,
            'explain_flag': self._run_explain_flag,
            'network_query': self._run_network_query,
            'comparative_query': self._run_comparative,
            'escalation_query': self._run_escalation_query,
            'broad_exploration': self._run_broad_exploration,
        }
        handler = dispatch.get(intent, self._run_broad_exploration)
        try:
            return handler(parsed_query, execution_summary)
        except Exception as exc:
            return self.formatter.format_full_result(
                execution_summary_json=self.formatter.format_execution_summary(execution_summary),
                execution_summary=execution_summary,
                plain_text_answer=f"Error during execution: {exc}",
            )

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _run_pattern_detection(self, parsed_query: dict, execution_summary: dict) -> dict:
        self.load_data(parsed_query.get('filters'))
        target_pattern = parsed_query.get('target_pattern', 'structuring')

        features_df = self.fe_tool.compute_structuring_features(self._transactions_df)
        if len(features_df) > 0 and 'structuring_regularity_score' in features_df.columns:
            # A customer with only 2-4 transactions can score a "perfect"
            # regularity purely by chance (little data for the coefficient-
            # of-variation calc to actually vary), which would otherwise let
            # a statistically meaningless account outrank genuine repeat
            # structuring behaviour. Require a minimum sample size before a
            # customer is eligible to be the top candidate.
            MIN_TXNS_FOR_PATTERN = 5
            candidates = features_df[features_df['transaction_count'] >= MIN_TXNS_FOR_PATTERN]
            if candidates.empty:
                candidates = features_df
            features_df = candidates.sort_values(by='structuring_regularity_score', ascending=False)
        top_customer = (
            features_df.index[0]
            if len(features_df) > 0
            else self._transactions_df['sender_account'].iloc[0]
            if not self._transactions_df.empty
            else 'unknown'
        )

        benford_res = self.benford.analyze_customer(self._transactions_df, top_customer)
        clustering_res = self.clustering.analyze_customer(self._transactions_df, top_customer)

        features_dict = (
            features_df.loc[top_customer].to_dict()
            if top_customer in features_df.index
            else {}
        )
        risk = self.risk_class.classify_customer(
            top_customer, features_dict,
            benford_results=benford_res,
            clustering_results=clustering_res,
        )
        case_file = self._build_case_file(
            top_customer, parsed_query['user_query'], target_pattern,
            features_dict, benford_res, clustering_res, {}, {}, risk,
        )
        committee_res = self._run_committee(case_file, agents=['TransactionMonitoringAgent'])
        explanation = self.explainer.explain(
            top_customer, parsed_query['user_query'], target_pattern,
            risk, benford_res, clustering_res,
        )
        memo = self.formatter.format_risk_memo(
            case_file, risk,
            committee_res['agent_votes'], committee_res['chair_result'],
            explanation,
        )
        top_suspicious = self._build_top_suspicious_list(
            candidate_ids=list(features_df.index),
            metric_label='Structuring regularity score',
            metric_values=(
                features_df['structuring_regularity_score'].to_dict()
                if 'structuring_regularity_score' in features_df.columns else {}
            ),
        )
        return self.formatter.format_full_result(
            execution_summary_json=self.formatter.format_execution_summary(execution_summary),
            execution_summary=execution_summary,
            risk_memo=memo,
            committee_minutes=committee_res['meeting_minutes'],
            explanation=explanation,
            charts={'benford': benford_res.get('chart_path', '')},
            agent_votes=committee_res['agent_votes'],
            chair_result=committee_res['chair_result'],
            top_suspicious=top_suspicious,
        )

    def _run_aggregation(self, parsed_query: dict, execution_summary: dict) -> dict:
        self.load_data(parsed_query.get('filters'))
        rule_res = self.rule_engine.count_threshold_transactions(
            self._transactions_df, max_amount=10000, min_count=10
        )
        table = self.formatter.format_aggregation_result(rule_res, execution_summary)
        return self.formatter.format_full_result(
            execution_summary_json=self.formatter.format_execution_summary(execution_summary),
            execution_summary=execution_summary,
            aggregation_table=table,
            plain_text_answer=table,
        )

    def _run_single_entity_lookup(self, parsed_query: dict, execution_summary: dict) -> dict:
        self.load_data(parsed_query.get('filters'))
        entities = parsed_query.get('entities', [])
        entity = entities[0] if entities else 'unknown'

        features_df = self.fe_tool.compute_all_features(
            self._transactions_df, customer_id=entity
        )
        benford_res = self.benford.analyze_customer(self._transactions_df, entity)
        clustering_res = self.clustering.analyze_customer(self._transactions_df, entity)

        all_features = self.fe_tool.compute_all_features(self._transactions_df)
        if len(all_features) > 0:
            self.ml_model.fit(all_features)

        features_dict = (
            features_df.loc[entity].to_dict()
            if entity in features_df.index
            else {}
        )
        ml_res = self.ml_model.predict_single(features_dict) if len(all_features) > 0 else {}

        risk = self.risk_class.classify_customer(
            entity, features_dict, benford_res, clustering_res, ml_res, {}
        )
        case_file = self._build_case_file(
            entity, parsed_query['user_query'], 'none',
            features_dict, benford_res, clustering_res, ml_res, {}, risk,
        )
        committee_res = self._run_committee(case_file, agents='all')
        explanation = self.explainer.explain(
            entity, parsed_query['user_query'], 'none',
            risk, benford_res, clustering_res, ml_res,
        )
        memo = self.formatter.format_risk_memo(
            case_file, risk,
            committee_res['agent_votes'], committee_res['chair_result'],
            explanation,
        )
        return self.formatter.format_full_result(
            execution_summary_json=self.formatter.format_execution_summary(execution_summary),
            execution_summary=execution_summary,
            risk_memo=memo,
            committee_minutes=committee_res['meeting_minutes'],
            explanation=explanation,
            charts={'benford': benford_res.get('chart_path', '')},
            agent_votes=committee_res['agent_votes'],
            chair_result=committee_res['chair_result'],
        )

    def _run_explain_flag(self, parsed_query: dict, execution_summary: dict) -> dict:
        self.load_data()
        entities = parsed_query.get('entities', [])
        entity = entities[0] if entities else 'unknown'
        explanation = self.explainer.retrieve_and_explain_flag(entity, {})
        return self.formatter.format_full_result(
            execution_summary_json=self.formatter.format_execution_summary(execution_summary),
            execution_summary=execution_summary,
            explanation=explanation,
            plain_text_answer=explanation,
        )

    def _run_network_query(self, parsed_query: dict, execution_summary: dict) -> dict:
        self.load_data(parsed_query.get('filters'))
        entities = parsed_query.get('entities', [])
        entity = entities[0] if entities else None
        net_res = self.network_tool.run(
            self._transactions_df, self._customers_df, focus_customer=entity
        )
        text_res = self.formatter.format_network_result(net_res, execution_summary)

        # The ego-network chart is only produced when a focus entity was
        # requested — it lives under entity_analysis, not at the top level.
        chart_path = net_res.get('entity_analysis', {}).get('chart_path', '')

        # top_hubs is already a genuine ranked list (by betweenness
        # centrality) — reuse the official network-centrality scoring
        # formula from risk_classification.py so hub accounts get a
        # comparable composite score without re-running Benford/clustering
        # (those signals aren't relevant to a pure relationship query).
        top_suspicious = []
        for hub in net_res.get('top_hubs', [])[:10]:
            bc = float(hub.get('betweenness_centrality', 0.0))
            scoring = self.risk_class.score(network_score=min(bc * 2.0, 1.0))
            top_suspicious.append({
                'customer_id': str(hub.get('account_id', 'unknown')),
                'risk_score': scoring['composite_score'],
                'risk_tier': scoring['risk_tier'],
                'key_metric': f"Betweenness centrality: {bc:.4f} (network hub)",
            })

        return self.formatter.format_full_result(
            execution_summary_json=self.formatter.format_execution_summary(execution_summary),
            execution_summary=execution_summary,
            plain_text_answer=text_res,
            charts={'network': chart_path},
            top_suspicious=top_suspicious,
        )

    def _run_comparative(self, parsed_query: dict, execution_summary: dict) -> dict:
        self.load_data(parsed_query.get('filters'))
        comp = self.rule_engine.compare_flagged_vs_unflagged(
            self._transactions_df, 'amount'
        )
        text_res = self.formatter.format_comparison_result(comp, execution_summary)
        return self.formatter.format_full_result(
            execution_summary_json=self.formatter.format_execution_summary(execution_summary),
            execution_summary=execution_summary,
            plain_text_answer=text_res,
        )

    def _run_escalation_query(self, parsed_query: dict, execution_summary: dict) -> dict:
        self.load_data()
        entities = parsed_query.get('entities', [])
        entity = entities[0] if entities else 'unknown'
        risk = {'composite_score': 85, 'risk_tier': 'HIGH', 'score_components': {}}
        case_file = self._build_case_file(
            entity, parsed_query['user_query'], 'none',
            {}, {}, {}, {}, {}, risk,
        )
        committee_res = self._run_committee(case_file, agents='all')
        memo = self.formatter.format_risk_memo(
            case_file, risk,
            committee_res['agent_votes'], committee_res['chair_result'],
            'Escalation review based on prior alerts.',
        )
        return self.formatter.format_full_result(
            execution_summary_json=self.formatter.format_execution_summary(execution_summary),
            execution_summary=execution_summary,
            risk_memo=memo,
            committee_minutes=committee_res['meeting_minutes'],
            agent_votes=committee_res['agent_votes'],
            chair_result=committee_res['chair_result'],
        )

    def _run_broad_exploration(self, parsed_query: dict, execution_summary: dict) -> dict:
        self.load_data(parsed_query.get('filters'))

        eda_res = self.eda_tool.run(self._transactions_df, self._customers_df)
        all_features = self.fe_tool.compute_all_features(self._transactions_df)
        
        if len(all_features) > 0 and 'transaction_count' in all_features.columns:
            all_features = all_features.sort_values(by='transaction_count', ascending=False)

        if len(all_features) == 0:
            return self.formatter.format_full_result(
                execution_summary_json=self.formatter.format_execution_summary(execution_summary),
                execution_summary=execution_summary,
                plain_text_answer='No data available for broad exploration.',
            )

        self.ml_model.fit(all_features)
        top_customer = all_features.index[0]

        benford_res = self.benford.analyze_customer(self._transactions_df, top_customer)
        clustering_res = self.clustering.analyze_customer(self._transactions_df, top_customer)
        net_res = self.network_tool.run(
            self._transactions_df, self._customers_df, focus_customer=top_customer
        )
        # Bulk feature computation zeroes rapid_cash_out_ratio for performance
        # (see feature_engineering.py); recompute it exactly for the one
        # customer that was actually selected.
        features_dict = all_features.loc[top_customer].to_dict()
        features_dict['rapid_cash_out_ratio'] = self.fe_tool.rapid_cash_out_ratio(
            self._transactions_df, top_customer
        )
        ml_res = self.ml_model.predict_single(features_dict)

        risk = self.risk_class.classify_customer(
            top_customer, features_dict,
            benford_res, clustering_res, ml_res,
            net_res.get('entity_analysis', {}),
        )
        case_file = self._build_case_file(
            top_customer, parsed_query['user_query'], 'none',
            features_dict, benford_res, clustering_res, ml_res,
            net_res.get('entity_analysis', {}), risk,
        )
        committee_res = self._run_committee(case_file, agents='all')
        explanation = self.explainer.explain(
            top_customer, parsed_query['user_query'], 'none',
            risk, benford_res, clustering_res, ml_res,
            net_res.get('entity_analysis', {}),
        )
        memo = self.formatter.format_risk_memo(
            case_file, risk,
            committee_res['agent_votes'], committee_res['chair_result'],
            explanation,
        )
        charts = {
            'amount_distribution': eda_res.get('amount_distribution_plot_path', ''),
            'typology': eda_res.get('typology_breakdown_plot_path', ''),
            'benford': benford_res.get('chart_path', ''),
            'clustering': clustering_res.get('chart_path', ''),
        }
        # Shortlist by activity (cheap, avoids scoring all 855k+ accounts),
        # then the helper itself re-ranks the shortlist by actual computed
        # risk score — so the returned order reflects suspicion, not just volume.
        top_suspicious = self._build_top_suspicious_list(
            candidate_ids=list(all_features.index),
            metric_label='Transaction count',
            metric_values=(
                all_features['transaction_count'].to_dict()
                if 'transaction_count' in all_features.columns else {}
            ),
            all_features_df=all_features,
        )
        return self.formatter.format_full_result(
            execution_summary_json=self.formatter.format_execution_summary(execution_summary),
            execution_summary=execution_summary,
            risk_memo=memo,
            committee_minutes=committee_res['meeting_minutes'],
            explanation=explanation,
            charts=charts,
            top_suspicious=top_suspicious,
            # Previously omitted from this handler's return — the committee
            # deliberation ran and its content was baked into risk_memo, but
            # the structured agent_votes/chair_result were never surfaced at
            # the top level, so the Committee tab's vote cards were always
            # empty for broad_exploration despite the committee having voted.
            agent_votes=committee_res['agent_votes'],
            chair_result=committee_res['chair_result'],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_top_suspicious_list(
        self,
        candidate_ids: list,
        metric_label: str = '',
        metric_values: dict = None,
        all_features_df=None,
        top_n: int = 10,
    ) -> list:
        """Build a genuine ranked list of suspicious accounts, not just the
        single case that gets the full deep-dive treatment below.

        Running the full committee (Groq calls per specialist agent) for
        every candidate would be far too slow, so this uses only the cheap
        signals — Benford's Law and threshold clustering, plus the ML
        anomaly model if it's already fitted — to produce a defensible
        composite risk score per candidate. The single top-ranked account
        still separately gets the full committee deliberation and risk memo.

        Args:
            candidate_ids: Ordered list of customer/account IDs (most
                interesting first, per whatever cheap heuristic selected
                them — e.g. activity volume or a pattern-specific score).
            metric_label: Human label for the metric that produced the
                candidate shortlist (shown alongside each row for context).
            metric_values: Optional {id: value} map for that metric.
            all_features_df: Bulk feature DataFrame; enables cheap ML scoring
                via the already-fitted model if provided.
            top_n: How many candidates to score and return.

        Returns:
            List of dicts sorted by composite risk score descending:
            {customer_id, risk_score, risk_tier, key_metric}.
        """
        metric_values = metric_values or {}
        results = []
        for cid in candidate_ids[:top_n]:
            try:
                benford_res = self.benford.analyze_customer(self._transactions_df, cid)
                clustering_res = self.clustering.analyze_customer(self._transactions_df, cid)
                ml_res = {}
                if (
                    all_features_df is not None
                    and cid in all_features_df.index
                    and getattr(self.ml_model, '_is_fitted', False)
                ):
                    ml_res = self.ml_model.predict_single(all_features_df.loc[cid].to_dict())
                risk = self.risk_class.classify_customer(cid, {}, benford_res, clustering_res, ml_res, {})
                entry = {
                    'customer_id': str(cid),
                    'risk_score': risk.get('composite_score', 0),
                    'risk_tier': risk.get('risk_tier', 'LOW'),
                }
                if metric_label and cid in metric_values:
                    entry['key_metric'] = f"{metric_label}: {metric_values[cid]:.4f}"
                results.append(entry)
            except Exception as exc:
                logger.warning("Could not score candidate %s for top-suspicious list: %s", cid, exc)
        results.sort(key=lambda r: r['risk_score'], reverse=True)
        return results

    def _build_case_file(
        self,
        customer_id: str, query: str, pattern: str,
        features: dict, benford_results: dict, clustering_results: dict,
        ml_results: dict, network_results: dict, risk_classification: dict,
        customer_profile: dict = None,
    ) -> dict:
        """Assemble the unified case-file dict consumed by all committee agents."""
        return {
            'customer_id': customer_id,
            'query': query,
            'pattern': pattern,
            'risk_score': risk_classification.get('composite_score', 0),
            'risk_tier': risk_classification.get('risk_tier', 'LOW'),
            'features': features,
            'benford_results': benford_results,
            'clustering_results': clustering_results,
            'ml_results': ml_results,
            'network_results': network_results,
            'customer_profile': customer_profile or {},
        }

    def _run_committee(self, case_file: dict, agents='all') -> dict:
        """Instantiate and run the requested committee agents, then the Chair."""
        votes = []

        run_all = (agents == 'all')

        if run_all or 'TransactionMonitoringAgent' in agents:
            votes.append(TransactionMonitoringAgent().deliberate(case_file))
        if run_all or 'KycUboAgent' in agents:
            votes.append(KycUboAgent().deliberate(case_file))
        if run_all or 'SanctionsPepAgent' in agents:
            votes.append(SanctionsPepAgent().deliberate(case_file))
        if run_all or 'NetworkRelationshipAgent' in agents:
            votes.append(NetworkRelationshipAgent().deliberate(case_file))

        chair_res = ChairAgent().synthesize(case_file, votes)
        return {
            'agent_votes': votes,
            'chair_result': chair_res,
            'meeting_minutes': chair_res.get('meeting_minutes', ''),
        }
