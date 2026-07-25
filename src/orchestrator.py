import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
import glob

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
from src.committee.transaction_monitoring_agent import TransactionMonitoringAgent
from src.committee.kyc_ubo_agent import KYCUBOAgent
from src.committee.sanctions_pep_agent import SanctionsPEPAgent
from src.committee.network_relationship_agent import NetworkRelationshipAgent
from src.committee.chair_agent import ChairAgent
from src.output_formatter import OutputFormatter

class AMLOrchestrator:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.qut = QueryUnderstandingTool()
        self.planner = DynamicExecutionPlanner()
        self.formatter = OutputFormatter()
        
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
        
    def load_data(self, filters: dict = None):
        import data.load_data as ld
        self._transactions_df = ld.load_transactions(self.data_dir)
        self._customers_df = ld.load_customers(self.data_dir)
        self._jurisdictions_df = ld.load_jurisdictions(self.data_dir)
        
        if filters and filters.get('date_range'):
            self._transactions_df = self._apply_date_filter(self._transactions_df, filters['date_range'])
            
    def _apply_date_filter(self, df: pd.DataFrame, date_range: str) -> pd.DataFrame:
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            max_date = df['timestamp'].max()
            if not pd.isna(max_date):
                if date_range == 'last_30_days':
                    df = df[df['timestamp'] >= max_date - timedelta(days=30)]
                elif date_range == 'last_7_days':
                    df = df[df['timestamp'] >= max_date - timedelta(days=7)]
        return df
    
    def run(self, query: str) -> dict:
        parsed_query = self.qut.parse(query)
        parsed_query['user_query'] = query
        execution_summary = self.planner.plan(parsed_query)
        execution_summary['user_query'] = query
        
        intent = parsed_query.get('intent', 'broad_exploration')
        
        if intent == 'pattern_detection':
            return self._run_pattern_detection(parsed_query, execution_summary)
        elif intent == 'aggregation_query':
            return self._run_aggregation(parsed_query, execution_summary)
        elif intent == 'single_entity_lookup':
            return self._run_single_entity_lookup(parsed_query, execution_summary)
        elif intent == 'explain_flag':
            return self._run_explain_flag(parsed_query, execution_summary)
        elif intent == 'network_query':
            return self._run_network_query(parsed_query, execution_summary)
        elif intent == 'comparative_query':
            return self._run_comparative(parsed_query, execution_summary)
        elif intent == 'escalation_query':
            return self._run_escalation_query(parsed_query, execution_summary)
        else:
            return self._run_broad_exploration(parsed_query, execution_summary)
            
    def _run_pattern_detection(self, parsed_query: dict, execution_summary: dict) -> dict:
        self.load_data(parsed_query.get('filters'))
        target_pattern = parsed_query.get('target_pattern')
        
        features_df = self.fe_tool.compute_structuring_features(self._transactions_df)
        
        # Pick top customer to analyze
        if len(features_df) > 0:
            top_customer = features_df.index[0]
        else:
            top_customer = self._transactions_df['sender_account'].iloc[0] if not self._transactions_df.empty else 'unknown'
            
        benford_res = self.benford.analyze_customer(self._transactions_df, top_customer)
        clustering_res = self.clustering.analyze_customer(self._transactions_df, top_customer)
        
        features_dict = features_df.loc[top_customer].to_dict() if top_customer in features_df.index else {}
        
        risk = self.risk_class.classify_customer(
            top_customer, features_dict, benford_results=benford_res, clustering_results=clustering_res
        )
        
        case_file = self._build_case_file(top_customer, parsed_query['user_query'], target_pattern, features_dict, benford_res, clustering_res, {}, {}, risk)
        
        committee_res = self._run_committee(case_file, agents=['TransactionMonitoringAgent'])
        
        explanation = self.explainer.explain(top_customer, parsed_query['user_query'], target_pattern, risk, benford_res, clustering_res)
        
        memo = self.formatter.format_risk_memo(case_file, risk, committee_res['agent_votes'], committee_res['chair_result'], explanation)
        
        return self.formatter.format_full_result(
            execution_summary_json=self.formatter.format_execution_summary(execution_summary),
            execution_summary=execution_summary,
            risk_memo=memo,
            committee_minutes=committee_res['meeting_minutes'],
            explanation=explanation,
            charts={'benford': benford_res.get('chart_path')}
        )
        
    def _run_aggregation(self, parsed_query: dict, execution_summary: dict) -> dict:
        self.load_data(parsed_query.get('filters'))
        
        rule_res = self.rule_engine.count_threshold_transactions(self._transactions_df, max_amount=10000, min_count=10)
        
        table = self.formatter.format_aggregation_result(rule_res, execution_summary)
        
        return self.formatter.format_full_result(
            execution_summary_json=self.formatter.format_execution_summary(execution_summary),
            execution_summary=execution_summary,
            aggregation_table=table,
            plain_text_answer=table
        )
        
    def _run_single_entity_lookup(self, parsed_query: dict, execution_summary: dict) -> dict:
        self.load_data(parsed_query.get('filters'))
        entities = parsed_query.get('entities', [])
        entity = entities[0] if entities else 'unknown'
        
        features_df = self.fe_tool.compute_all_features(self._transactions_df, customer_id=entity)
        
        benford_res = self.benford.analyze_customer(self._transactions_df, entity)
        clustering_res = self.clustering.analyze_customer(self._transactions_df, entity)
        
        # Fit ML on all, predict single
        all_features = self.fe_tool.compute_all_features(self._transactions_df)
        self.ml_model.fit(all_features)
        
        features_dict = features_df.loc[entity].to_dict() if entity in features_df.index else {}
        ml_res = self.ml_model.predict_single(features_dict)
        
        risk = self.risk_class.classify_customer(
            entity, features_dict, benford_res, clustering_res, ml_res, {}
        )
        
        case_file = self._build_case_file(entity, parsed_query['user_query'], 'none', features_dict, benford_res, clustering_res, ml_res, {}, risk)
        committee_res = self._run_committee(case_file, agents='all')
        
        explanation = self.explainer.explain(entity, parsed_query['user_query'], 'none', risk, benford_res, clustering_res, ml_res)
        
        memo = self.formatter.format_risk_memo(case_file, risk, committee_res['agent_votes'], committee_res['chair_result'], explanation)
        
        return self.formatter.format_full_result(
            execution_summary_json=self.formatter.format_execution_summary(execution_summary),
            execution_summary=execution_summary,
            risk_memo=memo,
            committee_minutes=committee_res['meeting_minutes'],
            explanation=explanation,
            charts={'benford': benford_res.get('chart_path')}
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
            plain_text_answer=explanation
        )
        
    def _run_network_query(self, parsed_query: dict, execution_summary: dict) -> dict:
        self.load_data(parsed_query.get('filters'))
        entities = parsed_query.get('entities', [])
        entity = entities[0] if entities else None
        
        net_res = self.network_tool.run(self._transactions_df, self._customers_df, focus_customer=entity)
        
        text_res = self.formatter.format_network_result(net_res, execution_summary)
        
        return self.formatter.format_full_result(
            execution_summary_json=self.formatter.format_execution_summary(execution_summary),
            execution_summary=execution_summary,
            plain_text_answer=text_res,
            charts={'network': net_res.get('chart_path', '')}
        )
        
    def _run_comparative(self, parsed_query: dict, execution_summary: dict) -> dict:
        self.load_data(parsed_query.get('filters'))
        comp = self.rule_engine.compare_flagged_vs_unflagged(self._transactions_df, 'amount')
        
        text_res = self.formatter.format_comparison_result(comp, execution_summary)
        
        return self.formatter.format_full_result(
            execution_summary_json=self.formatter.format_execution_summary(execution_summary),
            execution_summary=execution_summary,
            plain_text_answer=text_res
        )
        
    def _run_escalation_query(self, parsed_query: dict, execution_summary: dict) -> dict:
        self.load_data()
        entities = parsed_query.get('entities', [])
        entity = entities[0] if entities else 'unknown'
        
        risk = {'composite_score': 85, 'risk_tier': 'HIGH'} # Mock retrieved
        case_file = self._build_case_file(entity, parsed_query['user_query'], 'none', {}, {}, {}, {}, {}, risk)
        
        committee_res = self._run_committee(case_file, agents='all')
        
        memo = self.formatter.format_risk_memo(case_file, risk, committee_res['agent_votes'], committee_res['chair_result'], "Escalation review based on prior alerts.")
        
        return self.formatter.format_full_result(
            execution_summary_json=self.formatter.format_execution_summary(execution_summary),
            execution_summary=execution_summary,
            risk_memo=memo,
            committee_minutes=committee_res['meeting_minutes']
        )
        
    def _run_broad_exploration(self, parsed_query: dict, execution_summary: dict) -> dict:
        self.load_data(parsed_query.get('filters'))
        
        eda_res = self.eda_tool.run(self._transactions_df, self._customers_df)
        
        all_features = self.fe_tool.compute_all_features(self._transactions_df)
        self.ml_model.fit(all_features)
        
        # Pick top customer
        top_customer = all_features.index[0] if len(all_features) > 0 else 'unknown'
        benford_res = self.benford.analyze_customer(self._transactions_df, top_customer)
        clustering_res = self.clustering.analyze_customer(self._transactions_df, top_customer)
        net_res = self.network_tool.run(self._transactions_df, self._customers_df, focus_customer=top_customer)
        
        features_dict = all_features.loc[top_customer].to_dict() if top_customer in all_features.index else {}
        ml_res = self.ml_model.predict_single(features_dict)
        
        risk = self.risk_class.classify_customer(
            top_customer, features_dict, benford_res, clustering_res, ml_res, net_res.get('entity_analysis', {})
        )
        
        case_file = self._build_case_file(top_customer, parsed_query['user_query'], 'none', features_dict, benford_res, clustering_res, ml_res, net_res.get('entity_analysis', {}), risk)
        
        committee_res = self._run_committee(case_file, agents='all')
        
        explanation = self.explainer.explain(top_customer, parsed_query['user_query'], 'none', risk, benford_res, clustering_res, ml_res, net_res.get('entity_analysis', {}))
        
        memo = self.formatter.format_risk_memo(case_file, risk, committee_res['agent_votes'], committee_res['chair_result'], explanation)
        
        charts = {
            'amount_dist': eda_res.get('amount_distribution_plot_path'),
            'typology_dist': eda_res.get('typology_breakdown_plot_path'),
            'benford': benford_res.get('chart_path'),
            'clustering': clustering_res.get('chart_path')
        }
        
        return self.formatter.format_full_result(
            execution_summary_json=self.formatter.format_execution_summary(execution_summary),
            execution_summary=execution_summary,
            risk_memo=memo,
            committee_minutes=committee_res['meeting_minutes'],
            explanation=explanation,
            charts=charts
        )
        
    def _build_case_file(self, customer_id: str, query: str, pattern: str,
                          features: dict, benford_results: dict, clustering_results: dict,
                          ml_results: dict, network_results: dict, risk_classification: dict,
                          customer_profile: dict = None) -> dict:
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
            'customer_profile': customer_profile or {}
        }
        
    def _run_committee(self, case_file: dict, agents: list = 'all') -> dict:
        votes = []
        if agents == 'all' or 'TransactionMonitoringAgent' in agents:
            votes.append(TransactionMonitoringAgent().deliberate(case_file))
        if agents == 'all' or 'KYCUBOAgent' in agents:
            votes.append(KYCUBOAgent().deliberate(case_file))
        if agents == 'all' or 'SanctionsPEPAgent' in agents:
            votes.append(SanctionsPEPAgent().deliberate(case_file))
        if agents == 'all' or 'NetworkRelationshipAgent' in agents:
            votes.append(NetworkRelationshipAgent().deliberate(case_file))
            
        chair_res = ChairAgent().synthesize(case_file, votes)
        
        return {
            'agent_votes': votes,
            'chair_result': chair_res,
            'meeting_minutes': chair_res.get('meeting_minutes', '')
        }
