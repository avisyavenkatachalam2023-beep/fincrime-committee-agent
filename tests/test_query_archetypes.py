import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.query_understanding import QueryUnderstandingTool
from src.planner import DynamicExecutionPlanner

def test_structuring_pattern_detection():
    qut = QueryUnderstandingTool()
    planner = DynamicExecutionPlanner()
    
    query = 'Find structuring patterns in the last 30 days'
    parsed = qut.parse(query)
    
    assert parsed['intent'] == 'pattern_detection'
    assert parsed['target_pattern'] == 'structuring'
    assert parsed['filters'].get('date_range') == 'last_30_days'
    
    plan = planner.plan(parsed)
    assert any('benford' in step for step in plan['execution_plan'])
    assert 'full_eda' in plan['tools_skipped']
    assert 'ml_isolation_forest' in plan['tools_skipped']

def test_aggregation_threshold_query():
    qut = QueryUnderstandingTool()
    planner = DynamicExecutionPlanner()
    
    query = 'Which customers made 10+ transactions under $10,000?'
    parsed = qut.parse(query)
    
    assert parsed['intent'] == 'aggregation_query'
    plan = planner.plan(parsed)
    assert any('rule_engine' in step for step in plan['execution_plan'])
    assert 'ml_isolation_forest' in plan['tools_skipped']
    assert 'committee' in plan['tools_skipped']

def test_single_entity_lookup():
    qut = QueryUnderstandingTool()
    planner = DynamicExecutionPlanner()
    
    query = 'Is customer ID 4521 suspicious?'
    parsed = qut.parse(query)
    
    assert parsed['intent'] == 'single_entity_lookup'
    assert '4521' in str(parsed['entities']) or 'customer_4521' in parsed['entities']
    
    plan = planner.plan(parsed)
    assert 'full_eda' in plan['tools_skipped']
    assert any('entity_scoped' in step for step in plan['execution_plan'])

def test_broad_exploration():
    qut = QueryUnderstandingTool()
    planner = DynamicExecutionPlanner()
    
    query = 'Analyse this dataset for suspicious activity'
    parsed = qut.parse(query)
    
    assert parsed['intent'] == 'broad_exploration'
    plan = planner.plan(parsed)
    assert len(plan['tools_skipped']) == 0
    assert any('eda_tool' in step for step in plan['execution_plan'])

def test_explain_flag():
    qut = QueryUnderstandingTool()
    planner = DynamicExecutionPlanner()
    
    query = 'Why was transaction TXN_88213 flagged?'
    parsed = qut.parse(query)
    
    assert parsed['intent'] == 'explain_flag'
    plan = planner.plan(parsed)
    assert 'anomaly_detection' in plan['tools_skipped']
    assert any('retrieve_and_explain' in step for step in plan['execution_plan'])

def test_network_query():
    qut = QueryUnderstandingTool()
    planner = DynamicExecutionPlanner()
    
    query = 'Who is coordinating transactions with customer 4521?'
    parsed = qut.parse(query)
    
    assert parsed['intent'] == 'network_query'
    plan = planner.plan(parsed)
    assert any('network_tool' in step for step in plan['execution_plan'])
    assert 'benford' in plan['tools_skipped']

if __name__ == '__main__':
    print("Running QUERY ARCHETYPE TEST SUITE...")
    
    tests = [
        test_structuring_pattern_detection,
        test_aggregation_threshold_query,
        test_single_entity_lookup,
        test_broad_exploration,
        test_explain_flag,
        test_network_query
    ]
    
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__} PASSED")
        except Exception as e:
            print(f"[FAIL] {t.__name__} FAILED: {e}")
    print("Done.")
