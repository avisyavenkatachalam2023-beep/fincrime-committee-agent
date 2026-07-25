"""
src/tools package initialisation.

Exposes all public tool classes for convenient top-level imports:

    from src.tools import (
        EDATool, EDAToolOutput,
        FeatureEngineeringTool,
        BenfordAnalyzer,
        ThresholdClusteringAnalyzer,
        MLAnomalyDetector,
        RuleEngine,
        NetworkAnalysisTool,
        RiskClassifier,
        ExplanationTool,
    )
"""

from src.tools.eda_tool import EDATool, EDAToolOutput
from src.tools.feature_engineering import FeatureEngineeringTool
from src.tools.anomaly_detection.benford import BenfordAnalyzer
from src.tools.anomaly_detection.threshold_clustering import ThresholdClusteringAnalyzer
from src.tools.anomaly_detection.ml_model import MLAnomalyDetector
from src.tools.anomaly_detection.rule_engine import RuleEngine
from src.tools.network_tool import NetworkAnalysisTool
from src.tools.risk_classification import RiskClassifier
from src.tools.explanation import ExplanationTool

__all__ = [
    "EDATool",
    "EDAToolOutput",
    "FeatureEngineeringTool",
    "BenfordAnalyzer",
    "ThresholdClusteringAnalyzer",
    "MLAnomalyDetector",
    "RuleEngine",
    "NetworkAnalysisTool",
    "RiskClassifier",
    "ExplanationTool",
]
