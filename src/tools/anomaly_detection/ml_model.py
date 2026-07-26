"""
ml_model.py
-----------
Machine-learning based anomaly detection using Isolation Forest for the AML
Financial Crime Committee Agent.

Isolation Forest isolates anomalies by recursively partitioning the feature
space with random splits.  Anomalous observations require fewer splits to
isolate, yielding a lower anomaly score.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MLAnomalyDetector:
    """Wrap scikit-learn's Isolation Forest for AML feature-based anomaly detection.

    Usage pattern::

        detector = MLAnomalyDetector(contamination=0.1)
        detector.fit(features_df)
        scored_df = detector.predict(features_df)
    """

    def __init__(self, contamination: float = 0.1) -> None:
        """Initialise the detector with the specified contamination rate.

        Args:
            contamination: Expected proportion of anomalies in the dataset
                (passed directly to IsolationForest).  Typically set to a
                conservative value such as 0.05–0.15 for AML use cases.
        """
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
            max_samples="auto",
        )
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = [
            "transaction_count",
            "avg_amount",
            "velocity",
            "sub_threshold_count",
            "structuring_regularity_score",
            "rolling_sum_30d",
        ]
        self._is_fitted: bool = False
        # Decision-function range observed at fit() time, used to normalise
        # scores to [0, 1] consistently for both batch and single-row
        # predictions (see predict()/predict_single() docstrings for why this
        # can't just be recomputed from whatever batch is being scored).
        self._train_score_min: float = 0.0
        self._train_score_max: float = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, features_df: pd.DataFrame) -> "MLAnomalyDetector":
        """Fit the scaler and Isolation Forest on the provided feature matrix.

        Missing feature columns are filled with zero.  The method is
        idempotent — calling fit() again resets the model.

        Args:
            features_df: DataFrame with at least one of the expected feature
                columns.  Should contain one row per customer.

        Returns:
            Self, to allow chaining.
        """
        X = self._extract_features(features_df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._is_fitted = True

        # Capture the training population's decision_function range so later
        # single-row scoring has something meaningful to normalise against.
        train_scores = self.model.decision_function(X_scaled)
        self._train_score_min = float(train_scores.min())
        self._train_score_max = float(train_scores.max())

        logger.info(
            "MLAnomalyDetector fitted on %d samples with %d features.",
            X.shape[0],
            X.shape[1],
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Score all rows in features_df for anomalousness.

        Two columns are appended to a copy of the input DataFrame:
        - ``ml_anomaly_score``: float in [0, 1], where 1 is most anomalous.
        - ``ml_is_outlier``: bool, True if Isolation Forest predicts outlier.

        Note that Isolation Forest returns ``-1`` for outliers and ``1`` for
        normal observations; this method converts that convention to a boolean.

        Args:
            features_df: DataFrame with customer feature columns.

        Returns:
            Copy of features_df with ``ml_anomaly_score`` and
            ``ml_is_outlier`` columns added.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "MLAnomalyDetector has not been fitted. Call fit() first."
            )
        result_df = features_df.copy()
        X = self._extract_features(features_df)
        X_scaled = self.scaler.transform(X)

        # Raw decision scores (more negative = more anomalous)
        raw_scores = self.model.decision_function(X_scaled)
        # IsolationForest labels: -1 = outlier, 1 = inlier
        labels = self.model.predict(X_scaled)

        # Normalise raw scores to [0, 1] where 1 = most anomalous, using the
        # range observed at fit() time rather than the range of *this* batch.
        # This matters because predict()/predict_single() are frequently
        # called with a single row (one customer) — a batch of 1 always has
        # min == max, which would force every single-row score to 0.0
        # regardless of how anomalous that customer actually is.
        min_score, max_score = self._train_score_min, self._train_score_max
        if max_score - min_score > 0:
            anomaly_scores = 1.0 - (raw_scores - min_score) / (max_score - min_score)
            anomaly_scores = np.clip(anomaly_scores, 0.0, 1.0)
        else:
            anomaly_scores = np.zeros(len(raw_scores))

        result_df["ml_anomaly_score"] = anomaly_scores.round(4)
        result_df["ml_is_outlier"] = labels == -1

        return result_df

    def predict_single(self, customer_features: dict[str, Any]) -> dict:
        """Score a single customer's feature dictionary.

        Args:
            customer_features: Dictionary mapping feature names to values.
                Missing features default to 0.

        Returns:
            Dictionary with keys:
            ml_anomaly_score (float), ml_is_outlier (bool),
            features_used (dict).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "MLAnomalyDetector has not been fitted. Call fit() first."
            )
        row_df = pd.DataFrame([customer_features])
        scored_df = self.predict(row_df)

        return {
            "ml_anomaly_score": float(scored_df["ml_anomaly_score"].iloc[0]),
            "ml_is_outlier": bool(scored_df["ml_is_outlier"].iloc[0]),
            "features_used": {
                col: float(row_df[col].iloc[0]) if col in row_df.columns else 0.0
                for col in self.feature_columns
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Extract and align the expected feature columns from the DataFrame.

        Columns missing from the input are filled with zero so that the model
        always receives the correct number of feature dimensions.

        Args:
            features_df: Feature DataFrame (any columns).

        Returns:
            2-D numpy array of shape (n_samples, n_features).
        """
        data: dict[str, pd.Series] = {}
        for col in self.feature_columns:
            if col in features_df.columns:
                data[col] = features_df[col].fillna(0.0)
            else:
                data[col] = pd.Series(np.zeros(len(features_df)), index=features_df.index)
        return pd.DataFrame(data).values.astype(float)
