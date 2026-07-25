"""
feature_engineering.py
-----------------------
Computes per-customer behavioural features used by downstream anomaly
detectors and risk classifiers in the AML Financial Crime Committee Agent.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineeringTool:
    """Compute AML-relevant behavioural features for every customer.

    All feature methods accept the full transactions DataFrame plus a
    ``customer_id`` string so that features can be computed either
    individually or in bulk via ``compute_all_features``.
    """

    # ------------------------------------------------------------------
    # Bulk computation
    # ------------------------------------------------------------------

    def compute_all_features(
        self,
        transactions_df: pd.DataFrame,
        customer_id: Optional[str] = None,
        window_days: int = 30,
    ) -> pd.DataFrame:
        """Compute the full feature matrix for all (or one) customers.

        Args:
            transactions_df: Full transactions DataFrame.
            customer_id: If provided, restrict computation to this customer only.
            window_days: Look-back window in days for velocity / rolling metrics.

        Returns:
            DataFrame with one row per customer and columns:
            customer_id, transaction_count, avg_amount, velocity,
            rolling_sum_30d, sub_threshold_count, structuring_regularity_score,
            amount_deviation_zscore, rapid_cash_out_ratio.
        """
        if customer_id is not None:
            cids = [customer_id]
        else:
            # Collect unique senders
            cids = transactions_df["sender_account"].dropna().unique().tolist()

        rows: list[dict] = []
        for cid in cids:
            try:
                rows.append(self._compute_for_customer(transactions_df, cid, window_days))
            except Exception as exc:
                logger.warning("Feature computation failed for %s: %s", cid, exc)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def compute_structuring_features(
        self,
        transactions_df: pd.DataFrame,
        customer_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compute structuring-specific features only.

        Subset: sub_threshold_count, structuring_regularity_score,
        rolling_sum_30d, velocity.

        Args:
            transactions_df: Full transactions DataFrame.
            customer_id: If provided, restrict to this customer.

        Returns:
            DataFrame with structuring features, one row per customer.
        """
        if customer_id is not None:
            cids = [customer_id]
        else:
            cids = transactions_df["sender_account"].dropna().unique().tolist()

        rows: list[dict] = []
        for cid in cids:
            try:
                cust_txns = self._sender_txns(transactions_df, cid)
                rows.append(
                    {
                        "customer_id": cid,
                        "sub_threshold_count": self.sub_threshold_count(transactions_df, cid),
                        "structuring_regularity_score": self.structuring_regularity_score(
                            transactions_df, cid
                        ),
                        "rolling_sum_30d": self.rolling_sum(transactions_df, cid, 30),
                        "velocity": self.velocity(transactions_df, cid, 30),
                        "transaction_count": int(len(cust_txns)),
                    }
                )
            except Exception as exc:
                logger.warning("Structuring feature failed for %s: %s", cid, exc)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Individual feature methods
    # ------------------------------------------------------------------

    def transaction_frequency(
        self, txns: pd.DataFrame, customer_id: str, window_days: int
    ) -> float:
        """Compute transactions per day for a customer within a rolling window.

        Args:
            txns: Full transactions DataFrame.
            customer_id: Sender account identifier.
            window_days: Number of days in the rolling look-back window.

        Returns:
            Float representing average transactions per calendar day.
        """
        cust_txns = self._sender_txns(txns, customer_id)
        if cust_txns.empty:
            return 0.0
        cutoff = cust_txns["timestamp"].max() - timedelta(days=window_days)
        windowed = cust_txns[cust_txns["timestamp"] >= cutoff]
        if window_days == 0:
            return 0.0
        return float(len(windowed) / window_days)

    def rolling_sum(
        self, txns: pd.DataFrame, customer_id: str, window_days: int
    ) -> float:
        """Compute total amount sent by a customer in the rolling window.

        Args:
            txns: Full transactions DataFrame.
            customer_id: Sender account identifier.
            window_days: Number of days in the look-back window.

        Returns:
            Sum of transaction amounts in the window.
        """
        cust_txns = self._sender_txns(txns, customer_id)
        if cust_txns.empty:
            return 0.0
        cutoff = cust_txns["timestamp"].max() - timedelta(days=window_days)
        return float(cust_txns.loc[cust_txns["timestamp"] >= cutoff, "amount"].sum())

    def amount_deviation_from_baseline(
        self, txns: pd.DataFrame, customer_id: str
    ) -> float:
        """Compute z-score of a customer's average amount relative to all senders.

        A high positive value indicates the customer transacts in amounts far
        above the population mean, a red flag for layering.

        Args:
            txns: Full transactions DataFrame.
            customer_id: Sender account identifier.

        Returns:
            Z-score float.  Returns 0.0 if standard deviation is zero.
        """
        cust_txns = self._sender_txns(txns, customer_id)
        if cust_txns.empty:
            return 0.0
        customer_avg = cust_txns["amount"].mean()
        global_mean = txns["amount"].mean()
        global_std = txns["amount"].std()
        if global_std == 0:
            return 0.0
        return float((customer_avg - global_mean) / global_std)

    def velocity(
        self, txns: pd.DataFrame, customer_id: str, window_days: int
    ) -> float:
        """Alias for transaction_frequency – transactions per day in window.

        Args:
            txns: Full transactions DataFrame.
            customer_id: Sender account identifier.
            window_days: Rolling look-back window in days.

        Returns:
            Float transactions per day.
        """
        return self.transaction_frequency(txns, customer_id, window_days)

    def rapid_cash_out_ratio(
        self, txns: pd.DataFrame, customer_id: str, hours: int = 48
    ) -> float:
        """Fraction of received amounts re-transmitted within N hours.

        A high ratio signals rapid layering behaviour where funds are received
        and immediately forwarded.

        Args:
            txns: Full transactions DataFrame.
            customer_id: Account identifier (used as both sender and receiver).
            hours: Look-ahead window in hours after each receipt.

        Returns:
            Float in [0, 1].  Returns 0.0 if no inbound transactions exist.
        """
        inbound = txns[txns["receiver_account"] == customer_id].copy()
        outbound = txns[txns["sender_account"] == customer_id].copy()

        if inbound.empty or outbound.empty:
            return 0.0

        inbound = inbound.sort_values("timestamp")
        outbound = outbound.sort_values("timestamp")

        window = timedelta(hours=hours)
        rapid_outflow: float = 0.0

        for _, in_row in inbound.iterrows():
            t_recv = in_row["timestamp"]
            mask = (outbound["timestamp"] >= t_recv) & (
                outbound["timestamp"] <= t_recv + window
            )
            rapid_outflow += outbound.loc[mask, "amount"].sum()

        total_inbound = float(inbound["amount"].sum())
        if total_inbound == 0:
            return 0.0
        return float(min(rapid_outflow / total_inbound, 1.0))

    def sub_threshold_count(
        self,
        txns: pd.DataFrame,
        customer_id: str,
        threshold: float = 10_000.0,
        window_days: int = 30,
    ) -> int:
        """Count transactions in the 85–99.9% band just below the reporting threshold.

        Structuring involves deliberately splitting deposits to stay just under
        regulatory reporting thresholds.

        Args:
            txns: Full transactions DataFrame.
            customer_id: Sender account identifier.
            threshold: Reporting threshold (default: USD 10,000).
            window_days: Rolling look-back window in days.

        Returns:
            Integer count of qualifying transactions.
        """
        cust_txns = self._sender_txns(txns, customer_id)
        if cust_txns.empty:
            return 0
        cutoff = cust_txns["timestamp"].max() - timedelta(days=window_days)
        windowed = cust_txns[cust_txns["timestamp"] >= cutoff]
        low = threshold * 0.85
        high = threshold * 0.999
        return int(((windowed["amount"] >= low) & (windowed["amount"] < high)).sum())

    def structuring_regularity_score(
        self, txns: pd.DataFrame, customer_id: str
    ) -> float:
        """Score how mechanically regular a customer's transactions are (0–1).

        Genuine structuring is characterised by near-identical amounts and very
        regular inter-transaction intervals.  The score combines:
        - Inverse coefficient of variation (CV) of inter-transaction times.
        - Inverse CV of transaction amounts.

        A score close to 1 means highly regular (suspicious); close to 0 means
        irregular (normal).

        Args:
            txns: Full transactions DataFrame.
            customer_id: Sender account identifier.

        Returns:
            Float in [0, 1].
        """
        cust_txns = self._sender_txns(txns, customer_id).sort_values("timestamp")
        if len(cust_txns) < 3:
            return 0.0

        # Inter-transaction time in seconds
        times = cust_txns["timestamp"].astype(np.int64) // 10**9
        gaps = np.diff(times.values).astype(float)
        amount_vals = cust_txns["amount"].values.astype(float)

        def inv_cv(arr: np.ndarray) -> float:
            """Return 1 - CV, clipped to [0,1].  CV = std/mean."""
            mean = arr.mean()
            if mean == 0:
                return 0.0
            cv = arr.std() / mean
            return float(np.clip(1.0 - cv, 0.0, 1.0))

        time_regularity = inv_cv(gaps) if len(gaps) >= 2 else 0.0
        amount_regularity = inv_cv(amount_vals)

        return float((time_regularity + amount_regularity) / 2.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sender_txns(self, txns: pd.DataFrame, customer_id: str) -> pd.DataFrame:
        """Filter the full DataFrame to rows where sender_account == customer_id.

        Args:
            txns: Full transactions DataFrame.
            customer_id: Sender account to filter on.

        Returns:
            Filtered DataFrame.
        """
        df = txns[txns["sender_account"] == customer_id].copy()
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def _compute_for_customer(
        self, txns: pd.DataFrame, customer_id: str, window_days: int
    ) -> dict:
        """Compute the complete feature dict for one customer.

        Args:
            txns: Full transactions DataFrame.
            customer_id: Sender account identifier.
            window_days: Rolling look-back window in days.

        Returns:
            Dictionary of feature values for the customer.
        """
        cust_txns = self._sender_txns(txns, customer_id)
        transaction_count = int(len(cust_txns))
        avg_amount = float(cust_txns["amount"].mean()) if transaction_count > 0 else 0.0

        return {
            "customer_id": customer_id,
            "transaction_count": transaction_count,
            "avg_amount": avg_amount,
            "velocity": self.velocity(txns, customer_id, window_days),
            "rolling_sum_30d": self.rolling_sum(txns, customer_id, 30),
            "sub_threshold_count": self.sub_threshold_count(txns, customer_id),
            "structuring_regularity_score": self.structuring_regularity_score(txns, customer_id),
            "amount_deviation_zscore": self.amount_deviation_from_baseline(txns, customer_id),
            "rapid_cash_out_ratio": self.rapid_cash_out_ratio(txns, customer_id),
        }
