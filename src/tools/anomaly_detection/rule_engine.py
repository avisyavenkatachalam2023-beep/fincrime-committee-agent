"""
rule_engine.py
--------------
Pattern-matching rule engine for the AML Financial Crime Committee Agent.

Supports structured natural-language-style rule queries over the transactions
DataFrame without requiring the caller to write pandas directly.  Intended to
be wired into the agent's tool-dispatch layer.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RuleEngine:
    """Execute structured rule queries over AML transaction data.

    Supported query patterns (case-insensitive):
    - ``count(transactions where amount < X) >= N``
      → Find customers with ≥ N transactions under amount X.
    - ``compare(metric) between flagged and non-flagged``
      → Statistical comparison of a metric between suspicious / non-suspicious
        transactions.
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self, transactions_df: pd.DataFrame, rule_query: str
    ) -> dict[str, Any]:
        """Parse and execute a rule query against the transactions DataFrame.

        Args:
            transactions_df: Full transactions DataFrame.
            rule_query: Natural-language-style rule string.  See class docstring
                for supported patterns.

        Returns:
            Dictionary with keys ``results`` (DataFrame or dict) and
            ``explanation`` (plain-English summary of what was found).
        """
        query = rule_query.strip().lower()

        try:
            count_match = re.search(
                r"count\(transactions\s+where\s+amount\s*<\s*([\d,\.]+)\)\s*>=\s*(\d+)",
                query,
            )
            if count_match:
                max_amount = float(count_match.group(1).replace(",", ""))
                min_count = int(count_match.group(2))
                results_df = self.count_threshold_transactions(
                    transactions_df, max_amount, min_count
                )
                explanation = (
                    f"Rule: transactions under ${max_amount:,.0f} occurring ≥ {min_count} times. "
                    f"Found {len(results_df)} customer(s) matching this pattern."
                )
                if not results_df.empty:
                    top = results_df.iloc[0]
                    explanation += (
                        f" Top offender: {top['customer_id']} with "
                        f"{top['count']} transactions totalling ${top['total_amount']:,.2f}."
                    )
                return {"results": results_df, "explanation": explanation}

            compare_match = re.search(
                r"compare\((\w+)\)\s+between\s+flagged\s+and\s+non.?flagged",
                query,
            )
            if compare_match:
                metric = compare_match.group(1)
                comparison = self.compare_flagged_vs_unflagged(transactions_df, metric)
                flagged_mean = comparison.get("flagged_mean", float("nan"))
                unflagged_mean = comparison.get("unflagged_mean", float("nan"))
                explanation = (
                    f"Comparison of '{metric}' between flagged and non-flagged transactions. "
                    f"Flagged mean: {flagged_mean:.4f}, "
                    f"Non-flagged mean: {unflagged_mean:.4f}. "
                    f"Ratio: {comparison.get('ratio', float('nan')):.3f}x."
                )
                return {"results": comparison, "explanation": explanation}

            # Fallback: unrecognised query — return descriptive error
            logger.warning("Unrecognised rule query: '%s'", rule_query)
            return {
                "results": {},
                "explanation": (
                    f"Rule query not recognised: '{rule_query}'. "
                    "Supported patterns: "
                    "'count(transactions where amount < X) >= N' or "
                    "'compare(metric) between flagged and non-flagged'."
                ),
            }

        except Exception as exc:
            logger.error("RuleEngine.run() failed for query '%s': %s", rule_query, exc, exc_info=True)
            return {
                "results": {},
                "explanation": f"Rule execution failed: {exc}",
            }

    # ------------------------------------------------------------------
    # Specialised rule implementations
    # ------------------------------------------------------------------

    def count_threshold_transactions(
        self,
        transactions_df: pd.DataFrame,
        max_amount: float,
        min_count: int,
    ) -> pd.DataFrame:
        """Find customers with ≥ min_count transactions under max_amount.

        Args:
            transactions_df: Full transactions DataFrame.
            max_amount: Upper bound on transaction amount (exclusive).
            min_count: Minimum number of qualifying transactions.

        Returns:
            DataFrame with columns:
            customer_id, count, total_amount, avg_amount,
            first_txn, last_txn — sorted by count descending.
        """
        sub = transactions_df[transactions_df["amount"] < max_amount].copy()

        if sub.empty:
            return pd.DataFrame(
                columns=["customer_id", "count", "total_amount", "avg_amount", "first_txn", "last_txn"]
            )

        agg = (
            sub.groupby("sender_account")
            .agg(
                count=("amount", "count"),
                total_amount=("amount", "sum"),
                avg_amount=("amount", "mean"),
                first_txn=("timestamp", "min"),
                last_txn=("timestamp", "max"),
            )
            .reset_index()
            .rename(columns={"sender_account": "customer_id"})
        )

        qualifying = agg[agg["count"] >= min_count].sort_values("count", ascending=False)
        qualifying = qualifying.reset_index(drop=True)
        qualifying["total_amount"] = qualifying["total_amount"].round(2)
        qualifying["avg_amount"] = qualifying["avg_amount"].round(2)
        return qualifying

    def compare_flagged_vs_unflagged(
        self,
        transactions_df: pd.DataFrame,
        metric: str,
    ) -> dict[str, Any]:
        """Compare a numeric metric between suspicious and non-suspicious transactions.

        Args:
            transactions_df: Full transactions DataFrame (must contain
                ``is_suspicious`` column).
            metric: Column name of the numeric metric to compare.  If the
                column doesn't exist, ``amount`` is used as a fallback.

        Returns:
            Dictionary with keys:
            metric, flagged_mean, unflagged_mean, flagged_median,
            unflagged_median, flagged_count, unflagged_count, ratio,
            interpretation.
        """
        if "is_suspicious" not in transactions_df.columns:
            return {
                "metric": metric,
                "error": "is_suspicious column not found in transactions_df.",
            }

        # Resolve the metric column — fall back to 'amount' if not present
        if metric not in transactions_df.columns:
            logger.warning(
                "Metric column '%s' not found; falling back to 'amount'.", metric
            )
            metric = "amount"

        flagged = transactions_df[transactions_df["is_suspicious"] == True][metric].dropna()
        unflagged = transactions_df[transactions_df["is_suspicious"] == False][metric].dropna()

        flagged_mean = float(flagged.mean()) if not flagged.empty else 0.0
        unflagged_mean = float(unflagged.mean()) if not unflagged.empty else 0.0
        flagged_median = float(flagged.median()) if not flagged.empty else 0.0
        unflagged_median = float(unflagged.median()) if not unflagged.empty else 0.0

        ratio = (
            flagged_mean / unflagged_mean
            if unflagged_mean != 0
            else float("inf")
        )

        if ratio > 2.0:
            interp = (
                f"Flagged transactions have {ratio:.1f}x higher '{metric}' than non-flagged — "
                "strongly suggestive of a structural difference."
            )
        elif ratio > 1.2:
            interp = (
                f"Flagged transactions have {ratio:.1f}x higher '{metric}' — modestly elevated."
            )
        else:
            interp = (
                f"'{metric}' shows little difference between flagged and non-flagged groups "
                f"(ratio: {ratio:.2f})."
            )

        return {
            "metric": metric,
            "flagged_mean": round(flagged_mean, 4),
            "unflagged_mean": round(unflagged_mean, 4),
            "flagged_median": round(flagged_median, 4),
            "unflagged_median": round(unflagged_median, 4),
            "flagged_count": int(len(flagged)),
            "unflagged_count": int(len(unflagged)),
            "ratio": round(ratio, 4) if not np.isinf(ratio) else None,
            "interpretation": interp,
        }
