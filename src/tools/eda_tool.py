"""
eda_tool.py
-----------
Exploratory Data Analysis tool for the AML Financial Crime Committee Agent.
Produces statistical summaries and visualisation charts for transaction and
customer datasets.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import TypedDict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CHARTS_DIR = Path("data/charts")


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class EDAToolOutput(TypedDict):
    """Typed schema for EDAToolOutput returned by EDATool.run()."""

    missing_value_summary: dict
    distribution_stats: dict
    country_breakdown: dict
    segment_breakdown: dict
    typology_breakdown: dict
    correlation_heatmap_path: Optional[str]
    amount_distribution_plot_path: Optional[str]
    country_pie_chart_path: Optional[str]
    typology_bar_chart_path: Optional[str]
    suspicious_rate: float


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

class EDATool:
    """Broad exploratory analysis over the full transactions and customers dataset.

    Intended to be called for the ``broad_exploration`` intent only.  Produces
    a rich statistics dictionary plus three PNG charts saved under
    ``data/charts/``.
    """

    def __init__(self, charts_dir: str | Path = CHARTS_DIR) -> None:
        """Initialise the tool, creating the charts output directory if needed.

        Args:
            charts_dir: Filesystem path where PNG charts are stored.
        """
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame,
    ) -> EDAToolOutput:
        """Execute the full EDA pipeline and return structured results.

        Args:
            transactions_df: Full transactions DataFrame matching the canonical schema.
            customers_df: Full customers DataFrame matching the canonical schema.

        Returns:
            EDAToolOutput dictionary with statistics and chart file paths.
        """
        try:
            missing_value_summary = self._missing_value_summary(transactions_df, customers_df)
            distribution_stats = self._distribution_stats(transactions_df)
            country_breakdown = self._country_breakdown(transactions_df)
            segment_breakdown = self._segment_breakdown(customers_df)
            typology_breakdown = self._typology_breakdown(transactions_df)
            suspicious_rate = self._suspicious_rate(transactions_df)

            amount_dist_path = self._plot_amount_distribution(transactions_df)
            country_pie_path = self._plot_country_pie(transactions_df)
            typology_bar_path = self._plot_typology_bar(transactions_df)
            heatmap_path = self._plot_correlation_heatmap(transactions_df)

            return EDAToolOutput(
                missing_value_summary=missing_value_summary,
                distribution_stats=distribution_stats,
                country_breakdown=country_breakdown,
                segment_breakdown=segment_breakdown,
                typology_breakdown=typology_breakdown,
                correlation_heatmap_path=heatmap_path,
                amount_distribution_plot_path=amount_dist_path,
                country_pie_chart_path=country_pie_path,
                typology_bar_chart_path=typology_bar_path,
                suspicious_rate=suspicious_rate,
            )
        except Exception as exc:
            logger.error("EDATool.run() failed: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def _missing_value_summary(
        self,
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame,
    ) -> dict:
        """Return per-column missing value counts for both dataframes.

        Args:
            transactions_df: Transactions data.
            customers_df: Customers data.

        Returns:
            Dictionary with keys ``transactions`` and ``customers``, each
            mapping column names to their null count.
        """
        tx_nulls = transactions_df.isnull().sum()
        cx_nulls = customers_df.isnull().sum()
        return {
            "transactions": tx_nulls[tx_nulls > 0].to_dict(),
            "customers": cx_nulls[cx_nulls > 0].to_dict(),
            "transactions_total_rows": int(len(transactions_df)),
            "customers_total_rows": int(len(customers_df)),
        }

    def _distribution_stats(self, transactions_df: pd.DataFrame) -> dict:
        """Compute descriptive statistics for the ``amount`` column.

        Args:
            transactions_df: Transactions data.

        Returns:
            Dictionary with mean, median, std, min, max, skewness, and
            percentile values for transaction amounts.
        """
        amounts = transactions_df["amount"].dropna()
        return {
            "mean": float(amounts.mean()),
            "median": float(amounts.median()),
            "std": float(amounts.std()),
            "min": float(amounts.min()),
            "max": float(amounts.max()),
            "skewness": float(amounts.skew()),
            "kurtosis": float(amounts.kurtosis()),
            "p25": float(amounts.quantile(0.25)),
            "p75": float(amounts.quantile(0.75)),
            "p90": float(amounts.quantile(0.90)),
            "p95": float(amounts.quantile(0.95)),
            "p99": float(amounts.quantile(0.99)),
            "total_transactions": int(len(transactions_df)),
        }

    def _country_breakdown(self, transactions_df: pd.DataFrame) -> dict:
        """Count transactions by sender country.

        Args:
            transactions_df: Transactions data.

        Returns:
            Dictionary mapping country codes to transaction counts and
            fraction of total.
        """
        counts = (
            transactions_df["sender_country"]
            .value_counts()
            .to_dict()
        )
        total = len(transactions_df)
        return {
            country: {"count": int(cnt), "fraction": round(cnt / total, 4)}
            for country, cnt in counts.items()
        }

    def _segment_breakdown(self, customers_df: pd.DataFrame) -> dict:
        """Break down customer counts by income band and PEP/high-risk flags.

        Args:
            customers_df: Customers data.

        Returns:
            Dictionary with income band counts, PEP count, and high-risk
            jurisdiction count.
        """
        income_counts = customers_df["declared_income_band"].value_counts().to_dict()
        pep_count = int(customers_df["is_pep"].sum()) if "is_pep" in customers_df.columns else 0
        high_risk_count = (
            int(customers_df["is_high_risk_jurisdiction"].sum())
            if "is_high_risk_jurisdiction" in customers_df.columns
            else 0
        )
        return {
            "income_band": {str(k): int(v) for k, v in income_counts.items()},
            "pep_count": pep_count,
            "high_risk_jurisdiction_count": high_risk_count,
            "total_customers": int(len(customers_df)),
        }

    def _typology_breakdown(self, transactions_df: pd.DataFrame) -> dict:
        """Count transactions per typology label and compute suspicious sub-totals.

        Args:
            transactions_df: Transactions data.

        Returns:
            Dictionary mapping each typology to its count and average amount.
        """
        if "typology" not in transactions_df.columns:
            return {}
        result: dict = {}
        for typology, grp in transactions_df.groupby("typology"):
            result[str(typology)] = {
                "count": int(len(grp)),
                "avg_amount": float(grp["amount"].mean()),
                "total_amount": float(grp["amount"].sum()),
            }
        return result

    def _suspicious_rate(self, transactions_df: pd.DataFrame) -> float:
        """Compute the overall fraction of transactions flagged as suspicious.

        Args:
            transactions_df: Transactions data.

        Returns:
            Float between 0 and 1.
        """
        if "is_suspicious" not in transactions_df.columns or len(transactions_df) == 0:
            return 0.0
        return float(transactions_df["is_suspicious"].mean())

    # ------------------------------------------------------------------
    # Chart generators
    # ------------------------------------------------------------------

    def _plot_amount_distribution(self, transactions_df: pd.DataFrame) -> str:
        """Plot a histogram of transaction amounts and save to disk.

        Uses a log-scale x-axis to handle the heavy-tailed nature of financial
        transaction amounts.

        Args:
            transactions_df: Transactions data.

        Returns:
            Absolute path to the saved PNG file.
        """
        save_path = str(self.charts_dir / "eda_amount_distribution.png")
        amounts = transactions_df["amount"].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Transaction Amount Distribution", fontsize=14, fontweight="bold")

        # Linear scale
        axes[0].hist(amounts, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
        axes[0].set_title("Linear Scale")
        axes[0].set_xlabel("Amount")
        axes[0].set_ylabel("Frequency")

        # Log scale (drop zeros / negatives)
        pos_amounts = amounts[amounts > 0]
        axes[1].hist(np.log10(pos_amounts), bins=60, color="#DD8452", edgecolor="white", linewidth=0.4)
        axes[1].set_title("Log₁₀ Scale")
        axes[1].set_xlabel("log₁₀(Amount)")
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved amount distribution chart → %s", save_path)
        return save_path

    def _plot_country_pie(self, transactions_df: pd.DataFrame) -> str:
        """Plot a pie chart of transactions by sender country.

        Countries with < 2% of transactions are grouped into an 'Other' slice.

        Args:
            transactions_df: Transactions data.

        Returns:
            Absolute path to the saved PNG file.
        """
        save_path = str(self.charts_dir / "eda_country_breakdown.png")
        counts = transactions_df["sender_country"].value_counts()

        total = counts.sum()
        threshold = 0.02 * total
        main = counts[counts >= threshold]
        other_sum = counts[counts < threshold].sum()
        if other_sum > 0:
            main = pd.concat([main, pd.Series({"Other": other_sum})])

        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            main,
            labels=main.index,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.82,
        )
        for autotext in autotexts:
            autotext.set_fontsize(8)
        ax.set_title("Transaction Volume by Sender Country", fontsize=13, fontweight="bold", pad=16)

        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved country pie chart → %s", save_path)
        return save_path

    def _plot_typology_bar(self, transactions_df: pd.DataFrame) -> str:
        """Plot a bar chart of transaction counts per typology label.

        Args:
            transactions_df: Transactions data.

        Returns:
            Absolute path to the saved PNG file.
        """
        save_path = str(self.charts_dir / "eda_typology_breakdown.png")

        if "typology" not in transactions_df.columns:
            # Still produce an empty placeholder so callers always get a path
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No typology column available", ha="center", va="center")
            fig.savefig(save_path, dpi=120)
            plt.close(fig)
            return save_path

        counts = transactions_df["typology"].value_counts().sort_values(ascending=False)
        colour_map = {
            "STRUCTURING": "#d62728",
            "SMURFING": "#ff7f0e",
            "LAYERING": "#9467bd",
            "NORMAL": "#2ca02c",
        }
        colours = [colour_map.get(str(t), "#1f77b4") for t in counts.index]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(counts.index, counts.values, color=colours, edgecolor="white")
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_title("Transaction Count by Typology", fontsize=13, fontweight="bold")
        ax.set_xlabel("Typology")
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)

        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved typology bar chart → %s", save_path)
        return save_path

    def _plot_correlation_heatmap(self, transactions_df: pd.DataFrame) -> str:
        """Plot a Pearson correlation heatmap for numeric transaction columns.

        Args:
            transactions_df: Transactions data.

        Returns:
            Absolute path to the saved PNG file.
        """
        save_path = str(self.charts_dir / "eda_correlation_heatmap.png")
        numeric_cols = transactions_df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.text(0.5, 0.5, "Insufficient numeric columns", ha="center", va="center")
            fig.savefig(save_path, dpi=120)
            plt.close(fig)
            return save_path

        corr = transactions_df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols)), max(5, len(numeric_cols) - 1)))

        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(numeric_cols, fontsize=9)

        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                ax.text(
                    j, i,
                    f"{corr.values[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=7,
                    color="black",
                )

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Correlation Heatmap – Numeric Features", fontsize=12, fontweight="bold", pad=12)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved correlation heatmap → %s", save_path)
        return save_path
