"""
threshold_clustering.py
------------------------
Detects suspicious clustering of transaction amounts near regulatory reporting
thresholds — a hallmark of structuring / smurfing activity in AML typologies.

Regulatory thresholds monitored:
  $10,000 – Currency Transaction Report (CTR) in the USA
  $5,000  – common secondary surveillance threshold
  $3,000  – MSB/prepaid card monitoring threshold
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CHARTS_DIR = Path("data/charts")


class ThresholdClusteringAnalyzer:
    """Identify suspicious clusters of transaction amounts near reporting thresholds.

    Two signals are computed:
    1. **Sub-threshold spike score** – whether transactions cluster just below
       known regulatory reporting limits far more than expected by chance.
    2. **Round-number score** – whether transactions use suspiciously round amounts,
       which can also indicate fabricated / layered flows.
    """

    STRUCTURING_BANDS: list[tuple[int, int]] = [
        (9_000, 9_999),    # Just under the USD 10,000 CTR threshold
        (4_500, 4_999),    # Just under the USD 5,000 threshold
        (2_900, 2_999),    # Just under the USD 3,000 threshold
    ]

    ROUND_NUMBER_MULTIPLES: list[int] = [
        1_000, 2_000, 5_000, 10_000, 25_000, 50_000
    ]

    def __init__(self, charts_dir: str | Path = CHARTS_DIR) -> None:
        """Initialise the analyser and create the charts directory if needed.

        Args:
            charts_dir: Directory path where PNG charts are saved.
        """
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core analysis methods
    # ------------------------------------------------------------------

    def sub_threshold_clustering_score(self, amounts: pd.Series) -> dict:
        """Detect clustering of amounts just below known reporting thresholds.

        For each structuring band the method counts observed transactions and
        computes a spike score: the ratio of observed band density vs the
        expected density if amounts were uniformly distributed across [0, max].

        Args:
            amounts: Transaction amount Series for one customer.

        Returns:
            Dictionary with keys:
            bands (per-band count and expected_count), spike_score (float),
            interpretation (str).
        """
        amounts = amounts.dropna()
        amounts = amounts[amounts > 0]
        n_total = len(amounts)
        if n_total == 0:
            return {"bands": {}, "spike_score": 0.0, "interpretation": "No transactions."}

        max_amount = float(amounts.max())
        band_results: dict[str, dict] = {}
        total_spike = 0.0

        for low, high in self.STRUCTURING_BANDS:
            band_mask = (amounts >= low) & (amounts <= high)
            observed = int(band_mask.sum())
            band_width = high - low + 1

            # Expected count assuming uniform distribution up to max_amount
            if max_amount > 0:
                expected = n_total * (band_width / max_amount)
            else:
                expected = 0.0

            spike = (observed / expected) if expected > 0 else 0.0
            band_key = f"{low}-{high}"
            band_results[band_key] = {
                "observed": observed,
                "expected": round(expected, 2),
                "spike_ratio": round(spike, 3),
            }
            total_spike += spike

        # Normalise spike score to [0, 1] range (cap at 10x expected)
        num_bands = len(self.STRUCTURING_BANDS)
        avg_spike = total_spike / num_bands if num_bands > 0 else 0.0
        normalised_spike = float(np.clip(avg_spike / 10.0, 0.0, 1.0))

        interpretation = self._interpret_spike(normalised_spike, band_results)

        return {
            "bands": band_results,
            "spike_score": round(normalised_spike, 4),
            "interpretation": interpretation,
        }

    def round_number_score(self, amounts: pd.Series) -> dict:
        """Detect unusually high frequency of round-number transaction amounts.

        Args:
            amounts: Transaction amount Series for one customer.

        Returns:
            Dictionary with keys:
            round_number_ratio (float), spike_score (float),
            common_round_amounts (list), interpretation (str).
        """
        amounts = amounts.dropna()
        amounts = amounts[amounts > 0]
        n_total = len(amounts)

        if n_total == 0:
            return {
                "round_number_ratio": 0.0,
                "spike_score": 0.0,
                "common_round_amounts": [],
                "interpretation": "No transactions.",
            }

        # Count transactions that are exact multiples of any round-number sentinel
        round_mask = amounts.apply(
            lambda x: any(x % m == 0 for m in self.ROUND_NUMBER_MULTIPLES)
        )
        round_count = int(round_mask.sum())
        round_ratio = float(round_count / n_total)

        # Common round amounts found
        common_rounds = sorted(
            amounts[round_mask]
            .value_counts()
            .head(5)
            .index
            .tolist()
        )

        # Expected fraction: ~1-2% of random transactions hit round multiples
        # Use 0.05 as a conservative baseline
        baseline_fraction = 0.05
        spike_score = float(np.clip(round_ratio / (baseline_fraction + 1e-9) / 10.0, 0.0, 1.0))

        if round_ratio > 0.5:
            interpretation = (
                f"Very high round-number ratio ({round_ratio:.1%}) — strongly indicative "
                "of fabricated / layered transactions."
            )
        elif round_ratio > 0.25:
            interpretation = (
                f"Elevated round-number ratio ({round_ratio:.1%}) — warrants investigation."
            )
        else:
            interpretation = (
                f"Round-number ratio ({round_ratio:.1%}) within normal range."
            )

        return {
            "round_number_ratio": round(round_ratio, 4),
            "spike_score": round(spike_score, 4),
            "common_round_amounts": [float(x) for x in common_rounds],
            "interpretation": interpretation,
        }

    # ------------------------------------------------------------------
    # Customer-level API
    # ------------------------------------------------------------------

    def analyze_customer(
        self, transactions_df: pd.DataFrame, customer_id: str
    ) -> dict:
        """Run the full threshold clustering analysis for a single customer.

        Args:
            transactions_df: Full transactions DataFrame.
            customer_id: Sender account identifier.

        Returns:
            Dictionary with keys:
            customer_id, sample_size, sub_threshold (dict),
            round_numbers (dict), composite_clustering_score (float),
            chart_path (str).
        """
        cust_txns = transactions_df[
            transactions_df["sender_account"] == customer_id
        ].copy()
        amounts = cust_txns["amount"].dropna()

        sub_thresh = self.sub_threshold_clustering_score(amounts)
        round_nums = self.round_number_score(amounts)

        # Composite: 70% sub-threshold, 30% round-number
        composite = float(
            0.70 * sub_thresh["spike_score"] + 0.30 * round_nums["spike_score"]
        )

        chart_path = str(self.charts_dir / f"threshold_cluster_{customer_id}.png")
        self.plot_amount_histogram(amounts, customer_id, chart_path)

        return {
            "customer_id": customer_id,
            "sample_size": int(len(amounts[amounts > 0])),
            "sub_threshold": sub_thresh,
            "round_numbers": round_nums,
            "composite_clustering_score": round(composite, 4),
            "chart_path": chart_path,
        }

    def analyze_all_customers(
        self, transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Run threshold clustering analysis for every customer in the dataset.

        Args:
            transactions_df: Full transactions DataFrame.

        Returns:
            DataFrame sorted by composite_clustering_score descending, with columns:
            customer_id, sample_size, spike_score_sub_threshold,
            spike_score_round_number, composite_clustering_score.
        """
        unique_senders = transactions_df["sender_account"].dropna().unique()
        rows: list[dict] = []

        for cid in unique_senders:
            try:
                amounts = transactions_df.loc[
                    transactions_df["sender_account"] == cid, "amount"
                ].dropna()
                if (amounts > 0).sum() == 0:
                    continue
                sub_thresh = self.sub_threshold_clustering_score(amounts)
                round_nums = self.round_number_score(amounts)
                composite = float(
                    0.70 * sub_thresh["spike_score"] + 0.30 * round_nums["spike_score"]
                )
                rows.append(
                    {
                        "customer_id": cid,
                        "sample_size": int((amounts > 0).sum()),
                        "spike_score_sub_threshold": sub_thresh["spike_score"],
                        "spike_score_round_number": round_nums["spike_score"],
                        "composite_clustering_score": round(composite, 4),
                    }
                )
            except Exception as exc:
                logger.warning("Clustering analysis failed for %s: %s", cid, exc)

        return (
            pd.DataFrame(rows)
            .sort_values("composite_clustering_score", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_amount_histogram(
        self, amounts: pd.Series, customer_id: str, save_path: str
    ) -> str:
        """Save a histogram of transaction amounts with threshold band overlays.

        The $9,000–$9,999 structuring band is shaded red.  Vertical dashed
        lines mark the $10,000, $5,000, and $3,000 thresholds.

        Args:
            amounts: Transaction amount Series.
            customer_id: Used in the chart title.
            save_path: Absolute filesystem path for the output PNG.

        Returns:
            The save_path string.
        """
        pos_amounts = amounts[amounts > 0].dropna()

        fig, ax = plt.subplots(figsize=(10, 5))

        if not pos_amounts.empty:
            upper = min(float(pos_amounts.quantile(0.98)), 15_000)
            bins = np.linspace(0, upper, 80)
            ax.hist(
                np.clip(pos_amounts, 0, upper),
                bins=bins,
                color="#4C72B0",
                edgecolor="white",
                linewidth=0.3,
                alpha=0.8,
                label="Transaction amounts",
            )

            # Red shaded structuring band (9,000–9,999)
            ax.axvspan(9_000, 9_999, alpha=0.25, color="red", label="CTR zone (9,000–9,999)")

            # Threshold lines
            for val, label, colour in [
                (10_000, "$10,000 CTR", "#d62728"),
                (5_000, "$5,000", "#ff7f0e"),
                (3_000, "$3,000", "#9467bd"),
            ]:
                if val <= upper:
                    ax.axvline(val, color=colour, linestyle="--", linewidth=1.4, label=label)

        ax.set_xlabel("Transaction Amount (USD)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(
            f"Threshold Clustering Analysis – {customer_id}", fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved threshold clustering chart → %s", save_path)
        return save_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _interpret_spike(
        self, normalised_spike: float, band_results: dict
    ) -> str:
        """Generate a text interpretation for the sub-threshold spike score.

        Args:
            normalised_spike: Normalised spike score in [0, 1].
            band_results: Per-band analysis dictionary.

        Returns:
            Human-readable interpretation string.
        """
        high_bands = [
            band
            for band, info in band_results.items()
            if info.get("spike_ratio", 0.0) > 2.0
        ]

        if normalised_spike > 0.6:
            severity = "HIGH – strong structuring signal."
        elif normalised_spike > 0.3:
            severity = "MEDIUM – elevated structuring risk."
        else:
            severity = "LOW – no significant clustering."

        band_note = ""
        if high_bands:
            band_note = f" Elevated activity in bands: {', '.join(high_bands)}."

        return f"Sub-threshold clustering score {normalised_spike:.4f} [{severity}]{band_note}"
