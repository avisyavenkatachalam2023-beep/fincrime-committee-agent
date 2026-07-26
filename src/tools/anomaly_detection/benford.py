"""
benford.py
----------
Benford's Law analysis for AML transaction amount anomaly detection.

Benford's Law states that in many naturally occurring datasets, the leading
digit d appears with probability log10(1 + 1/d).  Deliberate manipulation of
transaction amounts (e.g., structuring) often produces digit distributions
that deviate significantly from this expectation.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from data.load_data import match_entity_id

logger = logging.getLogger(__name__)

CHARTS_DIR = Path("data/charts")

# Expected Benford probabilities for digits 1–9
BENFORD_EXPECTED: dict[int, float] = {
    1: 0.301,
    2: 0.176,
    3: 0.125,
    4: 0.097,
    5: 0.079,
    6: 0.067,
    7: 0.058,
    8: 0.051,
    9: 0.046,
}

# MAD conformity thresholds (Nigrini 2012)
MAD_CLOSE_CONFORMITY = 0.006
MAD_ACCEPTABLE = 0.012
MAD_MARGINAL = 0.015

# Minimum transaction count for a Benford's Law analysis to be statistically
# meaningful. Below this, the observed leading-digit distribution is too
# sensitive to a handful of data points (e.g. 6 similarly-sized transactions
# can produce a false 100% spike on one digit) to support any conclusion —
# analyze_customer() refuses to render a chart or score below this threshold.
MIN_SAMPLE_SIZE = 50


class BenfordAnalyzer:
    """Analyse whether transaction amounts conform to Benford's Law.

    Deviation from Benford's Law can indicate artificially constructed amounts
    such as structuring, round-tripping, or phantom transactions.
    """

    def __init__(self, charts_dir: str | Path = CHARTS_DIR) -> None:
        """Initialise the analyser and ensure the charts output directory exists.

        Args:
            charts_dir: Directory path where PNG charts are saved.
        """
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core Benford analysis primitives
    # ------------------------------------------------------------------

    def get_first_digit(self, amount: float) -> Optional[int]:
        """Extract the first significant (leading) digit from a transaction amount.

        Args:
            amount: Positive numeric transaction amount.

        Returns:
            Integer digit 1–9, or None if amount is zero or negative.
        """
        if amount <= 0:
            return None
        # Shift to a number in [1, 10) then take the floor
        exp = np.floor(np.log10(amount))
        leading = amount / (10 ** exp)
        return int(np.floor(leading))

    def compute_digit_distribution(self, amounts: pd.Series) -> dict[int, float]:
        """Compute the observed frequency of each leading digit across a series.

        Args:
            amounts: Series of transaction amounts (positives only used).

        Returns:
            Dictionary mapping digit 1–9 to its observed relative frequency.
        """
        positive_amounts = amounts[amounts > 0].dropna()
        if positive_amounts.empty:
            return {d: 0.0 for d in range(1, 10)}

        digits = positive_amounts.apply(self.get_first_digit).dropna().astype(int)
        digit_counts = digits.value_counts().reindex(range(1, 10), fill_value=0)
        total = digit_counts.sum()
        if total == 0:
            return {d: 0.0 for d in range(1, 10)}
        return {int(d): float(cnt / total) for d, cnt in digit_counts.items()}

    def mad_score(self, observed_dist: dict[int, float]) -> float:
        """Compute the Mean Absolute Deviation (MAD) from Benford's distribution.

        MAD = mean(|observed_i - expected_i|) for i in 1..9.

        Args:
            observed_dist: Dictionary mapping digit to observed frequency.

        Returns:
            MAD float.  Typical conforming data has MAD < 0.006.
        """
        deviations = [
            abs(observed_dist.get(d, 0.0) - BENFORD_EXPECTED[d])
            for d in range(1, 10)
        ]
        return float(np.mean(deviations))

    def chi_square_score(
        self, observed_dist: dict[int, float], total_count: int
    ) -> tuple[float, float]:
        """Compute chi-square goodness-of-fit against Benford's distribution.

        Args:
            observed_dist: Dictionary mapping digit to observed relative frequency.
            total_count: Total number of transactions in the sample.

        Returns:
            Tuple of (chi2_statistic, p_value).  A low p-value (< 0.05) indicates
            significant deviation from Benford's Law.
        """
        expected_counts = np.array(
            [BENFORD_EXPECTED[d] * total_count for d in range(1, 10)]
        )
        observed_counts = np.array(
            [observed_dist.get(d, 0.0) * total_count for d in range(1, 10)]
        )

        # Guard against expected zeros which cause division errors
        mask = expected_counts > 0
        chi2, p_value = stats.chisquare(
            f_obs=observed_counts[mask],
            f_exp=expected_counts[mask],
        )
        return float(chi2), float(p_value)

    def deviation_score(self, amounts: pd.Series) -> float:
        """Compute a composite suspicion score in [0, 1] for an amount series.

        Combines:
        - Normalised MAD (using Nigrini's marginal threshold as ceiling).
        - Chi-square significance (1 - p_value), so highly significant = high score.

        Args:
            amounts: Transaction amount Series.

        Returns:
            Float in [0, 1], where 1 indicates maximum Benford deviation.
        """
        positive_amounts = amounts[amounts > 0].dropna()
        n = len(positive_amounts)
        if n < MIN_SAMPLE_SIZE:
            return 0.0

        observed = self.compute_digit_distribution(positive_amounts)
        mad = self.mad_score(observed)
        chi2, p_value = self.chi_square_score(observed, n)

        # Normalise MAD: 0 at perfect conformance, 1 at MAD >= MAD_MARGINAL ceiling
        mad_norm = float(np.clip(mad / MAD_MARGINAL, 0.0, 1.0))

        # Chi-square component: high significance → high score
        chi_component = float(1.0 - p_value)

        # Weighted average (MAD gets slightly more weight as it is more stable)
        score = 0.55 * mad_norm + 0.45 * chi_component
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Customer-level analysis
    # ------------------------------------------------------------------

    def analyze_customer(
        self, transactions_df: pd.DataFrame, customer_id: str
    ) -> dict:
        """Run the full Benford analysis for a single customer.

        Args:
            transactions_df: Full transactions DataFrame.
            customer_id: Sender account identifier.

        Returns:
            Dictionary with keys:
            customer_id, digit_distribution, benford_expected,
            mad_score, chi_square, p_value, deviation_score,
            chart_path, interpretation, sample_size, insufficient_sample,
            sample_warning.

            When sample_size is below MIN_SAMPLE_SIZE, no chart is generated
            (chart_path is ''), deviation_score is forced to 0.0, and
            insufficient_sample is True — callers should fall back to the
            threshold-clustering signal alone for this entity rather than
            treating the (statistically meaningless) Benford result as a
            real "conforms to Benford's Law" finding.
        """
        cust_txns = transactions_df[
            match_entity_id(transactions_df["sender_account"], customer_id)
        ].copy()
        amounts = cust_txns["amount"].dropna()
        n = len(amounts[amounts > 0])

        digit_dist = self.compute_digit_distribution(amounts)
        insufficient_sample = n < MIN_SAMPLE_SIZE

        if insufficient_sample:
            mad = 0.0
            chi2, p_value = 0.0, 1.0
            dev_score = 0.0
            chart_path = ""
            sample_warning = (
                f"Insufficient transaction volume (N={n}) for statistically "
                f"meaningful Benford analysis. Deviation score not computed; "
                f"a minimum of {MIN_SAMPLE_SIZE} transactions is required."
            )
            interpretation = sample_warning
        else:
            mad = self.mad_score(digit_dist)
            chi2, p_value = self.chi_square_score(digit_dist, n)
            dev_score = self.deviation_score(amounts)
            chart_path = str(self.charts_dir / f"benford_{customer_id}.png")
            self.plot_benford_chart(digit_dist, customer_id, chart_path, sample_size=n)
            sample_warning = None
            interpretation = self._interpret(mad, p_value, dev_score)

        return {
            "customer_id": customer_id,
            "sample_size": n,
            "insufficient_sample": insufficient_sample,
            "sample_warning": sample_warning,
            "digit_distribution": digit_dist,
            "benford_expected": BENFORD_EXPECTED,
            "mad_score": round(mad, 6),
            "chi_square": round(chi2, 4),
            "p_value": round(p_value, 6),
            "deviation_score": round(dev_score, 4),
            "chart_path": chart_path,
            "interpretation": interpretation,
        }

    def analyze_all_customers(
        self, transactions_df: pd.DataFrame, top_n: int = 10
    ) -> pd.DataFrame:
        """Run Benford analysis for every customer and return ranked results.

        Args:
            transactions_df: Full transactions DataFrame.
            top_n: Number of most suspicious customers to highlight (controls
                   how many per-customer charts are saved).

        Returns:
            DataFrame sorted by deviation_score descending, with columns:
            customer_id, sample_size, mad_score, chi_square, p_value,
            deviation_score, interpretation.
        """
        unique_senders = transactions_df["sender_account"].dropna().unique()
        rows: list[dict] = []

        for cid in unique_senders:
            try:
                amounts = transactions_df.loc[
                    transactions_df["sender_account"] == cid, "amount"
                ].dropna()
                n = int((amounts > 0).sum())
                if n < MIN_SAMPLE_SIZE:
                    continue  # Too few transactions for a statistically meaningful result
                digit_dist = self.compute_digit_distribution(amounts)
                mad = self.mad_score(digit_dist)
                chi2, p_value = self.chi_square_score(digit_dist, n)
                dev_score = self.deviation_score(amounts)
                rows.append(
                    {
                        "customer_id": cid,
                        "sample_size": n,
                        "mad_score": round(mad, 6),
                        "chi_square": round(chi2, 4),
                        "p_value": round(p_value, 6),
                        "deviation_score": round(dev_score, 4),
                        "interpretation": self._interpret(mad, p_value, dev_score),
                    }
                )
            except Exception as exc:
                logger.warning("Benford analysis failed for %s: %s", cid, exc)

        result_df = (
            pd.DataFrame(rows)
            .sort_values("deviation_score", ascending=False)
            .reset_index(drop=True)
        )

        # Save charts only for top N to avoid excessive disk use
        for _, row in result_df.head(top_n).iterrows():
            try:
                cid = row["customer_id"]
                amounts = transactions_df.loc[
                    transactions_df["sender_account"] == cid, "amount"
                ].dropna()
                dist = self.compute_digit_distribution(amounts)
                self.plot_benford_chart(
                    dist, cid, str(self.charts_dir / f"benford_{cid}.png"),
                    sample_size=int(row["sample_size"]),
                )
            except Exception as exc:
                logger.warning("Chart failed for %s: %s", cid, exc)

        return result_df

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_benford_chart(
        self,
        observed_dist: dict[int, float],
        customer_id: str,
        save_path: str,
        sample_size: int,
    ) -> str:
        """Save a bar chart comparing observed digit frequencies to Benford's expected.

        Blue bars represent observed frequencies; orange bars show the Benford
        expected values. The transaction count the "Observed" bars were
        computed from is always disclosed in the subtitle, since the chart is
        meaningless (and can look deceptively confident) without knowing N —
        callers must not invoke this for samples below MIN_SAMPLE_SIZE.

        Args:
            observed_dist: Observed digit frequency dictionary.
            customer_id: Used in the chart title.
            save_path: Absolute path where the PNG is saved.
            sample_size: Number of transactions the observed distribution was
                computed from. Always shown on the chart.

        Returns:
            The save_path string.
        """
        digits = list(range(1, 10))
        observed_vals = [observed_dist.get(d, 0.0) for d in digits]
        expected_vals = [BENFORD_EXPECTED[d] for d in digits]

        x = np.arange(len(digits))
        width = 0.38

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width / 2, observed_vals, width, label="Observed", color="#4C72B0", alpha=0.85)
        ax.bar(x + width / 2, expected_vals, width, label="Benford Expected", color="#DD8452", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in digits])
        ax.set_xlabel("Leading Digit", fontsize=11)
        ax.set_ylabel("Relative Frequency", fontsize=11)
        fig.suptitle(f"Benford's Law Analysis: {customer_id}", fontsize=13, fontweight="bold", y=0.98)
        ax.set_title(f"Based on N={sample_size:,} transactions", fontsize=9.5, color="#555555", style="italic", pad=10)
        ax.legend(fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(max(observed_vals), max(expected_vals)) * 1.25)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        # Ensure parent directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved Benford chart → %s", save_path)
        return save_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _interpret(self, mad: float, p_value: float, dev_score: float) -> str:
        """Produce a plain-English interpretation of the Benford analysis results.

        Args:
            mad: Mean Absolute Deviation from Benford distribution.
            p_value: Chi-square p-value.
            dev_score: Composite deviation score [0, 1].

        Returns:
            Human-readable interpretation string.
        """
        if dev_score < 0.3:
            conformity = "close conformity"
        elif dev_score < 0.6:
            conformity = "marginal conformity"
        else:
            conformity = "non-conformity (suspicious)"

        chi_note = (
            "Chi-square test is statistically significant (p < 0.05), reinforcing suspicion."
            if p_value < 0.05
            else "Chi-square test is not statistically significant (p ≥ 0.05)."
        )

        return (
            f"Benford MAD score of {mad:.4f} indicates {conformity}. "
            f"Composite deviation score: {dev_score:.4f}. "
            f"{chi_note}"
        )
