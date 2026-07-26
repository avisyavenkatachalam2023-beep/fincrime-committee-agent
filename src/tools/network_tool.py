"""
network_tool.py
---------------
Graph-based network analysis for the AML Financial Crime Committee Agent.

Builds a directed weighted graph of account-to-account transaction flows and
computes graph-theoretic metrics that surface money-mule hubs, coordinated
smurfing networks, and layering chains.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:
    from community import best_partition  # python-louvain
    _LOUVAIN_AVAILABLE = True
except ImportError:
    best_partition = None
    _LOUVAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

CHARTS_DIR = Path("data/charts")


class NetworkAnalysisTool:
    """Build and analyse the account transaction network for AML signals.

    Graph representation:
    - Nodes: unique account IDs (sender or receiver).
    - Directed edges: (sender → receiver), weighted by transaction amount.
    - Multiple transactions between the same pair are aggregated into a single
      edge with summed weight and transaction count stored as an attribute.
    """

    def __init__(self, charts_dir: str | Path = CHARTS_DIR) -> None:
        """Initialise the tool and ensure the charts directory exists.

        Args:
            charts_dir: Directory path where PNG charts are saved.
        """
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self, transactions_df: pd.DataFrame) -> nx.DiGraph:
        """Construct a directed weighted transaction graph from the DataFrame.

        Each unique (sender_account, receiver_account) pair becomes one edge.
        Multiple transactions are aggregated: the edge weight is the total
        amount transferred; ``txn_count`` stores the number of transactions.

        Args:
            transactions_df: Full transactions DataFrame.

        Returns:
            Directed graph (nx.DiGraph).
        """
        G = nx.DiGraph()

        # Vectorised edge aggregation instead of a Python-level row loop,
        # which does not scale to the Kaggle SAML-D dataset (200k+ rows).
        df = transactions_df.copy()
        df["sender_account"] = df["sender_account"].astype(str)
        df["receiver_account"] = df["receiver_account"].astype(str)
        amount_col = df["amount"].fillna(0.0) if "amount" in df.columns else pd.Series(0.0, index=df.index)
        df["_amount"] = amount_col

        edges = (
            df.groupby(["sender_account", "receiver_account"])["_amount"]
            .agg(weight="sum", txn_count="count")
            .reset_index()
        )
        for src, dst, weight, txn_count in edges.itertuples(index=False, name=None):
            G.add_edge(src, dst, weight=float(weight), txn_count=int(txn_count))

        # Attach account-level metadata (first non-null value per sender) if columns exist
        for col in ["sender_bank", "sender_country"]:
            if col in df.columns:
                attr_key = col.replace("sender_", "")
                first_vals = df.dropna(subset=[col]).drop_duplicates(subset=["sender_account"])
                for node, value in zip(first_vals["sender_account"], first_vals[col]):
                    if node in G.nodes:
                        G.nodes[node][attr_key] = value

        logger.info(
            "Built transaction graph: %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges()
        )
        return G

    # ------------------------------------------------------------------
    # Centrality computation
    # ------------------------------------------------------------------

    def compute_centrality(self, G: nx.DiGraph) -> dict[str, dict]:
        """Compute key centrality metrics for every node in the graph.

        Metrics computed:
        - ``degree_centrality``: Fraction of nodes connected to this node.
        - ``betweenness_centrality``: Fraction of shortest paths passing
          through this node — high values indicate coordinator / hub roles.
        - ``in_degree``: Number of incoming edges (money received from).
        - ``out_degree``: Number of outgoing edges (money sent to).

        Args:
            G: Directed transaction graph.

        Returns:
            Nested dictionary ``{account_id: {metric: value, ...}}``.
        """
        degree_cent = nx.degree_centrality(G)

        # Betweenness is expensive on large graphs (each sampled source runs a
        # full shortest-path search). Two levers keep it responsive on the
        # Kaggle-scale graph (100k+ nodes):
        #   1. k-sample approximation instead of exact all-pairs.
        #   2. Unweighted BFS instead of weighted Dijkstra — coordination
        #      structure (hop distance) matters more here than dollar-weighted
        #      paths, and BFS is an order of magnitude faster.
        n = G.number_of_nodes()
        if n > 500:
            k = 100 if n > 20_000 else 500
        else:
            k = None
        betweenness_cent = nx.betweenness_centrality(G, k=k, normalized=True, weight=None)

        centrality: dict[str, dict] = {}
        for node in G.nodes():
            centrality[node] = {
                "degree_centrality": round(degree_cent.get(node, 0.0), 6),
                "betweenness_centrality": round(betweenness_cent.get(node, 0.0), 6),
                "in_degree": int(G.in_degree(node)),
                "out_degree": int(G.out_degree(node)),
                "total_sent": round(
                    sum(d["weight"] for _, _, d in G.out_edges(node, data=True)), 2
                ),
                "total_received": round(
                    sum(d["weight"] for _, _, d in G.in_edges(node, data=True)), 2
                ),
            }
        return centrality

    # ------------------------------------------------------------------
    # Community detection
    # ------------------------------------------------------------------

    def detect_communities(self, G: nx.DiGraph) -> dict[str, int]:
        """Detect communities of tightly connected accounts.

        Preference order:
        1. Louvain method (``python-louvain`` package) on the undirected
           projection — best quality but requires an optional dependency.
        2. Greedy modularity maximisation (built-in NetworkX) as fallback.

        Args:
            G: Directed transaction graph.

        Returns:
            Dictionary mapping ``account_id`` → ``community_id`` (int).
        """
        undirected = G.to_undirected()

        if _LOUVAIN_AVAILABLE and best_partition is not None:
            try:
                partition = best_partition(undirected)
                logger.info("Community detection via Louvain method.")
                return {str(k): int(v) for k, v in partition.items()}
            except Exception as exc:
                logger.warning("Louvain failed (%s); falling back to greedy.", exc)

        # Greedy modularity maximisation
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(undirected)
            partition = {}
            for cid, comm in enumerate(communities):
                for node in comm:
                    partition[str(node)] = cid
            logger.info("Community detection via greedy modularity (fallback).")
            return partition
        except Exception as exc:
            logger.error("Community detection failed: %s", exc)
            # Last resort: assign each node its own community
            return {str(node): i for i, node in enumerate(G.nodes())}

    # ------------------------------------------------------------------
    # Hub detection
    # ------------------------------------------------------------------

    def find_hubs(
        self, centrality_scores: dict[str, dict], top_n: int = 10
    ) -> list[dict]:
        """Identify top-N hub accounts by betweenness centrality.

        High-betweenness nodes sit on many shortest paths and likely act as
        coordinators or layering intermediaries.

        Args:
            centrality_scores: Output of ``compute_centrality()``.
            top_n: How many top hubs to return.

        Returns:
            List of dicts ``{account_id, betweenness_centrality, in_degree,
            out_degree, total_sent, total_received}`` sorted descending.
        """
        sorted_nodes = sorted(
            centrality_scores.items(),
            key=lambda x: x[1]["betweenness_centrality"],
            reverse=True,
        )
        hubs = []
        for account_id, metrics in sorted_nodes[:top_n]:
            hubs.append(
                {
                    "account_id": account_id,
                    "betweenness_centrality": metrics["betweenness_centrality"],
                    "degree_centrality": metrics["degree_centrality"],
                    "in_degree": metrics["in_degree"],
                    "out_degree": metrics["out_degree"],
                    "total_sent": metrics.get("total_sent", 0.0),
                    "total_received": metrics.get("total_received", 0.0),
                }
            )
        return hubs

    # ------------------------------------------------------------------
    # Entity-scoped analysis
    # ------------------------------------------------------------------

    def analyze_entity(
        self,
        transactions_df: pd.DataFrame,
        customer_id: str,
        customers_df: Optional[pd.DataFrame] = None,
        G: Optional[nx.DiGraph] = None,
        centrality: Optional[dict[str, dict]] = None,
        communities: Optional[dict[str, int]] = None,
    ) -> dict:
        """Perform network analysis scoped to a specific customer account.

        Extracts the ego-network (the customer and all direct neighbours) for
        focused metrics.

        Args:
            transactions_df: Full transactions DataFrame.
            customer_id: Sender (or receiver) account identifier.
            customers_df: Optional customers DataFrame for PEP / risk enrichment.
            G: Precomputed graph. If omitted, it is built from
                ``transactions_df`` (expensive on large datasets — prefer
                passing the graph already built by ``run()``).
            centrality: Precomputed centrality scores; recomputed if omitted.
            communities: Precomputed community partition; recomputed if omitted.

        Returns:
            Dictionary with keys:
            customer_id, community_id, hub_score (betweenness centrality),
            is_hub (bool), centrality_scores (dict),
            connected_flagged_accounts (list), interpretation (str),
            ego_network_size (int), chart_path (str).
        """
        if G is None:
            G = self.build_graph(transactions_df)
        if centrality is None:
            centrality = self.compute_centrality(G)
        if communities is None:
            communities = self.detect_communities(G)

        node_centrality = centrality.get(customer_id, {})
        community_id = communities.get(customer_id, -1)
        hub_score = node_centrality.get("betweenness_centrality", 0.0)

        # Ego network: nodes within 1 hop
        ego_nodes: set[str] = {customer_id}
        if customer_id in G:
            ego_nodes |= set(G.successors(customer_id))
            ego_nodes |= set(G.predecessors(customer_id))

        # Flagged accounts in the neighbourhood
        flagged_accounts: list[str] = []
        if "is_suspicious" in transactions_df.columns:
            flagged_in_data = set(
                transactions_df.loc[
                    transactions_df["is_suspicious"] == True, "sender_account"
                ].dropna().tolist()
                + transactions_df.loc[
                    transactions_df["is_suspicious"] == True, "receiver_account"
                ].dropna().tolist()
            )
            flagged_accounts = [a for a in ego_nodes if a in flagged_in_data and a != customer_id]

        # Hub threshold: betweenness > 0.1 or top decile
        all_betweenness = [v["betweenness_centrality"] for v in centrality.values()]
        hub_threshold = float(np.percentile(all_betweenness, 90)) if all_betweenness else 0.1
        is_hub = hub_score >= hub_threshold

        interpretation = self._interpret_entity(
            customer_id, hub_score, is_hub, flagged_accounts, community_id, len(ego_nodes)
        )

        chart_path = str(self.charts_dir / f"network_{customer_id}.png")
        if customer_id in G:
            ego_graph = G.subgraph(ego_nodes).copy()
            self.plot_network_graph(
                ego_graph,
                highlight_node=customer_id,
                save_path=chart_path,
                flagged_nodes=flagged_accounts,
            )

        return {
            "customer_id": customer_id,
            "community_id": int(community_id),
            "hub_score": round(hub_score, 6),
            "is_hub": is_hub,
            "centrality_scores": node_centrality,
            "connected_flagged_accounts": flagged_accounts,
            "ego_network_size": len(ego_nodes),
            "interpretation": interpretation,
            "chart_path": chart_path,
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_network_graph(
        self,
        G: nx.DiGraph,
        highlight_node: str,
        save_path: str,
        flagged_nodes: Optional[list[str]] = None,
    ) -> str:
        """Plot the transaction network with colour-coded nodes and save to disk.

        Colour scheme:
        - Red: the highlighted focus node.
        - Orange: flagged / suspicious neighbouring nodes.
        - Light-blue: all other nodes.

        Args:
            G: (Sub-)graph to visualise.
            highlight_node: Account ID to display in red.
            save_path: Absolute path for the output PNG.
            flagged_nodes: List of account IDs to display in orange.

        Returns:
            The save_path string.
        """
        flagged_nodes = flagged_nodes or []
        fig, ax = plt.subplots(figsize=(10, 7))

        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "No network data available", ha="center", va="center")
            fig.savefig(save_path, dpi=100)
            plt.close(fig)
            return save_path

        # Limit to a manageable size for visualisation
        if G.number_of_nodes() > 80:
            # Keep the ego node plus the 79 highest-degree neighbours
            degrees = dict(G.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:79]
            if highlight_node not in top_nodes:
                top_nodes.append(highlight_node)
            G = G.subgraph(top_nodes).copy()

        pos = nx.spring_layout(G, seed=42, k=1.5 / max(G.number_of_nodes() ** 0.5, 1))

        node_colours = []
        for n in G.nodes():
            if n == highlight_node:
                node_colours.append("#d62728")  # Red
            elif n in flagged_nodes:
                node_colours.append("#ff7f0e")  # Orange
            else:
                node_colours.append("#AEC6E8")  # Light blue

        node_sizes = [
            600 if n == highlight_node else 200
            for n in G.nodes()
        ]

        edge_weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
        max_w = max(edge_weights) if edge_weights else 1.0
        edge_widths = [0.5 + 2.5 * (w / max_w) for w in edge_weights]

        nx.draw_networkx_nodes(G, pos, node_color=node_colours, node_size=node_sizes, ax=ax)
        nx.draw_networkx_edges(
            G, pos, width=edge_widths, edge_color="#888888", arrows=True,
            arrowsize=10, connectionstyle="arc3,rad=0.1", ax=ax,
        )

        # Label only the focus and flagged nodes to keep the chart readable
        label_subset = {highlight_node: highlight_node}
        label_subset.update({n: n for n in flagged_nodes if n in G.nodes()})
        nx.draw_networkx_labels(
            G, pos, labels=label_subset, font_size=8, font_color="black", ax=ax
        )

        ax.set_title(
            f"Transaction Network – Focus: {highlight_node}",
            fontsize=12,
            fontweight="bold",
        )
        ax.axis("off")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#d62728", label="Focus account"),
            Patch(facecolor="#ff7f0e", label="Flagged account"),
            Patch(facecolor="#AEC6E8", label="Other account"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved network chart → %s", save_path)
        return save_path

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        transactions_df: pd.DataFrame,
        customers_df: Optional[pd.DataFrame] = None,
        focus_customer: Optional[str] = None,
    ) -> dict:
        """Full network analysis pipeline.

        Builds the graph, computes centrality and communities, identifies hubs,
        and optionally performs entity-scoped analysis for a focus customer.

        Args:
            transactions_df: Full transactions DataFrame.
            customers_df: Optional customers DataFrame.
            focus_customer: If provided, run ``analyze_entity`` for this account.

        Returns:
            Dictionary with keys:
            graph_summary (node/edge counts), top_hubs (list),
            community_count (int), entity_analysis (dict, if focus_customer
            provided).
        """
        G = self.build_graph(transactions_df)
        centrality = self.compute_centrality(G)
        communities = self.detect_communities(G)
        top_hubs = self.find_hubs(centrality, top_n=10)

        n_communities = len(set(communities.values()))
        result: dict = {
            "graph_summary": {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "is_dag": nx.is_directed_acyclic_graph(G),
                "density": round(nx.density(G), 6),
            },
            "top_hubs": top_hubs,
            "community_count": n_communities,
            "centrality": centrality,
        }

        if focus_customer is not None:
            result["entity_analysis"] = self.analyze_entity(
                transactions_df, focus_customer, customers_df,
                G=G, centrality=centrality, communities=communities,
            )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _interpret_entity(
        self,
        customer_id: str,
        hub_score: float,
        is_hub: bool,
        flagged_accounts: list[str],
        community_id: int,
        ego_size: int,
    ) -> str:
        """Generate a plain-English interpretation for the entity analysis.

        Args:
            customer_id: Account identifier.
            hub_score: Betweenness centrality score.
            is_hub: Whether this account is classified as a hub.
            flagged_accounts: Neighbouring flagged account IDs.
            community_id: Detected community identifier.
            ego_size: Number of nodes in the ego network.

        Returns:
            Interpretation string.
        """
        hub_note = (
            f"Account {customer_id} is a network HUB (betweenness centrality: "
            f"{hub_score:.4f}), suggesting a coordinator or money-mule role."
            if is_hub
            else f"Account {customer_id} is NOT a primary hub (betweenness: {hub_score:.4f})."
        )

        flagged_note = (
            f" It is directly connected to {len(flagged_accounts)} flagged account(s): "
            f"{', '.join(flagged_accounts[:5])}{'…' if len(flagged_accounts) > 5 else ''}."
            if flagged_accounts
            else " No directly connected flagged accounts detected."
        )

        community_note = (
            f" Belongs to community {community_id} containing {ego_size - 1} neighbours."
        )

        return hub_note + flagged_note + community_note
