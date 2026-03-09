"""
Social Network Graph Module
Creates and manages a synthetic social network using NetworkX.
Uses the Barabási-Albert preferential attachment model for realistic topology.
"""
import random
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from config import (
    NETWORK_NUM_NODES,
    NETWORK_EDGES_PER_NODE,
    NETWORK_SEED,
    GRAPH_FIGURE_SIZE,
    GRAPH_DPI,
    NODE_SIZE_BASE,
    NODE_SIZE_SCALE,
    INFLUENCER_THRESHOLD,
    NUM_INFLUENCERS,
    NUM_FACT_CHECKERS,
    NUM_MODERATORS,
)


class SocialNetwork:
    """
    Represents a synthetic social network graph.
    Nodes represent users; edges represent follower/friend connections.
    """

    def __init__(self, num_nodes=NETWORK_NUM_NODES, edges_per_node=NETWORK_EDGES_PER_NODE, seed=NETWORK_SEED,
                 num_influencers=NUM_INFLUENCERS, num_fact_checkers=NUM_FACT_CHECKERS, num_moderators=NUM_MODERATORS):
        self.num_nodes = num_nodes
        self.edges_per_node = edges_per_node
        self.seed = seed
        self.num_influencers_cfg = num_influencers
        self.num_fact_checkers_cfg = num_fact_checkers
        self.num_moderators_cfg = num_moderators
        self.graph = self._create_graph()
        self._assign_roles()
        self.pos = nx.spring_layout(self.graph, seed=seed, k=1.5 / np.sqrt(num_nodes))

    def _create_graph(self):
        """Create a Barabási-Albert scale-free network."""
        G = nx.barabasi_albert_graph(self.num_nodes, self.edges_per_node, seed=self.seed)
        # Assign initial attributes to nodes
        for node in G.nodes():
            G.nodes[node]["label"] = f"User_{node}"
            G.nodes[node]["status"] = "clean"  # clean, infected, warned, blocked
            G.nodes[node]["role"] = "neutral"   # neutral, influencer, fact_checker, moderator
            G.nodes[node]["exposure_count"] = 0
            G.nodes[node]["shared"] = False
            G.nodes[node]["infection_time"] = -1
        return G

    def _assign_roles(self):
        """Assign special roles to nodes based on configured counts."""
        degrees = dict(self.graph.degree())

        # Sort nodes by degree (descending) for role assignment
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)

        # Use configured counts (clamped to valid range)
        num_influencers = max(1, min(self.num_influencers_cfg, self.num_nodes // 3))
        num_fact_checkers = max(1, min(self.num_fact_checkers_cfg, self.num_nodes // 3))
        num_moderators = max(1, min(self.num_moderators_cfg, self.num_nodes // 3))

        # Top-degree nodes as influencers
        for node in sorted_nodes[:num_influencers]:
            self.graph.nodes[node]["role"] = "influencer"

        # Next highest-degree nodes as fact-checkers
        for node in sorted_nodes[num_influencers:num_influencers + num_fact_checkers]:
            self.graph.nodes[node]["role"] = "fact_checker"

        # Medium-degree nodes as moderators
        mid_start = num_influencers + num_fact_checkers
        for node in sorted_nodes[mid_start:mid_start + num_moderators]:
            self.graph.nodes[node]["role"] = "moderator"

    def reset_statuses(self):
        """Reset all node statuses for a new simulation."""
        for node in self.graph.nodes():
            self.graph.nodes[node]["status"] = "clean"
            self.graph.nodes[node]["exposure_count"] = 0
            self.graph.nodes[node]["shared"] = False
            self.graph.nodes[node]["infection_time"] = -1

    def get_random_start_node(self):
        """Select a random starting node for misinformation spread."""
        return random.choice(list(self.graph.nodes()))

    def get_influencer_nodes(self):
        """Return all nodes with influencer role."""
        return [n for n in self.graph.nodes() if self.graph.nodes[n]["role"] == "influencer"]

    def get_fact_checker_nodes(self):
        """Return all nodes with fact_checker role."""
        return [n for n in self.graph.nodes() if self.graph.nodes[n]["role"] == "fact_checker"]

    def get_moderator_nodes(self):
        """Return all nodes with moderator role."""
        return [n for n in self.graph.nodes() if self.graph.nodes[n]["role"] == "moderator"]

    def get_network_stats(self):
        """Return network topology statistics."""
        degrees = dict(self.graph.degree())
        return {
            "total_nodes": self.num_nodes,
            "total_edges": self.graph.number_of_edges(),
            "avg_degree": round(np.mean(list(degrees.values())), 2),
            "max_degree": max(degrees.values()),
            "min_degree": min(degrees.values()),
            "density": round(nx.density(self.graph), 4),
            "avg_clustering": round(nx.average_clustering(self.graph), 4),
            "num_influencers": len(self.get_influencer_nodes()),
            "num_fact_checkers": len(self.get_fact_checker_nodes()),
            "num_moderators": len(self.get_moderator_nodes()),
            "connected": nx.is_connected(self.graph),
        }

    def visualize_network(self, spread_path=None, title="Social Network Graph", save_path="network_graph.png"):
        """
        Visualize the social network with role-based coloring.
        Optionally highlights the misinformation spread path.
        """
        fig, ax = plt.subplots(1, 1, figsize=GRAPH_FIGURE_SIZE, dpi=GRAPH_DPI)
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')

        # Role-based color mapping
        role_colors = {
            "neutral": "#4a90d9",       # Blue
            "influencer": "#f5a623",    # Orange
            "fact_checker": "#7ed321",  # Green
            "moderator": "#9b59b6",     # Purple
        }

        # Status-based overrides
        status_colors = {
            "infected": "#e74c3c",   # Red
            "warned": "#f39c12",     # Yellow-orange
            "blocked": "#95a5a6",    # Gray
        }

        # Determine node colors
        node_colors = []
        for node in self.graph.nodes():
            status = self.graph.nodes[node]["status"]
            role = self.graph.nodes[node]["role"]
            if spread_path and node in spread_path:
                node_colors.append(status_colors.get(status, "#e74c3c"))
            else:
                node_colors.append(role_colors.get(role, "#4a90d9"))

        # Node sizes based on degree
        degrees = dict(self.graph.degree())
        node_sizes = [NODE_SIZE_BASE + degrees[n] * NODE_SIZE_SCALE for n in self.graph.nodes()]

        # Draw edges
        nx.draw_networkx_edges(
            self.graph, self.pos, ax=ax,
            alpha=0.15, edge_color="#555555", width=0.5
        )

        # Highlight spread path edges
        if spread_path and len(spread_path) > 1:
            spread_edges = [(spread_path[i], spread_path[i + 1])
                           for i in range(len(spread_path) - 1)
                           if self.graph.has_edge(spread_path[i], spread_path[i + 1])]
            nx.draw_networkx_edges(
                self.graph, self.pos, edgelist=spread_edges, ax=ax,
                alpha=0.8, edge_color="#e74c3c", width=2.5, style="solid"
            )

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, self.pos, ax=ax,
            node_color=node_colors, node_size=node_sizes,
            edgecolors="#ffffff", linewidths=0.5, alpha=0.9
        )

        # Draw labels for important nodes only (influencers, fact_checkers, moderators, spread source)
        label_nodes = {}
        for node in self.graph.nodes():
            role = self.graph.nodes[node]["role"]
            if role != "neutral":
                label_nodes[node] = f"{node}"
            elif spread_path and node == spread_path[0]:
                label_nodes[node] = f"SRC:{node}"

        nx.draw_networkx_labels(
            self.graph, self.pos, label_nodes, ax=ax,
            font_size=6, font_color="white", font_weight="bold"
        )

        # Legend
        legend_elements = [
            mpatches.Patch(color="#4a90d9", label=f"Neutral Users"),
            mpatches.Patch(color="#f5a623", label=f"Influencers ({len(self.get_influencer_nodes())})"),
            mpatches.Patch(color="#7ed321", label=f"Fact-Checkers ({len(self.get_fact_checker_nodes())})"),
            mpatches.Patch(color="#9b59b6", label=f"Moderators ({len(self.get_moderator_nodes())})"),
            mpatches.Patch(color="#e74c3c", label="Infected / Spread Path"),
        ]
        legend = ax.legend(handles=legend_elements, loc="upper left", fontsize=8,
                          facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")

        ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=15)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", facecolor="#0a0a0a")
        plt.close(fig)
        return save_path

    def visualize_spread_analysis(self, spread_data, save_path="spread_analysis.png"):
        """
        Create analytical visualizations of the spread simulation.
        spread_data: dict with keys like 'spread_per_step', 'agent_scores', etc.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=GRAPH_DPI)
        fig.patch.set_facecolor('#0a0a0a')
        fig.suptitle("Misinformation Spread Analysis Dashboard", fontsize=16,
                     fontweight="bold", color="white", y=0.98)

        for ax in axes.flat:
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#333333')

        # ─── Chart 1: Cumulative Spread Over Time ─────────────────────────────
        ax1 = axes[0, 0]
        spread_steps = spread_data.get("spread_per_step", [])
        if spread_steps:
            cumulative = np.cumsum(spread_steps)
            steps = range(1, len(cumulative) + 1)
            ax1.plot(steps, cumulative, color="#e74c3c", linewidth=2.5, marker='o', markersize=6)
            ax1.fill_between(steps, cumulative, alpha=0.3, color="#e74c3c")
            ax1.set_xlabel("Time Step (BFS Level)")
            ax1.set_ylabel("Cumulative Nodes Reached")
            ax1.set_title("📈 Misinformation Spread Velocity")
            ax1.grid(True, alpha=0.2, color="#444444")

        # ─── Chart 2: Agent Influence Scores ──────────────────────────────────
        ax2 = axes[0, 1]
        agent_scores = spread_data.get("agent_scores", {})
        if agent_scores:
            agents = list(agent_scores.keys())
            scores = list(agent_scores.values())
            colors = ["#e74c3c", "#4a90d9", "#7ed321", "#f5a623", "#9b59b6"]
            bars = ax2.barh(agents, scores, color=colors[:len(agents)], edgecolor="white", linewidth=0.5)
            ax2.set_xlabel("Influence Score")
            ax2.set_title("🤖 Agent Influence Ranking")
            for bar, score in zip(bars, scores):
                ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{score:.1f}', va='center', color='white', fontsize=9)

        # ─── Chart 3: Network Penetration Pie Chart ───────────────────────────
        ax3 = axes[1, 0]
        total = spread_data.get("total_nodes", self.num_nodes)
        infected = spread_data.get("nodes_reached", 0)
        protected = total - infected
        ax3.pie([infected, protected],
                labels=[f"Reached\n({infected})", f"Protected\n({protected})"],
                colors=["#e74c3c", "#2ecc71"],
                autopct='%1.1f%%', startangle=90,
                textprops={'color': 'white', 'fontsize': 10},
                wedgeprops={'edgecolor': '#0a0a0a', 'linewidth': 2})
        ax3.set_title("🌐 Network Penetration Rate")

        # ─── Chart 4: Moderation Effectiveness ────────────────────────────────
        ax4 = axes[1, 1]
        mod_data = spread_data.get("moderation_stats", {})
        if mod_data:
            categories = list(mod_data.keys())
            values = list(mod_data.values())
            colors = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db"]
            ax4.bar(categories, values, color=colors[:len(categories)],
                   edgecolor="white", linewidth=0.5, width=0.6)
            ax4.set_ylabel("Count / Rate")
            ax4.set_title("🛡️ Moderation Effectiveness")
            ax4.grid(True, axis='y', alpha=0.2, color="#444444")
            for i, (cat, val) in enumerate(zip(categories, values)):
                ax4.text(i, val + 0.3, f'{val:.1f}', ha='center', color='white', fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, bbox_inches="tight", facecolor="#0a0a0a")
        plt.close(fig)
        return save_path


# ─── Convenience function ─────────────────────────────────────────────────────
def create_network(num_nodes=NETWORK_NUM_NODES, edges_per_node=NETWORK_EDGES_PER_NODE, seed=NETWORK_SEED,
                   num_influencers=NUM_INFLUENCERS, num_fact_checkers=NUM_FACT_CHECKERS, num_moderators=NUM_MODERATORS):
    """Create and return a SocialNetwork instance."""
    return SocialNetwork(num_nodes, edges_per_node, seed, num_influencers, num_fact_checkers, num_moderators)
