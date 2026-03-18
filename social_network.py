"""
Social Network Graph Module
============================
Creates and manages a synthetic social network using NetworkX.
Uses the Barabási–Albert preferential attachment model.
AI agent nodes (misinfo, influencer, fact-checker, moderator) are embedded
directly inside the graph. Supports sequential agent phases.
"""
import random

import networkx as nx
import matplotlib
matplotlib.use("Agg")
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
    ROLE_COLOURS,
    STATUS_COLOURS,
    calculate_agent_counts,
)


class SocialNetwork:
    """
    Scale-free social network with embedded AI agent nodes.

    Roles assigned to nodes (highest-degree first):
        misinfo      – single misinformation source node
        influencer   – amplify content (× amplification_factor neighbours)
        fact_checker – verify & label claims
        moderator    – BLOCK / FLAG / ALLOW content
        normal       – regular users reshare with base probability
    """

    def __init__(
        self,
        num_nodes=NETWORK_NUM_NODES,
        edges_per_node=NETWORK_EDGES_PER_NODE,
        seed=NETWORK_SEED,
    ):
        self.num_nodes = int(num_nodes)
        self.edges_per_node = int(edges_per_node)
        self.seed = seed

        # Auto-calculate agent counts based on graph size
        agent_counts = calculate_agent_counts(self.num_nodes)
        self.num_misinfo = agent_counts["num_misinfo"]
        self.num_influencers = agent_counts["num_influencers"]
        self.num_fact_checkers = agent_counts["num_fact_checkers"]
        self.num_moderators = agent_counts["num_moderators"]

        self.graph = self._create_graph()
        self._assign_roles()
        self.pos = nx.spring_layout(
            self.graph, seed=seed, k=2.0 / np.sqrt(self.num_nodes), iterations=50
        )

    # ── graph creation ────────────────────────────────────────────────────────

    def _create_graph(self):
        G = nx.barabasi_albert_graph(self.num_nodes, self.edges_per_node, seed=self.seed)
        for n in G.nodes():
            G.nodes[n].update({
                "label": f"User_{n}",
                "role": "normal",
                "status": "clean",        # clean | infected | influenced | warned | blocked
                "exposure_count": 0,
                "shared": False,
                "infection_time": -1,
                "claim_text": "",
                "warning_label": False,
                "blocked_by": None,
            })
        return G

    def _assign_roles(self):
        """Assign agent roles to nodes sorted by descending degree."""
        degrees = dict(self.graph.degree())
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)

        idx = 0
        # Influencers get the highest-degree slots
        for node in sorted_nodes[idx: idx + self.num_influencers]:
            self.graph.nodes[node]["role"] = "influencer"
        idx += self.num_influencers

        # Fact-checkers
        for node in sorted_nodes[idx: idx + self.num_fact_checkers]:
            self.graph.nodes[node]["role"] = "fact_checker"
        idx += self.num_fact_checkers

        # Moderators
        for node in sorted_nodes[idx: idx + self.num_moderators]:
            self.graph.nodes[node]["role"] = "moderator"
        idx += self.num_moderators

        # Misinformation agent – pick from remaining normal nodes (medium degree)
        available = [n for n in sorted_nodes if self.graph.nodes[n]["role"] == "normal"]
        if available:
            mid_idx = len(available) // 3  # Pick from upper-middle range
            self.graph.nodes[available[mid_idx]]["role"] = "misinfo"

    # ── accessors ─────────────────────────────────────────────────────────────

    def get_nodes_by_role(self, role):
        return [n for n in self.graph.nodes() if self.graph.nodes[n]["role"] == role]

    def get_misinfo_nodes(self):
        return self.get_nodes_by_role("misinfo")

    def get_influencer_nodes(self):
        return self.get_nodes_by_role("influencer")

    def get_fact_checker_nodes(self):
        return self.get_nodes_by_role("fact_checker")

    def get_moderator_nodes(self):
        return self.get_nodes_by_role("moderator")

    def get_normal_nodes(self):
        return self.get_nodes_by_role("normal")

    def get_infected_nodes(self):
        return [n for n in self.graph.nodes() if self.graph.nodes[n]["status"] == "infected"]

    def get_influenced_nodes(self):
        return [n for n in self.graph.nodes() if self.graph.nodes[n]["status"] == "influenced"]

    def get_warned_nodes(self):
        return [n for n in self.graph.nodes() if self.graph.nodes[n]["status"] == "warned"]

    def get_blocked_nodes(self):
        return [n for n in self.graph.nodes() if self.graph.nodes[n]["status"] == "blocked"]

    def get_clean_nodes(self):
        return [n for n in self.graph.nodes() if self.graph.nodes[n]["status"] == "clean"]

    def reset_statuses(self):
        """Reset all dynamic node attributes for a fresh simulation."""
        for n in self.graph.nodes():
            self.graph.nodes[n].update({
                "status": "clean",
                "exposure_count": 0,
                "shared": False,
                "infection_time": -1,
                "claim_text": "",
                "warning_label": False,
                "blocked_by": None,
            })

    # ── network statistics ────────────────────────────────────────────────────

    def get_network_stats(self) -> dict:
        degrees = dict(self.graph.degree())
        return {
            "total_nodes": self.num_nodes,
            "total_edges": self.graph.number_of_edges(),
            "avg_degree": round(np.mean(list(degrees.values())), 2),
            "max_degree": max(degrees.values()),
            "min_degree": min(degrees.values()),
            "density": round(nx.density(self.graph), 4),
            "avg_clustering": round(nx.average_clustering(self.graph), 4),
            "num_misinfo_agents": len(self.get_misinfo_nodes()),
            "num_influencers": len(self.get_influencer_nodes()),
            "num_fact_checkers": len(self.get_fact_checker_nodes()),
            "num_moderators": len(self.get_moderator_nodes()),
            "num_normal_users": len(self.get_normal_nodes()),
            "connected": nx.is_connected(self.graph),
        }

    def get_agent_stats_table(self) -> list:
        """Return table data for agent working stats."""
        status_counts = {}
        for n in self.graph.nodes():
            st = self.graph.nodes[n]["status"]
            status_counts[st] = status_counts.get(st, 0) + 1

        total = self.num_nodes
        infected = status_counts.get("infected", 0)
        influenced = status_counts.get("influenced", 0)
        warned = status_counts.get("warned", 0)
        blocked = status_counts.get("blocked", 0)
        clean = status_counts.get("clean", 0)

        return [
            ["🔴 Nodes Spread (Infected)", infected, f"{infected/total*100:.1f}%"],
            ["🟠 Nodes Influenced", influenced, f"{influenced/total*100:.1f}%"],
            ["⚠️ Nodes Warned (Fact-Checked)", warned, f"{warned/total*100:.1f}%"],
            ["🚫 Nodes Blocked (Moderated)", blocked, f"{blocked/total*100:.1f}%"],
            ["🟢 Nodes Uninformed (Clean)", clean, f"{clean/total*100:.1f}%"],
            ["📊 Total Nodes", total, "100%"],
        ]

    # ── visualisation ─────────────────────────────────────────────────────────

    def visualize_network(
        self,
        title="Social Network Graph",
        save_path="network_graph.png",
        show_cascade_edges=None,
    ) -> str:
        """
        Draw the full network. Nodes are coloured by status;
        agent nodes are highlighted to show the simulation state.
        """
        fig, ax = plt.subplots(1, 1, figsize=GRAPH_FIGURE_SIZE, dpi=GRAPH_DPI)
        fig.patch.set_facecolor("#0a0a0a")
        ax.set_facecolor("#0a0a0a")

        # ── node colours ──
        node_colors = []
        for n in self.graph.nodes():
            st = self.graph.nodes[n]["status"]
            rl = self.graph.nodes[n]["role"]
            if st == "blocked":
                node_colors.append(STATUS_COLOURS["blocked"])
            elif st == "warned":
                node_colors.append(STATUS_COLOURS["warned"])
            elif st == "influenced":
                node_colors.append(STATUS_COLOURS["influenced"])
            elif st == "infected":
                node_colors.append(STATUS_COLOURS["infected"])
            else:
                node_colors.append(ROLE_COLOURS.get(rl, ROLE_COLOURS["normal"]))

        # ── node sizes ──
        degrees = dict(self.graph.degree())
        node_sizes = [NODE_SIZE_BASE + degrees[n] * NODE_SIZE_SCALE for n in self.graph.nodes()]

        # ── draw edges ──
        nx.draw_networkx_edges(self.graph, self.pos, ax=ax, alpha=0.08,
                               edge_color="#555555", width=0.3)

        # highlight cascade edges
        if show_cascade_edges:
            nx.draw_networkx_edges(self.graph, self.pos, edgelist=show_cascade_edges,
                                   ax=ax, alpha=0.6, edge_color="#e74c3c",
                                   width=1.2, style="solid")

        # ── draw nodes ──
        nx.draw_networkx_nodes(self.graph, self.pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, edgecolors="#ffffff",
                               linewidths=0.3, alpha=0.92)

        # labels for agent nodes only
        label_map = {}
        for n in self.graph.nodes():
            rl = self.graph.nodes[n]["role"]
            if rl == "misinfo":
                label_map[n] = f"M{n}"
            elif rl == "influencer":
                label_map[n] = f"I{n}"
            elif rl == "fact_checker":
                label_map[n] = f"F{n}"
            elif rl == "moderator":
                label_map[n] = f"D{n}"
        if label_map:
            nx.draw_networkx_labels(self.graph, self.pos, label_map, ax=ax,
                                    font_size=5, font_color="white", font_weight="bold")

        # legend
        legend_elements = [
            mpatches.Patch(color=ROLE_COLOURS["misinfo"],
                           label=f"Misinfo Source ({len(self.get_misinfo_nodes())})"),
            mpatches.Patch(color=ROLE_COLOURS["influencer"],
                           label=f"Influencers ({len(self.get_influencer_nodes())})"),
            mpatches.Patch(color=ROLE_COLOURS["fact_checker"],
                           label=f"Fact-Checkers ({len(self.get_fact_checker_nodes())})"),
            mpatches.Patch(color=ROLE_COLOURS["moderator"],
                           label=f"Moderators ({len(self.get_moderator_nodes())})"),
            mpatches.Patch(color=ROLE_COLOURS["normal"],
                           label=f"Normal Users ({len(self.get_normal_nodes())})"),
            mpatches.Patch(color=STATUS_COLOURS["infected"], label="Infected"),
            mpatches.Patch(color=STATUS_COLOURS["influenced"], label="Influenced"),
            mpatches.Patch(color=STATUS_COLOURS["warned"], label="Warned / Flagged"),
            mpatches.Patch(color=STATUS_COLOURS["blocked"], label="Blocked"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=7,
                  facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")
        ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=12)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", facecolor="#0a0a0a")
        plt.close(fig)
        return save_path

    def visualize_spread_analysis(self, simulation_result: dict, save_path="spread_analysis.png") -> str:
        """
        Generate a 2×3 analytical dashboard from the simulation results.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=GRAPH_DPI)
        fig.patch.set_facecolor("#0a0a0a")
        fig.suptitle("Misinformation Simulation Analysis Dashboard",
                     fontsize=15, fontweight="bold", color="white", y=0.99)

        for ax in axes.flat:
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for sp in ax.spines.values():
                sp.set_color("#333333")

        spread_data = simulation_result.get("spread_result", {})
        influence_data = simulation_result.get("influence_result", {})
        fact_check_data = simulation_result.get("fact_check_result", {})
        moderation_data = simulation_result.get("moderation_result", {})

        # ── 1. Spread per step ────────────────────────────────────────────────
        ax1 = axes[0, 0]
        steps = spread_data.get("spread_per_step", [])
        if steps:
            cum = np.cumsum(steps)
            x = range(1, len(cum) + 1)
            ax1.plot(x, cum, color="#e74c3c", linewidth=2.5, marker="o", markersize=5)
            ax1.fill_between(x, cum, alpha=0.25, color="#e74c3c")
        ax1.set_xlabel("BFS Level")
        ax1.set_ylabel("Cumulative Nodes")
        ax1.set_title("Cumulative Spread Velocity")
        ax1.grid(True, alpha=0.2, color="#444")

        # ── 2. Node status distribution ───────────────────────────────────────
        ax2 = axes[0, 1]
        sc = moderation_data.get("final_status_counts", {})
        if not sc:
            sc = {"clean": 0, "infected": 0, "influenced": 0, "warned": 0, "blocked": 0}
            for n in self.graph.nodes():
                st = self.graph.nodes[n]["status"]
                sc[st] = sc.get(st, 0) + 1
        labels = list(sc.keys())
        values = list(sc.values())
        colors = ["#3498db", "#ff6b6b", "#ff9f43", "#f1c40f", "#95a5a6"]
        bars = ax2.bar(labels, values, color=colors[:len(labels)], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 1, str(v),
                     ha="center", color="white", fontsize=9)
        ax2.set_ylabel("Node Count")
        ax2.set_title("Final Node Status Distribution")
        ax2.grid(True, axis="y", alpha=0.2, color="#444")

        # ── 3. Penetration pie ────────────────────────────────────────────────
        ax3 = axes[0, 2]
        total = self.num_nodes
        affected = total - sc.get("clean", 0)
        clean = sc.get("clean", 0)
        ax3.pie([affected, clean],
                labels=[f"Affected\n({affected})", f"Uninformed\n({clean})"],
                colors=["#e74c3c", "#2ecc71"], autopct="%1.1f%%", startangle=90,
                textprops={"color": "white", "fontsize": 9},
                wedgeprops={"edgecolor": "#0a0a0a", "linewidth": 2})
        ax3.set_title("Network Penetration")

        # ── 4. Agent activity ─────────────────────────────────────────────────
        ax4 = axes[1, 0]
        agents = ["Misinfo\nSource", "Normal\nSpread", "Influencer\nAmplify",
                  "Fact-Check\nVerify", "Moderator\nAction"]
        agent_values = [
            1,
            spread_data.get("total_spread", 0),
            influence_data.get("additional_spread", 0),
            fact_check_data.get("nodes_warned", 0),
            moderation_data.get("nodes_blocked", 0) + moderation_data.get("nodes_flagged", 0),
        ]
        bar_colours = ["#e74c3c", "#3498db", "#f5a623", "#2ecc71", "#9b59b6"]
        bars = ax4.bar(agents, agent_values, color=bar_colours, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, agent_values):
            ax4.text(bar.get_x() + bar.get_width() / 2, v + 0.5, str(v),
                     ha="center", color="white", fontsize=9)
        ax4.set_ylabel("Nodes Affected")
        ax4.set_title("Agent Activity Breakdown")
        ax4.grid(True, axis="y", alpha=0.2, color="#444")

        # ── 5. Fact-check verdict ─────────────────────────────────────────────
        ax5 = axes[1, 1]
        verdict = fact_check_data.get("verdict", "Unknown")
        confidence = fact_check_data.get("confidence", 0)
        v_color = {"Real": "#2ecc71", "Fake": "#e74c3c", "Unverified": "#f1c40f"}.get(verdict, "#999")
        ax5.barh(["Verdict"], [confidence * 100], color=v_color, edgecolor="white", height=0.4)
        ax5.text(confidence * 100 + 2, 0, f"{verdict} ({confidence*100:.0f}%)",
                 va="center", color="white", fontsize=12, fontweight="bold")
        ax5.set_xlim(0, 110)
        ax5.set_xlabel("Confidence %")
        ax5.set_title("Fact-Check Verdict")

        # ── 6. Moderation decision ────────────────────────────────────────────
        ax6 = axes[1, 2]
        blocked = moderation_data.get("nodes_blocked", 0)
        flagged = moderation_data.get("nodes_flagged", 0)
        allowed = moderation_data.get("nodes_allowed", 0)
        mod_labels = ["Blocked", "Flagged", "Allowed"]
        mod_values = [blocked, flagged, allowed]
        mod_colors = ["#95a5a6", "#f1c40f", "#2ecc71"]
        ax6.bar(mod_labels, mod_values, color=mod_colors, edgecolor="white", linewidth=0.5)
        for i, v in enumerate(mod_values):
            ax6.text(i, v + 0.3, str(v), ha="center", color="white", fontsize=10)
        ax6.set_ylabel("Nodes")
        ax6.set_title("Moderation Actions")
        ax6.grid(True, axis="y", alpha=0.2, color="#444")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, bbox_inches="tight", facecolor="#0a0a0a")
        plt.close(fig)
        return save_path


# ── convenience factory ───────────────────────────────────────────────────────

def create_network(
    num_nodes=NETWORK_NUM_NODES,
    edges_per_node=NETWORK_EDGES_PER_NODE,
    seed=NETWORK_SEED,
) -> SocialNetwork:
    return SocialNetwork(num_nodes, edges_per_node, seed)
