"""
Social Network Graph Module
============================
Creates and manages a synthetic social network using NetworkX.
Uses the Barabási–Albert preferential attachment model.
AI agent nodes (misinfo, influencer, fact-checker, moderator) are embedded
directly inside the graph and participate in cascade-based propagation.
"""
import random
from collections import deque

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
    NUM_MISINFO_AGENTS,
    NUM_INFLUENCERS,
    NUM_FACT_CHECKERS,
    NUM_MODERATORS,
    SPREAD_PROBABILITY,
    AMPLIFICATION_FACTOR,
    MAX_SPREAD_DEPTH,
    FACT_CHECK_SLOW,
    MODERATOR_BLOCK_PROB,
    MODERATOR_FLAG_PROB,
    GRAPH_FIGURE_SIZE,
    GRAPH_DPI,
    NODE_SIZE_BASE,
    NODE_SIZE_SCALE,
    ROLE_COLOURS,
    STATUS_COLOURS,
)


class SocialNetwork:
    """
    Scale-free social network with embedded AI agent nodes.

    Roles assigned to nodes (highest-degree first):
        misinfo      – misinformation source nodes
        influencer   – amplify content (× amplification_factor neighbours)
        fact_checker – verify & label claims (reduce spread probability)
        moderator    – may BLOCK / FLAG / ALLOW (can stop cascade)
        normal       – regular users reshare with base probability
    """

    def __init__(
        self,
        num_nodes=NETWORK_NUM_NODES,
        edges_per_node=NETWORK_EDGES_PER_NODE,
        seed=NETWORK_SEED,
        num_misinfo=NUM_MISINFO_AGENTS,
        num_influencers=NUM_INFLUENCERS,
        num_fact_checkers=NUM_FACT_CHECKERS,
        num_moderators=NUM_MODERATORS,
        spread_prob=SPREAD_PROBABILITY,
        amplification=AMPLIFICATION_FACTOR,
    ):
        self.num_nodes = num_nodes
        self.edges_per_node = edges_per_node
        self.seed = seed
        self.num_misinfo_cfg = int(num_misinfo)
        self.num_influencers_cfg = int(num_influencers)
        self.num_fact_checkers_cfg = int(num_fact_checkers)
        self.num_moderators_cfg = int(num_moderators)
        self.spread_prob = spread_prob
        self.amplification = amplification

        self.graph = self._create_graph()
        self._assign_roles()
        self.pos = nx.spring_layout(
            self.graph, seed=seed, k=2.0 / np.sqrt(num_nodes), iterations=50
        )

    # ── graph creation ────────────────────────────────────────────────────────

    def _create_graph(self):
        G = nx.barabasi_albert_graph(self.num_nodes, self.edges_per_node, seed=self.seed)
        for n in G.nodes():
            G.nodes[n].update({
                "label": f"User_{n}",
                "role": "normal",
                "status": "clean",        # clean | infected | warned | blocked
                "exposure_count": 0,
                "shared": False,
                "infection_time": -1,
                "cascade_ids": [],         # which cascades reached this node
                "warning_label": False,    # fact-checker attached a warning
                "blocked_by": None,        # moderator that blocked the cascade here
            })
        return G

    def _assign_roles(self):
        """Assign agent roles to nodes sorted by descending degree."""
        degrees = dict(self.graph.degree())
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)

        n_total = self.num_nodes
        # clamp counts so agents never exceed 80 % of nodes
        budget = int(n_total * 0.8)
        n_misinfo = max(1, min(self.num_misinfo_cfg, budget))
        remaining = budget - n_misinfo
        n_inf = max(1, min(self.num_influencers_cfg, remaining))
        remaining -= n_inf
        n_fc = max(1, min(self.num_fact_checkers_cfg, remaining))
        remaining -= n_fc
        n_mod = max(1, min(self.num_moderators_cfg, remaining))

        # Misinformation agents are placed at random positions in the top
        # half of the degree distribution so they have reasonable reach
        # but are NOT the very highest-degree nodes (those are influencers).
        top_half = sorted_nodes[: max(n_total // 2, n_misinfo + n_inf + n_fc + n_mod)]

        # Influencers get the highest-degree slots
        idx = 0
        for node in sorted_nodes[idx: idx + n_inf]:
            self.graph.nodes[node]["role"] = "influencer"
        idx += n_inf

        # Fact-checkers
        for node in sorted_nodes[idx: idx + n_fc]:
            self.graph.nodes[node]["role"] = "fact_checker"
        idx += n_fc

        # Moderators
        for node in sorted_nodes[idx: idx + n_mod]:
            self.graph.nodes[node]["role"] = "moderator"
        idx += n_mod

        # Misinformation agents – pick from remaining nodes that are NOT
        # already assigned, preferring medium-degree nodes
        available = [n for n in sorted_nodes if self.graph.nodes[n]["role"] == "normal"]
        # pick spread-out positions for diversity
        step = max(1, len(available) // (n_misinfo + 1))
        selected = []
        for i in range(n_misinfo):
            pos = min(i * step, len(available) - 1)
            selected.append(available[pos])
        for node in selected:
            self.graph.nodes[node]["role"] = "misinfo"

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

    def reset_statuses(self):
        """Reset all dynamic node attributes for a fresh simulation."""
        for n in self.graph.nodes():
            self.graph.nodes[n].update({
                "status": "clean",
                "exposure_count": 0,
                "shared": False,
                "infection_time": -1,
                "cascade_ids": [],
                "warning_label": False,
                "blocked_by": None,
            })

    # ── multi-cascade BFS propagation ─────────────────────────────────────────

    def run_multi_cascade(
        self,
        claims: list[dict],
        spread_prob: float | None = None,
        amplification: float | None = None,
        max_depth: int | None = None,
    ) -> dict:
        """
        Run simultaneous BFS cascades from every misinformation-agent node.

        Each misinfo node injects one unique claim. Every node processes the
        claim according to its role:
            normal      → reshare with *spread_prob*
            influencer  → reshare with higher prob, fan-out × amplification
            fact_checker → verify & attach warning; low reshare prob
            moderator   → may BLOCK (stops cascade at this node),
                          FLAG (reduces further spread) or ALLOW

        Returns a rich dict with per-cascade and aggregate results.
        """
        self.reset_statuses()

        if spread_prob is None:
            spread_prob = self.spread_prob
        if amplification is None:
            amplification = self.amplification
        if max_depth is None:
            max_depth = MAX_SPREAD_DEPTH

        misinfo_nodes = self.get_misinfo_nodes()
        if len(claims) < len(misinfo_nodes):
            # pad with generic claims if fewer claims than misinfo nodes
            while len(claims) < len(misinfo_nodes):
                claims.append({"claim": f"[auto-generated fallback claim #{len(claims)+1}]",
                               "source_agent": "fallback"})

        cascade_results = []
        all_infected = set()

        for cascade_id, (src_node, claim_data) in enumerate(zip(misinfo_nodes, claims)):
            cr = self._run_single_cascade(
                cascade_id=cascade_id,
                src_node=src_node,
                claim_text=claim_data.get("claim", ""),
                spread_prob=spread_prob,
                amplification=amplification,
                max_depth=max_depth,
                already_infected=all_infected,
            )
            cascade_results.append(cr)
            all_infected.update(cr["reached_nodes"])

        # aggregate
        total_reached = len(all_infected)
        total_blocked = sum(c["blocked_count"] for c in cascade_results)
        total_warned = sum(c["warned_count"] for c in cascade_results)
        total_exposures = sum(
            self.graph.nodes[n]["exposure_count"] for n in self.graph.nodes()
        )

        # spread-per-step aggregate (max length across cascades, sum per step)
        max_steps = max((len(c["spread_per_step"]) for c in cascade_results), default=0)
        agg_spread_per_step = [0] * max_steps
        for c in cascade_results:
            for i, v in enumerate(c["spread_per_step"]):
                agg_spread_per_step[i] += v

        return {
            "cascade_results": cascade_results,
            "total_reached": total_reached,
            "total_nodes": self.num_nodes,
            "penetration_rate": round(total_reached / self.num_nodes * 100, 2),
            "total_blocked": total_blocked,
            "total_warned": total_warned,
            "total_exposures": total_exposures,
            "num_cascades": len(cascade_results),
            "spread_per_step": agg_spread_per_step,
            "max_depth_reached": max(
                (c["max_depth"] for c in cascade_results), default=0
            ),
            "viral_coefficient": round(
                np.mean([c["viral_coefficient"] for c in cascade_results]), 2
            ) if cascade_results else 0,
            "reached_nodes": list(all_infected),
        }

    def _run_single_cascade(
        self,
        cascade_id: int,
        src_node: int,
        claim_text: str,
        spread_prob: float,
        amplification: float,
        max_depth: int,
        already_infected: set,
    ) -> dict:
        """BFS cascade from a single misinformation source node."""
        G = self.graph
        queue = deque([(src_node, 0)])
        visited = {src_node}
        reached = [src_node]
        spread_per_step: list[int] = []
        current_depth = 0
        step_count = 0
        blocked_count = 0
        warned_count = 0
        influencer_amplifications = 0
        edges_in_cascade: list[tuple[int, int]] = []

        # mark source
        G.nodes[src_node]["status"] = "infected"
        G.nodes[src_node]["shared"] = True
        G.nodes[src_node]["infection_time"] = 0
        G.nodes[src_node]["cascade_ids"].append(cascade_id)

        while queue:
            node, depth = queue.popleft()
            if depth > max_depth:
                break

            # flush step counter when BFS moves to new level
            if depth > current_depth:
                spread_per_step.append(step_count)
                step_count = 0
                current_depth = depth

            neighbours = list(G.neighbors(node))
            random.shuffle(neighbours)

            # influencer amplification: expose more neighbours
            role_of_current = G.nodes[node]["role"]
            if role_of_current == "influencer":
                fan_out = min(len(neighbours), int(len(neighbours) * amplification))
                influencer_amplifications += 1
            else:
                fan_out = len(neighbours)

            for nb in neighbours[:fan_out]:
                G.nodes[nb]["exposure_count"] += 1

                if nb in visited:
                    continue
                if depth + 1 > max_depth:
                    continue

                nb_role = G.nodes[nb]["role"]

                # ── role-based processing ──
                reshare = False

                if nb_role == "normal":
                    reshare = random.random() < spread_prob

                elif nb_role == "influencer":
                    # influencers are eager to share (higher probability)
                    reshare = random.random() < min(1.0, spread_prob * 1.8)

                elif nb_role == "fact_checker":
                    # fact-checkers verify: attach warning, low reshare
                    G.nodes[nb]["warning_label"] = True
                    G.nodes[nb]["status"] = "warned"
                    warned_count += 1
                    reshare = random.random() < FACT_CHECK_SLOW

                elif nb_role == "moderator":
                    # moderators may BLOCK, FLAG or ALLOW
                    action = self._moderator_action()
                    if action == "BLOCK":
                        G.nodes[nb]["status"] = "blocked"
                        G.nodes[nb]["blocked_by"] = nb
                        blocked_count += 1
                        reshare = False           # cascade dies here
                    elif action == "FLAG":
                        G.nodes[nb]["status"] = "warned"
                        G.nodes[nb]["warning_label"] = True
                        warned_count += 1
                        reshare = random.random() < (spread_prob * 0.15)
                    else:
                        reshare = random.random() < spread_prob

                elif nb_role == "misinfo":
                    # another misinfo node receives – treat like normal user
                    reshare = random.random() < spread_prob

                if reshare:
                    visited.add(nb)
                    reached.append(nb)
                    step_count += 1
                    edges_in_cascade.append((node, nb))
                    if G.nodes[nb]["status"] == "clean":
                        G.nodes[nb]["status"] = "infected"
                    G.nodes[nb]["shared"] = True
                    G.nodes[nb]["infection_time"] = depth + 1
                    G.nodes[nb]["cascade_ids"].append(cascade_id)
                    queue.append((nb, depth + 1))

        # final step
        if step_count:
            spread_per_step.append(step_count)

        # viral coefficient
        viral_coeff = 0.0
        if len(spread_per_step) > 1 and spread_per_step[0] > 0:
            viral_coeff = sum(spread_per_step[1:]) / spread_per_step[0]

        return {
            "cascade_id": cascade_id,
            "source_node": src_node,
            "claim": claim_text,
            "reached_nodes": set(reached),
            "spread_path": reached,
            "spread_per_step": spread_per_step,
            "total_reached": len(reached),
            "max_depth": current_depth,
            "viral_coefficient": round(viral_coeff, 2),
            "blocked_count": blocked_count,
            "warned_count": warned_count,
            "influencer_amplifications": influencer_amplifications,
            "edges": edges_in_cascade,
        }

    @staticmethod
    def _moderator_action() -> str:
        """Probabilistic moderator decision during cascade."""
        r = random.random()
        if r < MODERATOR_BLOCK_PROB:
            return "BLOCK"
        elif r < MODERATOR_BLOCK_PROB + MODERATOR_FLAG_PROB * (1 - MODERATOR_BLOCK_PROB):
            return "FLAG"
        return "ALLOW"

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

    # ── visualisation ─────────────────────────────────────────────────────────

    def visualize_network(
        self,
        spread_result=None,
        title="Social Network Graph",
        save_path="network_graph.png",
    ) -> str:
        """
        Draw the full network. Nodes are coloured by role; infected /
        warned / blocked nodes are highlighted to show spread & containment.
        """
        fig, ax = plt.subplots(1, 1, figsize=GRAPH_FIGURE_SIZE, dpi=GRAPH_DPI)
        fig.patch.set_facecolor("#0a0a0a")
        ax.set_facecolor("#0a0a0a")

        infected_set = set()
        if spread_result:
            infected_set = set(spread_result.get("reached_nodes", []))

        # ── node colours ──
        node_colors = []
        for n in self.graph.nodes():
            st = self.graph.nodes[n]["status"]
            rl = self.graph.nodes[n]["role"]
            if st == "blocked":
                node_colors.append(STATUS_COLOURS["blocked"])
            elif st == "warned":
                node_colors.append(STATUS_COLOURS["warned"])
            elif n in infected_set or st == "infected":
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
        if spread_result:
            cascade_edges = []
            for cr in spread_result.get("cascade_results", []):
                cascade_edges.extend(cr.get("edges", []))
            if cascade_edges:
                nx.draw_networkx_edges(self.graph, self.pos, edgelist=cascade_edges,
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
                           label=f"Misinfo Agents ({len(self.get_misinfo_nodes())})"),
            mpatches.Patch(color=ROLE_COLOURS["influencer"],
                           label=f"Influencers ({len(self.get_influencer_nodes())})"),
            mpatches.Patch(color=ROLE_COLOURS["fact_checker"],
                           label=f"Fact-Checkers ({len(self.get_fact_checker_nodes())})"),
            mpatches.Patch(color=ROLE_COLOURS["moderator"],
                           label=f"Moderators ({len(self.get_moderator_nodes())})"),
            mpatches.Patch(color=ROLE_COLOURS["normal"],
                           label=f"Normal Users ({len(self.get_normal_nodes())})"),
            mpatches.Patch(color=STATUS_COLOURS["infected"], label="Infected"),
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

    def visualize_spread_analysis(self, spread_result, save_path="spread_analysis.png") -> str:
        """
        Generate a 2×3 analytical dashboard from the multi-cascade results.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=GRAPH_DPI)
        fig.patch.set_facecolor("#0a0a0a")
        fig.suptitle("Multi-Cascade Misinformation Spread Analysis",
                     fontsize=15, fontweight="bold", color="white", y=0.99)

        for ax in axes.flat:
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for sp in ax.spines.values():
                sp.set_color("#333333")

        cascades = spread_result.get("cascade_results", [])

        # ── 1. Cumulative spread ──────────────────────────────────────────────
        ax1 = axes[0, 0]
        agg = spread_result.get("spread_per_step", [])
        if agg:
            cum = np.cumsum(agg)
            steps = range(1, len(cum) + 1)
            ax1.plot(steps, cum, color="#e74c3c", linewidth=2.5, marker="o", markersize=5)
            ax1.fill_between(steps, cum, alpha=0.25, color="#e74c3c")
        ax1.set_xlabel("BFS Level")
        ax1.set_ylabel("Cumulative Nodes")
        ax1.set_title("Cumulative Spread Velocity")
        ax1.grid(True, alpha=0.2, color="#444")

        # ── 2. Per-cascade reach ──────────────────────────────────────────────
        ax2 = axes[0, 1]
        if cascades:
            ids = [f"C{c['cascade_id']}" for c in cascades]
            reached = [c["total_reached"] for c in cascades]
            colours = plt.cm.Reds(np.linspace(0.4, 0.9, len(cascades)))
            ax2.bar(ids, reached, color=colours, edgecolor="white", linewidth=0.5)
            for i, v in enumerate(reached):
                ax2.text(i, v + 1, str(v), ha="center", color="white", fontsize=8)
        ax2.set_ylabel("Nodes Reached")
        ax2.set_title("Reach per Cascade")
        ax2.grid(True, axis="y", alpha=0.2, color="#444")

        # ── 3. Penetration pie ────────────────────────────────────────────────
        ax3 = axes[0, 2]
        total = spread_result.get("total_nodes", self.num_nodes)
        inf = spread_result.get("total_reached", 0)
        prot = total - inf
        ax3.pie([inf, prot], labels=[f"Reached\n({inf})", f"Protected\n({prot})"],
                colors=["#e74c3c", "#2ecc71"], autopct="%1.1f%%", startangle=90,
                textprops={"color": "white", "fontsize": 9},
                wedgeprops={"edgecolor": "#0a0a0a", "linewidth": 2})
        ax3.set_title("Network Penetration")

        # ── 4. Role-based containment ─────────────────────────────────────────
        ax4 = axes[1, 0]
        blocked = spread_result.get("total_blocked", 0)
        warned = spread_result.get("total_warned", 0)
        infected = spread_result.get("total_reached", 0)
        ax4.bar(["Infected", "Warned", "Blocked"],
                [infected, warned, blocked],
                color=["#e74c3c", "#f1c40f", "#95a5a6"],
                edgecolor="white", linewidth=0.5, width=0.55)
        for i, v in enumerate([infected, warned, blocked]):
            ax4.text(i, v + 1, str(v), ha="center", color="white", fontsize=9)
        ax4.set_ylabel("Count")
        ax4.set_title("Containment Breakdown")
        ax4.grid(True, axis="y", alpha=0.2, color="#444")

        # ── 5. Agent influence scores ─────────────────────────────────────────
        ax5 = axes[1, 1]
        scores = {
            "Misinfo\nAgents": min(10, (inf / max(total * 0.05, 1)) * 2),
            "Influencers": min(10, sum(c.get("influencer_amplifications", 0)
                                       for c in cascades) * 0.8),
            "Fact-\nCheckers": min(10, warned * 0.5),
            "Moderators": min(10, blocked * 1.2),
            "Normal\nUsers": min(10, inf * 0.15),
        }
        agents_list = list(scores.keys())
        sc = list(scores.values())
        bar_colours = ["#e74c3c", "#f5a623", "#2ecc71", "#9b59b6", "#3498db"]
        bars = ax5.barh(agents_list, sc, color=bar_colours, edgecolor="white", linewidth=0.5)
        for bar, s in zip(bars, sc):
            ax5.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{s:.1f}", va="center", color="white", fontsize=8)
        ax5.set_xlabel("Score (0-10)")
        ax5.set_title("Agent Influence Scores")

        # ── 6. Spread depth per cascade ───────────────────────────────────────
        ax6 = axes[1, 2]
        if cascades:
            ids2 = [f"C{c['cascade_id']}" for c in cascades]
            depths = [c["max_depth"] for c in cascades]
            ax6.bar(ids2, depths, color=plt.cm.Blues(np.linspace(0.4, 0.9, len(cascades))),
                    edgecolor="white", linewidth=0.5)
            for i, v in enumerate(depths):
                ax6.text(i, v + 0.2, str(v), ha="center", color="white", fontsize=8)
        ax6.set_ylabel("Hops")
        ax6.set_title("Cascade Depth")
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
    num_misinfo=NUM_MISINFO_AGENTS,
    num_influencers=NUM_INFLUENCERS,
    num_fact_checkers=NUM_FACT_CHECKERS,
    num_moderators=NUM_MODERATORS,
    spread_prob=SPREAD_PROBABILITY,
    amplification=AMPLIFICATION_FACTOR,
) -> SocialNetwork:
    return SocialNetwork(
        num_nodes, edges_per_node, seed,
        num_misinfo, num_influencers, num_fact_checkers, num_moderators,
        spread_prob, amplification,
    )
