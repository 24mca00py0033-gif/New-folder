
import numpy as np


class SimulationAnalytics:


    def __init__(self, network):
        self.network = network

    def compute_spread_metrics(self, spread: dict) -> dict:
        steps = spread.get("spread_per_step", [])
        total_spread = spread.get("total_spread", 0)
        total_nodes = self.network.num_nodes
        cumulative = list(np.cumsum(steps)) if steps else [0]
        return {
            "total_spread": total_spread,
            "total_nodes": total_nodes,
            "penetration_rate": round(total_spread / max(total_nodes, 1) * 100, 2),
            "spread_per_step": steps,
            "cumulative_spread": cumulative,
            "avg_velocity": round(float(np.mean(steps)), 2) if steps else 0,
            "peak_step": (int(np.argmax(steps)) + 1) if steps else 0,
            "peak_spread_count": int(max(steps)) if steps else 0,
            "max_depth": spread.get("max_depth", 0),
        }

    def compute_agent_activity(self, state: dict) -> list:
      
        spread = state.get("spread_result", {})
        influence = state.get("influence_result", {})
        fc = state.get("fact_check_result", {})
        mod = state.get("moderation_result", {})

        return [
            ["🔴 Misinformation Agent", "Generate Claim", "1 claim injected",
             f"Source: User_{spread.get('source_node', '?')}"],
            ["🔵 Neutral Agent", "BFS Spread",
             f"{spread.get('total_spread', 0)} nodes infected",
             f"Depth: {spread.get('max_depth', 0)} hops"],
            ["🟠 Influencer Agent", "Amplify & Modify",
             f"{influence.get('additional_spread', 0)} new nodes",
             f"Score: {influence.get('amplification_score', 0)}/10"],
            ["🟢 Fact-Checker Agent", "Verify & Warn",
             f"{fc.get('nodes_checked', 0)} checked, {fc.get('nodes_warned', 0)} warned",
             f"Verdict: {fc.get('verdict', 'N/A')}"],
            ["🟣 Moderator Agent", "Block/Flag/Allow",
             f"{mod.get('nodes_blocked', 0)} blocked, {mod.get('nodes_flagged', 0)} flagged",
             f"Decision: {mod.get('decision', 'N/A')}"],
        ]

    def compute_node_breakdown(self, state: dict) -> list:
      
        mod = state.get("moderation_result", {})
        sc = mod.get("final_status_counts", {})
        if not sc:
            sc = {"clean": 0, "infected": 0, "influenced": 0, "warned": 0, "blocked": 0}
            for n in self.network.graph.nodes():
                st = self.network.graph.nodes[n]["status"]
                sc[st] = sc.get(st, 0) + 1

        total = self.network.num_nodes
        return [
            ["Nodes Didn't Get Info", sc.get("clean", 0),
             f"{sc.get('clean', 0)/total*100:.1f}%", "Did not receive the claim"],
            ["Nodes Spread", sc.get("infected", 0),
             f"{sc.get('infected', 0)/total*100:.1f}%", "Received and shared the claim"],
            ["Nodes Influenced", sc.get("influenced", 0),
             f"{sc.get('influenced', 0)/total*100:.1f}%", "Received modified claim from influencers"],
            ["Nodes Checked the Info", sc.get("warned", 0),
             f"{sc.get('warned', 0)/total*100:.1f}%", "Fact-checked and warned"],
            ["Nodes Blocked", sc.get("blocked", 0),
             f"{sc.get('blocked', 0)/total*100:.1f}%", "Blocked by moderators"],
        ]

    def generate_full_analytics(self, state: dict) -> dict:
        spread = state.get("spread_result", {})
        return {
            "spread_metrics": self.compute_spread_metrics(spread),
            "agent_activity": self.compute_agent_activity(state),
            "node_breakdown": self.compute_node_breakdown(state),
        }

    def generate_analytics_report(self, state: dict) -> str:
        spread = state.get("spread_result", {})
        influence = state.get("influence_result", {})
        fc = state.get("fact_check_result", {})
        mod = state.get("moderation_result", {})

        steps = spread.get("spread_per_step", [])
        avg_vel = round(float(np.mean(steps)), 2) if steps else 0
        peak = (int(np.argmax(steps)) + 1) if steps else 0

        sc = mod.get("final_status_counts", {})
        total = self.network.num_nodes

        return f"""
{'='*60}
          DETAILED ANALYTICS REPORT
{'='*60}

📈 1. SPREAD VELOCITY
{'─'*60}
   Total Nodes Infected  : {spread.get('total_spread', 0)} / {total}
   Network Penetration   : {spread.get('penetration_rate', 0)}%
   Average Velocity      : {avg_vel} nodes/step
   Peak Spread Step      : Step {peak}
   Maximum Depth         : {spread.get('max_depth', 0)} hops

📣 2. INFLUENCER IMPACT
{'─'*60}
   Active Influencers    : {influence.get('active_influencers', 0)}
   Additional Spread     : {influence.get('additional_spread', 0)} nodes
   Amplification Score   : {influence.get('amplification_score', 0)}/10

🔍 3. FACT-CHECK IMPACT
{'─'*60}
   Verdict               : {fc.get('verdict', 'N/A')}
   Confidence            : {fc.get('confidence', 0)*100:.0f}%
   Nodes Checked         : {fc.get('nodes_checked', 0)}
   Nodes Warned          : {fc.get('nodes_warned', 0)}
   Active Fact-Checkers  : {fc.get('active_checkers', 0)} / {fc.get('total_fact_checkers', 0)}

🛡️ 4. MODERATION & CONTAINMENT
{'─'*60}
   Decision              : {mod.get('decision', 'N/A')}
   Severity              : {mod.get('severity', 'N/A')}
   Nodes Blocked         : {mod.get('nodes_blocked', 0)}
   Nodes Flagged         : {mod.get('nodes_flagged', 0)}
   Active Moderators     : {mod.get('active_moderators', 0)} / {mod.get('total_moderators', 0)}

📊 5. FINAL NODE STATUS
{'─'*60}
   Clean (Uninformed)    : {sc.get('clean', 0)} ({sc.get('clean', 0)/max(total,1)*100:.1f}%)
   Infected              : {sc.get('infected', 0)} ({sc.get('infected', 0)/max(total,1)*100:.1f}%)
   Influenced            : {sc.get('influenced', 0)} ({sc.get('influenced', 0)/max(total,1)*100:.1f}%)
   Warned                : {sc.get('warned', 0)} ({sc.get('warned', 0)/max(total,1)*100:.1f}%)
   Blocked               : {sc.get('blocked', 0)} ({sc.get('blocked', 0)/max(total,1)*100:.1f}%)

{'='*60}"""
