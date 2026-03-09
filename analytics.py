"""
Analytics Module
Provides comprehensive analytical capabilities for the misinformation simulation.
Covers: Spread Velocity, Verification Impact, Agent Influence, Moderation Effectiveness.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx


class SimulationAnalytics:
    """
    Generates detailed analytics and visualizations for simulation results.
    """

    def __init__(self, network):
        self.network = network

    def compute_spread_metrics(self, spread_data):
        """
        Compute detailed spread velocity metrics.
        """
        spread_per_step = spread_data.get("spread_per_step", [])
        total_reached = spread_data.get("total_reached", 0)
        total_nodes = self.network.num_nodes

        cumulative = list(np.cumsum(spread_per_step)) if spread_per_step else [0]

        # Average spread velocity (nodes per BFS level)
        avg_velocity = np.mean(spread_per_step) if spread_per_step else 0

        # Peak spread step
        peak_step = (np.argmax(spread_per_step) + 1) if spread_per_step else 0
        peak_spread = max(spread_per_step) if spread_per_step else 0

        return {
            "total_reached": total_reached,
            "total_nodes": total_nodes,
            "penetration_rate": round(total_reached / total_nodes * 100, 2),
            "spread_per_step": spread_per_step,
            "cumulative_spread": cumulative,
            "avg_velocity": round(avg_velocity, 2),
            "peak_step": peak_step,
            "peak_spread_count": peak_spread,
            "max_depth": spread_data.get("max_depth_reached", 0),
            "viral_coefficient": spread_data.get("viral_coefficient", 0),
        }

    def compute_verification_impact(self, spread_data, verification_data, moderation_data):
        """
        Compute the impact of verification on spread containment.
        """
        verdict = verification_data.get("verdict", "Unverified")
        confidence = verification_data.get("confidence", 0.5)
        decision = moderation_data.get("decision", "FLAG")

        total_reached = spread_data.get("total_reached", 0)
        total_nodes = self.network.num_nodes

        # Simulate post-verification metrics
        if verdict == "Fake" and decision == "BLOCK":
            post_verification_spread = 0
            containment = "Complete"
            reduction_rate = 100.0
        elif verdict == "Unverified" or decision == "FLAG":
            post_verification_spread = max(1, total_reached // 4)
            containment = "Partial"
            reduction_rate = 50.0
        else:
            post_verification_spread = total_reached
            containment = "None"
            reduction_rate = 0.0

        protected_nodes = total_nodes - total_reached - post_verification_spread
        protected_nodes = max(0, protected_nodes)

        return {
            "verdict": verdict,
            "confidence": confidence,
            "pre_verification_spread": total_reached,
            "post_verification_spread": post_verification_spread,
            "containment_type": containment,
            "spread_reduction_rate": reduction_rate,
            "protected_nodes": protected_nodes,
            "network_immunization_rate": round(protected_nodes / total_nodes * 100, 2),
        }

    def compute_agent_influence(self, spread_data, verification_data, influencer_data, moderation_data):
        """
        Compute which agent had the most influence on the simulation outcome.
        """
        total_reached = spread_data.get("total_reached", 0)
        total_nodes = self.network.num_nodes
        confidence = verification_data.get("confidence", 0.5)
        amp_score = influencer_data.get("amplification_score", 5.0)
        containment_rate = moderation_data.get("spread_impact", {}).get("containment_rate", 0)

        # Agent impact scores (normalized 0-10)
        scores = {
            "Misinformation Agent": 5.0,  # Baseline content creator
            "Neutral Agent (Spreader)": min(10.0, (total_reached / max(total_nodes * 0.1, 1)) * 2),
            "Fact-Checker Agent": confidence * 8.0,
            "Influencer Agent": amp_score,
            "Moderator Agent": (containment_rate / 100.0) * 10.0,
        }

        # Determine most influential agent
        most_influential = max(scores, key=scores.get)

        # Network centrality of spread path
        spread_path = spread_data.get("spread_path", [])
        most_central_node = None
        if spread_path and len(spread_path) > 1:
            subgraph = self.network.graph.subgraph(spread_path)
            if subgraph.number_of_nodes() > 1:
                centrality = nx.betweenness_centrality(subgraph)
                most_central_node = max(centrality, key=centrality.get)

        return {
            "agent_scores": scores,
            "most_influential_agent": most_influential,
            "most_influential_score": scores[most_influential],
            "most_central_node": most_central_node,
            "amplification_score": amp_score,
        }

    def compute_moderation_effectiveness(self, verification_data, moderation_data, spread_data):
        """
        Compute moderation effectiveness metrics.
        """
        verdict = verification_data.get("verdict", "Unverified")
        decision = moderation_data.get("decision", "FLAG")
        total_reached = spread_data.get("total_reached", 0)
        total_nodes = self.network.num_nodes

        # Determine if moderation was correct
        if verdict == "Fake" and decision in ["BLOCK", "FLAG"]:
            correct_decision = True
            decision_type = "True Positive"
        elif verdict == "Real" and decision == "ALLOW":
            correct_decision = True
            decision_type = "True Negative"
        elif verdict == "Real" and decision in ["BLOCK", "FLAG"]:
            correct_decision = False
            decision_type = "False Positive"
        elif verdict == "Fake" and decision == "ALLOW":
            correct_decision = False
            decision_type = "False Negative"
        else:
            correct_decision = True
            decision_type = "Cautionary (Unverified→Flag)"

        # Calculate containment completeness
        impact = moderation_data.get("spread_impact", {})
        containment_rate = impact.get("containment_rate", 0)

        return {
            "decision": decision,
            "verdict": verdict,
            "correct_decision": correct_decision,
            "decision_type": decision_type,
            "containment_rate": containment_rate,
            "nodes_protected": int(total_nodes * containment_rate / 100),
            "exposure_reduction": f"{containment_rate:.0f}%",
            "severity": moderation_data.get("severity", "MEDIUM"),
        }

    def generate_full_analytics(self, state):
        """
        Generate complete analytics from the simulation state.
        Returns a comprehensive analytics dict.
        """
        spread = state.get("spread_data", {})
        verification = state.get("verification_data", {})
        influencer = state.get("influencer_data", {})
        moderation = state.get("moderation_data", {})

        spread_metrics = self.compute_spread_metrics(spread)
        verification_impact = self.compute_verification_impact(spread, verification, moderation)
        agent_influence = self.compute_agent_influence(spread, verification, influencer, moderation)
        moderation_effectiveness = self.compute_moderation_effectiveness(verification, moderation, spread)

        return {
            "spread_metrics": spread_metrics,
            "verification_impact": verification_impact,
            "agent_influence": agent_influence,
            "moderation_effectiveness": moderation_effectiveness,
        }

    def generate_analytics_report(self, full_analytics):
        """
        Format analytics into a detailed text report.
        """
        sm = full_analytics["spread_metrics"]
        vi = full_analytics["verification_impact"]
        ai = full_analytics["agent_influence"]
        me = full_analytics["moderation_effectiveness"]

        report = f"""
{'='*60}
          DETAILED ANALYTICS REPORT
{'='*60}

📈 1. SPREAD VELOCITY ANALYSIS
{'─'*60}
   Total Nodes Reached    : {sm['total_reached']} / {sm['total_nodes']}
   Network Penetration    : {sm['penetration_rate']}%
   Average Velocity       : {sm['avg_velocity']} nodes/step
   Peak Spread Step       : Step {sm['peak_step']} ({sm['peak_spread_count']} nodes)
   Maximum Depth          : {sm['max_depth']} hops
   Viral Coefficient      : {sm['viral_coefficient']}
   Nodes Per Step         : {sm['spread_per_step']}

🔍 2. VERIFICATION IMPACT
{'─'*60}
   Verdict                : {vi['verdict']}
   Confidence             : {vi['confidence']*100:.0f}%
   Pre-Verification Spread: {vi['pre_verification_spread']} nodes
   Post-Verification Spread: {vi['post_verification_spread']} nodes
   Containment Type       : {vi['containment_type']}
   Spread Reduction       : {vi['spread_reduction_rate']}%
   Protected Nodes        : {vi['protected_nodes']}
   Network Immunization   : {vi['network_immunization_rate']}%

🤖 3. AGENT INFLUENCE RANKING
{'─'*60}
   Most Influential Agent : {ai['most_influential_agent']}
   Most Central Node      : User_{ai['most_central_node']}
   Amplification Score    : {ai['amplification_score']}/10

   Agent Scores:"""

        for agent, score in ai["agent_scores"].items():
            bar = "█" * int(score) + "░" * (10 - int(score))
            report += f"\n   {agent:28s} [{bar}] {score:.1f}/10"

        report += f"""

🛡️ 4. MODERATION EFFECTIVENESS
{'─'*60}
   Decision               : {me['decision']}
   Verdict Match          : {me['decision_type']}
   Correct Decision       : {'✅ Yes' if me['correct_decision'] else '❌ No'}
   Containment Rate       : {me['containment_rate']:.0f}%
   Nodes Protected        : {me['nodes_protected']}
   Exposure Reduction     : {me['exposure_reduction']}
   Severity Level         : {me['severity']}

{'='*60}"""

        return report
