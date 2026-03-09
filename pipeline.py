"""
LangGraph Pipeline for Multi-Agent Misinformation Simulation
Orchestrates all agents in a sequential pipeline with state management.
"""
import time
import random
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from social_network import SocialNetwork, create_network
from agents.misinformation_agent import MisinformationAgent
from agents.neutral_agent import NeutralAgent
from agents.fact_checker_agent import FactCheckerAgent
from agents.influencer_agent import InfluencerAgent
from agents.moderator_agent import ModeratorAgent
from config import NETWORK_NUM_NODES, NETWORK_EDGES_PER_NODE, NETWORK_SEED, NUM_INFLUENCERS, NUM_FACT_CHECKERS, NUM_MODERATORS


# ─── State Definition ─────────────────────────────────────────────────────────
class SimulationState(TypedDict):
    """State that flows through the LangGraph pipeline."""
    # Network
    network_stats: Dict[str, Any]

    # Agent outputs
    claim_data: Dict[str, Any]
    spread_data: Dict[str, Any]
    verification_data: Dict[str, Any]
    influencer_data: Dict[str, Any]
    moderation_data: Dict[str, Any]

    # Summaries
    claim_summary: str
    spread_summary: str
    verification_summary: str
    influencer_summary: str
    moderation_summary: str

    # Analytics
    analytics: Dict[str, Any]

    # Visualizations
    network_graph_path: str
    analysis_chart_path: str

    # Pipeline metadata
    pipeline_log: List[str]
    current_step: str
    start_time: float
    elapsed_time: float
    error: str


class MisinformationPipeline:
    """
    LangGraph-based pipeline that orchestrates the multi-agent simulation.
    
    Pipeline flow:
    generate_claim → spread_claim → verify_claim → influence_claim → moderate_claim → analyze_results
    """

    def __init__(self, num_nodes=NETWORK_NUM_NODES, edges_per_node=NETWORK_EDGES_PER_NODE, seed=NETWORK_SEED,
                 num_influencers=NUM_INFLUENCERS, num_fact_checkers=NUM_FACT_CHECKERS, num_moderators=NUM_MODERATORS):
        # Create social network
        self.network = create_network(num_nodes, edges_per_node, seed, num_influencers, num_fact_checkers, num_moderators)

        # Initialize agents
        self.misinfo_agent = MisinformationAgent()
        self.neutral_agent = NeutralAgent(self.network)
        self.fact_checker = FactCheckerAgent()
        self.influencer_agent = InfluencerAgent()
        self.moderator_agent = ModeratorAgent()

        # Build the LangGraph pipeline
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph state graph with all agent nodes."""
        workflow = StateGraph(SimulationState)

        # Add nodes
        workflow.add_node("generate_claim", self._step_generate_claim)
        workflow.add_node("spread_claim", self._step_spread_claim)
        workflow.add_node("verify_claim", self._step_verify_claim)
        workflow.add_node("influence_claim", self._step_influence_claim)
        workflow.add_node("moderate_claim", self._step_moderate_claim)
        workflow.add_node("analyze_results", self._step_analyze_results)

        # Define edges (sequential pipeline)
        workflow.set_entry_point("generate_claim")
        workflow.add_edge("generate_claim", "spread_claim")
        workflow.add_edge("spread_claim", "verify_claim")
        workflow.add_edge("verify_claim", "influence_claim")
        workflow.add_edge("influence_claim", "moderate_claim")
        workflow.add_edge("moderate_claim", "analyze_results")
        workflow.add_edge("analyze_results", END)

        return workflow.compile()

    # ─── Pipeline Steps ───────────────────────────────────────────────────────

    def _step_generate_claim(self, state: SimulationState) -> dict:
        """Step 1: Generate a misinformation claim."""
        log = state.get("pipeline_log", [])
        log.append("🔴 Step 1: Misinformation Agent generating claim...")

        try:
            claim_data = self.misinfo_agent.generate_claim()
            claim_text = claim_data["claim"]
            log.append(f"   ✅ Claim generated: '{claim_text[:80]}...'")

            return {
                "claim_data": claim_data,
                "claim_summary": f"📰 Generated Claim: {claim_text}",
                "current_step": "generate_claim",
                "pipeline_log": log,
            }
        except Exception as e:
            log.append(f"   ❌ Error: {str(e)}")
            return {
                "claim_data": {"claim": "Error generating claim", "error": str(e)},
                "claim_summary": f"Error: {str(e)}",
                "current_step": "generate_claim",
                "pipeline_log": log,
                "error": str(e),
            }

    def _step_spread_claim(self, state: SimulationState) -> dict:
        """Step 2: Spread the claim through the network."""
        log = state.get("pipeline_log", [])
        log.append("🔵 Step 2: Neutral Agent spreading claim through network...")

        claim_text = state["claim_data"]["claim"]
        start_node = self.network.get_random_start_node()

        try:
            spread_data = self.neutral_agent.spread_claim(start_node, claim_text)
            spread_summary = self.neutral_agent.get_spread_summary(spread_data)
            log.append(f"   ✅ Spread complete: {spread_data['total_reached']} nodes reached ({spread_data['penetration_rate']}%)")

            # Generate network visualization
            graph_path = self.network.visualize_network(
                spread_path=spread_data["spread_path"],
                title=f"Misinformation Spread from Node {start_node} — {spread_data['total_reached']} Nodes Reached",
                save_path="network_graph.png"
            )

            return {
                "spread_data": spread_data,
                "spread_summary": spread_summary,
                "network_graph_path": graph_path,
                "network_stats": self.network.get_network_stats(),
                "current_step": "spread_claim",
                "pipeline_log": log,
            }
        except Exception as e:
            log.append(f"   ❌ Error: {str(e)}")
            return {
                "spread_data": {"total_reached": 0, "spread_path": [], "error": str(e)},
                "spread_summary": f"Error: {str(e)}",
                "current_step": "spread_claim",
                "pipeline_log": log,
                "error": str(e),
            }

    def _step_verify_claim(self, state: SimulationState) -> dict:
        """Step 3: Fact-check the claim."""
        log = state.get("pipeline_log", [])
        log.append("🟢 Step 3: Fact-Checker Agent verifying claim...")

        claim_text = state["claim_data"]["claim"]

        try:
            verification_data = self.fact_checker.verify_claim(claim_text)
            verification_summary = self.fact_checker.get_verdict_summary(verification_data)
            log.append(f"   ✅ Verdict: {verification_data['verdict']} (confidence: {verification_data['confidence']*100:.0f}%)")

            return {
                "verification_data": verification_data,
                "verification_summary": verification_summary,
                "current_step": "verify_claim",
                "pipeline_log": log,
            }
        except Exception as e:
            log.append(f"   ❌ Error: {str(e)}")
            return {
                "verification_data": {"verdict": "Unverified", "confidence": 0.5, "error": str(e)},
                "verification_summary": f"Error: {str(e)}",
                "current_step": "verify_claim",
                "pipeline_log": log,
                "error": str(e),
            }

    def _step_influence_claim(self, state: SimulationState) -> dict:
        """Step 4: Influencer rewrites the claim."""
        log = state.get("pipeline_log", [])
        log.append("🟠 Step 4: Influencer Agent rewriting content...")

        claim_text = state["claim_data"]["claim"]
        verdict = state["verification_data"]["verdict"]
        confidence = state["verification_data"]["confidence"]

        try:
            influencer_data = self.influencer_agent.rewrite_claim(claim_text, verdict, confidence)
            influencer_summary = self.influencer_agent.get_influencer_summary(influencer_data)
            log.append(f"   ✅ Content rewritten ({influencer_data['action_type']}, score: {influencer_data['amplification_score']}/10)")

            return {
                "influencer_data": influencer_data,
                "influencer_summary": influencer_summary,
                "current_step": "influence_claim",
                "pipeline_log": log,
            }
        except Exception as e:
            log.append(f"   ❌ Error: {str(e)}")
            return {
                "influencer_data": {"rewritten_content": claim_text, "amplification_score": 0, "error": str(e)},
                "influencer_summary": f"Error: {str(e)}",
                "current_step": "influence_claim",
                "pipeline_log": log,
                "error": str(e),
            }

    def _step_moderate_claim(self, state: SimulationState) -> dict:
        """Step 5: Moderator makes a decision."""
        log = state.get("pipeline_log", [])
        log.append("🟣 Step 5: Moderator Agent reviewing content...")

        claim_text = state["claim_data"]["claim"]
        verdict = state["verification_data"]["verdict"]
        confidence = state["verification_data"]["confidence"]
        evidence = state["verification_data"].get("evidence", "")
        rewritten = state["influencer_data"].get("rewritten_content", "")

        try:
            moderation_data = self.moderator_agent.moderate_content(
                claim_text=claim_text,
                verdict=verdict,
                confidence=confidence,
                evidence=evidence,
                rewritten_content=rewritten,
                spread_data=state.get("spread_data", {}),
            )
            moderation_summary = self.moderator_agent.get_moderation_summary(moderation_data)
            log.append(f"   ✅ Decision: {moderation_data['decision']} (severity: {moderation_data.get('severity', 'N/A')})")

            return {
                "moderation_data": moderation_data,
                "moderation_summary": moderation_summary,
                "current_step": "moderate_claim",
                "pipeline_log": log,
            }
        except Exception as e:
            log.append(f"   ❌ Error: {str(e)}")
            return {
                "moderation_data": {"decision": "FLAG", "error": str(e)},
                "moderation_summary": f"Error: {str(e)}",
                "current_step": "moderate_claim",
                "pipeline_log": log,
                "error": str(e),
            }

    def _step_analyze_results(self, state: SimulationState) -> dict:
        """Step 6: Generate comprehensive analytics."""
        log = state.get("pipeline_log", [])
        log.append("📊 Step 6: Generating analytics and visualizations...")

        spread = state.get("spread_data", {})
        verification = state.get("verification_data", {})
        influencer = state.get("influencer_data", {})
        moderation = state.get("moderation_data", {})

        # Compute analytics
        analytics = self._compute_analytics(spread, verification, influencer, moderation)

        # Generate analysis chart
        try:
            analysis_data = {
                "spread_per_step": spread.get("spread_per_step", []),
                "total_nodes": self.network.num_nodes,
                "nodes_reached": spread.get("total_reached", 0),
                "agent_scores": analytics.get("agent_influence_scores", {}),
                "moderation_stats": analytics.get("moderation_stats", {}),
            }
            chart_path = self.network.visualize_spread_analysis(
                analysis_data, save_path="spread_analysis.png"
            )
        except Exception:
            chart_path = ""

        elapsed = time.time() - state.get("start_time", time.time())
        log.append(f"   ✅ Analysis complete in {elapsed:.1f}s")

        return {
            "analytics": analytics,
            "analysis_chart_path": chart_path,
            "current_step": "analyze_results",
            "elapsed_time": elapsed,
            "pipeline_log": log,
        }

    def _compute_analytics(self, spread, verification, influencer, moderation):
        """Compute comprehensive simulation analytics."""
        total_nodes = self.network.num_nodes
        nodes_reached = spread.get("total_reached", 0)
        verdict = verification.get("verdict", "Unverified")
        decision = moderation.get("decision", "FLAG")
        amp_score = influencer.get("amplification_score", 5.0)

        # Spread velocity
        spread_per_step = spread.get("spread_per_step", [])
        avg_velocity = sum(spread_per_step) / max(len(spread_per_step), 1) if spread_per_step else 0

        # Containment metrics
        mod_impact = moderation.get("spread_impact", {})
        containment_rate = mod_impact.get("containment_rate", 0)
        protected_nodes = int(total_nodes * (containment_rate / 100))

        # Agent influence scores
        agent_scores = {
            "Misinformation": 5.0,  # Baseline creator
            "Neutral\n(Spreader)": min(10, nodes_reached / (total_nodes / 10)),
            "Fact-Checker": verification.get("confidence", 0.5) * 8,
            "Influencer": amp_score,
            "Moderator": (containment_rate / 100) * 10,
        }

        # Moderation effectiveness stats
        moderation_stats = {
            "Blocked": 1 if decision == "BLOCK" else 0,
            "Flagged": 1 if decision == "FLAG" else 0,
            "Allowed": 1 if decision == "ALLOW" else 0,
            "Containment\n%": containment_rate,
        }

        # Network analysis
        spread_path = spread.get("spread_path", [])
        most_influential_node = None
        if spread_path:
            subgraph = self.network.graph.subgraph(spread_path)
            if len(subgraph) > 0:
                betweenness = dict(subgraph.degree())
                most_influential_node = max(betweenness, key=betweenness.get)

        return {
            # Spread metrics
            "spread_velocity_avg": round(avg_velocity, 2),
            "nodes_reached": nodes_reached,
            "penetration_rate": spread.get("penetration_rate", 0),
            "max_depth": spread.get("max_depth_reached", 0),
            "viral_coefficient": spread.get("viral_coefficient", 0),
            "total_exposures": spread.get("total_exposures", 0),

            # Verification metrics
            "verdict": verdict,
            "confidence": verification.get("confidence", 0),
            "verification_impact": "Spread Halted" if decision == "BLOCK" else (
                "Spread Reduced" if decision == "FLAG" else "No Containment"
            ),

            # Influence metrics
            "amplification_score": amp_score,
            "most_influential_node": most_influential_node,
            "agent_influence_scores": agent_scores,

            # Moderation metrics
            "moderation_decision": decision,
            "containment_rate": containment_rate,
            "protected_nodes": protected_nodes,
            "moderation_success": decision in ["BLOCK", "FLAG"] and verdict in ["Fake", "Unverified"],
            "moderation_stats": moderation_stats,
        }

    def run_simulation(self):
        """
        Execute the complete multi-agent simulation pipeline.
        
        Returns:
            dict: Complete simulation state with all results
        """
        initial_state = {
            "network_stats": {},
            "claim_data": {},
            "spread_data": {},
            "verification_data": {},
            "influencer_data": {},
            "moderation_data": {},
            "claim_summary": "",
            "spread_summary": "",
            "verification_summary": "",
            "influencer_summary": "",
            "moderation_summary": "",
            "analytics": {},
            "network_graph_path": "",
            "analysis_chart_path": "",
            "pipeline_log": ["🚀 Starting Multi-Agent Misinformation Simulation..."],
            "current_step": "init",
            "start_time": time.time(),
            "elapsed_time": 0.0,
            "error": "",
        }

        # Run the LangGraph pipeline
        result = self.graph.invoke(initial_state)
        return result

    def get_full_report(self, state):
        """Generate a complete simulation report from the final state."""
        analytics = state.get("analytics", {})
        claim = state.get("claim_data", {}).get("claim", "N/A")
        verdict = analytics.get("verdict", "N/A")
        decision = analytics.get("moderation_decision", "N/A")

        report = f"""
{'='*60}
   AI MULTI-AGENT MISINFORMATION SIMULATION REPORT
{'='*60}

📰 ORIGINAL CLAIM:
   "{claim}"

📊 SIMULATION RESULTS:
{'─'*60}

1️⃣  SPREAD ANALYSIS (How fast did fake news spread?)
   • Nodes Reached: {analytics.get('nodes_reached', 0)} / {self.network.num_nodes}
   • Network Penetration: {analytics.get('penetration_rate', 0)}%
   • Spread Depth: {analytics.get('max_depth', 0)} hops
   • Viral Coefficient: {analytics.get('viral_coefficient', 0)}
   • Avg Spread Velocity: {analytics.get('spread_velocity_avg', 0)} nodes/step
   • Total Exposures: {analytics.get('total_exposures', 0)}

2️⃣  VERIFICATION IMPACT (How verification slowed/stopped spread)
   • Verdict: {verdict}
   • Confidence: {analytics.get('confidence', 0)*100:.0f}%
   • Impact: {analytics.get('verification_impact', 'N/A')}

3️⃣  AGENT INFLUENCE (Which agent influenced spread most?)
   • Most Influential Node: User_{analytics.get('most_influential_node', 'N/A')}
   • Amplification Score: {analytics.get('amplification_score', 0)}/10
   • Agent Rankings:"""

        for agent, score in analytics.get("agent_influence_scores", {}).items():
            bar = "█" * int(score) + "░" * (10 - int(score))
            report += f"\n     {agent:20s} [{bar}] {score:.1f}/10"

        report += f"""

4️⃣  MODERATION EFFECTIVENESS (How moderation reduced misinformation)
   • Decision: {decision}
   • Containment Rate: {analytics.get('containment_rate', 0):.0f}%
   • Protected Nodes: {analytics.get('protected_nodes', 0)}
   • Moderation Successful: {'✅ Yes' if analytics.get('moderation_success') else '❌ No'}

⏱️  Simulation Time: {state.get('elapsed_time', 0):.1f}s
{'='*60}
"""
        return report
