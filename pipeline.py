import time
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from social_network import SocialNetwork, create_network
from agents.misinformation_agent import MisinformationAgent
from agents.neutral_agent import NeutralAgent
from agents.fact_checker_agent import FactCheckerAgent
from agents.influencer_agent import InfluencerAgent
from agents.moderator_agent import ModeratorAgent
from config import (
    NETWORK_NUM_NODES, NETWORK_EDGES_PER_NODE, NETWORK_SEED,
    SPREAD_PROBABILITY, AMPLIFICATION_FACTOR,
)



class SimulationState(TypedDict):
    network_stats: Dict[str, Any]

    
    claim: Dict[str, Any]
    injection_result: Dict[str, Any]

   
    spread_result: Dict[str, Any]
    influence_result: Dict[str, Any]
    fact_check_result: Dict[str, Any]
    moderation_result: Dict[str, Any]

    agent_stats_table: List[List[Any]]


    analytics: Dict[str, Any]
    network_graph_path: str
    analysis_chart_path: str

   
    pipeline_log: List[str]
    current_step: str
    start_time: float
    elapsed_time: float
    error: str


class MisinformationPipeline:
  

    def __init__(
        self,
        num_nodes=NETWORK_NUM_NODES,
        edges_per_node=NETWORK_EDGES_PER_NODE,
        seed=NETWORK_SEED,
        spread_prob=SPREAD_PROBABILITY,
        amplification=AMPLIFICATION_FACTOR,
    ):
        self.spread_prob = spread_prob
        self.amplification = amplification

        self.network = create_network(num_nodes, edges_per_node, seed)
        self.misinfo_agent = MisinformationAgent()
        self.neutral_agent = NeutralAgent(self.network)
        self.fact_checker = FactCheckerAgent()
        self.influencer_agent = InfluencerAgent()
        self.moderator_agent = ModeratorAgent()

        self.graph = self._build_graph()

    def _build_graph(self):
        wf = StateGraph(SimulationState)
        wf.add_node("generate_claim",  self._step_generate)
        wf.add_node("spread_claim",    self._step_spread)
        wf.add_node("influence_claim", self._step_influence)
        wf.add_node("check_claim",     self._step_verify)
        wf.add_node("moderate_claim",  self._step_moderate)
        wf.add_node("analyse",         self._step_analyse)

        wf.set_entry_point("generate_claim")
        wf.add_edge("generate_claim",  "spread_claim")
        wf.add_edge("spread_claim",    "influence_claim")
        wf.add_edge("influence_claim", "check_claim")
        wf.add_edge("check_claim",     "moderate_claim")
        wf.add_edge("moderate_claim",  "analyse")
        wf.add_edge("analyse",         END)
        return wf.compile()



    def _step_generate(self, state: SimulationState) -> dict:
        log = list(state.get("pipeline_log", []))
        log.append("🔴 Step 1: Misinformation Agent generating claim via LLM …")
        try:
            claim = self.misinfo_agent.generate_claim()
            injection = self.misinfo_agent.inject_into_graph(self.network, claim)
            log.append(f"   ✅ Claim generated: {claim['claim'][:80]}…")
            log.append(f"   📍 Injected at node User_{injection.get('source_node', '?')}")
            return {
                "claim": claim,
                "injection_result": injection,
                "current_step": "generate_claim",
                "pipeline_log": log,
            }
        except Exception as e:
            log.append(f"   ❌ Error: {e}")
            return {"claim": {}, "injection_result": {},
                    "pipeline_log": log, "error": str(e)}

    def _step_spread(self, state: SimulationState) -> dict:
        log = list(state.get("pipeline_log", []))
        log.append("🔵 Step 2: Neutral Agent spreading claim through BFS …")
        try:
            spread_result = self.neutral_agent.spread_claim(
                spread_prob=self.spread_prob,
            )
            log.append(
                f"   ✅ Spread to {spread_result.get('total_spread', 0)} nodes "
                f"({spread_result.get('penetration_rate', 0)}% penetration)"
            )
            return {
                "spread_result": spread_result,
                "current_step": "spread_claim",
                "pipeline_log": log,
            }
        except Exception as e:
            log.append(f"   ❌ Error: {e}")
            return {"spread_result": {}, "pipeline_log": log, "error": str(e)}

    def _step_influence(self, state: SimulationState) -> dict:
        log = list(state.get("pipeline_log", []))
        log.append("🟠 Step 3: Influencer Agent modifying & amplifying claim …")
        try:
            influence_result = self.influencer_agent.influence_graph(
                self.network, amplification=self.amplification,
            )
            log.append(
                f"   ✅ {influence_result.get('active_influencers', 0)} influencers active, "
                f"{influence_result.get('additional_spread', 0)} additional nodes influenced"
            )
            return {
                "influence_result": influence_result,
                "current_step": "influence_claim",
                "pipeline_log": log,
            }
        except Exception as e:
            log.append(f"   ❌ Error: {e}")
            return {"influence_result": {}, "pipeline_log": log, "error": str(e)}

    def _step_verify(self, state: SimulationState) -> dict:
        log = list(state.get("pipeline_log", []))
        log.append("🟢 Step 4: Fact-Checker Agent verifying claim in graph …")
        try:
            fact_check_result = self.fact_checker.check_graph(self.network)
            emoji = {"Real": "✅", "Fake": "❌", "Unverified": "⚠️"}.get(
                fact_check_result["verdict"], "❓"
            )
            log.append(
                f"   {emoji} Verdict: {fact_check_result['verdict']} "
                f"({fact_check_result['confidence']*100:.0f}% confidence)"
            )
            log.append(
                f"   📋 Checked {fact_check_result.get('nodes_checked', 0)} nodes, "
                f"warned {fact_check_result.get('nodes_warned', 0)} nodes"
            )
            return {
                "fact_check_result": fact_check_result,
                "current_step": "check_claim",
                "pipeline_log": log,
            }
        except Exception as e:
            log.append(f"   ❌ Error: {e}")
            return {"fact_check_result": {}, "pipeline_log": log, "error": str(e)}

    def _step_moderate(self, state: SimulationState) -> dict:
        log = list(state.get("pipeline_log", []))
        fact_check = state.get("fact_check_result", {})
        log.append("🟣 Step 5: Moderator Agent taking action …")
        try:
            moderation_result = self.moderator_agent.moderate_graph(
                self.network, fact_check,
            )
            d_emoji = {"BLOCK": "🚫", "FLAG": "⚠️", "ALLOW": "✅"}.get(
                moderation_result["decision"], "❓"
            )
            log.append(
                f"   {d_emoji} Decision: {moderation_result['decision']} "
                f"({moderation_result.get('severity', '?')})"
            )
            log.append(
                f"   🚫 Blocked: {moderation_result.get('nodes_blocked', 0)} | "
                f"⚠️ Flagged: {moderation_result.get('nodes_flagged', 0)}"
            )
            return {
                "moderation_result": moderation_result,
                "current_step": "moderate_claim",
                "pipeline_log": log,
            }
        except Exception as e:
            log.append(f"   ❌ Error: {e}")
            return {"moderation_result": {}, "pipeline_log": log, "error": str(e)}

    def _step_analyse(self, state: SimulationState) -> dict:
        log = list(state.get("pipeline_log", []))
        log.append("📊 Step 6: Generating analytics & visualisations …")

        spread_edges = state.get("spread_result", {}).get("edges", [])
        influence_edges = state.get("influence_result", {}).get("edges", [])
        all_edges = spread_edges + influence_edges

        try:
            graph_path = self.network.visualize_network(
                title=f"Misinformation Simulation — {self.network.num_nodes} Nodes",
                save_path="network_graph.png",
                show_cascade_edges=all_edges if all_edges else None,
            )
        except Exception:
            graph_path = ""

        try:
            chart_path = self.network.visualize_spread_analysis(
                state, save_path="spread_analysis.png",
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            chart_path = ""

        elapsed = time.time() - state.get("start_time", time.time())
        log.append(f"   ✅ Analysis complete ({elapsed:.1f}s)")

        agent_stats = self.network.get_agent_stats_table()

        return {
            "network_stats": self.network.get_network_stats(),
            "agent_stats_table": agent_stats,
            "network_graph_path": graph_path,
            "analysis_chart_path": chart_path,
            "current_step": "analyse",
            "elapsed_time": elapsed,
            "pipeline_log": log,
        }


    def run_simulation(self) -> dict:
        self.network.reset_statuses()
        initial = {
            "network_stats": {},
            "claim": {},
            "injection_result": {},
            "spread_result": {},
            "influence_result": {},
            "fact_check_result": {},
            "moderation_result": {},
            "agent_stats_table": [],
            "analytics": {},
            "network_graph_path": "",
            "analysis_chart_path": "",
            "pipeline_log": ["🚀 Starting Sequential Misinformation Simulation …"],
            "current_step": "init",
            "start_time": time.time(),
            "elapsed_time": 0.0,
            "error": "",
        }
        return self.graph.invoke(initial)

    def get_full_report(self, state: dict) -> str:
        claim = state.get("claim", {})
        spread = state.get("spread_result", {})
        influence = state.get("influence_result", {})
        fc = state.get("fact_check_result", {})
        mod = state.get("moderation_result", {})
        sc = mod.get("final_status_counts", {})

        return f"""
{'='*65}
   MISINFORMATION SIMULATION REPORT
{'='*65}

📰 CLAIM
   "{claim.get('claim', 'N/A')}"
   Topic: {claim.get('topic', 'N/A')} | Generated by: {claim.get('generated_by', 'N/A')}

📡 NEUTRAL SPREAD (Phase 1)
   Nodes Infected   : {spread.get('total_spread', 0)} / {spread.get('total_nodes', 0)}
   Penetration      : {spread.get('penetration_rate', 0)}%
   Max Depth        : {spread.get('max_depth', 0)} hops

📣 INFLUENCER (Phase 2)
   Active Influencers   : {influence.get('active_influencers', 0)}
   Additional Spread    : {influence.get('additional_spread', 0)} nodes
   Amplification Score  : {influence.get('amplification_score', 0)}/10
   Modified Claim       : {(influence.get('modified_claim', 'N/A'))[:120]}

🔍 FACT-CHECK (Phase 3)
   Verdict    : {fc.get('verdict', 'N/A')}
   Confidence : {fc.get('confidence', 0)*100:.0f}%
   Evidence   : {fc.get('evidence', 'N/A')[:150]}
   Nodes Checked : {fc.get('nodes_checked', 0)}
   Nodes Warned  : {fc.get('nodes_warned', 0)}

🛡️ MODERATION (Phase 4)
   Decision   : {mod.get('decision', 'N/A')}
   Severity   : {mod.get('severity', 'N/A')}
   Reason     : {mod.get('reason', 'N/A')}
   Blocked    : {mod.get('nodes_blocked', 0)} nodes
   Flagged    : {mod.get('nodes_flagged', 0)} nodes

📊 FINAL GRAPH STATUS
   Clean: {sc.get('clean', 0)} | Infected: {sc.get('infected', 0)} | \
Influenced: {sc.get('influenced', 0)} | Warned: {sc.get('warned', 0)} | \
Blocked: {sc.get('blocked', 0)}

⏱️ Simulation Time: {state.get('elapsed_time', 0):.1f}s
{'='*65}
"""
