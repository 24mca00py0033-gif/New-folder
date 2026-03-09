"""
LangGraph Pipeline for Multi-Agent Misinformation Simulation
=============================================================
Orchestrates:
  1. Generate N unique misinformation claims (one per misinfo-agent node)
  2. Run simultaneous BFS cascades through the network
  3. Fact-check every claim
  4. Influencer rewrite every claim
  5. Moderator verdict for every claim
  6. Aggregate analytics + visualisations
"""
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
    NUM_MISINFO_AGENTS, NUM_INFLUENCERS, NUM_FACT_CHECKERS, NUM_MODERATORS,
    SPREAD_PROBABILITY, AMPLIFICATION_FACTOR,
)


# ─── State Definition ─────────────────────────────────────────────────────────

class SimulationState(TypedDict):
    network_stats: Dict[str, Any]

    # per-claim lists
    claims: List[Dict[str, Any]]
    verification_results: List[Dict[str, Any]]
    influencer_results: List[Dict[str, Any]]
    moderation_results: List[Dict[str, Any]]

    # multi-cascade aggregate
    spread_result: Dict[str, Any]

    # summaries
    claim_summary: str
    spread_summary: str
    verification_summary: str
    influencer_summary: str
    moderation_summary: str

    # analytics + visuals
    analytics: Dict[str, Any]
    network_graph_path: str
    analysis_chart_path: str

    # meta
    pipeline_log: List[str]
    current_step: str
    start_time: float
    elapsed_time: float
    error: str


# ─── Pipeline ────────────────────────────────────────────────────────────────

class MisinformationPipeline:
    """
    LangGraph pipeline:
      generate_claims → run_cascades → verify_claims →
      influence_claims → moderate_claims → analyse
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
        self.network = create_network(
            num_nodes, edges_per_node, seed,
            num_misinfo, num_influencers, num_fact_checkers, num_moderators,
            spread_prob, amplification,
        )
        self.misinfo_agent = MisinformationAgent()
        self.neutral_agent = NeutralAgent(self.network)
        self.fact_checker = FactCheckerAgent()
        self.influencer_agent = InfluencerAgent()
        self.moderator_agent = ModeratorAgent()

        self.graph = self._build_graph()

    def _build_graph(self):
        wf = StateGraph(SimulationState)
        wf.add_node("generate_claims",  self._step_generate)
        wf.add_node("run_cascades",     self._step_spread)
        wf.add_node("verify_claims",    self._step_verify)
        wf.add_node("influence_claims", self._step_influence)
        wf.add_node("moderate_claims",  self._step_moderate)
        wf.add_node("analyse",          self._step_analyse)

        wf.set_entry_point("generate_claims")
        wf.add_edge("generate_claims",  "run_cascades")
        wf.add_edge("run_cascades",     "verify_claims")
        wf.add_edge("verify_claims",    "influence_claims")
        wf.add_edge("influence_claims", "moderate_claims")
        wf.add_edge("moderate_claims",  "analyse")
        wf.add_edge("analyse",          END)
        return wf.compile()

    # ── steps ─────────────────────────────────────────────────────────────────

    def _step_generate(self, state: SimulationState) -> dict:
        log = list(state.get("pipeline_log", []))
        n_misinfo = len(self.network.get_misinfo_nodes())
        log.append(f"🔴 Step 1: Generating {n_misinfo} misinformation claims …")
        try:
            claims = self.misinfo_agent.generate_batch(n_misinfo)
            summary_lines = [f"  • C{i}: {c['claim'][:80]}…" for i, c in enumerate(claims)]
            log.append(f"   ✅ {len(claims)} claims generated")
            return {
                "claims": claims,
                "claim_summary": "📰 Generated Claims:\n" + "\n".join(summary_lines),
                "current_step": "generate_claims",
                "pipeline_log": log,
            }
        except Exception as e:
            log.append(f"   ❌ Error: {e}")
            return {"claims": [], "claim_summary": f"Error: {e}",
                    "pipeline_log": log, "error": str(e)}

    def _step_spread(self, state: SimulationState) -> dict:
        log = list(state.get("pipeline_log", []))
        log.append("🔵 Step 2: Running simultaneous BFS cascades …")
        claims = state.get("claims", [])
        try:
            spread_result = self.network.run_multi_cascade(claims)
            summary = self.neutral_agent.get_spread_summary(spread_result)
            log.append(
                f"   ✅ {spread_result['num_cascades']} cascades → "
                f"{spread_result['total_reached']} nodes reached "
                f"({spread_result['penetration_rate']}%)"
            )
            graph_path = self.network.visualize_network(
                spread_result=spread_result,
                title=(
                    f"Multi-Cascade Spread — "
                    f"{spread_result['total_reached']}/{self.network.num_nodes} nodes reached"
                ),
                save_path="network_graph.png",
            )
            return {
                "spread_result": spread_result,
                "spread_summary": summary,
                "network_graph_path": graph_path,
                "network_stats": self.network.get_network_stats(),
                "current_step": "run_cascades",
                "pipeline_log": log,
            }
        except Exception as e:
            log.append(f"   ❌ Error: {e}")
            return {"spread_result": {}, "spread_summary": f"Error: {e}",
                    "pipeline_log": log, "error": str(e)}

    def _step_verify(self, state: SimulationState) -> dict:
        log = list(state.get("pipeline_log", []))
        claims = state.get("claims", [])
        log.append(f"🟢 Step 3: Fact-checking {len(claims)} claims …")
        try:
            verdicts = self.fact_checker.verify_batch(claims)
            lines = []
            for v in verdicts:
                emoji = {"Real": "✅", "Fake": "❌", "Unverified": "⚠️"}.get(v["verdict"], "❓")
                lines.append(f"  {emoji} {v['verdict']} ({v['confidence']*100:.0f}%) — {v.get('claim','')[:60]}")
            log.append(f"   ✅ Verdicts: " + ", ".join(v["verdict"] for v in verdicts))
            return {
                "verification_results": verdicts,
                "verification_summary": "🔍 Fact-Check Results:\n" + "\n".join(lines),
                "current_step": "verify_claims",
                "pipeline_log": log,
            }
        except Exception as e:
            log.append(f"   ❌ Error: {e}")
            return {"verification_results": [], "verification_summary": f"Error: {e}",
                    "pipeline_log": log, "error": str(e)}

    def _step_influence(self, state: SimulationState) -> dict:
        log = list(state.get("pipeline_log", []))
        claims = state.get("claims", [])
        verdicts = state.get("verification_results", [])
        log.append(f"🟠 Step 4: Influencer rewriting {len(claims)} claims …")
        try:
            results = self.influencer_agent.rewrite_batch(claims, verdicts)
            lines = [f"  • {r['action_type']} (score {r['amplification_score']}/10)"
                     for r in results]
            log.append(f"   ✅ {len(results)} claims rewritten")
            return {
                "influencer_results": results,
                "influencer_summary": "📣 Influencer Rewrites:\n" + "\n".join(lines),
                "current_step": "influence_claims",
                "pipeline_log": log,
            }
        except Exception as e:
            log.append(f"   ❌ Error: {e}")
            return {"influencer_results": [], "influencer_summary": f"Error: {e}",
                    "pipeline_log": log, "error": str(e)}

    def _step_moderate(self, state: SimulationState) -> dict:
        log = list(state.get("pipeline_log", []))
        claims = state.get("claims", [])
        verdicts = state.get("verification_results", [])
        inf_results = state.get("influencer_results", [])
        spread = state.get("spread_result", {})
        log.append(f"🟣 Step 5: Moderating {len(claims)} claims …")
        try:
            results = self.moderator_agent.moderate_batch(claims, verdicts, inf_results, spread)
            lines = [f"  {'🚫' if r['decision']=='BLOCK' else '⚠️' if r['decision']=='FLAG' else '✅'} "
                     f"{r['decision']} ({r.get('severity','?')})"
                     for r in results]
            log.append(f"   ✅ Decisions: " + ", ".join(r["decision"] for r in results))
            return {
                "moderation_results": results,
                "moderation_summary": "🛡️ Moderation Decisions:\n" + "\n".join(lines),
                "current_step": "moderate_claims",
                "pipeline_log": log,
            }
        except Exception as e:
            log.append(f"   ❌ Error: {e}")
            return {"moderation_results": [], "moderation_summary": f"Error: {e}",
                    "pipeline_log": log, "error": str(e)}

    def _step_analyse(self, state: SimulationState) -> dict:
        log = list(state.get("pipeline_log", []))
        log.append("📊 Step 6: Generating analytics & visualisations …")
        spread = state.get("spread_result", {})

        try:
            chart_path = self.network.visualize_spread_analysis(
                spread, save_path="spread_analysis.png"
            )
        except Exception:
            chart_path = ""

        elapsed = time.time() - state.get("start_time", time.time())
        log.append(f"   ✅ Analysis complete ({elapsed:.1f}s)")

        analytics = self._compute_analytics(state)

        return {
            "analytics": analytics,
            "analysis_chart_path": chart_path,
            "current_step": "analyse",
            "elapsed_time": elapsed,
            "pipeline_log": log,
        }

    # ── analytics computation ─────────────────────────────────────────────────

    def _compute_analytics(self, state: SimulationState) -> dict:
        spread = state.get("spread_result", {})
        verdicts = state.get("verification_results", [])
        mod_results = state.get("moderation_results", [])
        inf_results = state.get("influencer_results", [])
        cascades = spread.get("cascade_results", [])

        total_nodes = self.network.num_nodes
        total_reached = spread.get("total_reached", 0)
        total_blocked = spread.get("total_blocked", 0)
        total_warned = spread.get("total_warned", 0)

        # containment effectiveness
        containment_rate = round(
            (total_blocked + total_warned) / max(total_reached, 1) * 100, 1
        )

        # per-cascade summaries
        cascade_summaries = []
        for c in cascades:
            cascade_summaries.append({
                "id": c["cascade_id"],
                "source": c["source_node"],
                "claim": c["claim"][:100],
                "reached": c["total_reached"],
                "depth": c["max_depth"],
                "blocked": c["blocked_count"],
                "warned": c["warned_count"],
                "viral_coeff": c["viral_coefficient"],
            })

        return {
            "total_nodes": total_nodes,
            "total_reached": total_reached,
            "penetration_rate": spread.get("penetration_rate", 0),
            "max_depth": spread.get("max_depth_reached", 0),
            "viral_coefficient": spread.get("viral_coefficient", 0),
            "total_exposures": spread.get("total_exposures", 0),
            "total_blocked": total_blocked,
            "total_warned": total_warned,
            "containment_rate": containment_rate,
            "num_cascades": len(cascades),
            "cascade_summaries": cascade_summaries,
            "verdict_counts": {
                "Fake": sum(1 for v in verdicts if v.get("verdict") == "Fake"),
                "Real": sum(1 for v in verdicts if v.get("verdict") == "Real"),
                "Unverified": sum(1 for v in verdicts if v.get("verdict") == "Unverified"),
            },
            "moderation_counts": {
                "BLOCK": sum(1 for m in mod_results if m.get("decision") == "BLOCK"),
                "FLAG": sum(1 for m in mod_results if m.get("decision") == "FLAG"),
                "ALLOW": sum(1 for m in mod_results if m.get("decision") == "ALLOW"),
            },
            "avg_amplification": round(
                sum(r.get("amplification_score", 0) for r in inf_results) / max(len(inf_results), 1), 1
            ),
        }

    # ── public interface ──────────────────────────────────────────────────────

    def run_simulation(self) -> dict:
        initial = {
            "network_stats": {},
            "claims": [],
            "verification_results": [],
            "influencer_results": [],
            "moderation_results": [],
            "spread_result": {},
            "claim_summary": "",
            "spread_summary": "",
            "verification_summary": "",
            "influencer_summary": "",
            "moderation_summary": "",
            "analytics": {},
            "network_graph_path": "",
            "analysis_chart_path": "",
            "pipeline_log": ["🚀 Starting Multi-Cascade Misinformation Simulation …"],
            "current_step": "init",
            "start_time": time.time(),
            "elapsed_time": 0.0,
            "error": "",
        }
        return self.graph.invoke(initial)

    def get_full_report(self, state: dict) -> str:
        a = state.get("analytics", {})
        cascades = a.get("cascade_summaries", [])
        vc = a.get("verdict_counts", {})
        mc = a.get("moderation_counts", {})

        cascade_lines = ""
        for c in cascades:
            cascade_lines += (
                f"  C{c['id']}: src=User_{c['source']:>4d} | reached={c['reached']:>4d} "
                f"| depth={c['depth']:>2d} | blocked={c['blocked']} "
                f"| warned={c['warned']} | viral={c['viral_coeff']}\n"
                f"{'':>8s}claim: {c['claim'][:70]}{'…' if len(c['claim'])>70 else ''}\n"
            )

        return f"""
{'='*65}
   MULTI-AGENT MISINFORMATION SIMULATION REPORT
{'='*65}

📊 NETWORK
   Nodes: {a.get('total_nodes',0)} | Cascades: {a.get('num_cascades',0)}

📡 SPREAD
   Nodes Reached   : {a.get('total_reached',0)} / {a.get('total_nodes',0)}
   Penetration     : {a.get('penetration_rate',0)}%
   Max Depth       : {a.get('max_depth',0)} hops
   Viral Coefficient: {a.get('viral_coefficient',0)}
   Total Exposures : {a.get('total_exposures',0)}

🔍 FACT-CHECK VERDICTS
   Fake: {vc.get('Fake',0)} | Real: {vc.get('Real',0)} | Unverified: {vc.get('Unverified',0)}

📣 INFLUENCER
   Avg Amplification Score: {a.get('avg_amplification',0)}/10

🛡️ MODERATION
   BLOCK: {mc.get('BLOCK',0)} | FLAG: {mc.get('FLAG',0)} | ALLOW: {mc.get('ALLOW',0)}
   Nodes Blocked in Cascade: {a.get('total_blocked',0)}
   Nodes Warned in Cascade : {a.get('total_warned',0)}
   Containment Rate        : {a.get('containment_rate',0)}%

📋 PER-CASCADE BREAKDOWN
{cascade_lines}

⏱️ Simulation Time: {state.get('elapsed_time',0):.1f}s
{'='*65}
"""
