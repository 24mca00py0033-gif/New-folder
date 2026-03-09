"""
Analytics Module
================
Comprehensive analytics for the multi-cascade misinformation simulation.
Computes spread velocity, verification impact, agent influence,
moderation effectiveness, and generates a formatted report.
"""
import numpy as np


class SimulationAnalytics:
    """
    Produces detailed analytics from a completed simulation state
    (as returned by MisinformationPipeline.run_simulation()).
    """

    def __init__(self, network):
        self.network = network

    # ── metric groups ─────────────────────────────────────────────────────────

    def compute_spread_metrics(self, spread: dict) -> dict:
        steps = spread.get("spread_per_step", [])
        total_reached = spread.get("total_reached", 0)
        total_nodes = self.network.num_nodes
        cumulative = list(np.cumsum(steps)) if steps else [0]
        return {
            "total_reached": total_reached,
            "total_nodes": total_nodes,
            "penetration_rate": round(total_reached / max(total_nodes, 1) * 100, 2),
            "spread_per_step": steps,
            "cumulative_spread": cumulative,
            "avg_velocity": round(float(np.mean(steps)), 2) if steps else 0,
            "peak_step": (int(np.argmax(steps)) + 1) if steps else 0,
            "peak_spread_count": int(max(steps)) if steps else 0,
            "max_depth": spread.get("max_depth_reached", 0),
            "viral_coefficient": spread.get("viral_coefficient", 0),
            "num_cascades": spread.get("num_cascades", 0),
        }

    def compute_containment_metrics(self, spread: dict) -> dict:
        total_reached = spread.get("total_reached", 0)
        total_blocked = spread.get("total_blocked", 0)
        total_warned = spread.get("total_warned", 0)
        total_nodes = self.network.num_nodes
        protected = total_nodes - total_reached
        containment_rate = round(
            (total_blocked + total_warned) / max(total_reached, 1) * 100, 1
        )
        return {
            "total_blocked": total_blocked,
            "total_warned": total_warned,
            "protected_nodes": protected,
            "containment_rate": containment_rate,
            "network_immunization_rate": round(protected / max(total_nodes, 1) * 100, 2),
        }

    def compute_verification_summary(self, verdicts: list) -> dict:
        counts = {"Fake": 0, "Real": 0, "Unverified": 0}
        total_conf = 0.0
        for v in verdicts:
            vd = v.get("verdict", "Unverified")
            counts[vd] = counts.get(vd, 0) + 1
            total_conf += v.get("confidence", 0.5)
        avg_conf = total_conf / max(len(verdicts), 1)
        return {
            "verdict_counts": counts,
            "avg_confidence": round(avg_conf, 2),
            "total_claims": len(verdicts),
        }

    def compute_moderation_summary(self, mod_results: list) -> dict:
        counts = {"BLOCK": 0, "FLAG": 0, "ALLOW": 0}
        for m in mod_results:
            d = m.get("decision", "FLAG")
            counts[d] = counts.get(d, 0) + 1
        return {"decision_counts": counts}

    def compute_influencer_summary(self, inf_results: list) -> dict:
        scores = [r.get("amplification_score", 5.0) for r in inf_results]
        return {
            "avg_amplification": round(float(np.mean(scores)), 1) if scores else 0,
            "max_amplification": round(float(max(scores)), 1) if scores else 0,
            "counter_messages": sum(
                1 for r in inf_results if r.get("action_type") == "counter_messaging"
            ),
            "amplifications": sum(
                1 for r in inf_results if r.get("action_type") == "amplification"
            ),
        }

    # ── aggregate ─────────────────────────────────────────────────────────────

    def generate_full_analytics(self, state: dict) -> dict:
        spread = state.get("spread_result", {})
        verdicts = state.get("verification_results", [])
        inf_results = state.get("influencer_results", [])
        mod_results = state.get("moderation_results", [])

        return {
            "spread_metrics": self.compute_spread_metrics(spread),
            "containment": self.compute_containment_metrics(spread),
            "verification": self.compute_verification_summary(verdicts),
            "moderation": self.compute_moderation_summary(mod_results),
            "influencer": self.compute_influencer_summary(inf_results),
        }

    # ── formatted report ──────────────────────────────────────────────────────

    def generate_analytics_report(self, full: dict) -> str:
        sm = full["spread_metrics"]
        ct = full["containment"]
        vr = full["verification"]
        md = full["moderation"]
        inf = full["influencer"]

        return f"""
{'='*60}
          DETAILED ANALYTICS REPORT
{'='*60}

📈 1. SPREAD VELOCITY
{'─'*60}
   Cascades Launched    : {sm['num_cascades']}
   Total Nodes Reached  : {sm['total_reached']} / {sm['total_nodes']}
   Network Penetration  : {sm['penetration_rate']}%
   Average Velocity     : {sm['avg_velocity']} nodes/step
   Peak Spread Step     : Step {sm['peak_step']} ({sm['peak_spread_count']} nodes)
   Maximum Depth        : {sm['max_depth']} hops
   Viral Coefficient    : {sm['viral_coefficient']}

🔍 2. FACT-CHECK IMPACT
{'─'*60}
   Claims Analysed      : {vr['total_claims']}
   Verdicts — Fake      : {vr['verdict_counts'].get('Fake',0)}
              Real      : {vr['verdict_counts'].get('Real',0)}
              Unverified: {vr['verdict_counts'].get('Unverified',0)}
   Avg Confidence       : {vr['avg_confidence']*100:.0f}%

📣 3. INFLUENCER IMPACT
{'─'*60}
   Avg Amplification    : {inf['avg_amplification']}/10
   Max Amplification    : {inf['max_amplification']}/10
   Counter-Messages     : {inf['counter_messages']}
   Amplifications       : {inf['amplifications']}

🛡️ 4. MODERATION & CONTAINMENT
{'─'*60}
   Decisions — BLOCK    : {md['decision_counts'].get('BLOCK',0)}
               FLAG     : {md['decision_counts'].get('FLAG',0)}
               ALLOW    : {md['decision_counts'].get('ALLOW',0)}
   Nodes Blocked (BFS)  : {ct['total_blocked']}
   Nodes Warned  (BFS)  : {ct['total_warned']}
   Protected Nodes      : {ct['protected_nodes']}
   Containment Rate     : {ct['containment_rate']}%
   Immunisation Rate    : {ct['network_immunization_rate']}%

{'='*60}"""
