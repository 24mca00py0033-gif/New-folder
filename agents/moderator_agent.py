"""
Moderator Agent
===============
After fact-checking, moderator nodes take action on infected/influenced nodes.
Based on the fact-checker verdict, moderators can BLOCK, FLAG, or ALLOW content.
This is the final agent phase that stops or reduces misinformation spread.
"""
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL, MODERATOR_SYSTEM_PROMPT, TEMPERATURE


class ModeratorAgent:
    """
    Moderator nodes take action on the network based on fact-checker results.
    They can BLOCK (stop spread), FLAG (warn users), or ALLOW content.
    Each moderator scans its neighbourhood and applies its decision.
    """

    def __init__(self):
        self.llm = None
        if GROQ_API_KEY:
            try:
                self.llm = ChatGroq(
                    api_key=GROQ_API_KEY,
                    model=GROQ_MODEL,
                    temperature=0.1,
                    max_tokens=512,
                )
            except Exception:
                self.llm = None

    def moderate_graph(self, network, fact_check_result: dict) -> dict:
        """
        Moderator nodes take action on the graph based on fact-checker results.
        They scan their neighbourhoods and apply moderation decisions.
        """
        G = network.graph
        mod_nodes = network.get_moderator_nodes()

        claim_text = fact_check_result.get("claim", "")
        verdict = fact_check_result.get("verdict", "Unverified")
        confidence = fact_check_result.get("confidence", 0.5)
        evidence = fact_check_result.get("evidence", "")

        # Get LLM moderation decision
        decision_result = self._make_decision(claim_text, verdict, confidence, evidence)

        # Apply moderation to graph
        nodes_blocked = 0
        nodes_flagged = 0
        nodes_allowed = 0
        blocked_node_ids = []
        flagged_node_ids = []
        active_moderators = 0

        for mod_node in mod_nodes:
            # Check if moderator is near infected content
            neighbours_infected = [
                nb for nb in G.neighbors(mod_node)
                if G.nodes[nb]["status"] in ("infected", "influenced")
            ]

            if not neighbours_infected and G.nodes[mod_node]["exposure_count"] == 0:
                continue

            active_moderators += 1
            G.nodes[mod_node]["status"] = "warned"  # Moderator is aware

            for nb in neighbours_infected:
                if decision_result["decision"] == "BLOCK":
                    G.nodes[nb]["status"] = "blocked"
                    G.nodes[nb]["blocked_by"] = mod_node
                    G.nodes[nb]["shared"] = False
                    nodes_blocked += 1
                    blocked_node_ids.append(nb)
                elif decision_result["decision"] == "FLAG":
                    if G.nodes[nb]["status"] != "blocked":  # Don't override block
                        G.nodes[nb]["status"] = "warned"
                        G.nodes[nb]["warning_label"] = True
                        nodes_flagged += 1
                        flagged_node_ids.append(nb)
                else:  # ALLOW
                    nodes_allowed += 1

        # Count final node statuses across the entire graph
        status_counts = {"clean": 0, "infected": 0, "influenced": 0, "warned": 0, "blocked": 0}
        for n in G.nodes():
            st = G.nodes[n]["status"]
            status_counts[st] = status_counts.get(st, 0) + 1

        return {
            "success": True,
            "decision": decision_result["decision"],
            "reason": decision_result["reason"],
            "severity": decision_result.get("severity", "MEDIUM"),
            "action_taken": decision_result.get("action_taken", ""),
            "nodes_blocked": nodes_blocked,
            "nodes_flagged": nodes_flagged,
            "nodes_allowed": nodes_allowed,
            "blocked_node_ids": blocked_node_ids,
            "flagged_node_ids": flagged_node_ids,
            "active_moderators": active_moderators,
            "total_moderators": len(mod_nodes),
            "final_status_counts": status_counts,
            "verdict_used": verdict,
            "confidence_used": confidence,
        }

    def _make_decision(self, claim_text, verdict, confidence, evidence) -> dict:
        """Make moderation decision using LLM or rules."""
        if self.llm and claim_text:
            try:
                prompt = (
                    f'ORIGINAL CLAIM: "{claim_text}"\n\n'
                    f"FACT-CHECK: Verdict={verdict}, Confidence={confidence*100:.0f}%\n"
                    f"Evidence: {evidence}\n\n"
                    "Respond ONLY with valid JSON."
                )
                messages = [
                    SystemMessage(content=MODERATOR_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
                response = self.llm.invoke(messages)
                return self._parse_decision(response.content, verdict, confidence)
            except Exception:
                pass

        return self._rule_based(verdict, confidence)

    @staticmethod
    def _parse_decision(response_text, verdict, confidence):
        try:
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            result = json.loads(text)
            decision = result.get("decision", "FLAG").upper()
            if decision not in ("BLOCK", "FLAG", "ALLOW"):
                decision = "FLAG"
            return {
                "decision": decision,
                "reason": result.get("reason", "Policy-based"),
                "action_taken": result.get("action_taken", f"Content {decision.lower()}ed"),
                "severity": result.get("severity", "MEDIUM"),
            }
        except (json.JSONDecodeError, ValueError):
            return ModeratorAgent._rule_based(verdict, confidence)

    @staticmethod
    def _rule_based(verdict, confidence):
        if verdict == "Fake" and confidence > 0.6:
            return {"decision": "BLOCK", "severity": "HIGH",
                    "reason": "Confirmed fake with high confidence",
                    "action_taken": "Content blocked; spread halted."}
        if verdict == "Fake":
            return {"decision": "FLAG", "severity": "HIGH",
                    "reason": "Likely fake, low confidence",
                    "action_taken": "Content flagged for review."}
        if verdict == "Unverified":
            return {"decision": "FLAG", "severity": "MEDIUM",
                    "reason": "Content unverified",
                    "action_taken": "Content marked for review."}
        return {"decision": "ALLOW", "severity": "LOW",
                "reason": "Content verified as real",
                "action_taken": "Content allowed to spread."}

    @staticmethod
    def get_moderation_summary(result: dict) -> str:
        d_emoji = {"BLOCK": "🚫", "FLAG": "⚠️", "ALLOW": "✅"}.get(result["decision"], "❓")
        s_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(result.get("severity", ""), "⚪")
        sc = result.get("final_status_counts", {})
        return (
            f"\n🛡️ MODERATOR DECISION\n{'='*50}\n"
            f"{d_emoji} Decision : {result['decision']}\n"
            f"{s_emoji} Severity : {result.get('severity', 'N/A')}\n"
            f"📝 Reason  : {result['reason']}\n"
            f"⚡ Action  : {result.get('action_taken', '')}\n"
            f"🚫 Blocked : {result.get('nodes_blocked', 0)} nodes\n"
            f"⚠️ Flagged : {result.get('nodes_flagged', 0)} nodes\n"
            f"✅ Allowed : {result.get('nodes_allowed', 0)} nodes\n"
            f"👮 Active  : {result.get('active_moderators', 0)} / "
            f"{result.get('total_moderators', 0)}\n\n"
            f"📊 Final Graph Status:\n"
            f"  Clean: {sc.get('clean', 0)} | Infected: {sc.get('infected', 0)} | "
            f"Influenced: {sc.get('influenced', 0)} | Warned: {sc.get('warned', 0)} | "
            f"Blocked: {sc.get('blocked', 0)}\n"
        )
