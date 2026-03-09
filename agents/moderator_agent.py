"""
Moderator Agent
===============
Makes BLOCK / FLAG / ALLOW decisions using LLM policy reasoning.
During the BFS cascade, moderator nodes probabilistically block or flag
content (handled in social_network.py). This class provides the detailed
LLM-based analysis run *after* the cascade by the pipeline for reporting.
"""
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL, MODERATOR_SYSTEM_PROMPT, TEMPERATURE


class ModeratorAgent:
    """
    Content-moderation agent.  Produces per-claim moderation verdicts
    with severity and reasoning.
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

    def moderate_content(
        self,
        claim_text: str,
        verdict: str,
        confidence: float,
        evidence: str = "",
        rewritten_content: str = "",
        spread_data: dict | None = None,
    ) -> dict:
        """Return a moderation decision dict for one claim."""
        spread_info = ""
        if spread_data:
            spread_info = (
                f"\nSPREAD DATA:\n"
                f"- Nodes Reached: {spread_data.get('total_reached', '?')}\n"
                f"- Penetration: {spread_data.get('penetration_rate', '?')}%\n"
            )

        if self.llm:
            try:
                prompt = (
                    f'ORIGINAL CLAIM: "{claim_text}"\n\n'
                    f"FACT-CHECK: Verdict={verdict}, Confidence={confidence*100:.0f}%\n"
                    f"Evidence: {evidence}\n"
                    f'{f"INFLUENCER VERSION: {rewritten_content}" if rewritten_content else ""}'
                    f"{spread_info}\n"
                    "Respond ONLY with valid JSON."
                )
                messages = [
                    SystemMessage(content=MODERATOR_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
                response = self.llm.invoke(messages)
                result = self._parse_decision(response.content, verdict, confidence)
            except Exception:
                result = self._rule_based(verdict, confidence)
        else:
            result = self._rule_based(verdict, confidence)

        result.update({
            "claim": claim_text,
            "verdict_used": verdict,
            "confidence_used": confidence,
            "source_agent": "ModeratorAgent",
            "spread_impact": self._spread_impact(result["decision"], spread_data),
        })
        return result

    def moderate_batch(self, claims, verdicts, influencer_results, spread_result):
        """Moderate every claim."""
        results = []
        for cl, ver, inf in zip(claims, verdicts, influencer_results):
            results.append(self.moderate_content(
                claim_text=cl.get("claim", ""),
                verdict=ver.get("verdict", "Unverified"),
                confidence=ver.get("confidence", 0.5),
                evidence=ver.get("evidence", ""),
                rewritten_content=inf.get("rewritten_content", ""),
                spread_data=spread_result,
            ))
        return results

    # ── internal helpers ──────────────────────────────────────────────────────

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
    def _spread_impact(decision, spread_data):
        reached = (spread_data or {}).get("total_reached", 0)
        if decision == "BLOCK":
            return {"containment": "COMPLETE", "containment_rate": 100.0,
                    "additional_spread": 0,
                    "description": "All further spread halted."}
        if decision == "FLAG":
            return {"containment": "PARTIAL", "containment_rate": 50.0,
                    "additional_spread": max(1, reached // 4),
                    "description": "Spread reduced by ~50 %."}
        return {"containment": "NONE", "containment_rate": 0.0,
                "additional_spread": reached,
                "description": "Content spreads normally."}

    @staticmethod
    def get_moderation_summary(result: dict) -> str:
        d_emoji = {"BLOCK": "🚫", "FLAG": "⚠️", "ALLOW": "✅"}.get(result["decision"], "❓")
        s_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(result.get("severity", ""), "⚪")
        imp = result.get("spread_impact", {})
        return (
            f"\n🛡️ MODERATOR DECISION\n{'='*50}\n"
            f"{d_emoji} Decision : {result['decision']}\n"
            f"{s_emoji} Severity : {result.get('severity', 'N/A')}\n"
            f"📝 Reason  : {result['reason']}\n"
            f"⚡ Action  : {result['action_taken']}\n"
            f"🔒 Contain.: {imp.get('containment', 'N/A')} "
            f"({imp.get('containment_rate', 0):.0f}%)\n"
        )
