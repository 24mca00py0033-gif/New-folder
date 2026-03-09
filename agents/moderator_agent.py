"""
Moderator Agent
Makes intelligent decisions to flag, block, or allow content based on verification results.
"""
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL, MODERATOR_SYSTEM_PROMPT, TEMPERATURE


class ModeratorAgent:
    """
    Agent responsible for content moderation decisions.
    Uses LLM-based policy reasoning to determine appropriate action.
    """

    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.1,  # Very low temperature for consistent decisions
            max_tokens=512,
        )

    def moderate_content(self, claim_text, verdict, confidence, evidence,
                         rewritten_content=None, spread_data=None):
        """
        Make a moderation decision on the content.
        
        Args:
            claim_text: The original claim
            verdict: Fact-check verdict (Real/Fake/Unverified)
            confidence: Fact-check confidence
            evidence: Evidence from fact-checking
            rewritten_content: Influencer-modified version (if any)
            spread_data: Information about how far the claim has spread
            
        Returns:
            dict with moderation decision and analysis
        """
        spread_info = ""
        if spread_data:
            spread_info = f"""
SPREAD DATA:
- Nodes Reached: {spread_data.get('total_reached', 'Unknown')}
- Network Penetration: {spread_data.get('penetration_rate', 'Unknown')}%
- Viral Coefficient: {spread_data.get('viral_coefficient', 'Unknown')}"""

        prompt = f"""Review the following content for moderation:

ORIGINAL CLAIM: "{claim_text}"

FACT-CHECK RESULTS:
- Verdict: {verdict}
- Confidence: {confidence*100:.0f}%
- Evidence: {evidence}

{f'INFLUENCER VERSION: "{rewritten_content}"' if rewritten_content else ''}
{spread_info}

MODERATION POLICY:
- BLOCK: If verdict is "Fake" with confidence > 60%, immediately block content
- FLAG: If verdict is "Unverified" OR confidence is low, flag for manual review
- ALLOW: If verdict is "Real" with sufficient confidence, allow to spread

Make your moderation decision based on the above policy. Respond ONLY with valid JSON."""

        messages = [
            SystemMessage(content=MODERATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            result = self._parse_decision(response.content, verdict, confidence)
            result["claim"] = claim_text
            result["verdict_used"] = verdict
            result["confidence_used"] = confidence
            result["source_agent"] = "ModeratorAgent"

            # Apply moderation effects
            result["spread_impact"] = self._calculate_spread_impact(result["decision"], spread_data)

            return result
        except Exception as e:
            # Fallback: Rule-based decision
            return self._rule_based_decision(claim_text, verdict, confidence, spread_data, str(e))

    def _parse_decision(self, response_text, verdict, confidence):
        """Parse the LLM moderation response."""
        try:
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            result = json.loads(text)

            decision = result.get("decision", "FLAG").upper()
            if decision not in ["BLOCK", "FLAG", "ALLOW"]:
                decision = "FLAG"

            return {
                "decision": decision,
                "reason": result.get("reason", "Policy-based decision"),
                "action_taken": result.get("action_taken", f"Content {decision.lower()}ed"),
                "severity": result.get("severity", "MEDIUM"),
            }
        except (json.JSONDecodeError, ValueError):
            # If LLM parsing fails, use rule-based logic
            if verdict == "Fake" and confidence > 0.6:
                decision = "BLOCK"
            elif verdict == "Unverified":
                decision = "FLAG"
            else:
                decision = "ALLOW"

            return {
                "decision": decision,
                "reason": f"Rule-based: verdict={verdict}, confidence={confidence:.0%}",
                "action_taken": f"Content {decision.lower()}ed based on verification results",
                "severity": "HIGH" if decision == "BLOCK" else ("MEDIUM" if decision == "FLAG" else "LOW"),
            }

    def _rule_based_decision(self, claim_text, verdict, confidence, spread_data, error_msg):
        """Fallback rule-based moderation when LLM is unavailable."""
        if verdict == "Fake" and confidence > 0.6:
            decision = "BLOCK"
            severity = "HIGH"
            reason = "Content confirmed as fake with high confidence"
            action = "Content blocked and removed from circulation. Spread halted."
        elif verdict == "Fake" and confidence <= 0.6:
            decision = "FLAG"
            severity = "HIGH"
            reason = "Content likely fake but confidence is low"
            action = "Content flagged for manual review. Spread velocity reduced."
        elif verdict == "Unverified":
            decision = "FLAG"
            severity = "MEDIUM"
            reason = "Content could not be verified"
            action = "Content marked for review. Users warned about unverified claims."
        else:
            decision = "ALLOW"
            severity = "LOW"
            reason = "Content verified as real"
            action = "Content allowed to spread normally."

        return {
            "claim": claim_text,
            "decision": decision,
            "reason": reason,
            "action_taken": action,
            "severity": severity,
            "verdict_used": verdict,
            "confidence_used": confidence,
            "source_agent": "ModeratorAgent",
            "spread_impact": self._calculate_spread_impact(decision, spread_data),
            "used_fallback": True,
            "error": error_msg,
        }

    def _calculate_spread_impact(self, decision, spread_data):
        """Calculate the impact of the moderation decision on content spread."""
        if not spread_data:
            spread_data = {"total_reached": 0, "penetration_rate": 0}

        nodes_reached = spread_data.get("total_reached", 0)

        if decision == "BLOCK":
            return {
                "containment": "COMPLETE",
                "containment_rate": 100.0,
                "additional_spread": 0,
                "description": "All further spread immediately halted. Content removed from circulation.",
            }
        elif decision == "FLAG":
            return {
                "containment": "PARTIAL",
                "containment_rate": 50.0,
                "additional_spread": max(1, nodes_reached // 4),
                "description": "Spread velocity reduced by 50%. Content flagged with warning labels.",
            }
        else:
            return {
                "containment": "NONE",
                "containment_rate": 0.0,
                "additional_spread": nodes_reached,
                "description": "Content continues to spread normally. No restrictions applied.",
            }

    def get_moderation_summary(self, result):
        """Format moderation results into a human-readable summary."""
        r = result
        decision_emoji = {"BLOCK": "🚫", "FLAG": "⚠️", "ALLOW": "✅"}
        severity_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}

        d_emoji = decision_emoji.get(r["decision"], "❓")
        s_emoji = severity_emoji.get(r.get("severity", "MEDIUM"), "⚪")

        impact = r.get("spread_impact", {})

        summary = f"""
🛡️ MODERATOR DECISION
{'='*50}
{d_emoji} Decision: {r['decision']}
{s_emoji} Severity: {r.get('severity', 'N/A')}
📝 Reason: {r['reason']}
⚡ Action: {r['action_taken']}

📊 Spread Impact:
  🔒 Containment: {impact.get('containment', 'N/A')} ({impact.get('containment_rate', 0):.0f}%)
  📉 Additional Spread Allowed: {impact.get('additional_spread', 'N/A')} nodes
  📋 {impact.get('description', '')}
"""
        return summary
