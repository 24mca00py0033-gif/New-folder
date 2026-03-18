"""
Fact-Checker Agent
==================
After influencer amplification, fact-checker nodes scan the graph
for infected/influenced nodes and verify the claim using LLM.
They attach warning labels to nodes they check.
"""
import json
import random
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL, FACT_CHECKER_SYSTEM_PROMPT, TEMPERATURE


class FactCheckerAgent:
    """
    Fact-checker nodes in the graph verify the circulating claim.
    They scan their neighbourhoods for infected/influenced nodes,
    verify the claim using LLM, and attach warning labels.
    """

    def __init__(self):
        self.llm = None
        if GROQ_API_KEY:
            try:
                self.llm = ChatGroq(
                    api_key=GROQ_API_KEY,
                    model=GROQ_MODEL,
                    temperature=0.3,
                    max_tokens=512,
                )
            except Exception:
                self.llm = None

    def check_graph(self, network) -> dict:
        """
        Fact-checker nodes scan the graph, verify the claim,
        and attach warnings to infected nodes in their neighbourhood.
        """
        G = network.graph
        fc_nodes = network.get_fact_checker_nodes()

        # Get the claim text from any infected node
        claim_text = ""
        for n in G.nodes():
            if G.nodes[n].get("claim_text"):
                claim_text = G.nodes[n]["claim_text"]
                break

        if not claim_text:
            return {
                "success": True,
                "verdict": "Unverified",
                "confidence": 0.0,
                "nodes_checked": 0,
                "nodes_warned": 0,
                "evidence": "No claim found in the network",
                "red_flags": [],
                "active_checkers": 0,
            }

        # Verify the claim using LLM
        verdict_result = self.verify_claim(claim_text)

        # Fact-checker nodes scan their neighbourhoods
        nodes_checked = 0
        nodes_warned = 0
        warned_node_ids = []
        active_checkers = 0

        for fc_node in fc_nodes:
            # Check if this fact-checker was exposed to the claim
            if G.nodes[fc_node]["exposure_count"] > 0 or any(
                G.nodes[nb]["status"] in ("infected", "influenced")
                for nb in G.neighbors(fc_node)
            ):
                active_checkers += 1
                G.nodes[fc_node]["status"] = "warned"
                G.nodes[fc_node]["warning_label"] = True

                # Scan neighbours and attach warnings
                for nb in G.neighbors(fc_node):
                    if G.nodes[nb]["status"] in ("infected", "influenced"):
                        nodes_checked += 1
                        if verdict_result["verdict"] in ("Fake", "Unverified"):
                            G.nodes[nb]["warning_label"] = True
                            G.nodes[nb]["status"] = "warned"
                            nodes_warned += 1
                            warned_node_ids.append(nb)

        return {
            "success": True,
            "claim": claim_text,
            "verdict": verdict_result["verdict"],
            "confidence": verdict_result["confidence"],
            "evidence": verdict_result["evidence"],
            "red_flags": verdict_result.get("red_flags", []),
            "nodes_checked": nodes_checked,
            "nodes_warned": nodes_warned,
            "warned_node_ids": warned_node_ids,
            "active_checkers": active_checkers,
            "total_fact_checkers": len(fc_nodes),
        }

    def verify_claim(self, claim_text: str) -> dict:
        """Verify a claim using LLM or rule-based fallback."""
        search_evidence = self._simulate_web_search(claim_text)

        if self.llm:
            try:
                prompt = (
                    f'Analyze the following claim for truthfulness:\n\n'
                    f'CLAIM: "{claim_text}"\n\n'
                    f'SEARCH EVIDENCE: {search_evidence}\n\n'
                    f'Respond ONLY with valid JSON in the specified format.'
                )
                messages = [
                    SystemMessage(content=FACT_CHECKER_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
                response = self.llm.invoke(messages)
                result = self._parse_verdict(response.content)
                result["claim"] = claim_text
                result["search_evidence"] = search_evidence
                return result
            except Exception:
                pass

        # Fallback rule-based
        return self._fallback_verdict(claim_text, search_evidence)

    @staticmethod
    def _simulate_web_search(claim_text: str) -> str:
        return (
            f"Simulated web search for: '{claim_text[:80]}…'\n"
            "- No official sources confirm this claim.\n"
            "- Similar claims flagged by fact-checking organisations.\n"
            "- Specific details could not be independently verified."
        )

    @staticmethod
    def _parse_verdict(response_text: str) -> dict:
        try:
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            result = json.loads(text)
            verdict = result.get("verdict", "Unverified")
            if verdict not in ("Real", "Fake", "Unverified"):
                verdict = "Unverified"
            return {
                "verdict": verdict,
                "confidence": float(result.get("confidence", 0.5)),
                "evidence": result.get("evidence", "No evidence provided"),
                "red_flags": result.get("red_flags", []),
            }
        except (json.JSONDecodeError, ValueError):
            lower = response_text.lower()
            if "fake" in lower:
                v = "Fake"
            elif "real" in lower or "true" in lower:
                v = "Real"
            else:
                v = "Unverified"
            return {
                "verdict": v,
                "confidence": 0.5,
                "evidence": response_text[:200],
                "red_flags": ["Could not parse structured response"],
            }

    @staticmethod
    def _fallback_verdict(claim_text: str, evidence: str) -> dict:
        red_flags = []
        score = 0.5
        lower = claim_text.lower()
        triggers = ["breaking", "secret", "leaked", "shocking", "banned",
                     "they don't want you to know", "100%", "miracle"]
        for t in triggers:
            if t in lower:
                red_flags.append(f'Contains trigger word: "{t}"')
                score += 0.08
        verdict = "Fake" if score >= 0.6 else "Unverified"
        return {
            "claim": claim_text,
            "verdict": verdict,
            "confidence": round(min(score, 0.95), 2),
            "evidence": "Rule-based analysis (LLM unavailable). " + evidence,
            "red_flags": red_flags or ["No specific red flags detected"],
            "search_evidence": evidence,
        }

    @staticmethod
    def get_verdict_summary(result: dict) -> str:
        emoji = {"Real": "✅", "Fake": "❌", "Unverified": "⚠️"}.get(result["verdict"], "❓")
        flags = "\n".join(f"  🚩 {f}" for f in result.get("red_flags", []))
        return (
            f"\n🔍 FACT-CHECK RESULTS\n{'='*50}\n"
            f"{emoji} Verdict : {result['verdict']}\n"
            f"📊 Confidence: {result['confidence']*100:.0f}%\n"
            f"📝 Evidence  : {result['evidence']}\n"
            f"🚩 Red Flags :\n{flags or '  None'}\n"
            f"📋 Nodes Checked: {result.get('nodes_checked', 0)}\n"
            f"⚠️ Nodes Warned : {result.get('nodes_warned', 0)}\n"
            f"🔍 Active Checkers: {result.get('active_checkers', 0)} / "
            f"{result.get('total_fact_checkers', 0)}\n"
        )
