"""
Fact-Checker Agent
==================
Verifies claims using LLM reasoning. During the BFS cascade, fact-checker
nodes inside the graph automatically slow spread and attach warnings.
This class provides the detailed LLM-based analysis that the pipeline
runs *after* the cascade to produce a human-readable report for every claim.
"""
import json
import random
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL, FACT_CHECKER_SYSTEM_PROMPT, TEMPERATURE


class FactCheckerAgent:
    """
    Analyses a claim with the Groq LLM and returns a structured verdict.
    Called once per cascade by the pipeline to produce analysis reports.
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

    def verify_claim(self, claim_text: str) -> dict:
        """Verify a single claim; return structured verdict."""
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
                result.update({
                    "claim": claim_text,
                    "search_evidence": search_evidence,
                    "source_agent": "FactCheckerAgent",
                })
                return result
            except Exception:
                pass

        # Fallback rule-based
        return self._fallback_verdict(claim_text, search_evidence)

    def verify_batch(self, claims: list[dict]) -> list[dict]:
        """Verify every claim in a list."""
        return [self.verify_claim(c.get("claim", "")) for c in claims]

    # ── helpers ───────────────────────────────────────────────────────────────

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
        """Simple heuristic when LLM is unavailable."""
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
            "source_agent": "FactCheckerAgent",
        }

    @staticmethod
    def get_verdict_summary(verdict_result: dict) -> str:
        vr = verdict_result
        emoji = {"Real": "✅", "Fake": "❌", "Unverified": "⚠️"}.get(vr["verdict"], "❓")
        flags = "\n".join(f"  🚩 {f}" for f in vr.get("red_flags", []))
        return (
            f"\n🔍 FACT-CHECK RESULTS\n{'='*50}\n"
            f"{emoji} Verdict : {vr['verdict']}\n"
            f"📊 Confidence: {vr['confidence']*100:.0f}%\n"
            f"📝 Evidence  : {vr['evidence']}\n"
            f"🚩 Red Flags :\n{flags or '  None'}\n"
        )
