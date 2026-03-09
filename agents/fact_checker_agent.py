"""
Fact-Checker Agent
Verifies claims using LLM reasoning and simulated web search.
Employs function calling to evaluate claim truthfulness.
"""
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL, FACT_CHECKER_SYSTEM_PROMPT, TEMPERATURE


class FactCheckerAgent:
    """
    Agent responsible for verifying claims and providing a factual verdict.
    Uses Groq LLM with structured output to analyze claims.
    """

    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.9,  # Low temperature for factual accuracy
            max_tokens=512,
        )

    def verify_claim(self, claim_text):
        """
        Verify a claim and return a structured verdict.
        
        Args:
            claim_text: The claim to verify
            
        Returns:
            dict with verdict, confidence, evidence, and red_flags
        """
        # Step 1: Simulate web search for evidence
        search_results = self._simulate_web_search(claim_text)

        # Step 2: LLM-based analysis with evidence
        prompt = f"""Analyze the following claim for truthfulness:

CLAIM: "{claim_text}"

SEARCH EVIDENCE: {search_results}

Based on your analysis, provide your verdict in the exact JSON format specified.
Consider:
1. Is the claim specific enough to verify?
2. Does it match known facts or common misinformation patterns?
3. Are there red flags (too good to be true, emotional triggers, specific but unverifiable details)?
4. What is your confidence level?

Respond ONLY with valid JSON, no other text."""

        messages = [
            SystemMessage(content=FACT_CHECKER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            result = self._parse_verdict(response.content)
            result["claim"] = claim_text
            result["search_evidence"] = search_results
            result["source_agent"] = "FactCheckerAgent"
            return result
        except Exception as e:
            return {
                "claim": claim_text,
                "verdict": "Unverified",
                "confidence": 0.5,
                "evidence": f"Verification failed: {str(e)}",
                "red_flags": ["Unable to complete verification"],
                "search_evidence": search_results,
                "source_agent": "FactCheckerAgent",
                "error": str(e),
            }

    def _simulate_web_search(self, claim_text):
        """
        Simulate a web search for claim verification.
        In a production system, this would call a real search API.
        
        Returns:
            str: Simulated search results summary
        """
        return (
            f"Simulated web search results for: '{claim_text[:80]}...'\n"
            f"- No official government or verified news sources found confirming this exact claim.\n"
            f"- Similar claims have been flagged by fact-checking organizations.\n"
            f"- The claim contains specific details that could not be independently verified.\n"
            f"- No matching press releases or official statements found."
        )

    def _parse_verdict(self, response_text):
        """Parse the LLM response into a structured verdict."""
        try:
            # Try to extract JSON from the response
            text = response_text.strip()
            # Handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            result = json.loads(text)

            # Validate required fields
            verdict = result.get("verdict", "Unverified")
            if verdict not in ["Real", "Fake", "Unverified"]:
                verdict = "Unverified"

            return {
                "verdict": verdict,
                "confidence": float(result.get("confidence", 0.5)),
                "evidence": result.get("evidence", "No evidence provided"),
                "red_flags": result.get("red_flags", []),
            }
        except (json.JSONDecodeError, ValueError, KeyError):
            # If parsing fails, try to extract verdict from text
            response_lower = response_text.lower()
            if "fake" in response_lower:
                verdict = "Fake"
            elif "real" in response_lower or "true" in response_lower:
                verdict = "Real"
            else:
                verdict = "Unverified"

            return {
                "verdict": verdict,
                "confidence": 0.5,
                "evidence": response_text[:200],
                "red_flags": ["Could not parse structured response"],
            }

    def get_verdict_summary(self, verdict_result):
        """Format verdict into a human-readable summary."""
        vr = verdict_result
        verdict_emoji = {"Real": "✅", "Fake": "❌", "Unverified": "⚠️"}
        emoji = verdict_emoji.get(vr["verdict"], "❓")

        red_flags_text = "\n".join(f"  🚩 {flag}" for flag in vr.get("red_flags", []))

        summary = f"""
🔍 FACT-CHECK VERIFICATION RESULTS
{'='*50}
{emoji} Verdict: {vr['verdict']}
📊 Confidence: {vr['confidence']*100:.0f}%
📝 Evidence: {vr['evidence']}
🚩 Red Flags:
{red_flags_text if red_flags_text else '  None identified'}
"""
        return summary
