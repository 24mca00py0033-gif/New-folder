"""
Influencer Agent
================
Rewrites claims for maximum viral spread or counter-messaging.
During the BFS cascade, influencer nodes amplify spread automatically
(handled in social_network.py). This class provides the LLM-based
rewriting that the pipeline calls *after* the cascade for reporting.
"""
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL, INFLUENCER_SYSTEM_PROMPT, TEMPERATURE


class InfluencerAgent:
    """
    Rewrites a claim to maximise engagement (if Real) or to counter
    misinformation (if Fake / Unverified).
    """

    def __init__(self):
        self.llm = None
        if GROQ_API_KEY:
            try:
                self.llm = ChatGroq(
                    api_key=GROQ_API_KEY,
                    model=GROQ_MODEL,
                    temperature=TEMPERATURE + 0.1,
                    max_tokens=512,
                )
            except Exception:
                self.llm = None

    def rewrite_claim(self, claim_text: str, verdict: str, confidence: float = 0.5) -> dict:
        """Rewrite a claim based on fact-check verdict."""
        is_counter = verdict in ("Fake", "Unverified")

        if self.llm:
            try:
                if is_counter:
                    prompt = (
                        f"The following claim is {verdict.upper()} "
                        f"(confidence {confidence*100:.0f}%):\n\n"
                        f'CLAIM: "{claim_text}"\n\n'
                        "Rewrite as a WARNING post with urgent language, "
                        "hashtags, and a call to verify before sharing. "
                        "Write ONLY the rewritten post."
                    )
                else:
                    prompt = (
                        f"The following claim is VERIFIED REAL "
                        f"(confidence {confidence*100:.0f}%):\n\n"
                        f'CLAIM: "{claim_text}"\n\n'
                        "Rewrite to maximise viral engagement with hooks, "
                        "emotional triggers, and hashtags. "
                        "Write ONLY the rewritten post."
                    )
                messages = [
                    SystemMessage(content=INFLUENCER_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
                response = self.llm.invoke(messages)
                rewritten = response.content.strip()
                amp_score = self._calc_amplification(claim_text, rewritten, verdict)
                return {
                    "original_claim": claim_text,
                    "rewritten_content": rewritten,
                    "verdict_used": verdict,
                    "action_type": "counter_messaging" if is_counter else "amplification",
                    "amplification_score": amp_score,
                    "source_agent": "InfluencerAgent",
                }
            except Exception:
                pass

        # Fallback
        if is_counter:
            rewritten = (
                f"⚠️ FACT CHECK WARNING: The following claim is {verdict.upper()}: "
                f"'{claim_text}' — Please verify before sharing! #FactCheck #StopMisinfo"
            )
        else:
            rewritten = f"🔥 VERIFIED: {claim_text} — Share this! #Breaking #Verified"

        return {
            "original_claim": claim_text,
            "rewritten_content": rewritten,
            "verdict_used": verdict,
            "action_type": "counter_messaging" if is_counter else "amplification",
            "amplification_score": 5.0,
            "source_agent": "InfluencerAgent",
        }

    def rewrite_batch(self, claims: list[dict], verdicts: list[dict]) -> list[dict]:
        """Rewrite every claim using its corresponding verdict."""
        results = []
        for claim, ver in zip(claims, verdicts):
            results.append(self.rewrite_claim(
                claim.get("claim", ""),
                ver.get("verdict", "Unverified"),
                ver.get("confidence", 0.5),
            ))
        return results

    @staticmethod
    def _calc_amplification(original, rewritten, verdict):
        score = 5.0
        ratio = len(rewritten) / max(len(original), 1)
        if 1.5 <= ratio <= 3.0:
            score += 1.0
        elif ratio > 3.0:
            score += 0.5
        markers = ["!", "?", "🔥", "⚠️", "❌", "✅", "#", "BREAKING",
                    "WARNING", "URGENT", "SHARE", "VERIFIED", "FACT CHECK"]
        score += min(sum(1 for m in markers if m.upper() in rewritten.upper()) * 0.3, 2.0)
        score += min(sum(1 for c in rewritten if ord(c) > 0x1F600) * 0.2, 1.0)
        return round(min(10.0, max(1.0, score)), 1)

    @staticmethod
    def get_influencer_summary(result: dict) -> str:
        r = result
        emoji = "🛡️" if r["action_type"] == "counter_messaging" else "📣"
        return (
            f"\n{emoji} INFLUENCER RESULTS\n{'='*50}\n"
            f"Action : {r['action_type'].replace('_', ' ').title()}\n"
            f"Score  : {r['amplification_score']}/10\n"
            f"Verdict: {r['verdict_used']}\n\n"
            f"✍️ Rewritten:\n{r['rewritten_content']}\n"
        )
