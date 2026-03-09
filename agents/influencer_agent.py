"""
Influencer Agent
Rewrites content for maximum viral spread or counter-messaging using advanced prompt engineering.
"""
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL, INFLUENCER_SYSTEM_PROMPT, TEMPERATURE


class InfluencerAgent:
    """
    Agent that simulates social media influencer behavior.
    Rewrites content to maximize engagement or to counter misinformation.
    """

    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=TEMPERATURE + 0.1,
            max_tokens=512,
        )

    def rewrite_claim(self, claim_text, verdict, confidence=0.5):
        """
        Rewrite a claim based on the fact-check verdict.
        
        - If FAKE/UNVERIFIED: Rewrite as a warning to counter misinformation
        - If REAL: Rewrite to maximize viral engagement
        
        Args:
            claim_text: The original claim text
            verdict: "Real", "Fake", or "Unverified"
            confidence: Confidence level of the fact-check
            
        Returns:
            dict with rewritten content and analysis
        """
        if verdict in ["Fake", "Unverified"]:
            prompt = f"""The following claim has been fact-checked and found to be {verdict.upper()} 
(confidence: {confidence*100:.0f}%):

CLAIM: "{claim_text}"

Rewrite this as a WARNING post that:
1. Clearly labels it as {verdict.upper()} information
2. Uses urgent, attention-grabbing language
3. Explains why people should NOT share it
4. Includes a call to action (verify before sharing)
5. Uses hashtags for visibility

Write ONLY the rewritten warning post, nothing else."""
        else:
            prompt = f"""The following claim has been verified as REAL (confidence: {confidence*100:.0f}%):

CLAIM: "{claim_text}"

Rewrite this to maximize viral engagement:
1. Use a compelling hook in the first line
2. Add emotional triggers (surprise, urgency, importance)
3. Make it share-worthy with clear, impactful language
4. Include relevant hashtags
5. Keep it concise but attention-grabbing

Write ONLY the rewritten viral post, nothing else."""

        messages = [
            SystemMessage(content=INFLUENCER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            rewritten = response.content.strip()

            # Calculate amplification score
            amplification_score = self._calculate_amplification(claim_text, rewritten, verdict)

            return {
                "original_claim": claim_text,
                "rewritten_content": rewritten,
                "verdict_used": verdict,
                "action_type": "counter_messaging" if verdict in ["Fake", "Unverified"] else "amplification",
                "amplification_score": amplification_score,
                "source_agent": "InfluencerAgent",
            }
        except Exception as e:
            # Fallback rewritten content
            if verdict in ["Fake", "Unverified"]:
                fallback = f"⚠️ FACT CHECK WARNING: The following claim is {verdict.upper()}: '{claim_text}' - Please verify before sharing! #FactCheck #StopMisinformation"
            else:
                fallback = f"🔥 VERIFIED: {claim_text} - Share this important news! #Breaking #Verified"

            return {
                "original_claim": claim_text,
                "rewritten_content": fallback,
                "verdict_used": verdict,
                "action_type": "counter_messaging" if verdict in ["Fake", "Unverified"] else "amplification",
                "amplification_score": 5.0,
                "source_agent": "InfluencerAgent",
                "error": str(e),
            }

    def _calculate_amplification(self, original, rewritten, verdict):
        """
        Calculate an amplification score (1-10) based on content transformation.
        Higher scores indicate more viral potential.
        """
        score = 5.0  # Base score

        # Length ratio: longer ≠ always better, but expansion shows effort
        len_ratio = len(rewritten) / max(len(original), 1)
        if 1.5 <= len_ratio <= 3.0:
            score += 1.0
        elif len_ratio > 3.0:
            score += 0.5

        # Check for engagement markers
        engagement_markers = ["!", "?", "🔥", "⚠️", "❌", "✅", "#", "BREAKING",
                             "WARNING", "URGENT", "SHARE", "VERIFIED", "FACT CHECK"]
        marker_count = sum(1 for m in engagement_markers if m.upper() in rewritten.upper())
        score += min(marker_count * 0.3, 2.0)

        # Emoji usage
        emoji_count = sum(1 for c in rewritten if ord(c) > 0x1F600)
        score += min(emoji_count * 0.2, 1.0)

        # Cap score
        score = min(10.0, max(1.0, score))

        return round(score, 1)

    def get_influencer_summary(self, result):
        """Format influencer results into a human-readable summary."""
        r = result
        action_emoji = "🛡️" if r["action_type"] == "counter_messaging" else "📣"

        summary = f"""
{action_emoji} INFLUENCER AGENT RESULTS
{'='*50}
📝 Action Type: {r['action_type'].replace('_', ' ').title()}
📊 Amplification Score: {r['amplification_score']}/10
🔍 Based on Verdict: {r['verdict_used']}

✍️ Rewritten Content:
{r['rewritten_content']}
"""
        return summary
