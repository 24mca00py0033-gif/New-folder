"""
Misinformation Agent
Generates realistic fake news and misleading claims using Groq LLM API.
"""
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL, MISINFORMATION_SYSTEM_PROMPT, TEMPERATURE


class MisinformationAgent:
    """
    Agent responsible for generating realistic misinformation claims.
    Uses Groq LLM with creative prompting to produce plausible false claims.
    """

    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=TEMPERATURE + 0.2,  # Slightly higher temperature for creativity
            max_tokens=256,
        )
        self.claim_topics = [
            "government policy", "health and medicine", "technology",
            "environment", "education", "economy", "science",
            "social media", "international relations", "public safety",
        ]

    def generate_claim(self, topic=None):
        """
        Generate a realistic news-like claim that may be real or fake.
        
        Args:
            topic: Optional topic for the claim. If None, a general claim is generated.
            
        Returns:
            dict with 'claim' text and metadata
        """
        topic_instruction = f" about {topic}" if topic else ""
        
        prompt = f"""Generate a short, realistic news-like claim{topic_instruction} that could be either 
real or fake. The claim should be:
- 1-2 sentences maximum
- Sound like a real news headline or social media post
- Be specific with names, numbers, or dates to seem credible
- Cover a topic that would generate engagement and shares

Generate ONLY the claim text, nothing else."""

        messages = [
            SystemMessage(content=MISINFORMATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            claim_text = response.content.strip().strip('"').strip("'")
            
            return {
                "claim": claim_text,
                "source_agent": "MisinformationAgent",
                "topic": topic or "general",
                "generated": True,
            }
        except Exception as e:
            # Fallback claim if LLM fails
            return {
                "claim": "Breaking: Government announces free 5G internet for all students starting January 2026.",
                "source_agent": "MisinformationAgent",
                "topic": topic or "general",
                "generated": False,
                "error": str(e),
            }

    def generate_multiple_claims(self, count=3):
        """Generate multiple diverse claims."""
        import random
        claims = []
        topics = random.sample(self.claim_topics, min(count, len(self.claim_topics)))
        for topic in topics:
            claim = self.generate_claim(topic)
            claims.append(claim)
        return claims
