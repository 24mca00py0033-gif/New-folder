"""
Misinformation Agent
====================
Generates realistic fake news claims using Groq LLM.
Supports batch generation for multiple misinformation-source nodes.
"""
import random
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL, MISINFORMATION_SYSTEM_PROMPT, TEMPERATURE


class MisinformationAgent:
    """
    Generates realistic misinformation claims.
    Each misinformation-source node in the graph calls *generate_claim*
    to produce its unique claim before the cascade starts.
    """

    TOPICS = [
        "government policy", "health and medicine", "technology",
        "environment", "education", "economy", "science",
        "social media", "international relations", "public safety",
        "artificial intelligence", "climate change", "cybersecurity",
    ]

    FALLBACK_CLAIMS = [
        "Breaking: Government announces free 5G internet for all students starting January 2026.",
        "Scientists discover high-frequency Wi-Fi signals can accelerate plant growth by 300%.",
        "New study reveals that drinking cold water after a meal causes cancer, says WHO.",
        "Central bank to replace all physical currency with blockchain tokens by 2027.",
        "Leaked documents show top social media platform secretly sells user DNA data.",
        "NASA confirms a second moon has been captured by Earth's gravitational field.",
        "Major airline found injecting sleeping agents into cabin air on long-haul flights.",
        "World Health Organization declares coffee a controlled substance due to new research.",
        "Government plans to monitor all private messages for national security starting next month.",
        "Researchers claim 5G towers are responsible for a 40% decline in bee populations.",
    ]

    def __init__(self):
        self.llm = None
        if GROQ_API_KEY:
            try:
                self.llm = ChatGroq(
                    api_key=GROQ_API_KEY,
                    model=GROQ_MODEL,
                    temperature=TEMPERATURE + 0.2,
                    max_tokens=256,
                )
            except Exception:
                self.llm = None

    def generate_claim(self, topic=None):
        """Generate a single realistic misinformation claim."""
        topic = topic or random.choice(self.TOPICS)

        if self.llm:
            try:
                prompt = (
                    f"Generate a short, realistic news-like claim about {topic} that could be either "
                    "real or fake. 1-2 sentences, specific with names/numbers/dates. "
                    "Generate ONLY the claim text."
                )
                messages = [
                    SystemMessage(content=MISINFORMATION_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
                response = self.llm.invoke(messages)
                claim_text = response.content.strip().strip('"').strip("'")
                return {
                    "claim": claim_text,
                    "source_agent": "MisinformationAgent",
                    "topic": topic,
                    "generated": True,
                }
            except Exception as e:
                pass  # fall through to fallback

        # Fallback
        claim_text = random.choice(self.FALLBACK_CLAIMS)
        return {
            "claim": claim_text,
            "source_agent": "MisinformationAgent",
            "topic": topic,
            "generated": False,
        }

    def generate_batch(self, count: int) -> list[dict]:
        """
        Generate *count* unique misinformation claims using diverse topics.
        Each misinformation-source node in the graph gets one.
        """
        topics = random.sample(self.TOPICS, min(count, len(self.TOPICS)))
        while len(topics) < count:
            topics.append(random.choice(self.TOPICS))

        claims = []
        used_fallbacks = set()
        for topic in topics:
            claim = self.generate_claim(topic)
            # ensure fallback claims are unique
            if not claim["generated"]:
                attempts = 0
                while claim["claim"] in used_fallbacks and attempts < 20:
                    claim = self.generate_claim(topic)
                    attempts += 1
                used_fallbacks.add(claim["claim"])
            claims.append(claim)
        return claims
