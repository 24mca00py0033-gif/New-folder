"""
Misinformation Agent
====================
Generates a single realistic misinformation claim using Groq LLM.
This agent represents the source of fake news in the social network.
It injects exactly ONE claim into the network via the misinfo source node.
"""
import random
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL, MISINFORMATION_SYSTEM_PROMPT, TEMPERATURE


class MisinformationAgent:
    """
    Generates a single misinformation claim using Groq LLM.
    The claim is injected into the misinfo source node in the graph
    to kick off the simulation cascade.
    """

    TOPICS = [
        "government policy", "health and medicine", "technology",
        "environment", "education", "economy", "science",
        "social media", "international relations", "public safety",
        "artificial intelligence", "climate change", "cybersecurity",
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

    def generate_claim(self, topic=None) -> dict:
        """Generate a single misinformation claim using the LLM."""
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
                    "generated_by": "LLM",
                }
            except Exception as e:
                return {
                    "claim": f"[LLM Error: {str(e)[:80]}] — Unable to generate claim. Check your GROQ_API_KEY in .env file.",
                    "source_agent": "MisinformationAgent",
                    "topic": topic,
                    "generated_by": "error",
                }
        else:
            return {
                "claim": "⚠️ No GROQ_API_KEY found. Please set your API key in the .env file to generate claims.",
                "source_agent": "MisinformationAgent",
                "topic": topic,
                "generated_by": "error",
            }

    def inject_into_graph(self, network, claim: dict) -> dict:
        """Inject the claim into the misinfo source node in the graph."""
        misinfo_nodes = network.get_misinfo_nodes()
        if not misinfo_nodes:
            return {"success": False, "error": "No misinfo source node found in graph"}

        src_node = misinfo_nodes[0]  # Always use first misinfo node (single source)
        G = network.graph

        # Mark the source node as infected
        G.nodes[src_node]["status"] = "infected"
        G.nodes[src_node]["shared"] = True
        G.nodes[src_node]["infection_time"] = 0
        G.nodes[src_node]["claim_text"] = claim["claim"]

        return {
            "success": True,
            "source_node": src_node,
            "source_label": G.nodes[src_node]["label"],
            "claim": claim["claim"],
            "topic": claim.get("topic", ""),
            "degree": G.degree(src_node),
        }
