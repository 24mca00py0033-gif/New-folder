import random
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL, MISINFORMATION_SYSTEM_PROMPT, TEMPERATURE


class MisinformationAgent:

    TOPICS = [
        "Indian general elections and campaign narratives",
        "state elections and coalition politics in India",
        "LPG cylinder shortage and fuel distribution in India",
        "India border tension and regional security updates",
        "Indian parliament policy debates and political affairs",
        "NEET, UPSC, SSC and other major Indian exam controversies",
        "public hospital load and healthcare access in India",
        "inflation, food prices, and household costs in Indian cities",
        "Indian digital policy, UPI, and cybersecurity incidents",
        "farmer protests, MSP discussions, and rural welfare in India",
        "railway recruitment, government jobs, and youth employment in India",
        "monsoon, flooding, and disaster response in Indian states",
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

        topic = topic or random.choice(self.TOPICS)

        if self.llm:
            try:
                prompt = (
                    f"Generate a short, realistic India-specific news-like claim about {topic} that could be either "
                    "real or fake inside a closed simulation. 1-2 sentences, specific with names/numbers/dates, "
                    "and Indian place/context references. "
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
        misinfo_nodes = network.get_misinfo_nodes()
        if not misinfo_nodes:
            return {"success": False, "error": "No misinfo source node found in graph"}

        src_node = misinfo_nodes[0]  
        G = network.graph

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
