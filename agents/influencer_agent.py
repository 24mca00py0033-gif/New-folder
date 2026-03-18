"""
Influencer Agent
================
After neutral spread, influencer nodes that were exposed to the claim
modify/amplify it using LLM and re-spread to their additional neighbours.
This simulates how social media influencers amplify content.
"""
import random
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import (
    GROQ_API_KEY, GROQ_MODEL, INFLUENCER_SYSTEM_PROMPT,
    TEMPERATURE, AMPLIFICATION_FACTOR,
)


class InfluencerAgent:
    """
    Influencer nodes modify the claim and amplify its spread.
    After neutral users have spread the claim, influencers who were
    exposed to it rewrite it for maximum engagement and spread it
    to their neighbours.
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

    def influence_graph(self, network, amplification: float | None = None) -> dict:
        """
        Influencer nodes that were exposed to the claim modify it
        and spread to their additional neighbours.
        """
        if amplification is None:
            amplification = AMPLIFICATION_FACTOR

        G = network.graph
        influencer_nodes = network.get_influencer_nodes()

        # Find influencers who were exposed (have exposure_count > 0)
        active_influencers = [
            n for n in influencer_nodes
            if G.nodes[n]["exposure_count"] > 0
        ]

        if not active_influencers:
            return {
                "success": True,
                "active_influencers": 0,
                "additional_spread": 0,
                "modified_claim": "",
                "amplification_score": 0,
                "influenced_nodes": [],
            }

        # Get the original claim from any infected node
        claim_text = ""
        for n in G.nodes():
            if G.nodes[n].get("claim_text"):
                claim_text = G.nodes[n]["claim_text"]
                break

        # Use LLM to modify the claim
        modified_claim = self._modify_claim(claim_text)

        # Spread from influencer nodes to their neighbours
        influenced_nodes = []
        edges = []

        for inf_node in active_influencers:
            G.nodes[inf_node]["status"] = "influenced"
            G.nodes[inf_node]["shared"] = True
            G.nodes[inf_node]["claim_text"] = modified_claim

            neighbours = list(G.neighbors(inf_node))
            random.shuffle(neighbours)

            # Amplified fan-out
            fan_out = min(len(neighbours), int(len(neighbours) * amplification))

            for nb in neighbours[:fan_out]:
                if G.nodes[nb]["status"] == "clean":
                    # Higher spread probability for influencer-amplified content
                    if random.random() < 0.6:
                        G.nodes[nb]["status"] = "influenced"
                        G.nodes[nb]["shared"] = True
                        G.nodes[nb]["claim_text"] = modified_claim
                        G.nodes[nb]["exposure_count"] += 1
                        influenced_nodes.append(nb)
                        edges.append((inf_node, nb))

        amp_score = self._calc_amplification(claim_text, modified_claim)

        return {
            "success": True,
            "active_influencers": len(active_influencers),
            "additional_spread": len(influenced_nodes),
            "modified_claim": modified_claim,
            "original_claim": claim_text,
            "amplification_score": amp_score,
            "influenced_nodes": influenced_nodes,
            "influencer_node_ids": active_influencers,
            "edges": edges,
        }

    def _modify_claim(self, claim_text: str) -> str:
        """Use LLM to rewrite the claim for maximum viral spread."""
        if not claim_text:
            return ""

        if self.llm:
            try:
                prompt = (
                    f"The following claim is circulating on social media:\n\n"
                    f'CLAIM: "{claim_text}"\n\n'
                    "Rewrite it to maximize viral engagement with hooks, "
                    "emotional triggers, and hashtags. Make it attention-grabbing. "
                    "Write ONLY the rewritten post."
                )
                messages = [
                    SystemMessage(content=INFLUENCER_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
                response = self.llm.invoke(messages)
                return response.content.strip()
            except Exception:
                pass

        # Fallback: simple amplification
        return f"🔥 BREAKING: {claim_text} — Share this NOW! #Viral #Breaking"

    @staticmethod
    def _calc_amplification(original, rewritten):
        score = 5.0
        if not original or not rewritten:
            return score
        ratio = len(rewritten) / max(len(original), 1)
        if 1.5 <= ratio <= 3.0:
            score += 1.0
        elif ratio > 3.0:
            score += 0.5
        markers = ["!", "?", "🔥", "⚠️", "❌", "✅", "#", "BREAKING",
                    "WARNING", "URGENT", "SHARE", "VERIFIED", "FACT CHECK"]
        score += min(sum(1 for m in markers if m.upper() in rewritten.upper()) * 0.3, 2.0)
        return round(min(10.0, max(1.0, score)), 1)

    @staticmethod
    def get_influencer_summary(result: dict) -> str:
        return (
            f"\n📣 INFLUENCER RESULTS\n{'='*50}\n"
            f"Active Influencers : {result.get('active_influencers', 0)}\n"
            f"Additional Spread  : {result.get('additional_spread', 0)} nodes\n"
            f"Amplification Score: {result.get('amplification_score', 0)}/10\n\n"
            f"✍️ Modified Claim:\n{result.get('modified_claim', 'N/A')}\n"
        )
