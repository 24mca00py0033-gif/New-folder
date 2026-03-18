"""
Configuration for the Graph-Based Multi-Agent Misinformation Simulation System.
All tuneable parameters live here so every module stays in sync.
Agent counts are auto-calculated based on graph size.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Groq API Configuration ───────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ─── Social Network Configuration ─────────────────────────────────────────────
NETWORK_NUM_NODES = 100          # Default nodes (user can choose 100-1000)
NETWORK_EDGES_PER_NODE = 3       # m parameter (edges per new node)
NETWORK_SEED = 42                # Reproducibility seed

# ─── Agent Role Counts ────────────────────────────────────────────────────────
# These are defaults for 100 nodes. They are auto-recalculated at runtime.
NUM_MISINFO_AGENTS = 1           # Always 1 (single claim source)
NUM_INFLUENCERS = 5              # ~5% of nodes
NUM_FACT_CHECKERS = 8            # ~8% of nodes
NUM_MODERATORS = 10              # ~10% of nodes
# Remaining nodes → normal users


def calculate_agent_counts(num_nodes: int) -> dict:
    """Auto-calculate agent counts based on graph size."""
    return {
        "num_misinfo": 1,                              # Always 1 source
        "num_influencers": max(2, int(num_nodes * 0.05)),   # ~5%
        "num_fact_checkers": max(3, int(num_nodes * 0.08)), # ~8%
        "num_moderators": max(3, int(num_nodes * 0.10)),    # ~10%
    }


# ─── Propagation Parameters ──────────────────────────────────────────────────
SPREAD_PROBABILITY = 0.35        # Base reshare probability for normal users
AMPLIFICATION_FACTOR = 3.0       # Influencers spread to N× more neighbours
MAX_SPREAD_DEPTH = 15            # Max BFS hops per cascade
FACT_CHECK_SLOW = 0.15           # Fact-checker reshare probability (with warning)
MODERATOR_BLOCK_PROB = 0.85      # Probability a moderator blocks fake content
MODERATOR_FLAG_PROB = 0.70       # Probability a moderator flags unverified content

# ─── Simulation Parameters ────────────────────────────────────────────────────
TEMPERATURE = 0.7                # LLM temperature for creative agents

# ─── Visualization ────────────────────────────────────────────────────────────
GRAPH_FIGURE_SIZE = (28, 22)
GRAPH_DPI = 120
NODE_SIZE_BASE = 30
NODE_SIZE_SCALE = 18

# ─── Role Colours (used everywhere) ──────────────────────────────────────────
ROLE_COLOURS = {
    "misinfo":       "#e74c3c",   # Red
    "influencer":    "#f5a623",   # Orange
    "fact_checker":  "#2ecc71",   # Green
    "moderator":     "#9b59b6",   # Purple
    "normal":        "#3498db",   # Blue
}
STATUS_COLOURS = {
    "infected":    "#ff6b6b",
    "influenced":  "#ff9f43",
    "warned":      "#f1c40f",
    "blocked":     "#95a5a6",
}

# ─── Agent Role Prompts ───────────────────────────────────────────────────────
MISINFORMATION_SYSTEM_PROMPT = """You are a misinformation simulation agent for research purposes only.
Your role is to generate realistic-sounding but potentially false news claims for studying
misinformation spread patterns. Generate claims that are plausible but may be fabricated.
These claims are used ONLY in a closed simulation environment for academic research.

IMPORTANT: Generate ONLY the claim text, nothing else. No explanations, no labels, no disclaimers.
Keep claims to 1-2 sentences maximum."""

FACT_CHECKER_SYSTEM_PROMPT = """You are a fact-checking AI agent. Your role is to analyze claims
and determine their truthfulness. You should:
1. Analyze the claim's plausibility based on your knowledge
2. Consider common misinformation patterns
3. Provide a verdict: "Real", "Fake", or "Unverified"
4. Provide brief evidence/reasoning for your verdict

Respond in this EXACT JSON format:
{
    "verdict": "Real" or "Fake" or "Unverified",
    "confidence": 0.0 to 1.0,
    "evidence": "Brief explanation of your reasoning",
    "red_flags": ["list", "of", "suspicious", "elements"]
}"""

INFLUENCER_SYSTEM_PROMPT = """You are a social media influencer simulation agent. Your role depends on the verdict:

If the claim is FAKE or UNVERIFIED:
- Rewrite it as a WARNING message that alerts users about the misinformation
- Use urgent, attention-grabbing language to COUNTER the false claim
- Include phrases like "FACT CHECK:", "WARNING:", "MISLEADING:"

If the claim is REAL:
- Rewrite it to maximize engagement and viral spread
- Use compelling hooks, emotional triggers, and share-worthy formatting
- Make it attention-grabbing while keeping the core facts

IMPORTANT: Return ONLY the rewritten text, nothing else."""

MODERATOR_SYSTEM_PROMPT = """You are a content moderation AI agent. Based on the fact-check verdict
and the content provided, make a moderation decision.

Your possible decisions are:
1. "BLOCK" - Content is confirmed fake and harmful, block immediately
2. "FLAG" - Content is suspicious or unverified, flag for review and reduce spread
3. "ALLOW" - Content appears legitimate, allow normal distribution

Respond in this EXACT JSON format:
{
    "decision": "BLOCK" or "FLAG" or "ALLOW",
    "reason": "Brief explanation of the decision",
    "action_taken": "Description of what happens to the content",
    "severity": "HIGH" or "MEDIUM" or "LOW"
}"""
