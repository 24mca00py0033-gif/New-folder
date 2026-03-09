"""
Configuration for the AI Multi-Agent Misinformation System.
"""
import os

# ─── Groq API Configuration ───────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # Set your Groq API key here or via env var
GROQ_MODEL = "llama-3.3-70b-versatile"  # Groq model to use

# ─── Social Network Configuration ─────────────────────────────────────────────
NETWORK_NUM_NODES = 100          # Number of nodes in the social network
NETWORK_EDGES_PER_NODE = 3       # Edges per new node (Barabási-Albert model)
NETWORK_SEED = 42                # Random seed for reproducibility

# ─── Agent Configuration ──────────────────────────────────────────────────────
MAX_SPREAD_DEPTH = 50             # Max BFS depth for neutral agent spread
SPREAD_PROBABILITY = 0.6         # Probability a neutral node reshares content
INFLUENCER_THRESHOLD = 10        # Min degree for a node to be an influencer

# ─── Agent Role Counts (Configurable) ─────────────────────────────────────────
NUM_INFLUENCERS = 25              # Number of influencer nodes in the network
NUM_FACT_CHECKERS = 35            # Number of fact-checker nodes in the network
NUM_MODERATORS = 50               # Number of moderator nodes in the network

# ─── Simulation Parameters ────────────────────────────────────────────────────
DEFAULT_NUM_SIMULATIONS = 10     # Default simulations per run
TEMPERATURE = 0.7                # LLM temperature for creative generation

# ─── Visualization ────────────────────────────────────────────────────────────
GRAPH_FIGURE_SIZE = (75, 85)
GRAPH_DPI = 100
NODE_SIZE_BASE = 80
NODE_SIZE_SCALE = 55

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
