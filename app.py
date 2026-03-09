"""
Gradio UI — Graph-Based Multi-Agent Misinformation Simulation System
=====================================================================
Provides an interactive web interface with configurable network size,
agent counts, spread probability, amplification factor, and analytical
dashboards for the multi-cascade simulation.
"""
import os
import traceback

import gradio as gr
from PIL import Image

from pipeline import MisinformationPipeline
from analytics import SimulationAnalytics
from config import (
    NETWORK_NUM_NODES, NETWORK_EDGES_PER_NODE, GROQ_API_KEY,
    NUM_MISINFO_AGENTS, NUM_INFLUENCERS, NUM_FACT_CHECKERS, NUM_MODERATORS,
    SPREAD_PROBABILITY, AMPLIFICATION_FACTOR, MAX_SPREAD_DEPTH,
)


# ─── Global state ─────────────────────────────────────────────────────────────
pipeline_instance = None


def _set_api_key(api_key: str):
    if api_key and api_key.strip():
        os.environ["GROQ_API_KEY"] = api_key.strip()
        import config
        config.GROQ_API_KEY = api_key.strip()


def initialize_pipeline(
    num_nodes, edges_per_node, api_key,
    num_misinfo, num_influencers, num_fact_checkers, num_moderators,
    spread_prob, amplification,
):
    """Create (or recreate) the pipeline with the given parameters."""
    global pipeline_instance
    _set_api_key(api_key)

    pipeline_instance = MisinformationPipeline(
        num_nodes=int(num_nodes),
        edges_per_node=int(edges_per_node),
        num_misinfo=int(num_misinfo),
        num_influencers=int(num_influencers),
        num_fact_checkers=int(num_fact_checkers),
        num_moderators=int(num_moderators),
        spread_prob=float(spread_prob),
        amplification=float(amplification),
    )
    return pipeline_instance


# ── Simulation runner ─────────────────────────────────────────────────────────

def run_simulation(
    num_nodes, edges_per_node, api_key,
    num_misinfo, num_influencers, num_fact_checkers, num_moderators,
    spread_prob, amplification,
):
    """Execute the full multi-cascade simulation and return UI outputs."""
    empty = ("", "", "", "", "", "", "", None, None, "")
    try:
        pipe = initialize_pipeline(
            num_nodes, edges_per_node, api_key,
            num_misinfo, num_influencers, num_fact_checkers, num_moderators,
            spread_prob, amplification,
        )
        result = pipe.run_simulation()

        # Analytics
        analytics_engine = SimulationAnalytics(pipe.network)
        full_analytics = analytics_engine.generate_full_analytics(result)
        analytics_report = analytics_engine.generate_analytics_report(full_analytics)

        # ── Claim summary ────────────────────────────────────────────
        claims = result.get("claims", [])
        claim_lines = []
        for i, c in enumerate(claims, 1):
            claim_lines.append(f"C{i}: {c.get('claim', 'N/A')}")
        claim_output = "📰 Generated Claims:\n" + "\n".join(claim_lines)

        # ── Spread summary ───────────────────────────────────────────
        spread_output = result.get("spread_summary", "N/A")

        # ── Verification summary ─────────────────────────────────────
        verdicts = result.get("verification_results", [])
        ver_lines = []
        for v in verdicts:
            emoji = {"Real": "✅", "Fake": "❌", "Unverified": "⚠️"}.get(v.get("verdict"), "❓")
            ver_lines.append(
                f"{emoji} {v.get('verdict','?')} ({v.get('confidence',0)*100:.0f}%) "
                f"— {v.get('claim','')[:80]}"
            )
        verification_output = "🔍 Fact-Check Results:\n" + "\n".join(ver_lines)

        # ── Influencer summary ───────────────────────────────────────
        inf_results = result.get("influencer_results", [])
        inf_lines = []
        for r in inf_results:
            inf_lines.append(
                f"• {r.get('action_type','?')} (score {r.get('amplification_score',0)}/10)\n"
                f"  → {r.get('rewritten_content','')[:120]}"
            )
        influencer_output = "📣 Influencer Rewrites:\n" + "\n".join(inf_lines)

        # ── Moderation summary ───────────────────────────────────────
        mod_results = result.get("moderation_results", [])
        mod_lines = []
        for m in mod_results:
            emoji = {"BLOCK": "🚫", "FLAG": "⚠️", "ALLOW": "✅"}.get(m.get("decision"), "❓")
            mod_lines.append(
                f"{emoji} {m.get('decision','?')} [{m.get('severity','?')}] — {m.get('reason','')[:100]}"
            )
        moderation_output = "🛡️ Moderation Decisions:\n" + "\n".join(mod_lines)

        # ── Pipeline log ─────────────────────────────────────────────
        log = "\n".join(result.get("pipeline_log", []))
        log += f"\n\n⏱️ Total Time: {result.get('elapsed_time', 0):.1f}s"

        # ── Images ───────────────────────────────────────────────────
        network_img = None
        analysis_img = None
        graph_path = result.get("network_graph_path", "network_graph.png")
        chart_path = result.get("analysis_chart_path", "spread_analysis.png")
        if os.path.exists(graph_path):
            network_img = Image.open(graph_path)
        if os.path.exists(chart_path):
            analysis_img = Image.open(chart_path)

        # ── Full report ──────────────────────────────────────────────
        full_report = pipe.get_full_report(result)

        return (
            claim_output,
            spread_output,
            verification_output,
            influencer_output,
            moderation_output,
            log,
            analytics_report,
            network_img,
            analysis_img,
            full_report,
        )

    except Exception as e:
        err = f"❌ Simulation Error: {e}\n\n{traceback.format_exc()}"
        return (err, "", "", "", "", err, "", None, None, err)


def preview_network(
    num_nodes, edges_per_node, api_key,
    num_misinfo, num_influencers, num_fact_checkers, num_moderators,
    spread_prob, amplification,
):
    """Generate and preview the social network graph (no cascade)."""
    try:
        pipe = initialize_pipeline(
            num_nodes, edges_per_node, api_key,
            num_misinfo, num_influencers, num_fact_checkers, num_moderators,
            spread_prob, amplification,
        )
        path = pipe.network.visualize_network(
            title=f"Social Network Preview — {int(num_nodes)} Nodes",
            save_path="network_preview.png",
        )
        stats = pipe.network.get_network_stats()
        stats_text = (
            f"📊 Network Statistics\n{'─'*40}\n"
            f"Total Nodes: {stats['total_nodes']}\n"
            f"Total Edges: {stats['total_edges']}\n"
            f"Avg Degree:  {stats['avg_degree']}\n"
            f"Max Degree:  {stats['max_degree']}\n"
            f"Min Degree:  {stats['min_degree']}\n"
            f"Density:     {stats['density']}\n"
            f"Avg Clustering: {stats['avg_clustering']}\n"
            f"Misinfo Agents: {stats.get('num_misinfo_agents', '?')}\n"
            f"Influencers:    {stats.get('num_influencers', '?')}\n"
            f"Fact-Checkers:  {stats.get('num_fact_checkers', '?')}\n"
            f"Moderators:     {stats.get('num_moderators', '?')}\n"
            f"Connected: {'✅' if stats.get('connected') else '❌'}"
        )
        if os.path.exists(path):
            return Image.open(path), stats_text
    except Exception as e:
        return None, f"❌ Preview failed: {e}"
    return None, "❌ Failed to generate preview"


# ─── Build the Gradio Interface ──────────────────────────────────────────────

CUSTOM_CSS = """
.gradio-container { max-width: 1500px !important; }
"""

GRADIO_THEME = gr.themes.Soft(
    primary_hue="blue", secondary_hue="purple", neutral_hue="slate",
)


def create_ui():
    """Create and return the Gradio Blocks app."""

    with gr.Blocks(title="Multi-Agent Misinformation Simulation") as app:

        # ─── Header ──────────────────────────────────────────────────
        gr.Markdown("""
# 🛡️ Graph-Based Multi-Agent Misinformation Simulation System
### Simultaneous BFS Cascades · Embedded Agent Nodes · LangGraph + Groq LLMs
*MCA Final Year Project — Simulating Social Network Information Dynamics*
---
        """)

        with gr.Tabs():
            # ━━━ TAB 1: SIMULATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("🚀 Run Simulation", id="simulation"):

                with gr.Row():
                    # ── Left column: Config ──────────────────────────
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Network & API")
                        api_key_input = gr.Textbox(
                            label="Groq API Key", type="password",
                            placeholder="gsk_...",
                            value=GROQ_API_KEY or "",
                            info="console.groq.com → free key",
                        )
                        num_nodes_slider = gr.Slider(
                            minimum=50, maximum=2000, value=NETWORK_NUM_NODES, step=50,
                            label="Number of Nodes",
                            info="Network size (Barabási-Albert)",
                        )
                        edges_slider = gr.Slider(
                            minimum=1, maximum=8, value=NETWORK_EDGES_PER_NODE, step=1,
                            label="Edges per Node (m)",
                            info="Connection density",
                        )

                        gr.Markdown("### 🤖 Embedded Agent Counts")
                        misinfo_slider = gr.Slider(
                            minimum=1, maximum=15, value=NUM_MISINFO_AGENTS, step=1,
                            label="🔴 Misinformation Sources",
                            info="Nodes that inject unique fake claims",
                        )
                        influencer_slider = gr.Slider(
                            minimum=1, maximum=50, value=NUM_INFLUENCERS, step=1,
                            label="🟠 Influencers",
                            info="High-degree nodes that amplify content",
                        )
                        fact_checker_slider = gr.Slider(
                            minimum=1, maximum=50, value=NUM_FACT_CHECKERS, step=1,
                            label="🟢 Fact-Checkers",
                            info="Verify claims & attach warnings",
                        )
                        moderator_slider = gr.Slider(
                            minimum=1, maximum=50, value=NUM_MODERATORS, step=1,
                            label="🟣 Moderators",
                            info="BLOCK / FLAG / ALLOW at their nodes",
                        )

                        gr.Markdown("### 📡 Propagation")
                        spread_prob_slider = gr.Slider(
                            minimum=0.05, maximum=0.90, value=SPREAD_PROBABILITY, step=0.05,
                            label="Base Spread Probability",
                            info="Chance a normal user reshares",
                        )
                        amplification_slider = gr.Slider(
                            minimum=1.0, maximum=5.0, value=AMPLIFICATION_FACTOR, step=0.5,
                            label="Influencer Amplification Factor",
                            info="Multiplier on neighbour fan-out",
                        )

                        with gr.Row():
                            preview_btn = gr.Button("👁️ Preview Network", variant="secondary", size="lg")
                            run_btn = gr.Button("▶️ Run Simulation", variant="primary", size="lg")

                    # ── Right column: Preview ────────────────────────
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Network Preview")
                        preview_stats = gr.Textbox(label="Network Statistics", lines=14, interactive=False)

                preview_image = gr.Image(label="Network Graph Preview", type="pil", height=500)

                gr.Markdown("---")
                gr.Markdown("### 📋 Simulation Results")

                with gr.Row():
                    with gr.Column():
                        claim_output = gr.Textbox(
                            label="📰 1. Generated Claims (Misinformation Agents)",
                            lines=8, interactive=False,
                        )
                    with gr.Column():
                        spread_output = gr.Textbox(
                            label="📡 2. Multi-Cascade Spread (Neutral Agent / BFS)",
                            lines=12, interactive=False,
                        )

                with gr.Row():
                    with gr.Column():
                        verification_output = gr.Textbox(
                            label="🔍 3. Fact-Check Verdicts",
                            lines=10, interactive=False,
                        )
                    with gr.Column():
                        influencer_output = gr.Textbox(
                            label="📣 4. Influencer Rewrites",
                            lines=10, interactive=False,
                        )

                moderation_output = gr.Textbox(
                    label="🛡️ 5. Moderation Decisions",
                    lines=8, interactive=False,
                )
                pipeline_log = gr.Textbox(
                    label="📝 Pipeline Execution Log",
                    lines=15, interactive=False,
                )

            # ━━━ TAB 2: VISUALIZATIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("📊 Visualizations", id="visualizations"):
                gr.Markdown("### 🌐 Social Network — Multi-Cascade Spread")
                network_image = gr.Image(label="Network Graph", type="pil", height=650)
                gr.Markdown("### 📈 Spread Analysis Dashboard (2×3)")
                analysis_image = gr.Image(label="Analysis Charts", type="pil", height=650)

            # ━━━ TAB 3: ANALYTICS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("📈 Detailed Analytics", id="analytics"):
                gr.Markdown("### 📊 Comprehensive Multi-Cascade Analytics")
                analytics_output = gr.Textbox(label="Analytics Report", lines=50, interactive=False)

            # ━━━ TAB 4: FULL REPORT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("📄 Full Report", id="report"):
                gr.Markdown("### 📄 Complete Simulation Report")
                report_output = gr.Textbox(label="Full Report", lines=55, interactive=False)

            # ━━━ TAB 5: ABOUT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("ℹ️ About", id="about"):
                gr.Markdown("""
## About This Project

### 🎯 Project Title
**Graph-Based Multi-Agent Misinformation Simulation System**
A research platform for studying how misinformation spreads, gets verified,
amplified, and moderated inside social networks.

### 🏗️ Architecture
Agents are **embedded as nodes** inside the graph. Each misinformation source
injects a unique claim that propagates via BFS with role-based processing
at every hop:

| Agent Node | Role | Behaviour during BFS |
|------------|------|---------------------|
| 🔴 **Misinformation** | Cascade origin | Injects claim, always reshares |
| 🟠 **Influencer** | Amplifier | Higher reshare prob, N× fan-out |
| 🟢 **Fact-Checker** | Verifier | Attaches warning label, low reshare |
| 🟣 **Moderator** | Gatekeeper | BLOCK / FLAG / ALLOW — stops or slows cascade |
| 🔵 **Normal User** | Spreader | Base probability reshare |

### 🔄 Pipeline (LangGraph StateGraph)
```
Generate N Claims → Run Simultaneous BFS Cascades → Fact-Check All Claims
→ Influencer Rewrite → Moderation Verdict → Analytics & Visualisations
```

### 🛠️ Technology Stack
Python · NetworkX (Barabási-Albert) · LangGraph · Groq (LLaMA 3.3 70B)
· Gradio · Matplotlib · NumPy
                """)

        # ─── Inputs list (shared by preview & run) ───────────────────
        all_inputs = [
            num_nodes_slider, edges_slider, api_key_input,
            misinfo_slider, influencer_slider, fact_checker_slider, moderator_slider,
            spread_prob_slider, amplification_slider,
        ]

        # ─── Event Handlers ──────────────────────────────────────────
        preview_btn.click(
            fn=preview_network,
            inputs=all_inputs,
            outputs=[preview_image, preview_stats],
        )

        run_btn.click(
            fn=run_simulation,
            inputs=all_inputs,
            outputs=[
                claim_output, spread_output,
                verification_output, influencer_output,
                moderation_output, pipeline_log,
                analytics_output,
                network_image, analysis_image,
                report_output,
            ],
        )

    return app


# ─── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = create_ui()
    app.launch(
        share=False, server_name="0.0.0.0", server_port=7860,
        theme=GRADIO_THEME, css=CUSTOM_CSS,
    )
