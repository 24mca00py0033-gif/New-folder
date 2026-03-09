"""
Gradio UI for the AI Multi-Agent Misinformation Simulation System.
Provides an interactive web interface for running simulations and viewing results.
"""
import os
import gradio as gr
from PIL import Image

from pipeline import MisinformationPipeline
from analytics import SimulationAnalytics
from config import NETWORK_NUM_NODES, NETWORK_EDGES_PER_NODE, GROQ_API_KEY, NUM_INFLUENCERS, NUM_FACT_CHECKERS, NUM_MODERATORS


# ─── Global state ─────────────────────────────────────────────────────────────
pipeline_instance = None
last_result = None


def initialize_pipeline(num_nodes, edges_per_node, api_key, num_influencers=NUM_INFLUENCERS, num_fact_checkers=NUM_FACT_CHECKERS, num_moderators=NUM_MODERATORS):
    """Initialize or re-initialize the pipeline with given parameters."""
    global pipeline_instance

    # Set API key if provided
    if api_key and api_key.strip():
        os.environ["GROQ_API_KEY"] = api_key.strip()
        import config
        config.GROQ_API_KEY = api_key.strip()

    num_nodes = int(num_nodes)
    edges_per_node = int(edges_per_node)
    num_influencers = int(num_influencers)
    num_fact_checkers = int(num_fact_checkers)
    num_moderators = int(num_moderators)

    pipeline_instance = MisinformationPipeline(
        num_nodes=num_nodes,
        edges_per_node=edges_per_node,
        num_influencers=num_influencers,
        num_fact_checkers=num_fact_checkers,
        num_moderators=num_moderators,
    )
    return f"✅ Network initialized: {num_nodes} nodes, {edges_per_node} edges/node, {num_influencers} influencers, {num_fact_checkers} fact-checkers, {num_moderators} moderators"


def run_simulation(num_nodes, edges_per_node, api_key, num_influencers, num_fact_checkers, num_moderators):
    """Execute the full multi-agent simulation."""
    global pipeline_instance, last_result

    # Ensure pipeline is initialized
    init_msg = initialize_pipeline(num_nodes, edges_per_node, api_key, num_influencers, num_fact_checkers, num_moderators)

    if not pipeline_instance:
        return ("❌ Pipeline not initialized", "", "", "", "", "", "", None, None, "", "")

    try:
        # Run simulation
        result = pipeline_instance.run_simulation()
        last_result = result

        # Generate analytics
        analytics_engine = SimulationAnalytics(pipeline_instance.network)
        full_analytics = analytics_engine.generate_full_analytics(result)
        analytics_report = analytics_engine.generate_analytics_report(full_analytics)

        # Extract outputs
        claim = result.get("claim_data", {}).get("claim", "N/A")
        spread_summary = result.get("spread_summary", "N/A")
        verdict = result.get("verification_data", {}).get("verdict", "N/A")
        confidence = result.get("verification_data", {}).get("confidence", 0)
        evidence = result.get("verification_data", {}).get("evidence", "N/A")
        red_flags = result.get("verification_data", {}).get("red_flags", [])
        influencer_content = result.get("influencer_data", {}).get("rewritten_content", "N/A")
        amp_score = result.get("influencer_data", {}).get("amplification_score", 0)
        decision = result.get("moderation_data", {}).get("decision", "N/A")
        reason = result.get("moderation_data", {}).get("reason", "N/A")
        action = result.get("moderation_data", {}).get("action_taken", "N/A")
        severity = result.get("moderation_data", {}).get("severity", "N/A")
        spread_path = result.get("spread_data", {}).get("spread_path", [])
        total_reached = result.get("spread_data", {}).get("total_reached", 0)
        penetration = result.get("spread_data", {}).get("penetration_rate", 0)

        # Format outputs
        verdict_emoji = {"Real": "✅", "Fake": "❌", "Unverified": "⚠️"}.get(verdict, "❓")
        decision_emoji = {"BLOCK": "🚫", "FLAG": "⚠️", "ALLOW": "✅"}.get(decision, "❓")

        claim_output = f"📰 {claim}"

        spread_output = f"""📊 SPREAD RESULTS
{'─'*40}
🚀 Starting Node: User_{result.get('spread_data', {}).get('start_node', '?')}
📡 Nodes Reached: {total_reached} / {pipeline_instance.network.num_nodes}
📈 Penetration: {penetration}%
🔄 Max Depth: {result.get('spread_data', {}).get('max_depth_reached', 0)} hops
📊 Viral Coefficient: {result.get('spread_data', {}).get('viral_coefficient', 0)}
👁️ Total Exposures: {result.get('spread_data', {}).get('total_exposures', 0)}
📋 Path (first 15): {spread_path[:15]}"""

        verification_output = f"""{verdict_emoji} VERDICT: {verdict}
{'─'*40}
📊 Confidence: {confidence*100:.0f}%
📝 Evidence: {evidence}
🚩 Red Flags: {', '.join(red_flags) if red_flags else 'None'}"""

        influencer_output = f"""{'🛡️' if result.get('influencer_data', {}).get('action_type') == 'counter_messaging' else '📣'} INFLUENCER REWRITE
{'─'*40}
📊 Amplification Score: {amp_score}/10
Type: {result.get('influencer_data', {}).get('action_type', 'N/A')}

✍️ Rewritten:
{influencer_content}"""

        moderation_output = f"""{decision_emoji} DECISION: {decision}
{'─'*40}
📋 Reason: {reason}
⚡ Action: {action}
🔴 Severity: {severity}
🔒 Containment: {result.get('moderation_data', {}).get('spread_impact', {}).get('containment', 'N/A')} ({result.get('moderation_data', {}).get('spread_impact', {}).get('containment_rate', 0):.0f}%)"""

        # Pipeline log
        log = "\n".join(result.get("pipeline_log", []))
        log += f"\n\n⏱️ Total Time: {result.get('elapsed_time', 0):.1f}s"

        # Load images
        network_img = None
        analysis_img = None

        graph_path = result.get("network_graph_path", "network_graph.png")
        chart_path = result.get("analysis_chart_path", "spread_analysis.png")

        if os.path.exists(graph_path):
            network_img = Image.open(graph_path)
        if os.path.exists(chart_path):
            analysis_img = Image.open(chart_path)

        # Full report
        full_report = pipeline_instance.get_full_report(result)

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
        import traceback
        error_msg = f"❌ Simulation Error: {str(e)}\n\n{traceback.format_exc()}"
        return (error_msg, "", "", "", "", error_msg, "", None, None, error_msg)


def preview_network(num_nodes, edges_per_node, api_key, num_influencers, num_fact_checkers, num_moderators):
    """Generate and preview the social network graph before simulation."""
    init_msg = initialize_pipeline(num_nodes, edges_per_node, api_key, num_influencers, num_fact_checkers, num_moderators)

    if pipeline_instance:
        path = pipeline_instance.network.visualize_network(
            title=f"Social Network Preview — {int(num_nodes)} Nodes, {int(edges_per_node)} Edges/Node",
            save_path="network_preview.png"
        )
        stats = pipeline_instance.network.get_network_stats()
        stats_text = f"""📊 Network Statistics
{'─'*40}
Total Nodes: {stats['total_nodes']}
Total Edges: {stats['total_edges']}
Avg Degree: {stats['avg_degree']}
Max Degree: {stats['max_degree']}
Min Degree: {stats['min_degree']}
Density: {stats['density']}
Avg Clustering: {stats['avg_clustering']}
Influencers: {stats['num_influencers']}
Fact-Checkers: {stats['num_fact_checkers']}
Moderators: {stats['num_moderators']}
Connected: {'✅' if stats['connected'] else '❌'}"""

        if os.path.exists(path):
            return Image.open(path), stats_text
    return None, "❌ Failed to generate preview"


# ─── Build the Gradio Interface ──────────────────────────────────────────────

CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
}
.main-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    margin-bottom: 20px;
}
"""

GRADIO_THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="purple", neutral_hue="slate")

def create_ui():
    """Create and return the Gradio interface."""

    with gr.Blocks(
        title="AI Multi-Agent Misinformation System",
    ) as app:

        # ─── Header ──────────────────────────────────────────────────
        gr.Markdown("""
# 🛡️ AI Multi-Agent Misinformation Spread, Verification & Moderation System
### A Graph-Based Multi-Agent Simulation Using LangGraph and LLMs
*MCA Final Year Project — Simulating Social Network Information Dynamics*
---
        """)

        with gr.Tabs():
            # ━━━ TAB 1: SIMULATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("🚀 Run Simulation", id="simulation"):

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configuration")
                        api_key_input = gr.Textbox(
                            label="Groq API Key",
                            type="password",
                            placeholder="Enter your Groq API key (gsk_...)",
                            value=GROQ_API_KEY if GROQ_API_KEY else "",
                            info="Get your free API key from console.groq.com",
                        )
                        num_nodes_slider = gr.Slider(
                            minimum=20, maximum=300, value=100, step=10,
                            label="Number of Nodes",
                            info="Size of the social network graph",
                        )
                        edges_slider = gr.Slider(
                            minimum=1, maximum=6, value=3, step=1,
                            label="Edges per Node",
                            info="Connection density (Barabási-Albert model)",
                        )

                        gr.Markdown("### 🤖 Agent Counts")
                        influencer_slider = gr.Slider(
                            minimum=1, maximum=20, value=NUM_INFLUENCERS, step=1,
                            label="Number of Influencers 🟠",
                            info="High-degree nodes that amplify content",
                        )
                        fact_checker_slider = gr.Slider(
                            minimum=1, maximum=20, value=NUM_FACT_CHECKERS, step=1,
                            label="Number of Fact-Checkers 🟢",
                            info="Nodes that verify claims and reduce spread",
                        )
                        moderator_slider = gr.Slider(
                            minimum=1, maximum=20, value=NUM_MODERATORS, step=1,
                            label="Number of Moderators 🟣",
                            info="Nodes that enforce content moderation policies",
                        )

                        with gr.Row():
                            preview_btn = gr.Button(
                                "👁️ Preview Network", variant="secondary", size="lg",
                            )
                            run_btn = gr.Button(
                                "▶️ Generate & Run Simulation", variant="primary", size="lg",
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Network Preview")
                        preview_stats = gr.Textbox(
                            label="Network Statistics", lines=14,
                            interactive=False,
                        )

                preview_image = gr.Image(
                    label="Network Graph Preview", type="pil", height=500,
                )

                gr.Markdown("---")
                gr.Markdown("### 📋 Simulation Results")

                with gr.Row():
                    with gr.Column():
                        claim_output = gr.Textbox(
                            label="📰 1. Generated Claim (Misinformation Agent)",
                            lines=3, interactive=False,
                        )
                    with gr.Column():
                        spread_output = gr.Textbox(
                            label="📡 2. Spread Results (Neutral Agent)",
                            lines=10, interactive=False,
                        )

                with gr.Row():
                    with gr.Column():
                        verification_output = gr.Textbox(
                            label="🔍 3. Fact-Check Verdict (Fact-Checker Agent)",
                            lines=7, interactive=False,
                        )
                    with gr.Column():
                        influencer_output = gr.Textbox(
                            label="📣 4. Influencer Rewrite (Influencer Agent)",
                            lines=7, interactive=False,
                        )

                moderation_output = gr.Textbox(
                    label="🛡️ 5. Moderation Decision (Moderator Agent)",
                    lines=7, interactive=False,
                )

                pipeline_log = gr.Textbox(
                    label="📝 Pipeline Execution Log",
                    lines=12, interactive=False,
                )

            # ━━━ TAB 2: VISUALIZATIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("📊 Visualizations", id="visualizations"):
                gr.Markdown("### 🌐 Social Network Graph with Spread Path")
                network_image = gr.Image(
                    label="Network Graph", type="pil", height=600,
                )

                gr.Markdown("### 📈 Spread Analysis Dashboard")
                analysis_image = gr.Image(
                    label="Analysis Charts", type="pil", height=600,
                )

            # ━━━ TAB 3: ANALYTICS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("📈 Detailed Analytics", id="analytics"):
                gr.Markdown("### 📊 Comprehensive Simulation Analytics")
                analytics_output = gr.Textbox(
                    label="Analytics Report",
                    lines=45, interactive=False,
                )

            # ━━━ TAB 4: FULL REPORT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("📄 Full Report", id="report"):
                gr.Markdown("### 📄 Complete Simulation Report")
                report_output = gr.Textbox(
                    label="Full Report",
                    lines=50, interactive=False,
                )

            # ━━━ TAB 5: ABOUT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("ℹ️ About", id="about"):
                gr.Markdown("""
## About This Project

### 🎯 Project Title
**AI Multi-Agent Misinformation Spread, Verification & Moderation System**  
A Graph-Based Simulation Platform for Social Network Information Dynamics

### 🏗️ Architecture
This system deploys **5 specialized AI agents** within a graph-based social network:

| Agent | Role | Technology |
|-------|------|-----------|
| 🔴 **Misinformation Agent** | Generates realistic fake news claims | Groq LLM API |
| 🔵 **Neutral Agent** | Simulates viral content spread via BFS | NetworkX Graph Traversal |
| 🟢 **Fact-Checker Agent** | Verifies claims with evidence-based reasoning | LLM + Tool Calling |
| 🟠 **Influencer Agent** | Rewrites content for viral impact or counter-messaging | Advanced Prompt Engineering |
| 🟣 **Moderator Agent** | Makes flag/block/allow decisions | LLM Policy Reasoning |

### 🔄 Pipeline Flow
```
Claim Generation → Network Spread → Fact Verification → Influencer Rewrite → Moderation → Analytics
```

### 📊 Analysis Dimensions
1. **Spread Velocity** — How fast fake news spreads through the network
2. **Verification Impact** — How fact-checking slows or stops misinformation
3. **Agent Influence** — Which agents have the most impact on information flow
4. **Moderation Effectiveness** — How content moderation reduces harmful spread

### 🛠️ Technology Stack
- **Language**: Python 3.x
- **LLM Provider**: Groq (LLaMA 3.3 70B)
- **Agent Orchestration**: LangGraph
- **Graph Library**: NetworkX (Barabási-Albert model)
- **UI Framework**: Gradio
- **Visualization**: Matplotlib
                """)

        # ─── Event Handlers ───────────────────────────────────────────
        preview_btn.click(
            fn=preview_network,
            inputs=[num_nodes_slider, edges_slider, api_key_input, influencer_slider, fact_checker_slider, moderator_slider],
            outputs=[preview_image, preview_stats],
        )

        run_btn.click(
            fn=run_simulation,
            inputs=[num_nodes_slider, edges_slider, api_key_input, influencer_slider, fact_checker_slider, moderator_slider],
            outputs=[
                claim_output,
                spread_output,
                verification_output,
                influencer_output,
                moderation_output,
                pipeline_log,
                analytics_output,
                network_image,
                analysis_image,
                report_output,
            ],
        )

    return app


# ─── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = create_ui()
    app.launch(
        share=False, server_name="0.0.0.0", server_port=7860,
        theme=GRADIO_THEME,
        css=CUSTOM_CSS,
    )
