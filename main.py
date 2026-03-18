"""
CLI entry point for the Misinformation Simulation.
Run: python main.py
Or use the Gradio UI: python app.py
"""
from pipeline import MisinformationPipeline
from config import calculate_agent_counts


def main():
    print("\n🚀 Starting Sequential Misinformation Simulation …\n")

    num_nodes = 100
    agents = calculate_agent_counts(num_nodes)
    print(f"📊 Network: {num_nodes} nodes")
    print(f"🤖 Agents: {agents}\n")

    pipe = MisinformationPipeline(num_nodes=num_nodes)
    result = pipe.run_simulation()

    # Print pipeline log
    for line in result.get("pipeline_log", []):
        print(line)

    # Print full report
    print(pipe.get_full_report(result))


if __name__ == "__main__":
    main()
