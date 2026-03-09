"""
Main entry point — Graph-Based Multi-Agent Misinformation Simulation System
============================================================================
Launches the Gradio web interface.

Usage:
    python main.py
    python app.py          (alternative)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Verify all required packages are installed."""
    required = {
        "networkx": "networkx",
        "matplotlib": "matplotlib",
        "gradio": "gradio",
        "langgraph": "langgraph",
        "langchain_groq": "langchain-groq",
        "langchain_core": "langchain-core",
        "groq": "groq",
        "numpy": "numpy",
        "PIL": "Pillow",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("❌ Missing dependencies:")
        print(f"   pip install {' '.join(missing)}")
        return False
    print("✅ All dependencies satisfied.")
    return True


def check_api_key():
    from config import GROQ_API_KEY
    if GROQ_API_KEY:
        print(f"✅ Groq API key set (starts with {GROQ_API_KEY[:8]}…)")
    else:
        print("⚠️  GROQ_API_KEY not set — enter it in the UI or export as env var.")


def main():
    print("=" * 60)
    print("  Graph-Based Multi-Agent Misinformation Simulation System")
    print("=" * 60)
    print()

    if not check_dependencies():
        sys.exit(1)

    check_api_key()
    print()
    print("🚀 Launching Gradio interface …")
    print("   Open http://localhost:7860 in your browser")
    print()

    from app import create_ui, GRADIO_THEME, CUSTOM_CSS
    app = create_ui()
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=GRADIO_THEME,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
