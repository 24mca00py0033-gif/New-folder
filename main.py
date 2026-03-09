"""
Main entry point for the AI Multi-Agent Misinformation System.
Run this file to launch the Gradio web interface.

Usage:
    python main.py
    
Or run the app directly:
    python app.py
"""
import sys
import os

# Add project root to path
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
        print("❌ Missing dependencies detected!")
        print(f"   Run: pip install {' '.join(missing)}")
        print(f"   Or:  pip install -r requirements.txt")
        return False

    print("✅ All dependencies satisfied.")
    return True


def check_api_key():
    """Check if Groq API key is configured."""
    from config import GROQ_API_KEY
    if GROQ_API_KEY:
        print(f"✅ Groq API key configured (starts with: {GROQ_API_KEY[:8]}...)")
        return True
    else:
        print("⚠️  Groq API key not set in environment.")
        print("   You can enter it in the Gradio UI or set GROQ_API_KEY environment variable.")
        print("   Get your free key at: https://console.groq.com")
        return True  # Allow running without key (can enter in UI)


def main():
    """Launch the application."""
    print("=" * 60)
    print("  AI Multi-Agent Misinformation Simulation System")
    print("  MCA Final Year Project")
    print("=" * 60)
    print()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check API key
    check_api_key()

    print()
    print("🚀 Launching Gradio interface...")
    print("   Open http://localhost:7860 in your browser")
    print()

    # Import and launch
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
