"""
Setup file for the AI Multi-Agent Misinformation System.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-misinformation-simulation",
    version="1.0.0",
    author="MCA Final Year Project",
    description=(
        "AI Multi-Agent Misinformation Spread, Verification & Moderation System: "
        "A Graph-Based Simulation Platform for Social Network Information Dynamics"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    py_modules=[
        "config",
        "social_network",
        "pipeline",
        "analytics",
        "app",
        "main",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "misinfo-sim=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "misinformation",
        "multi-agent",
        "social-network",
        "fact-checking",
        "content-moderation",
        "langgraph",
        "llm",
        "groq",
        "simulation",
        "graph-algorithms",
    ],
)
