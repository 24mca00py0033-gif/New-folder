# 🛡️ AI Multi-Agent Misinformation Spread, Verification & Moderation System

> A Graph-Based Multi-Agent Simulation Using LangGraph and LLMs  
> **MCA Final Year Project**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Agents](#agents)
- [Pipeline Flow](#pipeline-flow)
- [Analytics](#analytics)
- [Screenshots](#screenshots)
- [License](#license)

---

## 🎯 Overview

This project develops an intelligent **multi-agent AI system** that simulates real-world social media dynamics to understand, analyze, and mitigate the spread of misinformation. The system creates a **synthetic social network** (100 nodes, Barabási-Albert model) where autonomous AI agents interact, share information, verify claims, influence content, and moderate harmful posts—mimicking the complex ecosystem of modern social platforms.

Unlike traditional rule-based content moderation systems, this project leverages **Large Language Models (LLMs)** and **multi-agent orchestration via LangGraph** to create a dynamic, adaptive simulation that can test various scenarios of information spread, fact-checking interventions, influencer amplification, and moderation strategies.

---

## ✨ Features

- **🔴 Misinformation Generation** — AI-powered realistic fake news claim generation using Groq LLM
- **🔵 Viral Spread Simulation** — BFS-based propagation through a 100-node social network graph
- **🟢 Automated Fact-Checking** — LLM-based claim verification with structured verdicts (Real/Fake/Unverified)
- **🟠 Influencer Amplification** — Content rewriting for virality or counter-messaging based on verdicts
- **🟣 Intelligent Moderation** — Policy-based BLOCK/FLAG/ALLOW decisions with containment analysis
- **📊 Comprehensive Analytics** — Spread velocity, verification impact, agent influence, moderation effectiveness
- **🌐 Graph Visualization** — Interactive network graphs showing spread paths and node roles
- **📈 Analysis Dashboard** — Multi-chart dashboard with spread curves, agent rankings, and moderation stats
- **🖥️ Gradio Web Interface** — 5-tab interactive UI for simulation, visualization, analytics, and reporting

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        GRADIO WEB UI                             │
│  [Simulation] [Visualizations] [Analytics] [Report] [About]     │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│                   LANGGRAPH PIPELINE                              │
│                                                                   │
│  ┌──────────┐  ┌─────────┐  ┌───────────┐  ┌──────────┐  ┌────┐│
│  │ Misinfo  │→│ Neutral  │→│ Fact-Check │→│Influencer│→│ Mod ││
│  │ Agent    │  │ Agent    │  │  Agent     │  │  Agent   │  │Agent││
│  │(Generate)│  │(Spread)  │  │ (Verify)   │  │(Rewrite) │  │(Act)││
│  └──────────┘  └─────────┘  └───────────┘  └──────────┘  └────┘│
│                                                                   │
│  State: claim → spread_path → verdict → rewritten → decision     │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│              SOCIAL NETWORK GRAPH (NetworkX)                      │
│         100 Nodes · Barabási-Albert Topology                      │
│    Configurable Influencers · Fact-Checkers · Moderators         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component            | Technology                     | Purpose                                    |
|----------------------|--------------------------------|--------------------------------------------|
| Programming Language | Python 3.x                     | Core development language                  |
| LLM Provider         | Groq (LLaMA 3.3 70B Versatile)| Claim generation, verification, rewriting  |
| Agent Orchestration  | LangGraph                      | Multi-agent coordination & state mgmt      |
| Graph Library        | NetworkX                       | Social network modeling & BFS traversal    |
| UI Framework         | Gradio                         | Interactive web-based interface             |
| Visualization        | Matplotlib                     | Graph rendering & analytics charts          |
| LLM Integration      | LangChain + LangChain-Groq     | LLM abstraction and message handling        |

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- A free Groq API key ([Get one here](https://console.groq.com))

### Step 1: Clone / Download the Project

```bash
cd "path/to/project"
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install via setup.py:

```bash
pip install -e .
```

### Step 3: Configure API Key

**Option A** — Set as environment variable:
```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_your_api_key_here"

# Linux/Mac
export GROQ_API_KEY="gsk_your_api_key_here"
```

**Option B** — Edit `config.py` directly:
```python
GROQ_API_KEY = "gsk_your_api_key_here"
```

**Option C** — Enter it in the Gradio UI when prompted.

---

## ▶️ Usage

### Launch the Web Interface

```bash
python main.py
```

Then open **http://localhost:7860** in your browser.

### Using the Interface

1. **Enter your Groq API Key** in the configuration panel
2. **Adjust network parameters** (nodes: 20–300, edges: 1–6) or keep defaults (100 nodes)
3. Click **"👁️ Preview Network"** to see the social graph before simulation
4. Click **"▶️ Generate & Run Simulation"** to execute the full 6-step pipeline
5. Explore results across tabs:
   - **🚀 Run Simulation** — Agent outputs (claim, spread, verification, influencer rewrite, moderation)
   - **📊 Visualizations** — Network graph with spread path + 4-chart analysis dashboard
   - **📈 Detailed Analytics** — Full numerical report on all 4 analysis dimensions
   - **📄 Full Report** — Comprehensive simulation summary with agent rankings

---

## 📁 Project Structure

```
├── agents/                          # AI Agent modules
│   ├── __init__.py                  # Package exports
│   ├── misinformation_agent.py      # Generates fake claims via Groq LLM
│   ├── neutral_agent.py             # BFS spread simulation through network
│   ├── fact_checker_agent.py        # Verifies claims with LLM reasoning
│   ├── influencer_agent.py          # Rewrites content for virality/warnings
│   └── moderator_agent.py           # BLOCK/FLAG/ALLOW decisions
├── analytics.py                     # 4-dimension analysis engine
├── app.py                           # Gradio UI (5 tabs)
├── config.py                        # Configuration, prompts, parameters
├── main.py                          # Application entry point
├── pipeline.py                      # LangGraph orchestration pipeline
├── social_network.py                # NetworkX graph (Barabási-Albert)
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup file
└── README.md                        # This file
```

---

## 🤖 Agents

### 1. Misinformation Agent 🔴
- **Role**: Generates realistic fake news claims
- **Technology**: Groq LLM with creative prompting (temperature: 0.9)
- **Output**: Short, plausible news-like claim text

### 2. Neutral Agent 🔵
- **Role**: Simulates viral content spreading through the social network
- **Technology**: Breadth-First Search (BFS) traversal on NetworkX graph
- **Behavior**: Role-adjusted spread probability (influencers share more, fact-checkers share less)
- **Output**: Spread path, nodes reached, penetration rate, viral coefficient

### 3. Fact-Checker Agent 🟢
- **Role**: Verifies claims and provides structured verdicts
- **Technology**: Groq LLM with low temperature (0.2) for factual analysis
- **Output**: Verdict (Real/Fake/Unverified), confidence score, evidence, red flags

### 4. Influencer Agent 🟠
- **Role**: Rewrites content based on verification results
- **Technology**: Advanced prompt engineering with Groq LLM
- **Behavior**: Counter-messaging for fake claims, amplification for real claims
- **Output**: Rewritten content, amplification score (1–10)

### 5. Moderator Agent 🟣
- **Role**: Makes content moderation decisions
- **Technology**: LLM-based policy reasoning with rule-based fallback
- **Decisions**: BLOCK (fake+high confidence), FLAG (unverified), ALLOW (real)
- **Output**: Decision, severity, containment rate, spread impact

---

## 🔄 Pipeline Flow

The LangGraph pipeline orchestrates all agents sequentially:

```
1. Claim Generation     → Misinformation Agent creates a claim
2. Network Propagation  → Neutral Agent spreads via BFS (100-node graph)
3. Fact Verification    → Fact-Checker Agent analyzes truthfulness
4. Content Amplification→ Influencer Agent rewrites based on verdict
5. Moderation Decision  → Moderator Agent decides BLOCK/FLAG/ALLOW
6. Analytics Generation → System computes metrics & visualizations
```

Each step passes its output to the next via a shared `SimulationState`.

---

## 📊 Analytics

The system provides **4 key analysis dimensions**:

### 1. 📈 Spread Velocity
- Nodes reached per BFS level
- Network penetration rate (%)
- Viral coefficient
- Peak spread step identification

### 2. 🔍 Verification Impact
- Pre vs. post-verification spread comparison
- Containment type (Complete/Partial/None)
- Spread reduction rate
- Network immunization percentage

### 3. 🤖 Agent Influence Ranking
- Per-agent influence score (0–10 scale)
- Most influential agent identification
- Network centrality analysis (betweenness centrality)
- Amplification score from influencer rewriting

### 4. 🛡️ Moderation Effectiveness
- Decision accuracy (True Positive/Negative, False Positive/Negative)
- Containment rate
- Nodes protected count
- Severity assessment

---

## 📸 Screenshots

After running `python main.py`, the following views are available at `http://localhost:7860`:

- **Simulation Tab**: Configure network, run pipeline, see all 5 agent outputs
- **Visualizations Tab**: Interactive network graph + 4-chart analysis dashboard
- **Analytics Tab**: Full numerical report with agent rankings
- **Report Tab**: Complete copy-paste-ready simulation report

---

## ⚙️ Configuration

Key parameters in `config.py`:

| Parameter              | Default | Description                              |
|------------------------|---------|------------------------------------------|
| `NETWORK_NUM_NODES`    | 100     | Number of nodes in the social network    |
| `NETWORK_EDGES_PER_NODE`| 3     | Edges per new node (BA model)            |
| `NUM_INFLUENCERS`      | 5       | Number of influencer nodes               |
| `NUM_FACT_CHECKERS`    | 5       | Number of fact-checker nodes             |
| `NUM_MODERATORS`       | 3       | Number of moderator nodes                |
| `MAX_SPREAD_DEPTH`     | 5       | Maximum BFS depth for spread             |
| `SPREAD_PROBABILITY`   | 0.6     | Probability a node reshares content      |
| `GROQ_MODEL`           | llama-3.3-70b-versatile | Groq model name       |
| `TEMPERATURE`          | 0.7     | LLM temperature for generation           |

---

## 📄 License

This project is developed as an academic final year project for MCA.  
For educational and research purposes only.

---

## 🙏 Acknowledgments

- **Groq** — For providing fast LLM inference API
- **LangGraph** — For multi-agent orchestration framework
- **NetworkX** — For graph modeling and algorithms
- **Gradio** — For rapid UI development
