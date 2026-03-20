# ✈️ Agentic Travel Agent (A2A)

A **multi-agent AI travel planner** built with [CrewAI](https://docs.crewai.com/) and [Groq](https://groq.com/) (Llama 3.3 70B), deployed as a [Streamlit](https://streamlit.io/) web app.

Five specialized AI agents collaborate in sequence to deliver a complete, personalized travel plan — from research and itinerary to bookings, cultural tips, and budget optimization.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit&logoColor=white)
![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-green)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-orange)

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│                        CrewAI Crew                             │
│                  (Sequential Processing)                       │
│                                                                │
│  ┌──────────────────┐   ┌──────────────────┐                  │
│  │  1. Travel        │──▶│  2. Itinerary     │                 │
│  │     Researcher    │   │     Planner       │                 │
│  │  (web search)     │   │  (day-by-day)     │                 │
│  └──────────────────┘   └──────────────────┘                  │
│           │                       │                            │
│           ▼                       ▼                            │
│  ┌──────────────────┐   ┌──────────────────┐                  │
│  │  3. Booking       │   │  4. Culture       │                 │
│  │     Advisor       │   │     Expert        │                 │
│  │  (flights/hotels) │   │  (local tips)     │                 │
│  └──────────────────┘   └──────────────────┘                  │
│                    │                                           │
│                    ▼                                           │
│           ┌──────────────────┐                                 │
│           │  5. Budget        │                                │
│           │     Optimizer     │                                │
│           │  (costs/packing)  │                                │
│           └──────────────────┘                                 │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
Complete Travel Plan (Markdown)
```

---

## 🤖 The Five Agents

| # | Agent | Role | Tools |
|---|-------|------|-------|
| 1 | **Senior Travel Researcher** | Researches attractions, weather, visa info, safety, events | Web Search (Serper) |
| 2 | **Expert Itinerary Planner** | Builds day-by-day schedule with meals, timings, activities | — |
| 3 | **Booking & Transport Advisor** | Recommends flights, hotels, transport with prices | Web Search (Serper) |
| 4 | **Local Culture Expert** | Provides customs, phrases, food guide, scam alerts | Web Search (Serper) |
| 5 | **Budget Optimizer & Packing Advisor** | Cost breakdown, savings tips, packing checklist | — |

Each agent passes its output as context to the next, building a progressively richer travel plan.

---

## ✨ Features

- **Multi-agent collaboration** — five agents with distinct expertise work together
- **Live web search** — agents search the web via Serper API for real-time information
- **Customizable inputs** — destination, origin, dates, interests, travel style, budget, group size
- **Three travel tiers** — Budget, Mid-range, or Luxury style
- **Chat interface** — conversational UI with message history
- **Downloadable output** — export the full plan as JSON
- **Robust error handling** — graceful handling of API failures and rate limits
- **Streamlit Cloud ready** — deploy with `secrets.toml` configuration

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/keys) (free tier available)
- (Optional) A [Serper API key](https://serper.dev/) for web search capabilities

### 1. Clone the repository

```bash
git clone https://github.com/Somroy1993/agentic-travel-agent-A2A-.git
cd agentic-travel-agent-A2A-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

**Option A — Streamlit secrets (recommended for deployment):**

Create `.streamlit/secrets.toml`:

```toml
[openai]
api_key = "sk-dummy-key"  # Required by CrewAI internals — dummy value is fine

[groq]
api_key = "your_groq_api_key_here"

[serper]
api_key = "your_serper_api_key_here"  # Optional — enables web search
```

**Option B — Environment variables:**

```bash
export OPENAI_API_KEY="sk-dummy-key"
export GROQ_API_KEY="your_groq_api_key_here"
export SERPER_API_KEY="your_serper_api_key_here"  # Optional
```

### 4. Run the app

```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub repo and select `streamlit_app.py`
4. Add your secrets in **App Settings → Secrets**:

```toml
[openai]
api_key = "sk-dummy-key"

[groq]
api_key = "your_groq_api_key_here"

[serper]
api_key = "your_serper_api_key_here"
```

5. Deploy — the app will be live in seconds.

---

## 🔧 Configuration

### Sidebar Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| Destination | Where you want to go | Paris, France |
| Traveling from | Your departure city | New Delhi, India |
| Start Date | Trip start date | 2026-04-01 |
| End Date | Trip end date | 2026-04-07 |
| Interests | Activities you enjoy | sightseeing, local food, culture, museums |
| Travel Style | Budget / Mid-range / Luxury | Mid-range |
| Budget (USD) | Total trip budget | $1,500 |
| Travelers | Number of people | 2 |

### API Keys

| Key | Required | Purpose |
|-----|----------|---------|
| `groq.api_key` | ✅ Yes | Powers the Llama 3.3 LLM via Groq inference |
| `openai.api_key` | ⚠️ Dummy OK | CrewAI internal dependency — `sk-dummy-key` works |
| `serper.api_key` | ❌ Optional | Enables live web search for agents |

---

## 📁 Project Structure

```
agentic-travel-agent-A2A-/
├── streamlit_app.py      # Main application — agents, tasks, UI
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
└── .devcontainer/
    └── devcontainer.json # GitHub Codespaces config
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | [CrewAI](https://docs.crewai.com/) — multi-agent orchestration |
| **LLM** | [Groq](https://groq.com/) — Llama 3.3 70B Versatile (fast inference) |
| **Web Search** | [Serper](https://serper.dev/) — Google search API for agents |
| **Frontend** | [Streamlit](https://streamlit.io/) — Python web app framework |
| **Deployment** | Streamlit Cloud |

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `OPENAI_API_KEY` error | Set `openai.api_key = "sk-dummy-key"` in secrets — CrewAI requires it internally even when using Groq |
| Rate limit errors from Groq | Wait a moment and retry — Groq free tier has rate limits |
| `SerperDevTool` validation error | Ensure `SERPER_API_KEY` is set as an environment variable (the app does this automatically from secrets) |
| App loads but agents fail | Check that your Groq API key is valid at [console.groq.com](https://console.groq.com/) |
| Slow response | Normal — five agents run sequentially; each makes LLM calls. Expect 30-90 seconds. |

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [CrewAI](https://www.crewai.com/) for the multi-agent framework
- [Groq](https://groq.com/) for blazing-fast LLM inference
- [Serper](https://serper.dev/) for the search API
- [Streamlit](https://streamlit.io/) for the web app framework
