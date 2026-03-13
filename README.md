# Agentic Travel Agent (A2A)

A Streamlit travel planner powered by CrewAI + Groq.

## Run locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure credentials with either option:

- **Option A (recommended):** create `.streamlit/secrets.toml`

```toml
[groq]
api_key = "your_groq_api_key"

[serper]
api_key = "your_serper_api_key" # optional
```

- **Option B:** set environment variables

```bash
export GROQ_API_KEY="your_groq_api_key"
export SERPER_API_KEY="your_serper_api_key" # optional
```

3. Start app:

```bash
streamlit run streamlit_app.py
```

If `SERPER_API_KEY` is missing, the app still runs (without web search).
If `GROQ_API_KEY` is missing, the app loads and explains how to configure it before generating plans.
