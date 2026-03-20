"""
Agentic Travel Agent (A2A) — Streamlit App
Powered by CrewAI multi-agent orchestration + Groq LLM inference.

Agents:
  1. Travel Researcher     – gathers destination intel via web search
  2. Itinerary Planner     – builds a day-by-day plan
  3. Booking Advisor        – recommends flights, hotels, transport
  4. Local Culture Expert   – insider tips, etiquette, food, language
  5. Budget Optimizer       – cost breakdown and money-saving hacks
"""

import json
import os
import traceback
from datetime import date

import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import SerperDevTool

# ─── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Travel Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Secrets helper ────────────────────────────────────────────────────────
def _get_secret(name: str, *, nested_key: str = "api_key") -> str:
    """Load secret from Streamlit secrets first, then environment variables."""
    try:
        section = st.secrets.get(name, {})
        if isinstance(section, dict):
            value = section.get(nested_key)
        else:
            value = None
    except Exception:
        value = None

    if value:
        return value
    return os.getenv(f"{name.upper()}_{nested_key.upper()}", "")


# ─── Load API keys ────────────────────────────────────────────────────────
groq_api = _get_secret("groq")
serper_api = _get_secret("serper")

# CrewAI internally checks for OPENAI_API_KEY even when using other providers.
# Setting a dummy value prevents ImportError / KeyError during init.
if "OPENAI_API_KEY" not in os.environ:
    openai_secret = _get_secret("openai")
    os.environ["OPENAI_API_KEY"] = openai_secret if openai_secret else "sk-dummy-bypass"

# Propagate Serper key to env so SerperDevTool picks it up automatically.
if serper_api:
    os.environ["SERPER_API_KEY"] = serper_api

# ─── Header ────────────────────────────────────────────────────────────────
st.title("✈️ Agentic Travel Agent (A2A)")
st.markdown(
    "Multi-agent travel planner powered by **CrewAI** and **Groq Llama 3.3**. "
    "Five specialized AI agents collaborate to research, plan, and optimize your trip."
)

# ─── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🗺️ Trip Details")

    destination = st.text_input("Destination", "Paris, France")
    origin = st.text_input("Traveling from", "New Delhi, India")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", date(2026, 4, 1))
    with col2:
        end_date = st.date_input("End Date", date(2026, 4, 7))

    interests = st.text_area(
        "Interests & Preferences",
        "sightseeing, local food, culture, museums",
        height=80,
    )
    travel_style = st.selectbox(
        "Travel Style",
        ["Budget", "Mid-range", "Luxury"],
        index=1,
    )
    budget = st.slider("Total Budget (USD)", 300, 10_000, 1500, step=100)
    travelers = st.number_input("Number of Travelers", 1, 20, 2)

    st.divider()
    st.caption("Agents: Researcher · Planner · Booking · Culture · Budget")

# Validate dates
if end_date <= start_date:
    st.error("End date must be after start date.")
    st.stop()

trip_days = (end_date - start_date).days

# ─── API key warnings ─────────────────────────────────────────────────────
if not groq_api:
    st.warning(
        "⚠️ **Groq API key not found.** "
        "Add `groq.api_key` to `.streamlit/secrets.toml` or set `GROQ_API_KEY` env var."
    )

if not serper_api:
    st.info(
        "ℹ️ Serper API key not set — web search is disabled. "
        "The agents will still work but cannot fetch live data."
    )

# ─── Tools ─────────────────────────────────────────────────────────────────
tools = []
if serper_api:
    tools.append(SerperDevTool())

# ─── LLM ───────────────────────────────────────────────────────────────────
llm = None
if groq_api:
    llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=groq_api,
        temperature=0.7,
    )

# ─── Session state ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Chat history display ─────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ─── Agent & Crew builder ─────────────────────────────────────────────────
def build_crew(inputs: dict) -> Crew:
    """Create the five-agent travel crew with well-defined tasks."""

    # --- Agent definitions ---------------------------------------------------
    researcher = Agent(
        role="Senior Travel Researcher",
        goal=(
            f"Conduct thorough research on {inputs['destination']} for a "
            f"{inputs['trip_days']}-day trip. Find top attractions, hidden gems, "
            f"weather forecast, visa requirements, local transport options, "
            f"safety advisories, and seasonal events happening between "
            f"{inputs['start_date']} and {inputs['end_date']}."
        ),
        backstory=(
            "You are a veteran travel journalist who has visited 120+ countries. "
            "You combine official tourism data with local insider knowledge. "
            "You always verify facts and provide specific, actionable information."
        ),
        llm=llm,
        tools=tools,
        verbose=True,
        allow_delegation=False,
        memory=False,
    )

    planner = Agent(
        role="Expert Itinerary Planner",
        goal=(
            f"Design a detailed {inputs['trip_days']}-day itinerary for "
            f"{inputs['travelers']} traveler(s) visiting {inputs['destination']}. "
            f"Optimize for {inputs['interests']} within a {inputs['travel_style']} "
            f"travel style and ${inputs['budget']} total budget."
        ),
        backstory=(
            "You are a professional trip designer at a premium travel agency. "
            "You build itineraries that balance must-see sights with relaxation, "
            "account for travel time between locations, and include meal "
            "recommendations for every day. You always present a clear daily "
            "schedule with timings."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=False,
    )

    booker = Agent(
        role="Booking & Transport Advisor",
        goal=(
            f"Recommend the best flights from {inputs['origin']} to "
            f"{inputs['destination']}, hotels/accommodations, and local "
            f"transport options. Provide price estimates, booking platform "
            f"names, and tips to save money. Target {inputs['travel_style']} tier."
        ),
        backstory=(
            "You are a seasoned travel booking specialist who tracks airline "
            "deals, hotel promotions, and transport hacks worldwide. You always "
            "suggest at least 3 options (budget, mid, premium) with approximate "
            "prices and direct booking platform references."
        ),
        llm=llm,
        tools=tools,
        verbose=True,
        allow_delegation=False,
        memory=False,
    )

    culture_expert = Agent(
        role="Local Culture & Experience Expert",
        goal=(
            f"Provide essential cultural guidance for {inputs['destination']}: "
            f"local customs, etiquette, tipping norms, key phrases in the local "
            f"language, must-try foods and where to find them, neighborhoods to "
            f"explore, and safety advice."
        ),
        backstory=(
            "You are a cultural anthropologist and food critic who has lived in "
            "dozens of countries. You help travelers go beyond tourist traps to "
            "experience authentic local life. Your recommendations are always "
            "specific — exact dish names, neighborhood names, and practical tips."
        ),
        llm=llm,
        tools=tools,
        verbose=True,
        allow_delegation=False,
        memory=False,
    )

    budget_optimizer = Agent(
        role="Budget Optimizer & Packing Advisor",
        goal=(
            f"Create a complete cost breakdown for the {inputs['trip_days']}-day "
            f"trip to {inputs['destination']} for {inputs['travelers']} travelers. "
            f"Total budget: ${inputs['budget']}. Also provide a packing checklist "
            f"tailored to the destination's weather and planned activities."
        ),
        backstory=(
            "You are a financial planner specializing in travel budgets. You "
            "break costs into categories (flights, accommodation, food, "
            "activities, transport, emergency fund) and find ways to stretch "
            "every dollar. You also advise on packing smart for the destination."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=False,
    )

    # --- Task definitions ----------------------------------------------------
    research_task = Task(
        description=(
            f"Research {inputs['destination']} comprehensively for a trip from "
            f"{inputs['start_date']} to {inputs['end_date']}:\n"
            f"- Top attractions and hidden gems related to: {inputs['interests']}\n"
            f"- Expected weather and what to wear\n"
            f"- Visa/entry requirements for travelers from {inputs['origin']}\n"
            f"- Local transport options and costs\n"
            f"- Safety tips and areas to avoid\n"
            f"- Seasonal events or festivals during travel dates\n"
            f"- Average costs for meals, activities, and accommodation "
            f"({inputs['travel_style']} level)"
        ),
        expected_output=(
            "A structured research report with sections: Overview, Weather, "
            "Visa & Entry, Top Attractions (with brief descriptions), Hidden "
            "Gems, Local Transport, Safety Tips, Seasonal Events, and Average "
            "Cost Estimates. Include specific names, addresses or neighborhoods "
            "where possible."
        ),
        agent=researcher,
    )

    plan_task = Task(
        description=(
            f"Using the research provided, create a day-by-day itinerary for "
            f"{inputs['trip_days']} days in {inputs['destination']} for "
            f"{inputs['travelers']} traveler(s).\n"
            f"- Travel style: {inputs['travel_style']}\n"
            f"- Total budget: ${inputs['budget']}\n"
            f"- Interests: {inputs['interests']}\n"
            f"- Include morning, afternoon, and evening activities\n"
            f"- Add breakfast, lunch, and dinner recommendations with "
            f"restaurant names or food types\n"
            f"- Include travel time estimates between locations\n"
            f"- Add one flexible/rest slot per day"
        ),
        expected_output=(
            "A detailed day-by-day itinerary formatted as:\n"
            "**Day X: [Theme]**\n"
            "- Morning (time): Activity + location\n"
            "- Lunch: Restaurant/food recommendation\n"
            "- Afternoon (time): Activity + location\n"
            "- Dinner: Restaurant/food recommendation\n"
            "- Evening: Activity or rest\n"
            "- Estimated daily cost: $X\n\n"
            "End with a total estimated cost summary."
        ),
        agent=planner,
        context=[research_task],
    )

    booking_task = Task(
        description=(
            f"Recommend travel bookings for {inputs['travelers']} traveler(s) "
            f"from {inputs['origin']} to {inputs['destination']}:\n"
            f"- Dates: {inputs['start_date']} to {inputs['end_date']}\n"
            f"- Budget tier: {inputs['travel_style']}\n"
            f"- Total budget: ${inputs['budget']}\n\n"
            f"Provide:\n"
            f"1. Top 3 flight options (airline, approximate price, booking site)\n"
            f"2. Top 3 accommodation options (name, type, nightly rate, platform)\n"
            f"3. Airport transfer options\n"
            f"4. Local transport passes or recommendations\n"
            f"5. Money-saving booking tips (best time to book, coupon sites, etc.)"
        ),
        expected_output=(
            "A structured booking guide with:\n"
            "## Flights\n| Option | Airline | Price | Book On |\n"
            "## Accommodation\n| Option | Name | Price/Night | Book On |\n"
            "## Transport\nAirport transfers and local transport recommendations.\n"
            "## Booking Tips\nActionable money-saving advice."
        ),
        agent=booker,
        context=[research_task, plan_task],
    )

    culture_task = Task(
        description=(
            f"Provide a local culture guide for {inputs['destination']}:\n"
            f"- Essential local customs and etiquette (do's and don'ts)\n"
            f"- Tipping norms\n"
            f"- 10 useful phrases in the local language with pronunciation\n"
            f"- Top 10 must-try local dishes with where to find them\n"
            f"- Best neighborhoods for: food, nightlife, shopping, culture\n"
            f"- Common tourist scams and how to avoid them\n"
            f"- Emergency contacts (police, ambulance, embassy)"
        ),
        expected_output=(
            "A traveler-friendly culture guide with sections:\n"
            "## Customs & Etiquette\n"
            "## Tipping Guide\n"
            "## Useful Phrases (table with local language + pronunciation)\n"
            "## Must-Try Food (dish name, description, where to find it)\n"
            "## Best Neighborhoods\n"
            "## Scam Awareness\n"
            "## Emergency Contacts"
        ),
        agent=culture_expert,
        context=[research_task],
    )

    budget_task = Task(
        description=(
            f"Create a complete budget breakdown and packing list for a "
            f"{inputs['trip_days']}-day trip to {inputs['destination']} for "
            f"{inputs['travelers']} traveler(s).\n"
            f"- Total budget: ${inputs['budget']}\n"
            f"- Travel style: {inputs['travel_style']}\n\n"
            f"Budget breakdown should cover: flights, accommodation, food, "
            f"activities/attractions, local transport, travel insurance, "
            f"souvenirs, and emergency fund.\n\n"
            f"Packing list should be tailored to the weather and activities "
            f"planned for {inputs['start_date']} to {inputs['end_date']}."
        ),
        expected_output=(
            "## Budget Breakdown\n"
            "| Category | Estimated Cost | % of Budget | Tips |\n"
            "(Table with all categories, totals, and whether budget is feasible)\n\n"
            "## Money-Saving Tips\nBulleted list of specific tips.\n\n"
            "## Packing Checklist\n"
            "Categorized list: Clothing, Toiletries, Electronics, Documents, "
            "Miscellaneous — tailored to destination weather and activities."
        ),
        agent=budget_optimizer,
        context=[research_task, plan_task, booking_task],
    )

    # --- Crew ----------------------------------------------------------------
    crew = Crew(
        agents=[researcher, planner, booker, culture_expert, budget_optimizer],
        tasks=[research_task, plan_task, booking_task, culture_task, budget_task],
        process=Process.sequential,
        verbose=True,
        memory=False,
    )
    return crew


# ─── Chat input & execution ───────────────────────────────────────────────
if prompt := st.chat_input("Ask about your trip (e.g., 'Plan my Paris adventure!')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not llm:
            msg = (
                "❌ Cannot plan your trip — the **Groq API key** is not configured.\n\n"
                "Add `groq.api_key` to `.streamlit/secrets.toml` or set the "
                "`GROQ_API_KEY` environment variable and refresh the page."
            )
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        else:
            with st.spinner("🤖 Five AI agents are collaborating on your travel plan — this may take a minute..."):
                try:
                    inputs = {
                        "destination": destination,
                        "origin": origin,
                        "start_date": str(start_date),
                        "end_date": str(end_date),
                        "trip_days": trip_days,
                        "interests": interests,
                        "travel_style": travel_style,
                        "budget": budget,
                        "travelers": travelers,
                    }
                    crew = build_crew(inputs)
                    result = crew.kickoff(inputs=inputs)
                    result_text = str(result)

                except Exception as exc:
                    error_detail = traceback.format_exc()
                    result_text = (
                        f"⚠️ **An error occurred while planning your trip:**\n\n"
                        f"`{type(exc).__name__}: {exc}`\n\n"
                        f"<details><summary>Full traceback</summary>\n\n"
                        f"```\n{error_detail}\n```\n</details>\n\n"
                        f"**Common fixes:**\n"
                        f"- Check that your Groq API key is valid and has quota\n"
                        f"- Try again in a moment (Groq may be rate-limiting)\n"
                        f"- Ensure Serper API key is set for web search features"
                    )

            st.markdown(result_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": result_text}
            )

# ─── Download & clear ──────────────────────────────────────────────────────
if st.session_state.messages:
    col_dl, col_clear = st.columns([1, 1])
    with col_dl:
        st.download_button(
            "📥 Download Travel Plan",
            data=json.dumps(st.session_state.messages, indent=2),
            file_name="travel_plan.json",
            mime="application/json",
        )
    with col_clear:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
