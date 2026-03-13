import json
import os
from datetime import date

import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import SerperDevTool


st.set_page_config(page_title="Agentic Travel Agent", layout="wide")
st.title("✈️ Agentic Travel Agent A2A")
st.markdown("Powered by CrewAI + Groq Llama 3.3")


# Sidebar inputs
with st.sidebar:
    st.header("Trip Details")
    destination = st.text_input("Destination", "Paris, France")
    start_date = st.date_input("Start Date", date(2026, 4, 1))
    end_date = st.date_input("End Date", date(2026, 4, 7))
    interests = st.text_area("Interests", "sightseeing, food, culture", height=100)
    budget = st.slider("Budget ($)", 500, 5000, 1500)
    travelers = st.number_input("Travelers", 1, 10, 2)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []


# Chat display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


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


groq_api = _get_secret("groq")
serper_api = _get_secret("serper")

if not groq_api:
    st.warning(
        "Groq API key is missing. Add `groq.api_key` to `.streamlit/secrets.toml` "
        "or set `GROQ_API_KEY` in your environment to run itinerary generation."
    )


# Tools
tools = []
if serper_api:
    tools.append(SerperDevTool(api_key=serper_api))


# LLM
llm = None
if groq_api:
    llm = LLM(
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api,
    )


# Chat input
if prompt := st.chat_input("Ask about your trip (e.g., 'Plan my Paris adventure')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not llm:
            msg = (
                "I can't plan your trip yet because the Groq API key is not configured. "
                "Set `groq.api_key` in `.streamlit/secrets.toml` or `GROQ_API_KEY` and try again."
            )
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        else:
            with st.spinner("Agents planning your trip..."):
                researcher = Agent(
                    role="Travel Researcher",
                    goal="Research destinations, attractions, weather, and deals for {destination}",
                    backstory="Expert in global travel trends and hidden gems.",
                    llm=llm,
                    tools=tools,
                    verbose=True,
                    allow_default_llm=False,
                    memory=False,
                )

                planner = Agent(
                    role="Itinerary Planner",
                    goal="Create detailed daily plans based on research, interests, and budget.",
                    backstory="Master organizer for seamless trips.",
                    llm=llm,
                    verbose=True,
                    allow_default_llm=False,
                    memory=False,
                )

                booker = Agent(
                    role="Booking Advisor",
                    goal="Suggest flights/hotels with prices, links, and alternatives.",
                    backstory="Saves money on bookings.",
                    llm=llm,
                    verbose=True,
                    allow_default_llm=False,
                    memory=False,
                )

                research_task = Task(
                    description=(
                        "Research {destination} from {start_date} to {end_date}: attractions, "
                        "weather, costs for {travelers}. Focus on {interests}."
                    ),
                    expected_output=(
                        "Comprehensive research summary: top attractions, weather forecast, "
                        "average costs, tailored to interests and group size."
                    ),
                    agent=researcher,
                )

                plan_task = Task(
                    description="Plan itinerary using research, fit {budget} for {travelers}. Include meals/transport.",
                    expected_output=(
                        "Detailed 5-7 day itinerary with daily schedule, activities, meals, "
                        "transport, total estimated cost breakdown."
                    ),
                    agent=planner,
                    context=[research_task],
                )

                book_task = Task(
                    description="Recommend flights/hotels for {destination}, dates {start_date}-{end_date}, under {budget}.",
                    expected_output=(
                        "Top 3 flight/hotel options with prices, links/providers, booking "
                        "tips, and alternatives."
                    ),
                    agent=booker,
                    context=[research_task, plan_task],
                )

                crew = Crew(
                    agents=[researcher, planner, booker],
                    tasks=[research_task, plan_task, book_task],
                    process=Process.sequential,
                    verbose=2,
                    memory=False,
                )

                result = crew.kickoff(
                    inputs={
                        "destination": destination,
                        "start_date": str(start_date),
                        "end_date": str(end_date),
                        "interests": interests,
                        "budget": budget,
                        "travelers": travelers,
                    }
                )
                result_text = str(result)
                st.markdown(result_text)

            st.session_state.messages.append({"role": "assistant", "content": result_text})


# Download
if st.session_state.messages:
    st.download_button(
        "📥 Download Itinerary",
        data=json.dumps(st.session_state.messages, indent=2),
        file_name="travel_plan.json",
        mime="application/json",
    )
