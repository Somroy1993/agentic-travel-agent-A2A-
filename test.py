import streamlit as st
groq_api = st.secrets["groq"]["api_key"]
serper_api = st.secrets["serper"]["api_key"]
print(groq_api, serper_api)