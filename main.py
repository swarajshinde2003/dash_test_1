import streamlit as st
from model_interface.a_1_banana_harvest_forecasting import banana_harvest_forecasting
# -------------------------------
# Sidebar Project Selection
# -------------------------------
st.set_page_config(page_title="🌾 Agri ML Hub", layout="wide")
st.sidebar.title("🔘 Select DS Model")
project = st.sidebar.selectbox("Choose a Model:", [
    "2_Banana_Harvest_Forecasting"
])

if project == "2_Banana_Harvest_Forecasting":
    banana_harvest_forecasting()