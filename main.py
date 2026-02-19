import streamlit as st
from model_interface.a_2_banana_harvest_forecasting import banana_harvest_forecasting
from model_interface.a_1_sales_forecasting_vegpro import sales_forecasting
from model_interface.a_3_customer_scoring_vegpro import run_sales_score_app
# -------------------------------
# Sidebar Project Selection
# -------------------------------
st.set_page_config(page_title="🌾 Agri ML Hub", layout="wide")
st.sidebar.title("🔘 Select DS Model")
project = st.sidebar.selectbox("Choose a Model:", [
    "1_Sales_forecasting_vegpro",
    "2_Banana_Harvest_Forecasting",
    "3_Customer_scoring_vegpro"
])

if project == "2_Banana_Harvest_Forecasting":
    banana_harvest_forecasting()
elif project=="1_Sales_forecasting_vegpro":
    sales_forecasting()
elif project=="3_Customer_scoring_vegpro":
    run_sales_score_app()