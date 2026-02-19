import streamlit as st
import joblib
import pandas as pd

def run_sales_score_app():

    # -----------------------------
    # Load model + scaler
    # -----------------------------
    best_model = joblib.load("model_files/3_customer_scoring_vegpro/top_1_model_Gradient_Boosting.joblib")
    scaler = joblib.load("model_files/3_customer_scoring_vegpro/scale.joblib")

    # -----------------------------
    # UI
    # -----------------------------
    st.title("Sales Score Prediction App")
    st.write("Enter the required details below to predict the sales score.")

    # Inputs
    sales_order_quantity = st.number_input("Sales Order Quantity", min_value=0, step=1, value=10)
    total_order_count = st.number_input("Total Order Count", min_value=0, step=1, value=1)
    avg_spend_per_order_crop = st.number_input("Avg Spend per Order Crop", min_value=0.0, step=0.01, value=500.0)
    order_frequency_days = st.number_input("Order Frequency (Days)", min_value=1, step=1, value=1)
    total_amount = st.number_input("Total Amount", min_value=0.0, step=0.01, value=1000.0)

    # Predict button
    if st.button("Predict Sales Score"):

        # Create dataframe
        df = pd.DataFrame({
            "SalesOrder_Quantity": [sales_order_quantity],
            "Total_order_count": [total_order_count],
            "Avg_spend_per_order_crop": [avg_spend_per_order_crop],
            "Order_frequency_days": [order_frequency_days],
            "Total_Amount": [total_amount]
        })

        # Scale
        cols = df.columns
        df[cols] = scaler.transform(df[cols])

        # Predict
        prediction = best_model.predict(df)[0].round(2)

        st.success(f"Predicted Sales Score: {prediction}")



