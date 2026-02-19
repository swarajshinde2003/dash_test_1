import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# Load Weekly Model
# ---------------------------
def load_weekly_model():
    model = joblib.load("model_files/1_sales_forecasting_vegpro/weekly/week_best_model.joblib")
    scaler = joblib.load("model_files/1_sales_forecasting_vegpro/weekly/week_scaler.joblib")
    label_enc = joblib.load("model_files/1_sales_forecasting_vegpro/weekly/week_label_encoder.joblib")
    return model, scaler, label_enc

# ---------------------------
# Load Monthly Model
# ---------------------------
def load_monthly_model():
    model = joblib.load("model_files/1_sales_forecasting_vegpro/monthly/month_best_model.joblib")
    scaler = joblib.load("model_files/1_sales_forecasting_vegpro/monthly/month_scaler.joblib")
    label_enc = joblib.load("model_files/1_sales_forecasting_vegpro/monthly/month_label_encoder.joblib")
    return model, scaler, label_enc

# ---------------------------
# Sidebar – LHS Selection
# ---------------------------
def sales_forecasting():
    st.sidebar.title("📊 Prediction Dashboard")
    project = st.sidebar.radio(
        "Select Prediction Type",
        ["Weekly Prediction", "Monthly Prediction"]
    )

    # ---------------------------
    # WEEKLY PREDICTION SECTION
    # ---------------------------
    if project == "Weekly Prediction":

        st.title("🌾 Weekly Crop Order Quantity Prediction")

        def predict_weekly(input_data, model, scaler, label_enc):
            input_data = input_data.copy()

            crop_name = input_data["Crop_Name"].values[0]
            if crop_name not in label_enc.classes_:
                return f"❌ Error: Crop '{crop_name}' was not seen during training!"

            input_data["Crop_Name"] = label_enc.transform([crop_name])[0]

            feature_cols = [
                'Crop_Name', 'Weekly_rate_mean_t-5', 'Weekly_rate_mean_t-4', 'Weekly_rate_mean_t-3',
                'Weekly_rate_mean_t-2', 'Weekly_rate_mean_t-1',
                'Crop_wise_weekly_quantity_t-5', 'Crop_wise_weekly_quantity_t-4', 
                'Crop_wise_weekly_quantity_t-3', 'Crop_wise_weekly_quantity_t-2', 
                'Crop_wise_weekly_quantity_t-1'
            ]

            input_data = input_data[feature_cols]

            num_cols = [col for col in feature_cols if col != "Crop_Name"]
            input_data[num_cols] = scaler.transform(input_data[num_cols])

            predicted = model.predict(input_data)[0]
            return f"✅ Predicted Quantity: {predicted:.2f} KG"

        # Load weekly model
        model, scaler, label_enc = load_weekly_model()

        crop_name = st.selectbox("Crop Name:", label_enc.classes_)
        weekly_rates = [st.number_input(f"Weekly Rate Mean (t-{i})", step=1) for i in range(5,0,-1)]
        weekly_qty = [st.number_input(f"Crop-wise Quantity (t-{i})", step=1) for i in range(5,0,-1)]

        if st.button("Predict Weekly Quantity"):
            df = pd.DataFrame({
                "Crop_Name": [crop_name],
                "Weekly_rate_mean_t-5": [weekly_rates[0]],
                "Weekly_rate_mean_t-4": [weekly_rates[1]],
                "Weekly_rate_mean_t-3": [weekly_rates[2]],
                "Weekly_rate_mean_t-2": [weekly_rates[3]],
                "Weekly_rate_mean_t-1": [weekly_rates[4]],
                "Crop_wise_weekly_quantity_t-5": [weekly_qty[0]],
                "Crop_wise_weekly_quantity_t-4": [weekly_qty[1]],
                "Crop_wise_weekly_quantity_t-3": [weekly_qty[2]],
                "Crop_wise_weekly_quantity_t-2": [weekly_qty[3]],
                "Crop_wise_weekly_quantity_t-1": [weekly_qty[4]],
            })

            result = predict_weekly(df, model, scaler, label_enc)
            st.success(result)

    # ---------------------------
    # MONTHLY PREDICTION SECTION
    # ---------------------------
    elif project == "Monthly Prediction":

        st.title("🌱 Monthly Crop Order Quantity Prediction")

        def preprocess_input(data, label_encoders, scaler, categorical_cols, numerical_cols):
            data = data.copy()

            for col in categorical_cols:
                enc = label_encoders.get(col)
                data[col] = data[col].apply(lambda x: enc.transform([x])[0] if x in enc.classes_ else -1)

            data[numerical_cols] = scaler.transform(data[numerical_cols])
            return data

        categorical_cols = ["Crop_Name"]
        numerical_cols = [
            "Mean_per_kg_rate-3", "Mean_per_kg_rate-2", "Mean_per_kg_rate-1",
            "Mean_order_quantity-3", "Mean_order_quantity-2", "Mean_order_quantity-1"
        ]

        model, scaler, label_encoders = load_monthly_model()
        crop_names = list(label_encoders["Crop_Name"].classes_)

        selected_crop = st.selectbox("Select Crop Name", crop_names)

        rate_3 = st.number_input("Mean per kg rate (3 months ago)", step=1)
        rate_2 = st.number_input("Mean per kg rate (2 months ago)", step=1)
        rate_1 = st.number_input("Mean per kg rate (1 month ago)", step=1)

        qty_3 = st.number_input("Mean order quantity (3 months ago)", step=1)
        qty_2 = st.number_input("Mean order quantity (2 months ago)", step=1)
        qty_1 = st.number_input("Mean order quantity (1 month ago)", step=1)

        if st.button("Predict Monthly Quantity"):

            df = pd.DataFrame({
                "Crop_Name": [selected_crop],
                "Mean_per_kg_rate-3": [rate_3],
                "Mean_per_kg_rate-2": [rate_2],
                "Mean_per_kg_rate-1": [rate_1],
                "Mean_order_quantity-3": [qty_3],
                "Mean_order_quantity-2": [qty_2],
                "Mean_order_quantity-1": [qty_1],
            })

            X_new = preprocess_input(df, label_encoders, scaler, categorical_cols, numerical_cols)
            y_pred = model.predict(X_new)[0]

            st.success(f"Predicted Monthly Quantity: {y_pred:.2f} KG")

    # ---------------------------
    # CUSTOMER SCORE SECTION
    # ---------------------------
