import streamlit as st
import numpy as np
import pandas as pd
import joblib

@st.cache_resource
def load_files():
    best_model = joblib.load("model_files/2_banana_harvest_forecasting/best_regression_model.joblib")
    scaler = joblib.load("model_files/2_banana_harvest_forecasting/banana_scale.joblib")
    label_enc = joblib.load("model_files/2_banana_harvest_forecasting/banana_label.joblib")
    return best_model, scaler, label_enc

best_model, scaler, label_enc = load_files()

def banana_harvest_forecasting():
    
   
    # Feature names
    cat_col = ['Crop']
    num_col = [
        'Temperature_avg_day', 'Relative_humidity_avg_day', 'Rain_avg_day',
        'Wind_speed_avg_day', 'Wind_direction_avg_day',
        'Shortwave_radiation_avg_day', 'NDVI'
    ]

    # Streamlit UI
    st.set_page_config(page_title="Banana Yield Predictor", layout="centered")
    st.title("🍌 Banana Yield Prediction App")

    with st.form("prediction_form"):
        st.subheader("Enter the following inputs:")
        
        crop = st.selectbox("Crop", ["Banana"])  # Only Banana for now
        temp = st.number_input("Temperature Avg (°C)", value=25.0)
        humidity = st.number_input("Relative Humidity Avg (%)", value=86.0)
        rain = st.number_input("Rain Avg (mm)", value=1.0)
        wind_speed = st.number_input("Wind Speed Avg (km/h)", value=10.0)
        wind_dir = st.number_input("Wind Direction Avg (°)", value=114.0)
        radiation = st.number_input("Shortwave Radiation Avg", value=193.0)
        ndvi = st.number_input("NDVI", value=0.32)

        submitted = st.form_submit_button("Predict Yield")

    if submitted:
        # Prepare input DataFrame
        input_data = pd.DataFrame({
            "Crop": [crop],
            "Temperature_avg_day": [temp],
            "Relative_humidity_avg_day": [humidity],
            "Rain_avg_day": [rain],
            "Wind_speed_avg_day": [wind_speed],
            "Wind_direction_avg_day": [wind_dir],
            "Shortwave_radiation_avg_day": [radiation],
            "NDVI": [ndvi]
        })

        # Preprocess function
        def preprocess_data(df):
            for col in cat_col:
                if col in label_enc:
                    df[col] = label_enc[col].transform(df[col])
            df[num_col] = scaler.transform(df[num_col])
            return df

        # Prediction
        processed_input = preprocess_data(input_data)
        prediction = best_model.predict(processed_input)[0]
        
        st.success(f"✅ Predicted Yield: **{prediction:.2f} Kg/acre**")
