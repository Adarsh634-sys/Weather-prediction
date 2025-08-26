import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('weather_model.pkl')

st.title("🌤 Weather Prediction App")

# User inputs
temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=10.0)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'temp': [temp],
        'humidity': [humidity],
        'wind_speed': [wind_speed]
    })
    prediction = model.predict(input_data)[0]
    st.success(f"🌧 Predicted Rainfall: {prediction:.2f} mm")
