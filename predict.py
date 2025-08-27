import joblib
import pandas as pd

# Load model
model = joblib.load('weather.csv')

# Example new data for prediction
new_data = pd.DataFrame({
    'temp': [30],
    'humidity': [70],
    'wind_speed': [10]
})

prediction = model.predict(new_data)
print(f"🌦 Predicted Rainfall: {prediction[0]:.2f} mm")
