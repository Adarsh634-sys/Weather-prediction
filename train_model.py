# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv('weather.csv')

# Features (input variables)
X = df[['temp', 'humidity', 'wind_speed']]  # You can add more features if available

# Target (what we want to predict)
y = df['rainfall']  # Make sure your CSV has a 'rainfall' column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'weather_model.pkl')

print("✅ Model trained and saved as 'weather_model.pkl'")
