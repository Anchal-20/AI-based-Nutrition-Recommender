import joblib
import numpy as np

# Load saved model
model = joblib.load('calorie_predictor_model.pkl')
print("Model loaded.")

# Example input [age, gender, height_cm, weight_kg, activity_level_encoded]
sample_input = np.array([[25, 1, 175, 70, 1.55]])  # modify this as needed

# Predict
prediction = model.predict(sample_input)
print(f"Predicted Calorie Need: {prediction[0]:.2f} kcal")
