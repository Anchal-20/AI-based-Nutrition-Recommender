import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load('nutrition_multioutput_model.pkl')

st.title("🍽️ AI-Powered Indian Nutrition Recommender")

st.write("Get your daily calorie & macro needs — plus Indian meal suggestions!")

# Inputs
age = st.number_input("Age (years)", min_value=18, max_value=80, value=25)
gender = st.radio("Gender", ["Female", "Male"])
height = st.number_input("Height (cm)", min_value=140, max_value=210, value=170)
weight = st.number_input("Weight (kg)", min_value=40, max_value=150, value=65)
activity_level = st.selectbox("Activity Level", [
    "Sedentary (little or no exercise)",
    "Light (light exercise 1-3 days/week)",
    "Moderate (moderate exercise 3-5 days/week)",
    "Active (hard exercise 6-7 days/week)",
    "Very Active (very hard exercise & physical job)"
])

# Live BMI
bmi = weight / ((height / 100) ** 2)
st.write(f"Your BMI: **{bmi:.1f}**")

# Encode features
gender_encoded = 0 if gender == "Female" else 1
activity_map = {
    "Sedentary (little or no exercise)": 1.2,
    "Light (light exercise 1-3 days/week)": 1.375,
    "Moderate (moderate exercise 3-5 days/week)": 1.55,
    "Active (hard exercise 6-7 days/week)": 1.725,
    "Very Active (very hard exercise & physical job)": 1.9
}
activity_encoded = activity_map[activity_level]

features = np.array([[age, gender_encoded, height, weight, activity_encoded]])

# Indian meals dataset
meals = pd.read_csv('indian_meals.csv')

# Suggestion function
def suggest_meals(calories, protein, fat, carbs, meals_df):
    calorie_range = (calories * 0.6, calories * 1.4)
    filtered = meals_df[(meals_df['Calories'] >= calorie_range[0]) & (meals_df['Calories'] <= calorie_range[1])].copy()
    if filtered.empty:
        filtered = meals_df.copy()
    filtered['Protein_diff'] = abs(filtered['Protein_g'] - protein / 3)  # Suggesting per-meal macros
    suggested = filtered.sort_values('Protein_diff').head(3)
    return suggested[['Meal', 'Calories', 'Protein_g', 'Fat_g', 'Carb_g']]

# Predict and display
if st.button("Predict & Suggest Meals"):
    prediction = model.predict(features)[0]
    calories, protein, fat, carbs = prediction

    st.subheader("🧪 Predicted Daily Needs:")
    st.write(f"Calories: **{calories:.0f} kcal**")
    st.write(f"Protein: **{protein:.0f} g**")
    st.write(f"Fat: **{fat:.0f} g**")
    st.write(f"Carbohydrates: **{carbs:.0f} g**")

    st.subheader("🍱 Suggested Indian Meals:")
    suggestions = suggest_meals(calories, protein, fat, carbs, meals)
    st.table(suggestions)
#to run-----> streamlit run app.py 