import numpy as np
import pandas as pd

np.random.seed(42)

num_samples = 1000

# Generate features
age = np.random.randint(18, 65, num_samples)
gender = np.random.choice([0, 1], num_samples)  # 0=female, 1=male
height_cm = np.random.randint(150, 200, num_samples)
weight_kg = np.random.randint(50, 100, num_samples)
activity_levels = ['sedentary', 'light', 'moderate', 'active', 'very_active']
activity_level = np.random.choice(activity_levels, num_samples)

activity_map = {
    'sedentary': 1.2,
    'light': 1.375,
    'moderate': 1.55,
    'active': 1.725,
    'very_active': 1.9
}

def calculate_bmr(gender, weight, height, age):
    if gender == 1:
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

bmr = [calculate_bmr(g, w, h, a) for g, w, h, a in zip(gender, weight_kg, height_cm, age)]
calorie_needs = [b * activity_map[al] for b, al in zip(bmr, activity_level)]

protein_g = [0.2 * c / 4 for c in calorie_needs]
fat_g = [0.3 * c / 9 for c in calorie_needs]
carb_g = [0.5 * c / 4 for c in calorie_needs]

df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'height_cm': height_cm,
    'weight_kg': weight_kg,
    'activity_level': activity_level,
    'activity_level_encoded': [activity_map[al] for al in activity_level],
    'calorie_needs': calorie_needs,
    'protein_g': protein_g,
    'fat_g': fat_g,
    'carb_g': carb_g
})

df.to_csv('synthetic_nutrition_data.csv', index=False)
print("Data saved to synthetic_nutrition_data.csv")
