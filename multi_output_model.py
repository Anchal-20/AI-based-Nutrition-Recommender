import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load the synthetic data
df = pd.read_csv('synthetic_nutrition_data.csv')

# Define features (input variables)
features = ['age', 'gender', 'height_cm', 'weight_kg', 'activity_level_encoded']
X = df[features]

# Define targets (output variables)
y = df[['calorie_needs', 'protein_g', 'fat_g', 'carb_g']]

# Split into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Use MultiOutputRegressor with RandomForest
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate performance
errors = abs(y_pred - y_test.values)
mae_per_target = errors.mean(axis=0)

print("\n🔍 Mean Absolute Errors:")
print(f"Calories: {mae_per_target[0]:.2f} kcal")
print(f"Protein: {mae_per_target[1]:.2f} g")
print(f"Fat: {mae_per_target[2]:.2f} g")
print(f"Carbs: {mae_per_target[3]:.2f} g")

# Save model
joblib.dump(model, 'nutrition_multioutput_model.pkl')
print("\n✅ Model saved as 'nutrition_multioutput_model.pkl'")
