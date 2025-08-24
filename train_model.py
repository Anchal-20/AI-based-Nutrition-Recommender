import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load synthetic data generated earlier
df = pd.read_csv('synthetic_nutrition_data.csv')

# Select features and target variable (calorie needs)
features = ['age', 'gender', 'height_cm', 'weight_kg', 'activity_level_encoded']
X = df[features]
y = df['calorie_needs']

# Split data into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor and train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (Calories): {mae:.2f}")

# Plot Actual vs Predicted calorie needs
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Actual vs Predicted Calorie Needs')
plt.show()
import joblib

# Save the model
joblib.dump(model, 'calorie_predictor_model.pkl')
print("Model saved to calorie_predictor_model.pkl")
