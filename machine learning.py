import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\itzla\Documents\merged_smart_power_data.csv")

# Feature Engineering
df["People_Density"] = df["People_Count"] / df["Floor_Area_sqft"]
df["Normalized_Hours"] = df["Operating_Hours"] / 24
df["Power_per_Person"] = df["Mean_Power_kW"] / (df["People_Count"] + 1)  # Avoid division by zero
df["Interaction_Term"] = df["People_Count"] * df["Floor_Area_sqft"]

# Define features (X) and target (y)
X = df[['Operating_Hours', 'People_Count', 'Floor_Area_sqft', 'People_Density', 'Normalized_Hours', 'Power_per_Person', 'Interaction_Term']]
y = df['Mean_Power_kW']

# Check feature correlation
print("Feature Correlations:\n", df.corr())

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict and evaluate Random Forest
y_pred_rf = rf_model.predict(X_test_scaled)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest Model Performance: MAE = {mae_rf:.2f}, R² = {r2_rf:.3f}")

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Predict and evaluate Gradient Boosting
y_pred_gb = gb_model.predict(X_test_scaled)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
print(f"Gradient Boosting Model Performance: MAE = {mae_gb:.2f}, R² = {r2_gb:.3f}")

# Hyperparameter tuning for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gb_grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=5, scoring='r2')
gb_grid_search.fit(X_train_scaled, y_train)
print("Best Gradient Boosting Parameters:", gb_grid_search.best_params_)

# Use the best model from GridSearchCV
best_gb_model = gb_grid_search.best_estimator_
y_pred_best_gb = best_gb_model.predict(X_test_scaled)
mae_best_gb = mean_absolute_error(y_test, y_pred_best_gb)
r2_best_gb = r2_score(y_test, y_pred_best_gb)
print(f"Optimized Gradient Boosting Model: MAE = {mae_best_gb:.2f}, R² = {r2_best_gb:.3f}")

# Refined Hyperparameter tuning for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='r2')
grid_search_rf.fit(X_train_scaled, y_train)
print("Best Random Forest Parameters:", grid_search_rf.best_params_)

# Use the best model from GridSearchCV
best_rf_model = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test_scaled)
mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)
print(f"Optimized Random Forest Model: MAE = {mae_best_rf:.2f}, R² = {r2_best_rf:.3f}")

# Predict total power based on historical data
predicted_total_power = best_rf_model.predict(scaler.transform([[12, 300, 5000, 0.06, 0.5, 0.1, 1500000]]))[0]  # Example input

# Get user input for total power (if provided, override prediction)
user_total_power = input("Enter total available power (or press Enter to use ML prediction): ")
total_power_available = float(user_total_power) if user_total_power else predicted_total_power
print(f"Using Total Power: {total_power_available:.2f} kW")

# Get user input for number of floors and people per floor
num_floors = max(int(input("Enter total number of floors (minimum 20): ")), 20)
people_per_floor = [int(input(f"Enter number of people on floor {i+1}: ")) for i in range(num_floors)]

# Create floor-wise input data
floor_data = pd.DataFrame({"Floor": range(1, num_floors+1), "People_Count": people_per_floor})

# Roulette-Wheel Power Allocation
def roulette_wheel_power_allocation(people_counts, total_power):
    probabilities = people_counts / np.sum(people_counts)  # Compute probability weights
    allocated_power = probabilities * total_power  # Allocate power proportionally
    return {i+1: float(power) for i, power in enumerate(allocated_power)}

# Allocate power
allocated_power = roulette_wheel_power_allocation(floor_data['People_Count'].values, total_power=total_power_available)
print("Power Allocation (Roulette-Wheel Selection):", allocated_power)

import joblib
import os

# Define the save directory (modify as needed)
save_dir = r"C:\Users\itzla\Documents\SmartPowerModel"
os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist

# Save model and scaler
joblib.dump(best_rf_model, os.path.join(save_dir, "best_rf_model.pkl"))
joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

print(f"Model and scaler saved in: {save_dir}")
