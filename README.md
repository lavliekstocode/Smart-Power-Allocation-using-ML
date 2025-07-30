Smart Power Allocation System 

While the world's electricity demand keeps growing, conventional power distribution networks fail to handle peak loads, wastage of energy, and inefficiencies. A smart power allocation system uses digital technologies to allocate power dynamically depending on real-time consumption patterns, grid conditions, and renewable energy availability. Such a system maximizes power utilization while minimizing operating costs and environmental footprint.
Objectives:
To analyse real-time energy consumption data and identify patterns.
To predict future energy demand using machine learning algorithms.
To detect anomalies in power usage that may indicate wastage or faults.
To recommend strategies for energy optimisation and cost reduction.
To integrate renewable energy sources into existing power systems efficiently.

Methodology

1. Data Import & Preprocessing:
Import necessary libraries and read the dataset from a CSV file.
Create new features: People_Density, Normalized_Hours, Power_per_Person, and Interaction_Term.
2. Feature Selection:
Select features (X) and target (y) variables.
Print the correlation matrix.
3. Train-Test Split:
Split the data into training and testing sets with an 80-20 split.
4. Data Scaling:
Standardize the features using StandardScaler.
5. Model Training:
Train RandomForestRegressor and GradientBoostingRegressor on the scaled training data.
6. Model Evaluation:
Predict on the test set and calculate MAE and R² for both models.
7. Hyperparameter Tuning:
Tune hyperparameters using GridSearchCV for both models and train optimized models.
8. Predicted Power Calculation:
Predict total power for a given input set using the best model.
Allow user input for total available power.
9. Power Allocation:
Collect number of floors and people per floor from user input.
Allocate power for each floor using the roulette_wheel_power_allocation method.
10. Model Saving:
Save the trained model and scaler using joblib
