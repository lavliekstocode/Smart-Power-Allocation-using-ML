import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Processed Data
df = pd.read_csv("Processed_Power_Allocation.csv")

# Calculate optimized power allocation based on multiple factors
df["weighted_factor"] = (df["normalized_occupant_density"] * 0.5) + \
                        (df["normalized_floor_area"] * 0.3) + \
                        (df["operating hours"] * 0.2)

# Normalize weighted factor
df["weighted_factor"] /= df["weighted_factor"].sum()

# Compute optimized power allocation
total_power = df["power_consumption"].sum()
df["optimized_power"] = df["weighted_factor"] * total_power

# Save Optimized Data
df.to_csv("Optimized_Power_Allocation.csv", index=False)

# --- 📊 Visualization ---
plt.figure(figsize=(12, 6))
sns.kdeplot(df["power_consumption"], label="Current Allocation", fill=True)
sns.kdeplot(df["optimized_power"], label="Optimized Allocation", fill=True)
plt.xlabel("Power Allocation (kW)")
plt.ylabel("Density")
plt.title("Power Distribution: Current vs Optimized")
plt.legend()
plt.show()

# Print summary of changes
print("\n🔹 Power Allocation Summary 🔹")
summary = df[["power_consumption", "allocated_power", "optimized_power"]].describe()
print(summary)
