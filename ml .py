import pandas as pd
import numpy as np

# Number of random entries to generate
num_entries = 100  

# Generate random data
data = {
    "Operating_Hours": np.random.randint(6, 24, num_entries),
    "People_Count": np.random.randint(10, 500, num_entries),
    "Floor_Area_sqft": np.random.randint(500, 10000, num_entries),
    "Mean_Power_kW": np.random.uniform(5, 500, num_entries)
}

# Create DataFrame
df = pd.DataFrame(data)

# Define a specific file path
save_path = r"C:\Users\itzla\Documents\generated_smart_power_data.csv"
df.to_csv(save_path, index=False)

print(f"Generated dataset saved at: {save_path}")
df = pd.read_csv(r"C:\Users\itzla\Documents\generated_smart_power_data.csv")
print(f"Total entries in CSV: {df.shape[0]}")