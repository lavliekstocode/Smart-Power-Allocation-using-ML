import pandas as pd
import numpy as np

# Load existing dataset
existing_df = pd.read_csv(r"C:\Users\itzla\Downloads\smart_power_dataset_updated (1).csv")

# Load newly generated dataset
new_df = pd.read_csv(r"C:\Users\itzla\Documents\generated_smart_power_data.csv")

# Merge datasets
merged_df = pd.concat([existing_df, new_df], ignore_index=True)

# Fill missing values with a normal distribution around the mean ± standard deviation
for col in ["Peak_Power_kW", "Standard_Deviation_kW", "Percentile_25_kW", "Percentile_50_kW", "Percentile_75_kW"]:
    mean_val = existing_df[col].mean()
    std_dev = existing_df[col].std()
    
    # Generate realistic variations for missing values
    random_values = np.random.normal(mean_val, std_dev, merged_df[col].isna().sum())
    
    # Assign random values only to missing rows
    merged_df.loc[merged_df[col].isna(), col] = random_values

# Save the improved merged dataset
merged_df.to_csv(r"C:\Users\itzla\Documents\merged_smart_power_data.csv", index=False)

print(f"Updated merged dataset saved with {merged_df.shape[0]} entries.")
