import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
excel_file_url = r"C:\Users\itzla\OneDrive\Documents\EX-All Commercial Buildings-2025-02-11T09_50_08.829Z.xlsx"

# Load the sheets
try:
    operating_hours_df = pd.read_excel(excel_file_url, sheet_name="EX-All Commercial Buildings-202", header=1)
    floor_area_df = pd.read_excel(excel_file_url, sheet_name="Sheet1", header=1)
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit()

# Clean the data
operating_hours_df = operating_hours_df[operating_hours_df['Operating Hours'] != 'Summary'].dropna()
floor_area_df = floor_area_df[floor_area_df['Floor Area'] != 'Summary'].dropna()

# Convert 'Count' columns to numeric
operating_hours_df['Count'] = pd.to_numeric(operating_hours_df['Count'], errors='coerce')
floor_area_df['Count'] = pd.to_numeric(floor_area_df['Count'], errors='coerce')

# Merge data
merged_df = pd.merge(operating_hours_df, floor_area_df, on='Count', how='inner')

# Save cleaned data to CSV
output_csv_path = r"C:\Users\itzla\OneDrive\Documents\cleaned_data.csv"
merged_df.to_csv(output_csv_path, index=False)
print(f"Cleaned data saved to: {output_csv_path}")

# Ensure correct column names for visualization
if 'Operating Hours' in merged_df.columns and 'Power Consumption' in merged_df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Operating Hours', y='Power Consumption', data=merged_df)
    plt.title('Power Consumption vs Operating Hours')
    plt.xlabel('Operating Hours')
    plt.ylabel('Power Consumption')
    plt.show()

    # Correlation matrix
    correlation_matrix = merged_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
else:
    print("Error: Required columns not found in the merged data.")


