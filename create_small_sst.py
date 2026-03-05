import pandas as pd
import os

# Read the full dataset
df = pd.read_csv('datasets/sst/sst_patch_00_old.csv')

print(f"Full dataset shape: {df.shape}")

# Take 10 years (~3650 days) and only first 20 variables
# This reduces variables from 3601 → 20 while keeping more temporal data
n_rows = min(3650, len(df))  # ~10 years of daily data
df_small = df.iloc[:n_rows, :20].copy()  # First 20 columns only

# Save the small dataset
os.makedirs('datasets/sst', exist_ok=True)
df_small.to_csv('datasets/sst/sst_patch_00.csv', index=False)
print(f"Small dataset shape: {df_small.shape}")
print(f"Variables: {df_small.shape[1]}")
print(f"Temporal points: {df_small.shape[0]}")
print(f"Saved to: datasets/sst/sst_patch_00.csv")

