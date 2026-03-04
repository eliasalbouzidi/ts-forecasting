import pandas as pd
import os

# Read the full dataset
df = pd.read_csv('datasets/sst/sst_patch_00.csv')

print(f"Full dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"First few rows:")
print(df.head())

# Check if there's a date column or index
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    first_date = df['date'].min()
    two_years_later = first_date + pd.DateOffset(years=2)
    df_small = df[df['date'] < two_years_later].copy()
    print(f"\nFiltered by date: {first_date} to {two_years_later}")
elif 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    first_date = df['timestamp'].min()
    two_years_later = first_date + pd.DateOffset(years=2)
    df_small = df[df['timestamp'] < two_years_later].copy()
    print(f"\nFiltered by timestamp: {first_date} to {two_years_later}")
else:
    # Just take first 2 years worth of rows (estimate ~730 days if daily)
    # Or take first 30% of rows as a safe approach
    n_rows = len(df)
    df_small = df.iloc[:n_rows // 5].copy()  # 1/5 of data as small sample
    print(f"\nTook first {len(df_small)} rows out of {n_rows}")

# Save the small dataset
os.makedirs('datasets/sst', exist_ok=True)
df_small.to_csv('datasets/sst/sst_small.csv', index=False)
print(f"\nSmall dataset shape: {df_small.shape}")
print(f"Saved to: datasets/sst/sst_small.csv")
