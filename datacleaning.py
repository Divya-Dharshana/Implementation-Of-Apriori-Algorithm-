# ==========================================
# BAKERY DATASET CLEANING FOR ASSOCIATION RULE MINING
# ==========================================

import pandas as pd
import numpy as np

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------
df = pd.read_csv("C:/Users/divya/OneDrive/Desktop/Project/bakery_sales.csv")

# Display basic information
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ------------------------------------------
# 2. Remove duplicate rows
# ------------------------------------------
df = df.drop_duplicates()

# ------------------------------------------
# 3. Remove missing values
# ------------------------------------------
df = df.dropna()

# ------------------------------------------
# 4. Remove unwanted spaces
# ------------------------------------------
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.strip()

# ------------------------------------------
# 5. Convert column names to lowercase
# ------------------------------------------
df.columns = df.columns.str.lower()

print("\nColumns:")
print(df.columns)

# ------------------------------------------
# 6. Convert date column
# ------------------------------------------
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# ------------------------------------------
# 7. Convert time column
# ------------------------------------------
if 'time' in df.columns:
    df['time'] = pd.to_datetime(
        df['time'],
        format='%H:%M:%S',
        errors='coerce'
    ).dt.time

# ------------------------------------------
# 8. Create period_day column
# ------------------------------------------
def get_period(t):

    if pd.isnull(t):
        return np.nan

    hour = t.hour

    if 6 <= hour < 12:
        return 'Morning'

    elif 12 <= hour < 17:
        return 'Afternoon'

    elif 17 <= hour < 21:
        return 'Evening'

    else:
        return 'Night'

if 'time' in df.columns:
    df['period_day'] = df['time'].apply(get_period)

# ------------------------------------------
# 9. Create weekday_weekend column
# ------------------------------------------
if 'date' in df.columns:

    df['day_name'] = df['date'].dt.day_name()

    df['weekday_weekend'] = np.where(
        df['day_name'].isin(['Saturday','Sunday']),
        'Weekend',
        'Weekday'
    )

# ------------------------------------------
# 10. Standardize item names
# ------------------------------------------
if 'item' in df.columns:

    df['item'] = (
        df['item']
        .str.lower()
        .str.strip()
        .str.replace('-', ' ')
        .str.replace('_', ' ')
    )

# ------------------------------------------
# 11. Remove invalid transactions
# ------------------------------------------
if 'transaction' in df.columns:
    df = df[df['transaction'].notnull()]

if 'item' in df.columns:
    df = df[df['item'] != 'none']
    df = df[df['item'] != '']

# ------------------------------------------
# 12. Check missing values
# ------------------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------------------
# 13. Final cleaned dataset
# ------------------------------------------
print("\nFinal Shape:", df.shape)

print("\nSample Data:")
print(df.head())

# ------------------------------------------
# 14. Save cleaned dataset
# ------------------------------------------
df.to_csv(
    'bakery_cleaned.csv',
    index=False
)

print("\nCleaned dataset saved as bakery_cleaned.csv")
