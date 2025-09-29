#!/usr/bin/env python3
"""
Generate realistic sample dataset for timeseries forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Generate 60 days of hourly data
n_days = 60
n_hours = n_days * 24

# Create timestamp range (UTC)
start_date = datetime(2024, 1, 1)
dates = pd.date_range(start=start_date, periods=n_hours, freq='h', tz='UTC')

# Time-based patterns
hour = dates.hour.values
day_of_week = dates.dayofweek.values
day_of_year = dates.dayofyear.values

# Generate regressors with realistic patterns

# 1. Volatility Index (15-35 range, mean-reverting)
volatility = np.random.normal(20, 5, n_hours)
volatility = np.clip(volatility, 8, 40)

# 2. Interest Rate Differential (mean-reverting around 2.5%)
rate_diff = np.zeros(n_hours)
rate_diff[0] = 2.5
for i in range(1, n_hours):
    rate_diff[i] = rate_diff[i-1] * 0.99 + 0.01 * 2.5 + np.random.normal(0, 0.02)

# 3. Trading Volume (higher during business hours)
base_volume = 1000
hourly_volume_multiplier = 1 + 0.5 * np.sin(2 * np.pi * hour / 24)
volume = base_volume * hourly_volume_multiplier + np.random.normal(0, 100, n_hours)
volume = np.clip(volume, 100, 3000)

# 4. Currency Strength Indices
currency_a = 100 + 10 * np.sin(2 * np.pi * np.arange(n_hours) / (24*7)) + np.cumsum(np.random.normal(0, 0.05, n_hours))
currency_b = 100 + 8 * np.cos(2 * np.pi * np.arange(n_hours) / (24*5)) + np.cumsum(np.random.normal(0, 0.05, n_hours))

# 5. Economic Trend (slow moving)
economic_trend = np.cumsum(np.random.normal(0, 0.005, n_hours))

# 6. Market Session Dummies
session_asian = ((hour >= 1) & (hour <= 9)).astype(int)
session_european = ((hour >= 10) & (hour <= 16)).astype(int)
session_us = ((hour >= 17) & (hour <= 23)).astype(int)

# 7. High Volatility Periods
high_vol_threshold = np.percentile(volatility, 75)
high_volatility = (volatility > high_vol_threshold).astype(int)

# 8. Weekend Effect
is_weekend = (day_of_week >= 5).astype(int)

# Generate Premium (target variable)
base_premium = 1.2

# Combine effects to create realistic premium
premium = (
    base_premium
    + 0.3 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
    + 0.2 * np.sin(2 * np.pi * day_of_week / 7)  # Weekly cycle
    + 0.1 * np.sin(2 * np.pi * day_of_year / 365.25)  # Seasonal
    + 0.02 * (volatility - 20)  # Volatility effect
    + 0.05 * (rate_diff - 2.5)  # Rate differential effect
    + 0.0001 * (volume - 1000)  # Volume effect
    + 0.01 * (currency_a - currency_b)  # Currency differential
    + 0.02 * economic_trend  # Economic trend
    + 0.05 * session_asian  # Asian session premium
    + 0.03 * session_european  # European session premium
    + 0.08 * session_us  # US session premium
    + 0.04 * high_volatility  # High vol premium
    + 0.06 * is_weekend  # Weekend premium
    + np.random.normal(0, 0.08, n_hours)  # Random noise
)

# Add autoregressive component for realism
premium_ar = np.zeros_like(premium)
premium_ar[0] = premium[0]
for i in range(1, len(premium)):
    premium_ar[i] = 0.8 * premium_ar[i-1] + 0.2 * premium[i]

# Ensure premium is positive
premium_final = np.maximum(premium_ar, 0.01)

# Create DataFrame
df = pd.DataFrame({
    'valueDateTimeOffset': dates,
    'premium': np.round(premium_final, 4),
    'volatility_index': np.round(volatility, 2),
    'rate_differential': np.round(rate_diff, 4),
    'trading_volume': np.round(volume, 0).astype(int),
    'currency_a_strength': np.round(currency_a, 2),
    'currency_b_strength': np.round(currency_b, 2),
    'economic_trend': np.round(economic_trend, 4),
    'session_asian': session_asian,
    'session_european': session_european,
    'session_us': session_us,
    'high_volatility': high_volatility,
    'is_weekend': is_weekend
})

# Save to CSV
df.to_csv('sample_data.csv', index=False)

print("Sample data generated successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['valueDateTimeOffset'].min()} to {df['valueDateTimeOffset'].max()}")
print(f"Premium range: {df['premium'].min():.4f} to {df['premium'].max():.4f}")
print(f"Saved to: sample_data.csv")

print("\nDataset Structure:")
print(df.head())

print(f"\nDataset Summary:")
print(f"- Total records: {len(df):,}")
print(f"- Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
print(f"- Binary dummies: 5 (session_asian, session_european, session_us, high_volatility, is_weekend)")
print(f"- Missing values: {df.isnull().sum().sum()}")
print(f"- Premium mean: {df['premium'].mean():.4f}")
print(f"- Premium std: {df['premium'].std():.4f}")