#!/usr/bin/env python3
"""
Generate realistic sample dataset for timeseries forecasting
Creates synthetic premium spread data with multiple regressors
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

def generate_sample_data(n_days=30, freq='H', seed=42):
    """
    Generate synthetic timeseries data for premium spreads

    Parameters:
    - n_days: Number of days to generate
    - freq: Frequency ('H' for hourly, 'D' for daily)
    - seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    # Create timestamp range
    start_date = datetime(2024, 1, 1)
    if freq == 'H':
        periods = n_days * 24
        date_range = pd.date_range(start=start_date, periods=periods, freq='h', tz='UTC')
    elif freq == 'D':
        periods = n_days
        date_range = pd.date_range(start=start_date, periods=periods, freq='D', tz='UTC')
    else:
        raise ValueError("Frequency must be 'H' (hourly) or 'D' (daily)")

    n_points = len(date_range)

    # Generate time-based features for realistic patterns
    if freq == 'H':
        # Hourly patterns
        hour = date_range.hour
        day_of_week = date_range.dayofweek
        day_of_year = date_range.dayofyear

        # Create realistic hourly patterns
        hourly_pattern = 0.3 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
        weekly_pattern = 0.2 * np.sin(2 * np.pi * day_of_week / 7)  # Weekly cycle
        seasonal_pattern = 0.15 * np.sin(2 * np.pi * day_of_year / 365.25)  # Seasonal

    else:  # Daily
        day_of_week = date_range.dayofweek
        day_of_year = date_range.dayofyear

        # Create realistic daily patterns
        hourly_pattern = 0
        weekly_pattern = 0.2 * np.sin(2 * np.pi * day_of_week / 7)
        seasonal_pattern = 0.15 * np.sin(2 * np.pi * day_of_year / 365.25)

    # Generate base regressors with realistic relationships

    # Economic indicators (slower moving)
    economic_trend = np.cumsum(np.random.normal(0, 0.01, n_points))

    # Market volatility index
    volatility = np.abs(np.random.normal(15, 5, n_points))
    volatility = np.maximum(volatility, 5)  # Floor at 5

    # Interest rate differential (mean-reverting)
    rate_diff = np.zeros(n_points)
    rate_diff[0] = 2.5
    for i in range(1, n_points):
        rate_diff[i] = rate_diff[i-1] + np.random.normal(-0.01 * (rate_diff[i-1] - 2.5), 0.05)

    # Trading volume (with daily patterns)
    if freq == 'H':
        volume_pattern = 1000 + 500 * (1 + np.sin(2 * np.pi * hour / 24))  # Higher during trading hours
    else:
        volume_pattern = 1500 + 200 * np.sin(2 * np.pi * day_of_week / 7)

    volume = volume_pattern + np.random.normal(0, 100, n_points)
    volume = np.maximum(volume, 100)  # Floor at 100

    # Currency strength indices
    currency_a_strength = 100 + 10 * np.sin(2 * np.pi * np.arange(n_points) / (24*7)) + np.cumsum(np.random.normal(0, 0.1, n_points))
    currency_b_strength = 100 + 10 * np.cos(2 * np.pi * np.arange(n_points) / (24*5)) + np.cumsum(np.random.normal(0, 0.1, n_points))

    # Generate some dummy variables (categorical effects)
    # Market session (0=Off-hours, 1=Asian, 2=European, 3=US)
    if freq == 'H':
        market_session = np.zeros(n_points)
        for i, h in enumerate(hour):
            if 1 <= h <= 9:
                market_session[i] = 1  # Asian
            elif 10 <= h <= 16:
                market_session[i] = 2  # European
            elif 17 <= h <= 23:
                market_session[i] = 3  # US
            else:
                market_session[i] = 0  # Off-hours
    else:
        market_session = np.random.choice([0, 1, 2, 3], n_points, p=[0.1, 0.3, 0.3, 0.3])

    # Create dummy variables for market session
    session_asian = (market_session == 1).astype(int)
    session_european = (market_session == 2).astype(int)
    session_us = (market_session == 3).astype(int)

    # High volatility periods (dummy)
    high_vol_threshold = np.percentile(volatility, 75)
    high_volatility = (volatility > high_vol_threshold).astype(int)

    # Weekend effect (dummy)
    is_weekend = (day_of_week >= 5).astype(int)

    # Generate the target variable (premium spread)
    base_premium = 1.0  # Base spread

    # Premium is influenced by:
    premium = (
        base_premium
        + 0.8 * hourly_pattern  # Time of day effect
        + 0.6 * weekly_pattern  # Day of week effect
        + 0.4 * seasonal_pattern  # Seasonal effect
        + 0.3 * (volatility - 15) / 10  # Volatility effect
        + 0.2 * rate_diff / 5  # Interest rate effect
        + 0.15 * economic_trend  # Economic trend
        + 0.1 * (currency_a_strength - currency_b_strength) / 20  # Currency strength differential
        + 0.05 * (volume - 1500) / 500  # Volume effect
        + 0.1 * session_asian  # Asian session premium
        + 0.05 * session_european  # European session premium
        + 0.15 * session_us  # US session premium
        + 0.08 * high_volatility  # High volatility premium
        + 0.12 * is_weekend  # Weekend premium
        + np.random.normal(0, 0.1, n_points)  # Random noise
    )

    # Ensure premium is positive
    premium = np.maximum(premium, 0.01)

    # Add some autocorrelation to make it more realistic
    premium_ar = premium.copy()
    for i in range(1, len(premium_ar)):
        premium_ar[i] = 0.7 * premium_ar[i-1] + 0.3 * premium[i] + np.random.normal(0, 0.05)
    premium = premium_ar

    # Create the dataframe
    df = pd.DataFrame({
        'valueDateTimeOffset': date_range,
        'premium': np.round(premium, 4),
        'volatility_index': np.round(volatility, 2),
        'rate_differential': np.round(rate_diff, 4),
        'trading_volume': np.round(volume, 0).astype(int),
        'currency_a_strength': np.round(currency_a_strength, 2),
        'currency_b_strength': np.round(currency_b_strength, 2),
        'economic_trend': np.round(economic_trend, 4),
        'session_asian': session_asian,
        'session_european': session_european,
        'session_us': session_us,
        'high_volatility': high_volatility,
        'is_weekend': is_weekend
    })

    return df

def main():
    parser = argparse.ArgumentParser(description='Generate sample timeseries data')
    parser.add_argument('--days', type=int, default=30, help='Number of days to generate')
    parser.add_argument('--freq', type=str, default='H', choices=['H', 'D'],
                       help='Frequency: H (hourly) or D (daily)')
    parser.add_argument('--output', type=str, default='sample_data.csv',
                       help='Output CSV filename')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print(f"Generating {args.days} days of {args.freq} data...")

    # Generate data
    df = generate_sample_data(n_days=args.days, freq=args.freq, seed=args.seed)

    # Save to CSV
    df.to_csv(args.output, index=False)

    print(f"Sample data saved to {args.output}")
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['valueDateTimeOffset'].min()} to {df['valueDateTimeOffset'].max()}")
    print(f"Premium range: {df['premium'].min():.4f} to {df['premium'].max():.4f}")

    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    # Display data types and basic stats
    print("\nDataset info:")
    print(f"- Total records: {len(df):,}")
    print(f"- Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"- Missing values: {df.isnull().sum().sum()}")

    print("\nBasic statistics for premium (target):")
    print(f"- Mean: {df['premium'].mean():.4f}")
    print(f"- Std: {df['premium'].std():.4f}")
    print(f"- Min: {df['premium'].min():.4f}")
    print(f"- Max: {df['premium'].max():.4f}")

if __name__ == '__main__':
    main()