#!/usr/bin/env python3
"""
Data Pipeline for Time Series Forecasting
Builds the dataset required by train_forecast.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import argparse
from pathlib import Path
import requests
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ElexonBMRSFetcher:
    """
    Fetch day-ahead demand forecast data from Elexon BMRS API.

    API Endpoint: https://data.elexon.co.uk/bmrs/api/v1/forecast/demand/day-ahead/history
    """

    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/forecast/demand/day-ahead/history"

    def __init__(self, max_retries=3, backoff_factor=1):
        """
        Initialize the Elexon BMRS fetcher.

        Args:
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Base factor for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)

    def fetch_day_ahead_demand(self, publish_date):
        """
        Fetch day-ahead demand forecast for a specific publish date.

        Args:
            publish_date: datetime.date object for the publish date

        Returns:
            pd.DataFrame with columns: startTime, settlementPeriod,
                                      transmissionSystemDemand, nationalDemand
        """
        # Build publishTime parameter: publish_date at 16:30:00 UTC
        publish_time = f"{publish_date.strftime('%Y-%m-%d')}T16:30:00Z"

        # Target settlement date is day after publish date
        target_settlement_date = publish_date + timedelta(days=1)
        target_settlement_str = target_settlement_date.strftime('%Y-%m-%d')

        self.logger.info(f"Fetching data for publish date {publish_date} (settlement date: {target_settlement_str})")

        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                # Build request parameters
                params = {
                    'publishTime': publish_time,
                    'format': 'json'
                }

                # Make GET request
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()

                # Parse JSON response
                try:
                    data = response.json()
                except ValueError as e:
                    self.logger.warning(f"JSON parsing failed for {publish_date}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.backoff_factor * (2 ** attempt))
                        continue
                    return pd.DataFrame()

                # Extract data array
                if 'data' not in data or not isinstance(data['data'], list):
                    self.logger.warning(f"No data array found in response for {publish_date}")
                    return pd.DataFrame()

                records = data['data']

                if not records:
                    self.logger.warning(f"Empty data array for {publish_date}")
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(records)

                # Filter to target settlement date only
                if 'settlementDate' not in df.columns:
                    self.logger.warning(f"No settlementDate column for {publish_date}")
                    return pd.DataFrame()

                df_filtered = df[df['settlementDate'] == target_settlement_str].copy()

                if df_filtered.empty:
                    self.logger.warning(f"No rows matching settlement date {target_settlement_str} for publish date {publish_date}")
                    return pd.DataFrame()

                # Select required columns
                required_cols = ['startTime', 'settlementPeriod', 'transmissionSystemDemand', 'nationalDemand']
                missing_cols = [col for col in required_cols if col not in df_filtered.columns]

                if missing_cols:
                    self.logger.warning(f"Missing columns {missing_cols} for {publish_date}")
                    return pd.DataFrame()

                df_result = df_filtered[required_cols].copy()

                self.logger.info(f"Successfully fetched {len(df_result)} rows for {publish_date}")
                return df_result

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed for {publish_date} (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"All retry attempts failed for {publish_date}")
                    return pd.DataFrame()

        return pd.DataFrame()

    def fetch_date_range(self, start_date, end_date):
        """
        Fetch day-ahead demand forecasts for a date range.

        Args:
            start_date: datetime.date or string (YYYY-MM-DD) for start date (inclusive)
            end_date: datetime.date or string (YYYY-MM-DD) for end date (inclusive)

        Returns:
            pd.DataFrame with columns: startTime, settlementPeriod,
                                      transmissionSystemDemand, nationalDemand
        """
        # Convert strings to datetime.date if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        self.logger.info(f"Fetching Elexon BMRS data from {start_date} to {end_date}")

        # Collect all DataFrames
        all_dfs = []

        # Iterate over date range (inclusive)
        current_date = start_date
        while current_date <= end_date:
            df_day = self.fetch_day_ahead_demand(current_date)

            if not df_day.empty:
                all_dfs.append(df_day)

            current_date += timedelta(days=1)

        # Concatenate all results
        if not all_dfs:
            self.logger.warning("No data fetched for any date in range")
            return pd.DataFrame(columns=['startTime', 'settlementPeriod',
                                        'transmissionSystemDemand', 'nationalDemand'])

        df_combined = pd.concat(all_dfs, ignore_index=True)

        # Drop duplicates by startTime and settlementPeriod
        initial_rows = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['startTime', 'settlementPeriod'], keep='last')
        duplicates_removed = initial_rows - len(df_combined)

        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")

        # Sort by startTime
        df_combined = df_combined.sort_values('startTime').reset_index(drop=True)

        self.logger.info(f"Total rows fetched: {len(df_combined)}")

        return df_combined


class ElexonIndicatedImbalanceFetcher:
    """
    Fetch day-ahead indicated imbalance data by boundary from Elexon BMRS API.

    API Endpoint: https://data.elexon.co.uk/bmrs/api/v1/forecast/indicated/day-ahead/history
    """

    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/forecast/indicated/day-ahead/history"
    BOUNDARIES = [f"B{i}" for i in range(1, 18)]  # B1 to B17

    def __init__(self, max_retries=3, backoff_factor=1):
        """
        Initialize the Elexon Indicated Imbalance fetcher.

        Args:
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Base factor for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)

    def fetch_boundary_data(self, publish_date, boundary):
        """
        Fetch indicated imbalance data for a specific publish date and boundary.

        Args:
            publish_date: datetime.date object for the publish date
            boundary: Boundary identifier (e.g., "B1", "B2", ..., "B17")

        Returns:
            pd.DataFrame with columns: startTime, settlementPeriod, boundary,
                                      indicatedGeneration, indicatedDemand,
                                      indicatedMargin, indicatedImbalance
        """
        # Build publishTime parameter: publish_date at 16:30:00 UTC
        publish_time = f"{publish_date.strftime('%Y-%m-%d')}T16:30:00Z"

        # Target settlement date is day after publish date
        target_settlement_date = publish_date + timedelta(days=1)
        target_settlement_str = target_settlement_date.strftime('%Y-%m-%d')

        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                # Build request parameters
                params = {
                    'publishTime': publish_time,
                    'boundary': boundary,
                    'format': 'json'
                }

                # Make GET request
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()

                # Parse JSON response
                try:
                    data = response.json()
                except ValueError as e:
                    self.logger.warning(f"JSON parsing failed for {publish_date} {boundary}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.backoff_factor * (2 ** attempt))
                        continue
                    return pd.DataFrame()

                # Extract data array
                if 'data' not in data or not isinstance(data['data'], list):
                    self.logger.warning(f"No data array found for {publish_date} {boundary}")
                    return pd.DataFrame()

                records = data['data']

                if not records:
                    self.logger.warning(f"Empty data array for {publish_date} {boundary}")
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(records)

                # Filter to target settlement date only
                if 'settlementDate' not in df.columns:
                    self.logger.warning(f"No settlementDate column for {publish_date} {boundary}")
                    return pd.DataFrame()

                df_filtered = df[df['settlementDate'] == target_settlement_str].copy()

                if df_filtered.empty:
                    self.logger.warning(f"No rows matching settlement date {target_settlement_str} for {publish_date} {boundary}")
                    return pd.DataFrame()

                # Select required columns
                required_cols = ['startTime', 'settlementPeriod', 'boundary',
                               'indicatedGeneration', 'indicatedDemand',
                               'indicatedMargin', 'indicatedImbalance']
                missing_cols = [col for col in required_cols if col not in df_filtered.columns]

                if missing_cols:
                    self.logger.warning(f"Missing columns {missing_cols} for {publish_date} {boundary}")
                    return pd.DataFrame()

                df_result = df_filtered[required_cols].copy()

                return df_result

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed for {publish_date} {boundary} (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"All retry attempts failed for {publish_date} {boundary}")
                    return pd.DataFrame()

        return pd.DataFrame()

    def fetch_day_data(self, publish_date):
        """
        Fetch indicated imbalance data for all boundaries for a specific publish date.

        Args:
            publish_date: datetime.date object for the publish date

        Returns:
            pd.DataFrame with columns: startTime, settlementPeriod, boundary,
                                      indicatedGeneration, indicatedDemand,
                                      indicatedMargin, indicatedImbalance
        """
        self.logger.info(f"Fetching indicated imbalance for {publish_date} (all boundaries)...")

        all_boundary_dfs = []

        for boundary in self.BOUNDARIES:
            df_boundary = self.fetch_boundary_data(publish_date, boundary)

            if not df_boundary.empty:
                all_boundary_dfs.append(df_boundary)

        if not all_boundary_dfs:
            self.logger.warning(f"No boundary data fetched for {publish_date}")
            return pd.DataFrame()

        # Concatenate all boundary data
        df_combined = pd.concat(all_boundary_dfs, ignore_index=True)
        self.logger.info(f"Fetched {len(df_combined)} rows for {publish_date} ({len(all_boundary_dfs)} boundaries)")

        return df_combined

    def reshape_to_wide_format(self, df_long):
        """
        Reshape data from long format (one row per boundary) to wide format
        (one row per settlement period with all boundaries as columns).

        Args:
            df_long: DataFrame in long format with boundary column

        Returns:
            pd.DataFrame in wide format with boundary-prefixed columns
        """
        if df_long.empty:
            return pd.DataFrame()

        # Pivot the data
        # For each (startTime, settlementPeriod), create columns for each boundary's metrics
        metric_cols = ['indicatedGeneration', 'indicatedDemand', 'indicatedMargin', 'indicatedImbalance']

        # Create pivot for each metric
        pivoted_dfs = []

        for metric in metric_cols:
            df_pivot = df_long.pivot_table(
                index=['startTime', 'settlementPeriod'],
                columns='boundary',
                values=metric,
                aggfunc='last'  # In case of duplicates, take last
            )

            # Rename columns to include boundary prefix
            df_pivot.columns = [f"{boundary}_{metric}" for boundary in df_pivot.columns]

            pivoted_dfs.append(df_pivot)

        # Combine all pivoted metrics
        df_wide = pd.concat(pivoted_dfs, axis=1)

        # Reset index to make startTime and settlementPeriod regular columns
        df_wide = df_wide.reset_index()

        return df_wide

    def fetch_date_range(self, start_date, end_date):
        """
        Fetch indicated imbalance data for a date range, reshaped to wide format.

        Args:
            start_date: datetime.date or string (YYYY-MM-DD) for start date (inclusive)
            end_date: datetime.date or string (YYYY-MM-DD) for end date (inclusive)

        Returns:
            pd.DataFrame in wide format with columns: startTime, settlementPeriod,
                                                     B1_*, B2_*, ..., B17_*
        """
        # Convert strings to datetime.date if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        self.logger.info(f"Fetching Elexon Indicated Imbalance data from {start_date} to {end_date}")

        # Calculate total API calls
        num_days = (end_date - start_date).days + 1
        total_calls = num_days * len(self.BOUNDARIES)
        self.logger.info(f"Expected API calls: {total_calls} ({num_days} days × {len(self.BOUNDARIES)} boundaries)")

        # Collect all DataFrames
        all_dfs = []

        # Iterate over date range (inclusive)
        current_date = start_date
        while current_date <= end_date:
            df_day = self.fetch_day_data(current_date)

            if not df_day.empty:
                all_dfs.append(df_day)

            current_date += timedelta(days=1)

        # Concatenate all results (long format)
        if not all_dfs:
            self.logger.warning("No data fetched for any date in range")
            return pd.DataFrame()

        df_long = pd.concat(all_dfs, ignore_index=True)

        # Drop duplicates by startTime, settlementPeriod, and boundary
        initial_rows = len(df_long)
        df_long = df_long.drop_duplicates(subset=['startTime', 'settlementPeriod', 'boundary'], keep='last')
        duplicates_removed = initial_rows - len(df_long)

        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")

        # Reshape to wide format
        self.logger.info("Reshaping data to wide format (boundary columns)...")
        df_wide = self.reshape_to_wide_format(df_long)

        # Sort by startTime
        df_wide = df_wide.sort_values('startTime').reset_index(drop=True)

        self.logger.info(f"Total rows fetched: {len(df_wide)}")
        self.logger.info(f"Total columns: {len(df_wide.columns)}")

        return df_wide


class ElexonSystemPricesFetcher:
    """
    Fetch imbalance settlement system prices from Elexon BMRS API.

    API Endpoint: https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{date}?format=json
    """

    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices"

    def __init__(self, max_retries=3, backoff_factor=1):
        """
        Initialize the Elexon System Prices fetcher.

        Args:
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Base factor for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)

    def fetch_system_prices(self, settlement_date):
        """
        Fetch system prices for a specific settlement date.

        Args:
            settlement_date: datetime.date object for the settlement date

        Returns:
            pd.DataFrame with columns: startTime, settlementPeriod, imbalancePrice
        """
        settlement_date_str = settlement_date.strftime('%Y-%m-%d')

        self.logger.info(f"Fetching system prices for settlement date {settlement_date_str}")

        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                # Build URL with settlement date
                url = f"{self.BASE_URL}/{settlement_date_str}"

                # Build request parameters
                params = {
                    'format': 'json'
                }

                # Make GET request
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()

                # Parse JSON response
                try:
                    data = response.json()
                except ValueError as e:
                    self.logger.warning(f"JSON parsing failed for {settlement_date_str}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.backoff_factor * (2 ** attempt))
                        continue
                    return pd.DataFrame()

                # Extract data array
                if 'data' not in data or not isinstance(data['data'], list):
                    self.logger.warning(f"No data array found in response for {settlement_date_str}")
                    return pd.DataFrame()

                records = data['data']

                if not records:
                    self.logger.warning(f"Empty data array for {settlement_date_str}")
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(records)

                # Check required columns exist
                required_cols = ['startTime', 'settlementPeriod', 'systemSellPrice']
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    self.logger.warning(f"Missing columns {missing_cols} for {settlement_date_str}")
                    return pd.DataFrame()

                # Select and rename columns
                df_result = df[required_cols].copy()
                df_result = df_result.rename(columns={'systemSellPrice': 'imbalancePrice'})

                self.logger.info(f"Successfully fetched {len(df_result)} rows for {settlement_date_str}")
                return df_result

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed for {settlement_date_str} (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"All retry attempts failed for {settlement_date_str}")
                    return pd.DataFrame()

        return pd.DataFrame()

    def fetch_date_range(self, start_date, end_date):
        """
        Fetch system prices for a date range.

        Args:
            start_date: datetime.date or string (YYYY-MM-DD) for start date (inclusive)
            end_date: datetime.date or string (YYYY-MM-DD) for end date (inclusive)

        Returns:
            pd.DataFrame with columns: startTime, settlementPeriod, imbalancePrice
        """
        # Convert strings to datetime.date if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        self.logger.info(f"Fetching Elexon System Prices data from {start_date} to {end_date}")

        # Collect all DataFrames
        all_dfs = []

        # Iterate over date range (inclusive)
        current_date = start_date
        while current_date <= end_date:
            df_day = self.fetch_system_prices(current_date)

            if not df_day.empty:
                all_dfs.append(df_day)

            current_date += timedelta(days=1)

        # Concatenate all results
        if not all_dfs:
            self.logger.warning("No data fetched for any date in range")
            return pd.DataFrame(columns=['startTime', 'settlementPeriod', 'imbalancePrice'])

        df_combined = pd.concat(all_dfs, ignore_index=True)

        # Drop duplicates by startTime and settlementPeriod
        initial_rows = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['startTime', 'settlementPeriod'], keep='last')
        duplicates_removed = initial_rows - len(df_combined)

        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")

        # Sort by startTime
        df_combined = df_combined.sort_values('startTime').reset_index(drop=True)

        self.logger.info(f"Total rows fetched: {len(df_combined)}")

        return df_combined


class ElexonLOLPDRMFetcher:
    """
    Fetch Loss of Load Probability and Derated Margin data from Elexon BMRS API.

    API Endpoint: https://data.elexon.co.uk/bmrs/api/v1/datasets/LOLPDRM/stream
    """

    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/LOLPDRM/stream"

    def __init__(self, max_retries=3, backoff_factor=1):
        """
        Initialize the Elexon LOLPDRM fetcher.

        Args:
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Base factor for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)

    def fetch_lolpdrm_data(self, publish_date):
        """
        Fetch LOLP and Derated Margin data for a specific publish date.

        Fetches data published between 16:15 and 16:45 UTC on the publish date,
        containing forecasts for the next day (D+1).

        Args:
            publish_date: datetime.date object for the publish date

        Returns:
            pd.DataFrame with columns: startTime, settlementPeriod,
                                      lossOfLoadProbability, deratedMargin
        """
        # Build time window: 16:15 to 16:45 UTC on publish_date
        publish_from = f"{publish_date.strftime('%Y-%m-%d')}T16:15:00Z"
        publish_to = f"{publish_date.strftime('%Y-%m-%d')}T16:45:00Z"

        # Target settlement date is day after publish date
        target_settlement_date = publish_date + timedelta(days=1)
        target_settlement_str = target_settlement_date.strftime('%Y-%m-%d')

        self.logger.info(f"Fetching LOLPDRM data for publish date {publish_date} (settlement date: {target_settlement_str})")

        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                # Build request parameters
                params = {
                    'publishDateTimeFrom': publish_from,
                    'publishDateTimeTo': publish_to,
                    'format': 'json'
                }

                # Make GET request
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()

                # Parse JSON response
                try:
                    data = response.json()
                except ValueError as e:
                    self.logger.warning(f"JSON parsing failed for {publish_date}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.backoff_factor * (2 ** attempt))
                        continue
                    return pd.DataFrame()

                # LOLPDRM API returns a JSON array directly (not wrapped in {"data": []})
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict) and 'data' in data:
                    records = data['data']
                else:
                    self.logger.warning(f"Unexpected response format for {publish_date}")
                    return pd.DataFrame()

                if not records:
                    self.logger.warning(f"Empty data array for {publish_date}")
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(records)

                # Filter to target settlement date only
                if 'settlementDate' not in df.columns:
                    self.logger.warning(f"No settlementDate column for {publish_date}")
                    return pd.DataFrame()

                df_filtered = df[df['settlementDate'] == target_settlement_str].copy()

                if df_filtered.empty:
                    self.logger.warning(f"No rows matching settlement date {target_settlement_str} for publish date {publish_date}")
                    return pd.DataFrame()

                # Check required columns exist
                required_cols = ['startTime', 'settlementPeriod', 'lossOfLoadProbability', 'deratedMargin', 'publishTime']
                missing_cols = [col for col in required_cols if col not in df_filtered.columns]

                if missing_cols:
                    self.logger.warning(f"Missing columns {missing_cols} for {publish_date}")
                    return pd.DataFrame()

                # Convert publishTime to datetime for sorting
                df_filtered['publishTime'] = pd.to_datetime(df_filtered['publishTime'])

                # For each settlementPeriod, take the latest publishTime (most recent forecast)
                df_filtered = df_filtered.sort_values('publishTime', ascending=False)
                df_result = df_filtered.drop_duplicates(subset=['settlementPeriod'], keep='first')

                # Select final columns (drop publishTime after deduplication)
                df_result = df_result[['startTime', 'settlementPeriod', 'lossOfLoadProbability', 'deratedMargin']].copy()

                # Sort by settlementPeriod for consistency
                df_result = df_result.sort_values('settlementPeriod').reset_index(drop=True)

                self.logger.info(f"Successfully fetched {len(df_result)} rows for {publish_date}")
                return df_result

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed for {publish_date} (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"All retry attempts failed for {publish_date}")
                    return pd.DataFrame()

        return pd.DataFrame()

    def fetch_date_range(self, start_date, end_date):
        """
        Fetch LOLP and Derated Margin data for a date range.

        Args:
            start_date: datetime.date or string (YYYY-MM-DD) for start date (inclusive)
            end_date: datetime.date or string (YYYY-MM-DD) for end date (inclusive)

        Returns:
            pd.DataFrame with columns: startTime, settlementPeriod,
                                      lossOfLoadProbability, deratedMargin
        """
        # Convert strings to datetime.date if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        self.logger.info(f"Fetching Elexon LOLPDRM data from {start_date} to {end_date}")

        # Collect all DataFrames
        all_dfs = []

        # Iterate over date range (inclusive)
        current_date = start_date
        while current_date <= end_date:
            df_day = self.fetch_lolpdrm_data(current_date)

            if not df_day.empty:
                all_dfs.append(df_day)

            current_date += timedelta(days=1)

        # Concatenate all results
        if not all_dfs:
            self.logger.warning("No data fetched for any date in range")
            return pd.DataFrame(columns=['startTime', 'settlementPeriod',
                                        'lossOfLoadProbability', 'deratedMargin'])

        df_combined = pd.concat(all_dfs, ignore_index=True)

        # Drop duplicates by startTime and settlementPeriod
        initial_rows = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['startTime', 'settlementPeriod'], keep='last')
        duplicates_removed = initial_rows - len(df_combined)

        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")

        # Sort by startTime
        df_combined = df_combined.sort_values('startTime').reset_index(drop=True)

        self.logger.info(f"Total rows fetched: {len(df_combined)}")

        return df_combined


class DataBuilder:
    """
    Build and prepare time series data for forecasting.

    Output format required:
    - valueDateTimeOffset: UTC timestamp column
    - Additional feature columns
    - Optionally: premium (target variable)
    """

    def __init__(self, start_date, end_date):
        """
        Initialize DataBuilder with global date range.

        Args:
            start_date: Start date for data collection (YYYY-MM-DD or datetime.date)
            end_date: End date for data collection (YYYY-MM-DD or datetime.date)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.df = None
        self.sources = []
        self.logger = logging.getLogger(__name__)

    def load_source_1(self):
        """
        Load data from Elexon BMRS API (Source 1).

        Fetches day-ahead demand forecasts for the date range specified
        in __init__ (self.start_date to self.end_date).

        Returns:
            pd.DataFrame with columns: valueDateTimeOffset, settlementPeriod,
                                      transmissionSystemDemand, nationalDemand
        """
        self.logger.info("=" * 60)
        self.logger.info("Loading Source 1: Elexon BMRS Day-Ahead Demand")
        self.logger.info("=" * 60)

        # Initialize fetcher
        fetcher = ElexonBMRSFetcher()

        # Fetch data for date range
        df = fetcher.fetch_date_range(self.start_date, self.end_date)

        if df.empty:
            raise ValueError("No data fetched from Elexon BMRS API")

        # Convert startTime to UTC datetime as valueDateTimeOffset
        self.logger.info("Converting startTime to valueDateTimeOffset (UTC datetime)...")
        df['valueDateTimeOffset'] = pd.to_datetime(df['startTime'], utc=True)

        # Drop the original startTime column (keep valueDateTimeOffset)
        df = df.drop(columns=['startTime'])

        # Reorder columns: valueDateTimeOffset first, then features
        df = df[['valueDateTimeOffset', 'settlementPeriod', 'transmissionSystemDemand', 'nationalDemand']]

        self.logger.info(f"Source 1 loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        self.logger.info(f"Date range: {df['valueDateTimeOffset'].min()} to {df['valueDateTimeOffset'].max()}")

        return df

    def load_source_2(self):
        """
        Load data from Elexon BMRS Indicated Imbalance API (Source 2).

        Fetches day-ahead indicated imbalance by boundary for the date range
        specified in __init__ (self.start_date to self.end_date).

        Returns:
            pd.DataFrame with columns: valueDateTimeOffset, settlementPeriod,
                                      B1_*, B2_*, ..., B17_* (68 boundary columns)
        """
        self.logger.info("=" * 60)
        self.logger.info("Loading Source 2: Elexon BMRS Indicated Imbalance by Boundary")
        self.logger.info("=" * 60)

        # Initialize fetcher
        fetcher = ElexonIndicatedImbalanceFetcher()

        # Fetch data for date range (already in wide format)
        df = fetcher.fetch_date_range(self.start_date, self.end_date)

        if df.empty:
            raise ValueError("No data fetched from Elexon Indicated Imbalance API")

        # Convert startTime to UTC datetime as valueDateTimeOffset
        self.logger.info("Converting startTime to valueDateTimeOffset (UTC datetime)...")
        df['valueDateTimeOffset'] = pd.to_datetime(df['startTime'], utc=True)

        # Drop the original startTime column (keep valueDateTimeOffset)
        df = df.drop(columns=['startTime'])

        # Reorder columns: valueDateTimeOffset and settlementPeriod first, then boundary columns
        boundary_cols = [col for col in df.columns if col not in ['valueDateTimeOffset', 'settlementPeriod']]
        df = df[['valueDateTimeOffset', 'settlementPeriod'] + boundary_cols]

        self.logger.info(f"Source 2 loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        self.logger.info(f"Date range: {df['valueDateTimeOffset'].min()} to {df['valueDateTimeOffset'].max()}")
        self.logger.info(f"Boundary columns: {len(boundary_cols)} (17 boundaries × 4 metrics)")

        return df

    def load_source_3(self):
        """
        Load data from Elexon BMRS System Prices API (Source 3).

        Fetches imbalance settlement system prices for the date range
        specified in __init__ (self.start_date to self.end_date).

        Returns:
            pd.DataFrame with columns: valueDateTimeOffset, settlementPeriod, imbalancePrice
        """
        self.logger.info("=" * 60)
        self.logger.info("Loading Source 3: Elexon BMRS Imbalance Settlement Prices")
        self.logger.info("=" * 60)

        # Initialize fetcher
        fetcher = ElexonSystemPricesFetcher()

        # Fetch data for date range
        df = fetcher.fetch_date_range(self.start_date, self.end_date)

        if df.empty:
            raise ValueError("No data fetched from Elexon System Prices API")

        # Convert startTime to UTC datetime as valueDateTimeOffset
        self.logger.info("Converting startTime to valueDateTimeOffset (UTC datetime)...")
        df['valueDateTimeOffset'] = pd.to_datetime(df['startTime'], utc=True)

        # Drop the original startTime column (keep valueDateTimeOffset)
        df = df.drop(columns=['startTime'])

        # Reorder columns: valueDateTimeOffset first, then features
        df = df[['valueDateTimeOffset', 'settlementPeriod', 'imbalancePrice']]

        self.logger.info(f"Source 3 loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        self.logger.info(f"Date range: {df['valueDateTimeOffset'].min()} to {df['valueDateTimeOffset'].max()}")

        return df

    def load_source_4(self):
        """
        Load data from Elexon BMRS LOLPDRM API (Source 4).

        Fetches Loss of Load Probability and Derated Margin data for the date range
        specified in __init__ (self.start_date to self.end_date).

        Returns:
            pd.DataFrame with columns: valueDateTimeOffset, settlementPeriod,
                                      lossOfLoadProbability, deratedMargin
        """
        self.logger.info("=" * 60)
        self.logger.info("Loading Source 4: Elexon BMRS Loss of Load Probability & Derated Margin")
        self.logger.info("=" * 60)

        # Initialize fetcher
        fetcher = ElexonLOLPDRMFetcher()

        # Fetch data for date range
        df = fetcher.fetch_date_range(self.start_date, self.end_date)

        if df.empty:
            raise ValueError("No data fetched from Elexon LOLPDRM API")

        # Convert startTime to UTC datetime as valueDateTimeOffset
        self.logger.info("Converting startTime to valueDateTimeOffset (UTC datetime)...")
        df['valueDateTimeOffset'] = pd.to_datetime(df['startTime'], utc=True)

        # Drop the original startTime column (keep valueDateTimeOffset)
        df = df.drop(columns=['startTime'])

        # Reorder columns: valueDateTimeOffset first, then features
        df = df[['valueDateTimeOffset', 'settlementPeriod', 'lossOfLoadProbability', 'deratedMargin']]

        self.logger.info(f"Source 4 loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        self.logger.info(f"Date range: {df['valueDateTimeOffset'].min()} to {df['valueDateTimeOffset'].max()}")

        return df

    def merge_sources(self, df1, df2, merge_type='left'):
        """
        Merge multiple data sources on timestamp and settlement period.

        Args:
            df1: Primary dataframe with valueDateTimeOffset
            df2: Secondary dataframe with additional features
            merge_type: Type of merge ('left', 'inner', 'outer')

        Returns:
            pd.DataFrame: Merged dataset
        """
        self.logger.info(f"Merging data sources with {merge_type} join...")

        # Determine join keys (use both timestamp and settlementPeriod if available)
        join_keys = ['valueDateTimeOffset']
        if 'settlementPeriod' in df1.columns and 'settlementPeriod' in df2.columns:
            join_keys.append('settlementPeriod')
            self.logger.info(f"Merging on: {join_keys}")

        # Merge on join keys
        merged_df = pd.merge(
            df1,
            df2,
            on=join_keys,
            how=merge_type,
            suffixes=('', '_source2')
        )

        self.logger.info(f"Merged dataset shape: {merged_df.shape}")
        return merged_df

    def validate_data(self, df, require_premium=False):
        """
        Validate the dataset has required columns and proper format.

        Args:
            df: DataFrame to validate
            require_premium: If True, validate that 'premium' column exists and is numeric.
                           If False, only validate valueDateTimeOffset (features-only mode)

        Returns:
            bool: True if valid, raises error otherwise
        """
        self.logger.info("Validating dataset...")

        # Check required timestamp column
        required_cols = ['valueDateTimeOffset']
        if require_premium:
            required_cols.append('premium')

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check timestamp format
        if not pd.api.types.is_datetime64_any_dtype(df['valueDateTimeOffset']):
            raise ValueError("valueDateTimeOffset must be datetime type")

        # Check for timezone awareness
        if df['valueDateTimeOffset'].dt.tz is None:
            raise ValueError("valueDateTimeOffset must be timezone-aware (UTC)")

        # Check premium is numeric (only if required)
        if require_premium:
            if not pd.api.types.is_numeric_dtype(df['premium']):
                raise ValueError("premium must be numeric type")

        # Check for duplicates
        duplicates = df.duplicated(subset=['valueDateTimeOffset']).sum()
        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicate timestamps")

        # Check for missing values
        missing_timestamp = df['valueDateTimeOffset'].isna().sum()

        if missing_timestamp > 0:
            self.logger.warning(f"Found {missing_timestamp} missing timestamps")

        if require_premium:
            missing_premium = df['premium'].isna().sum()
            if missing_premium > 0:
                self.logger.warning(f"Found {missing_premium} missing premium values")

        self.logger.info("✓ Dataset validation passed")
        self.logger.info(f"  - Shape: {df.shape}")
        self.logger.info(f"  - Time range: {df['valueDateTimeOffset'].min()} to {df['valueDateTimeOffset'].max()}")
        self.logger.info(f"  - Columns: {list(df.columns)}")

        return True

    def clean_data(self, df, require_premium=False):
        """
        Clean the dataset before saving.

        Args:
            df: DataFrame to clean
            require_premium: If True, drop rows with missing premium values.
                           If False, only drop rows with missing timestamps.

        Returns:
            pd.DataFrame: Cleaned dataset
        """
        self.logger.info("Cleaning dataset...")

        initial_rows = len(df)

        # Remove rows with missing timestamps
        drop_cols = ['valueDateTimeOffset']
        if require_premium and 'premium' in df.columns:
            drop_cols.append('premium')

        df = df.dropna(subset=drop_cols)

        # Remove duplicate timestamps (keep last)
        df = df.drop_duplicates(subset=['valueDateTimeOffset'], keep='last')

        # Sort by timestamp
        df = df.sort_values('valueDateTimeOffset').reset_index(drop=True)

        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            self.logger.info(f"  - Removed {removed_rows} rows during cleaning")

        self.logger.info(f"  - Final shape: {df.shape}")

        return df

    def save_dataset(self, df, output_path, require_premium=False):
        """
        Save the final dataset to CSV.

        Args:
            df: DataFrame to save
            output_path: Path to save the CSV file
            require_premium: If True, validate that premium column exists
        """
        # Validate before saving
        self.validate_data(df, require_premium=require_premium)

        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        self.logger.info(f"✓ Dataset saved to {output_path}")
        self.logger.info(f"  - {len(df)} rows, {len(df.columns)} columns")

    def build_dataset(self, output_path, load_sources=[1], require_premium=False):
        """
        Main pipeline to build the dataset from multiple sources.

        Args:
            output_path: Path to save the final CSV
            load_sources: List of source numbers to load (e.g., [1, 2])
            require_premium: If True, validate that premium column exists in final dataset

        Returns:
            pd.DataFrame: Final dataset
        """
        self.logger.info("=" * 60)
        self.logger.info("Building Dataset for Time Series Forecasting")
        self.logger.info("=" * 60)
        self.logger.info(f"Date range: {self.start_date} to {self.end_date}")

        if not load_sources:
            raise ValueError("At least one source must be specified")

        # Load primary source (source 1)
        if 1 in load_sources:
            df = self.load_source_1()
        else:
            raise ValueError("Source 1 must be included as the primary data source")

        # Load and merge additional sources if specified
        for source_num in load_sources[1:]:
            self.logger.info(f"\nLoading source {source_num}...")
            loader_method = getattr(self, f"load_source_{source_num}", None)

            if loader_method is None:
                self.logger.warning(f"Source {source_num} not implemented, skipping...")
                continue

            df_additional = loader_method()
            df = self.merge_sources(df, df_additional, merge_type='left')

        # Clean the data
        df = self.clean_data(df, require_premium=require_premium)

        # Save the final dataset
        self.save_dataset(df, output_path, require_premium=require_premium)

        self.df = df
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Build dataset for time series forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch Elexon BMRS data for a week
  python data.py --start 2025-01-01 --end 2025-01-07 --output data/elexon_data.csv

  # Fetch data for multiple sources (when implemented)
  python data.py --start 2025-01-01 --end 2025-01-31 --sources 1 2 --output data/combined_data.csv

  # Require premium column validation (for complete datasets)
  python data.py --start 2025-01-01 --end 2025-01-31 --output data/full_data.csv --require-premium
        """
    )

    # Required arguments
    parser.add_argument('--start', type=str, required=True,
                       help='Start date for data collection (format: YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                       help='End date for data collection (format: YYYY-MM-DD, inclusive)')

    # Optional arguments
    parser.add_argument('--output', type=str, default='data/processed_data.csv',
                       help='Path to save the processed dataset (default: data/processed_data.csv)')
    parser.add_argument('--sources', type=int, nargs='+', default=[1],
                       help='List of data sources to load (default: 1). Example: --sources 1 2')
    parser.add_argument('--require-premium', action='store_true',
                       help='Validate that premium column exists in final dataset')

    args = parser.parse_args()

    try:
        # Parse dates
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

        # Validate date range
        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")

        # Initialize builder with date range
        builder = DataBuilder(start_date, end_date)

        # Build dataset
        df = builder.build_dataset(
            output_path=args.output,
            load_sources=args.sources,
            require_premium=args.require_premium
        )

        logging.info("\n" + "=" * 60)
        logging.info("Dataset building completed successfully!")
        logging.info("=" * 60)
        logging.info(f"\nDataset saved to: {args.output}")
        logging.info(f"Total rows: {len(df)}")
        logging.info(f"Columns: {list(df.columns)}")

        if not args.require_premium:
            logging.info("\nNote: This is a features-only dataset (no premium/target column).")
            logging.info("To train a model, you'll need to add a 'premium' column as the target variable.")
        else:
            logging.info(f"\nNext step: Train the model using:")
            logging.info(f"  python train_forecast.py {args.output}")

    except ValueError as e:
        logging.error(f"Error: {e}")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
