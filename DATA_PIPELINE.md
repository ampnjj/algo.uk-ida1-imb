# Data Pipeline Documentation

This document explains how to use `data.py` to build datasets for the time series forecasting model in `train_forecast.py`.

## Overview

The `data.py` script fetches data from multiple sources and combines them into a single dataset suitable for time series forecasting. All data sources use a **global date range** specified via `--start` and `--end` arguments.

## Quick Start

### Basic Usage

```bash
# Fetch Elexon BMRS data for a date range
python data.py --start 2025-01-01 --end 2025-01-31 --output data/elexon_data.csv
```

### Common Options

```bash
# Required arguments:
--start YYYY-MM-DD    # Start date (inclusive)
--end YYYY-MM-DD      # End date (inclusive)

# Optional arguments:
--output PATH         # Output CSV path (default: data/processed_data.csv)
--sources N [N ...]   # List of data sources to load (default: [1])
--require-premium     # Validate that 'premium' column exists (for complete datasets)
```

## Output Format

The generated dataset follows this structure required by `train_forecast.py`:

| Column Name | Type | Description | Required |
|------------|------|-------------|----------|
| `valueDateTimeOffset` | datetime (UTC) | Timestamp column | ✅ Yes |
| `premium` | float | Target variable to forecast | Only with `--require-premium` |
| Additional columns | various | Feature columns from data sources | No |

### Example Output Structure

```csv
valueDateTimeOffset,settlementPeriod,transmissionSystemDemand,nationalDemand
2025-01-01 00:00:00+00:00,1,23556,22700
2025-01-01 00:30:00+00:00,2,22848,22209
2025-01-01 01:00:00+00:00,3,22974,22200
...
```

## Data Sources

### Source 1: Elexon BMRS Day-Ahead Demand Forecast

**Status:** ✅ Implemented

**Description:** Fetches day-ahead electricity demand forecasts from the Elexon BMRS (Balancing Mechanism Reporting Service) API.

**API Endpoint:**
```
https://data.elexon.co.uk/bmrs/api/v1/forecast/demand/day-ahead/history
```

**How it Works:**
1. For each date `D` in the range `[start_date, end_date]`:
   - Requests data published at `D + T16:30:00Z` (4:30 PM UTC)
   - Filters results to only include settlement date = `D + 1 day` (day-ahead forecast)
   - Fetches data once per day

2. Data granularity: **30-minute intervals** (48 settlement periods per day)

3. Output columns:
   - `valueDateTimeOffset` - UTC timestamp
   - `settlementPeriod` - Settlement period number (1-48)
   - `transmissionSystemDemand` - Transmission system demand forecast (MW)
   - `nationalDemand` - National demand forecast (MW)

**Features:**
- ✅ HTTP retry logic (3 attempts with exponential backoff)
- ✅ Automatic duplicate removal
- ✅ Sorts by timestamp
- ✅ Robust error handling for missing data
- ✅ No API key required

**Example:**
```bash
# Fetch one month of Elexon data
python data.py --start 2025-01-01 --end 2025-01-31 --output data/elexon_jan.csv
```

**Expected Output Size:**
- 1 day = 48 rows (30-min intervals)
- 1 week = ~336 rows
- 1 month = ~1,440 rows
- 1 year = ~17,520 rows

---

### Source 2: Elexon BMRS Indicated Imbalance by Boundary

**Status:** ✅ Implemented

**Description:** Fetches day-ahead indicated imbalance forecasts by boundary zone from the Elexon BMRS API. The UK electricity grid is divided into 17 boundary zones (B1-B17), and this source provides generation, demand, margin, and imbalance metrics for each boundary.

**API Endpoint:**
```
https://data.elexon.co.uk/bmrs/api/v1/forecast/indicated/day-ahead/history
```

**How it Works:**
1. For each date `D` in the range `[start_date, end_date]`:
   - For each boundary `B1` through `B17`:
     - Requests data published at `D + T16:30:00Z` (4:30 PM UTC)
     - Filters results to only include settlement date = `D + 1 day` (day-ahead forecast)
   - **17 API requests per day** (one per boundary)

2. Data granularity: **30-minute intervals** (48 settlement periods per day)

3. Data reshaping: Converts from long format (one row per boundary) to wide format (one row per settlement period with all 17 boundaries as columns)

4. Output columns (70 total):
   - `valueDateTimeOffset` - UTC timestamp
   - `settlementPeriod` - Settlement period number (1-48)
   - For each boundary (B1-B17), 4 metrics × 17 boundaries = **68 columns**:
     - `{B}_indicatedGeneration` - Indicated generation for boundary (MW)
     - `{B}_indicatedDemand` - Indicated demand for boundary (MW)
     - `{B}_indicatedMargin` - Indicated margin for boundary (MW)
     - `{B}_indicatedImbalance` - Indicated imbalance for boundary (MW)

**Example column names:**
- `B1_indicatedGeneration`, `B1_indicatedDemand`, `B1_indicatedMargin`, `B1_indicatedImbalance`
- `B2_indicatedGeneration`, `B2_indicatedDemand`, `B2_indicatedMargin`, `B2_indicatedImbalance`
- ... (continues through B17)

**Features:**
- ✅ HTTP retry logic (3 attempts with exponential backoff)
- ✅ Boundary loop (fetches all 17 boundaries automatically)
- ✅ Automatic duplicate removal
- ✅ Reshapes data from long to wide format
- ✅ Sorts by timestamp
- ✅ Robust error handling for missing boundaries
- ✅ No API key required

**Join Keys:**
- Merges with Source 1 on: `['valueDateTimeOffset', 'settlementPeriod']`

**Example:**
```bash
# Fetch both Source 1 and Source 2
python data.py --start 2025-01-01 --end 2025-01-31 --sources 1 2 --output data/combined.csv
```

**Expected Output Size:**
- 1 day = 48 rows (30-min intervals)
- 1 week = ~336 rows
- 1 month = ~1,440 rows
- 1 year = ~17,520 rows

**Columns:** 72 total when combined with Source 1
- 4 columns from Source 1 (valueDateTimeOffset, settlementPeriod, transmissionSystemDemand, nationalDemand)
- 68 columns from Source 2 (17 boundaries × 4 metrics)

**API Call Volume:**
- 1 day = 17 API calls
- 1 week = 119 API calls
- 1 month = 527 API calls (31 days)
- 1 year = 6,205 API calls

**Performance Notes:**
⚠️ This source makes **17× more API calls** than Source 1. For a 1-month date range:
- Source 1 alone: ~31 API calls, ~30 seconds
- Source 2 alone: ~527 API calls, ~5-10 minutes
- Both sources: ~558 API calls, ~5-10 minutes

---

### Source 3: Elexon BMRS Imbalance Settlement System Prices

**Status:** ✅ Implemented

**Description:** Fetches imbalance settlement system prices from the Elexon BMRS API. This provides the system sell price (imbalance price) which represents the price at which the system is short and needs to buy energy to balance.

**API Endpoint:**
```
https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{date}?format=json
```

**How it Works:**
1. For each date `D` in the range `[start_date, end_date]`:
   - Requests data for settlement date = `D`
   - Fetches data once per day
   - Extracts `systemSellPrice` and renames it to `imbalancePrice`

2. Data granularity: **30-minute intervals** (48 settlement periods per day)

3. Output columns:
   - `valueDateTimeOffset` - UTC timestamp
   - `settlementPeriod` - Settlement period number (1-48)
   - `imbalancePrice` - System sell price (£/MWh), representing imbalance settlement price

**Features:**
- ✅ HTTP retry logic (3 attempts with exponential backoff)
- ✅ Automatic duplicate removal
- ✅ Sorts by timestamp
- ✅ Robust error handling for missing data
- ✅ No API key required

**Join Keys:**
- Merges with Source 1 and Source 2 on: `['valueDateTimeOffset', 'settlementPeriod']`

**Example:**
```bash
# Fetch Source 1, 2, and 3
python data.py --start 2025-01-01 --end 2025-01-31 --sources 1 2 3 --output data/with_imbalance.csv
```

**Expected Output Size:**
- 1 day = 48 rows (30-min intervals)
- 1 week = ~336 rows
- 1 month = ~1,440 rows
- 1 year = ~17,520 rows

**API Call Volume:**
- 1 day = 1 API call
- 1 week = 7 API calls
- 1 month = 31 API calls
- 1 year = 365 API calls

**Performance Notes:**
⚠️ This source makes the **same number of API calls** as Source 1. For a 1-month date range:
- Source 1 alone: ~31 API calls, ~30 seconds
- Source 3 alone: ~31 API calls, ~30 seconds
- Source 1 + 3: ~62 API calls, ~60 seconds
- All sources (1 + 2 + 3): ~589 API calls, ~5-10 minutes (Source 2 dominates)

**Use Case:**
This source provides one component needed to calculate the **premium** target variable:
- `imbalancePrice` (from this source, Source 3)
- `spotPrice` (from IDA1 HH spot price, to be implemented)
- `premium = imbalancePrice - spotPrice` (calculated after both sources are loaded)

---

### Source 4: Elexon BMRS Loss of Load Probability & Derated Margin (LOLPDRM)

**Status:** ✅ Implemented

**Description:** Fetches Loss of Load Probability (LOLP) and Derated Margin data from the Elexon BMRS streaming API. These metrics indicate the risk of power shortages and the available generation margin in the system.

**API Endpoint:**
```
https://data.elexon.co.uk/bmrs/api/v1/datasets/LOLPDRM/stream
```

**How it Works:**
1. For each date `D` in the range `[start_date, end_date]`:
   - Requests data published between `D + T16:15:00Z` and `D + T16:45:00Z` (30-minute window)
   - Filters results to only include settlement date = `D + 1 day` (day-ahead forecast)
   - Takes the latest `publishTime` for each `settlementPeriod` (most recent forecast within the window)
   - Fetches data once per day

2. Data granularity: **30-minute intervals** (48 settlement periods per day)

3. Output columns:
   - `valueDateTimeOffset` - UTC timestamp
   - `settlementPeriod` - Settlement period number (1-48)
   - `lossOfLoadProbability` - Probability of loss of load (0-1 or percentage)
   - `deratedMargin` - Derated margin (MW) - available generation capacity after accounting for forced outages

**Features:**
- ✅ HTTP retry logic (3 attempts with exponential backoff)
- ✅ Automatic duplicate removal (takes latest publishTime per settlementPeriod)
- ✅ Sorts by timestamp
- ✅ Robust error handling for missing data
- ✅ No API key required

**Join Keys:**
- Merges with other sources on: `['valueDateTimeOffset', 'settlementPeriod']`

**Example:**
```bash
# Fetch Source 1 and Source 4
python data.py --start 2025-01-01 --end 2025-01-31 --sources 1 4 --output data/with_lolpdrm.csv

# Fetch all sources (1, 2, 3, 4)
python data.py --start 2025-01-01 --end 2025-01-31 --sources 1 2 3 4 --output data/full_dataset.csv
```

**Expected Output Size:**
- 1 day = 48 rows (30-min intervals)
- 1 week = ~336 rows
- 1 month = ~1,440 rows
- 1 year = ~17,520 rows

**API Call Volume:**
- 1 day = 1 API call
- 1 week = 7 API calls
- 1 month = 31 API calls
- 1 year = 365 API calls

**Performance Notes:**
⚠️ This source makes the **same number of API calls** as Source 1 and Source 3. For a 1-month date range:
- Source 4 alone: ~31 API calls, ~30 seconds
- Source 1 + 4: ~62 API calls, ~60 seconds
- All sources (1 + 2 + 3 + 4): ~620 API calls, ~5-10 minutes (Source 2 dominates)

**Use Case:**
These metrics provide insights into system reliability and capacity margins, which can be valuable features for forecasting:
- `lossOfLoadProbability` - Indicates system stress and potential shortage risk
- `deratedMargin` - Shows available generation capacity after forced outages

---

## Usage Examples

### Example 1: Fetch Test Data (2 days)

```bash
python data.py --start 2025-10-10 --end 2025-10-11 --output data/test.csv
```

**Output:**
- 96 rows (48 rows per day × 2 days)
- Columns: `valueDateTimeOffset`, `settlementPeriod`, `transmissionSystemDemand`, `nationalDemand`

---

### Example 2: Fetch One Month of Data

```bash
python data.py --start 2025-01-01 --end 2025-01-31 --output data/january_data.csv
```

**Output:**
- ~1,488 rows (31 days × 48 rows/day)
- Same column structure as above

---

### Example 3: Fetch Multiple Data Sources

```bash
# Fetch Source 1, 2, 3, and 4 for a month
python data.py --start 2025-01-01 --end 2025-01-31 --sources 1 2 3 4 --output data/combined_dataset.csv
```

**Output:**
- ~1,488 rows (31 days × 48 rows/day)
- 75 columns total:
  - 2 join keys: `valueDateTimeOffset`, `settlementPeriod`
  - 2 from Source 1: `transmissionSystemDemand`, `nationalDemand`
  - 68 from Source 2: All boundary metrics (B1-B17 × 4 metrics)
  - 1 from Source 3: `imbalancePrice`
  - 2 from Source 4: `lossOfLoadProbability`, `deratedMargin`

This will merge all specified sources on `valueDateTimeOffset` and `settlementPeriod`.

---

### Example 4: Validate Complete Dataset with Target

```bash
# For datasets that include a 'premium' target column
python data.py --start 2025-01-01 --end 2025-01-31 --output data/complete.csv --require-premium
```

---

## Data Processing Pipeline

The `data.py` script follows this processing pipeline:

```
1. Parse date range (--start and --end)
   ↓
2. Initialize DataBuilder with date range
   ↓
3. Load Source 1 (Elexon BMRS)
   - Loop through each date in range
   - Fetch data from API with retry logic
   - Filter to target settlement dates
   - Combine all results
   ↓
4. [Optional] Load additional sources
   - Merge on valueDateTimeOffset
   ↓
5. Clean data
   - Remove missing timestamps
   - Remove duplicates
   - Sort by timestamp
   ↓
6. Validate data
   - Check required columns exist
   - Check timestamp format (UTC)
   - Check for missing values
   ↓
7. Save to CSV
```

---

## Error Handling & Robustness

### HTTP Request Failures

The script includes robust retry logic:
- **3 retry attempts** for each failed request
- **Exponential backoff**: 1s, 2s, 4s delays
- Logs warnings for all failures
- Continues processing remaining dates even if some fail

### Missing or Malformed Data

- **Empty API responses**: Logs warning, continues to next date
- **Missing columns**: Logs warning, skips that date
- **JSON parse errors**: Retries, then skips if persistent
- **Duplicate records**: Automatically removed (keeps last occurrence)

### Date Filtering

- Only includes rows where `settlementDate == publish_date + 1 day`
- Logs warning if no matching rows found for a given publish date

---

## Troubleshooting

### Issue: No data returned

**Possible causes:**
1. Date range is in the future (API only has historical data)
2. API endpoint is down
3. Network connectivity issues

**Solution:**
- Check the log output for specific error messages
- Try a known working date range (e.g., recent past weeks)
- Verify internet connection

---

### Issue: Missing columns in output

**Possible causes:**
1. API response format changed
2. Specific dates have incomplete data

**Solution:**
- Check the log warnings for which dates failed
- Try a different date range
- Report the issue if persistent

---

### Issue: Duplicate timestamps

**Possible causes:**
1. API returned overlapping data for consecutive publish dates
2. Multiple requests for same settlement periods

**Solution:**
- The script automatically removes duplicates (keeps last)
- If issues persist, check the log for duplicate counts

---

## Integration with train_forecast.py

### Current Status: Features Only

The current dataset from `data.py` contains **features only** (no target variable). To use it with `train_forecast.py`, you need to add a `premium` column as the target variable.

### Future Integration

When the dataset includes all necessary features and a target column:

```bash
# 1. Build the dataset
python data.py --start 2025-01-01 --end 2025-03-31 --output data/q1_data.csv --require-premium

# 2. Train the forecasting model
python train_forecast.py data/q1_data.csv --max-lag 96 --n-trials 100
```

---

## Adding New Data Sources

To add a new data source (e.g., Source 2):

1. **Implement the loader method** in the `DataBuilder` class:
   ```python
   def load_source_2(self):
       """
       Load data from Source 2.

       Returns:
           pd.DataFrame with valueDateTimeOffset and feature columns
       """
       # Fetch data for self.start_date to self.end_date
       # Return DataFrame with valueDateTimeOffset column
       pass
   ```

2. **Update this documentation** with:
   - Source description
   - API endpoint / data location
   - Output columns
   - Usage examples
   - Expected data size

3. **Use the new source**:
   ```bash
   python data.py --start 2025-01-01 --end 2025-01-31 --sources 1 2 --output data/combined.csv
   ```

---

## Command Line Reference

### Full Command Syntax

```bash
python data.py --start YYYY-MM-DD --end YYYY-MM-DD [OPTIONS]
```

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--start` | string | Start date (format: YYYY-MM-DD, inclusive) |
| `--end` | string | End date (format: YYYY-MM-DD, inclusive) |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output` | string | `data/processed_data.csv` | Output file path |
| `--sources` | int [int ...] | `[1]` | List of data sources to load |
| `--require-premium` | flag | `False` | Validate that 'premium' column exists |

### Help

```bash
python data.py --help
```

---

## Data Quality Checks

The script performs the following validation:

### Automatic Checks

1. ✅ **Timestamp validation**
   - Must be datetime type
   - Must be timezone-aware (UTC)
   - No missing timestamps

2. ✅ **Duplicate detection**
   - Warns if duplicates found
   - Automatically removes duplicates

3. ✅ **Column validation**
   - Checks required columns exist
   - Checks data types are correct

4. ✅ **Sorting**
   - Sorts by timestamp (ascending)

### Optional Checks (with `--require-premium`)

5. ✅ **Target variable validation**
   - Premium column exists
   - Premium column is numeric
   - No missing premium values

---

## Version History

### Version 1.3 (Current)
- ✅ Implemented Source 1: Elexon BMRS Day-Ahead Demand
- ✅ Implemented Source 2: Elexon BMRS Indicated Imbalance by Boundary
- ✅ Implemented Source 3: Elexon BMRS Imbalance Settlement System Prices
- ✅ Implemented Source 4: Elexon BMRS Loss of Load Probability & Derated Margin (LOLPDRM)
- ✅ Global date range parameters (--start, --end)
- ✅ HTTP retry logic with exponential backoff
- ✅ Features-only mode (no premium column required)
- ✅ Robust error handling and logging
- ✅ Multi-column join support (valueDateTimeOffset + settlementPeriod)
- ✅ Automatic data reshaping (long to wide format for boundaries)
- ✅ Time window support for publish date filtering (Source 4)

### Planned Features
- 🔲 Source 5: IDA1 HH Spot Price (for premium calculation)
- 🔲 Automatic premium/target column generation (imbalancePrice - spotPrice)
- 🔲 Data caching for faster re-runs
- 🔲 Parallel API requests for faster fetching (especially for Source 2)

---

## Contact & Support

For issues, questions, or feature requests:
1. Check the **Troubleshooting** section above
2. Review log output for specific error messages
3. Verify your date range and command syntax

---

**Last Updated:** 2025-10-14
