#!/usr/bin/env python3
"""
Enhanced Time Series Forecasting with Advanced Models and Features
Supports Ridge, Lasso, LightGBM, XGBoost, CatBoost, and Ensemble methods
"""

import pandas as pd
import numpy as np
import argparse
import warnings
from pathlib import Path
import json
import joblib
from datetime import datetime, timezone
import pytz

# ML imports
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class EnhancedTimeSeriesForecaster:
    def __init__(self, max_lag=96, freq=None, n_trials=100, cv_folds=3):
        self.max_lag = max_lag
        self.freq = freq
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.lag_set = [1, 2, 3, 6, 12, 24, 48, 96]
        self.lag_set = [lag for lag in self.lag_set if lag <= max_lag]
        self.feature_columns = []
        self.target_col = 'premium'
        self.timestamp_col = 'valueDateTimeOffset'
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        self.best_model_name = None
        self.ensemble_model = None

    def load_data(self, csv_path):
        """Load and initial processing of data"""
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)

        # Parse timestamps to UTC
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], utc=True)

        # Sort by time
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)

        # Drop missing timestamp or target
        initial_rows = len(df)
        df = df.dropna(subset=[self.timestamp_col, self.target_col])
        print(f"Dropped {initial_rows - len(df)} rows with missing timestamp/target")

        # Drop duplicate timestamps (keep last)
        df = df.drop_duplicates(subset=[self.timestamp_col], keep='last')
        print(f"Final dataset shape: {df.shape}")

        return df

    def regularize_frequency(self, df):
        """Regularize to a specific frequency if requested"""
        if self.freq:
            print(f"Regularizing to frequency: {self.freq}")
            df_indexed = df.set_index(self.timestamp_col)

            # Create regular grid
            start_time = df_indexed.index.min()
            end_time = df_indexed.index.max()
            regular_index = pd.date_range(start=start_time, end=end_time, freq=self.freq, tz=timezone.utc)

            # Reindex and forward fill
            df_regular = df_indexed.reindex(regular_index, method='ffill')
            df = df_regular.reset_index().rename(columns={'index': self.timestamp_col})

        return df

    def remove_outliers(self, df, std_threshold=3):
        """Remove outliers from target variable"""
        mean_val = df[self.target_col].mean()
        std_val = df[self.target_col].std()

        outlier_mask = np.abs(df[self.target_col] - mean_val) > (std_threshold * std_val)
        outliers_removed = outlier_mask.sum()

        if outliers_removed > 0:
            print(f"Removing {outliers_removed} outliers (>{std_threshold} std from mean)")
            df = df[~outlier_mask].reset_index(drop=True)

        return df

    def create_fourier_features(self, df, periods=[24, 168, 8760]):  # hourly, weekly, yearly
        """Create Fourier features for periodic patterns"""
        fourier_features = pd.DataFrame(index=df.index)

        # Create time index in hours from start
        time_idx = (df[self.timestamp_col] - df[self.timestamp_col].min()).dt.total_seconds() / 3600

        for period in periods:
            for k in range(1, 3):  # First 2 harmonics
                fourier_features[f'fourier_sin_{period}_{k}'] = np.sin(2 * np.pi * k * time_idx / period)
                fourier_features[f'fourier_cos_{period}_{k}'] = np.cos(2 * np.pi * k * time_idx / period)

        return fourier_features

    def create_rolling_features(self, df, windows=[6, 12, 24, 48]):
        """Create rolling statistical features"""
        rolling_features = pd.DataFrame(index=df.index)

        for col in [self.target_col] + self.feature_columns:
            for window in windows:
                if window <= len(df):
                    rolling_features[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                    rolling_features[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                    rolling_features[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                    rolling_features[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()

        return rolling_features

    def create_seasonal_features(self, df):
        """Create seasonal decomposition features"""
        seasonal_features = pd.DataFrame(index=df.index)

        # Only if we have enough data points
        if len(df) >= 48:  # At least 2 days for hourly data
            try:
                # Use a reasonable period for decomposition
                period = min(24, len(df) // 4)  # Daily cycle or quarter of data
                decomposition = seasonal_decompose(df[self.target_col].fillna(method='ffill'),
                                                 model='additive', period=period, extrapolate_trend='freq')

                seasonal_features[f'{self.target_col}_trend'] = decomposition.trend
                seasonal_features[f'{self.target_col}_seasonal'] = decomposition.seasonal
                seasonal_features[f'{self.target_col}_residual'] = decomposition.resid
            except:
                print("Warning: Could not create seasonal decomposition features")

        return seasonal_features

    def create_lag_features(self, df):
        """Create lag features for target and regressors"""
        feature_df = df[[self.timestamp_col, self.target_col] + self.feature_columns].copy()

        # Standard lag features
        for lag in self.lag_set:
            # Target lags
            feature_df[f'{self.target_col}_lag_{lag}'] = feature_df[self.target_col].shift(lag)

            # Regressor lags
            for col in self.feature_columns:
                feature_df[f'{col}_lag_{lag}'] = feature_df[col].shift(lag)

        return feature_df

    def create_interaction_features(self, df, max_interactions=10):
        """Create cross-lag interaction features"""
        interaction_features = pd.DataFrame(index=df.index)

        # Get lag columns
        target_lags = [col for col in df.columns if col.startswith(f'{self.target_col}_lag_')]
        regressor_lags = [col for col in df.columns if '_lag_' in col and not col.startswith(f'{self.target_col}_lag_')]

        # Create interactions between target lags and regressor lags
        interaction_count = 0
        for target_lag in target_lags[:5]:  # Limit to first 5 target lags
            for reg_lag in regressor_lags:
                if interaction_count >= max_interactions:
                    break
                interaction_features[f'{target_lag}_x_{reg_lag}'] = df[target_lag] * df[reg_lag]
                interaction_count += 1
            if interaction_count >= max_interactions:
                break

        return interaction_features

    def create_calendar_features(self, df):
        """Create calendar-based features"""
        calendar_features = pd.DataFrame(index=df.index)

        dt = df[self.timestamp_col]
        calendar_features['hour'] = dt.dt.hour
        calendar_features['dow'] = dt.dt.dayofweek  # 0=Monday
        calendar_features['dom'] = dt.dt.day
        calendar_features['month'] = dt.dt.month
        calendar_features['quarter'] = dt.dt.quarter
        calendar_features['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        calendar_features['is_month_start'] = dt.dt.is_month_start.astype(int)
        calendar_features['is_month_end'] = dt.dt.is_month_end.astype(int)

        return calendar_features

    def feature_engineering(self, df):
        """Complete feature engineering pipeline"""
        print("Starting feature engineering...")

        # Identify regressor columns (all except timestamp and target)
        self.feature_columns = [col for col in df.columns if col not in [self.timestamp_col, self.target_col]]
        print(f"Found {len(self.feature_columns)} regressor columns")

        # Create lag features
        print("Creating lag features...")
        feature_df = self.create_lag_features(df)

        # Create rolling features
        print("Creating rolling statistical features...")
        rolling_features = self.create_rolling_features(df)
        feature_df = pd.concat([feature_df, rolling_features], axis=1)

        # Create Fourier features
        print("Creating Fourier features...")
        fourier_features = self.create_fourier_features(df)
        feature_df = pd.concat([feature_df, fourier_features], axis=1)

        # Create seasonal features
        print("Creating seasonal decomposition features...")
        seasonal_features = self.create_seasonal_features(df)
        if not seasonal_features.empty:
            feature_df = pd.concat([feature_df, seasonal_features], axis=1)

        # Create calendar features
        print("Creating calendar features...")
        calendar_features = self.create_calendar_features(df)
        feature_df = pd.concat([feature_df, calendar_features], axis=1)

        # Create interaction features
        print("Creating interaction features...")
        interaction_features = self.create_interaction_features(feature_df)
        if not interaction_features.empty:
            feature_df = pd.concat([feature_df, interaction_features], axis=1)

        # Drop rows with NaNs caused by shifting
        initial_rows = len(feature_df)
        feature_df = feature_df.dropna()
        print(f"Dropped {initial_rows - len(feature_df)} rows due to NaN values from feature creation")

        print(f"Final feature matrix shape: {feature_df.shape}")
        return feature_df

    def expanding_window_split(self, df, min_train_size=0.6):
        """Create expanding window splits for time series validation"""
        n = len(df)
        min_train = int(n * min_train_size)

        splits = []
        # Create multiple expanding windows
        for i in range(self.cv_folds):
            train_end = min_train + int((n - min_train) * (i + 1) / (self.cv_folds + 1))
            test_start = train_end
            test_end = min_train + int((n - min_train) * (i + 2) / (self.cv_folds + 1))

            if test_end <= n:
                train_idx = np.arange(0, train_end)
                test_idx = np.arange(test_start, min(test_end, n))
                if len(test_idx) > 0:
                    splits.append((train_idx, test_idx))

        return splits

    def optimize_hyperparameters(self, X_train, y_train, model_type):
        """Optimize hyperparameters using Optuna"""
        print(f"Optimizing hyperparameters for {model_type}...")

        def objective(trial):
            if model_type == 'ridge':
                alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
                model = Ridge(alpha=alpha, random_state=42)

            elif model_type == 'lasso':
                alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
                model = Lasso(alpha=alpha, random_state=42, max_iter=2000)

            elif model_type == 'lightgbm':
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                }
                model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)

            elif model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                }
                model = xgb.XGBRegressor(**params, random_state=42, verbose=0)

            elif model_type == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 500),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                }
                model = cb.CatBoostRegressor(**params, random_state=42, verbose=False)

            # Cross-validation with expanding windows
            splits = self.expanding_window_split(pd.DataFrame(X_train))
            scores = []

            for train_idx, val_idx in splits:
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Apply preprocessing for linear models
                if model_type in ['ridge', 'lasso']:
                    # Use temporary scalers for CV
                    temp_scaler = StandardScaler()
                    temp_imputer = SimpleImputer(strategy='median')

                    X_tr_processed = temp_scaler.fit_transform(temp_imputer.fit_transform(X_tr))
                    X_val_processed = temp_scaler.transform(temp_imputer.transform(X_val))
                else:
                    temp_imputer = SimpleImputer(strategy='median')
                    X_tr_processed = temp_imputer.fit_transform(X_tr)
                    X_val_processed = temp_imputer.transform(X_val)

                model.fit(X_tr_processed, y_tr)
                y_pred = model.predict(X_val_processed)
                mae = mean_absolute_error(y_val, y_pred)
                scores.append(mae)

            return np.mean(scores)

        # Run optimization with reduced trials for faster execution
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=min(self.n_trials, 50))  # Limit trials

        return study.best_params

    def train_models(self, X_train, y_train):
        """Train all models with hyperparameter optimization"""
        print("Training models with hyperparameter optimization...")

        model_configs = {
            'ridge': Ridge,
            'lasso': Lasso,
            'lightgbm': lgb.LGBMRegressor,
            'xgboost': xgb.XGBRegressor,
            'catboost': cb.CatBoostRegressor
        }

        for model_name, model_class in model_configs.items():
            print(f"\nTraining {model_name}...")

            # Optimize hyperparameters
            best_params = self.optimize_hyperparameters(X_train, y_train, model_name)
            print(f"Best parameters for {model_name}: {best_params}")

            # Prepare data based on model type
            if model_name in ['ridge', 'lasso']:
                # Linear models need scaling and imputation
                imputer = SimpleImputer(strategy='median')
                scaler = StandardScaler()

                X_processed = scaler.fit_transform(imputer.fit_transform(X_train))

                self.imputers[model_name] = imputer
                self.scalers[model_name] = scaler

                if model_name == 'ridge':
                    model = Ridge(**best_params, random_state=42)
                else:  # lasso
                    model = Lasso(**best_params, random_state=42, max_iter=2000)

            else:
                # Tree-based models only need imputation
                imputer = SimpleImputer(strategy='median')
                X_processed = imputer.fit_transform(X_train)

                self.imputers[model_name] = imputer

                if model_name == 'lightgbm':
                    model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
                elif model_name == 'xgboost':
                    model = xgb.XGBRegressor(**best_params, random_state=42, verbose=0)
                else:  # catboost
                    model = cb.CatBoostRegressor(**best_params, random_state=42, verbose=False)

            # Train the model
            model.fit(X_processed, y_train)
            self.models[model_name] = model
            print(f"{model_name} training completed")

    def create_ensemble(self, X_train, y_train):
        """Create ensemble model from individual models"""
        print("Creating ensemble model...")

        # Create voting regressor with all models
        estimators = []
        for name, model in self.models.items():
            estimators.append((name, model))

        # We can't directly use VotingRegressor due to different preprocessing needs
        # Instead, we'll create a custom ensemble that averages predictions
        self.ensemble_model = 'weighted_average'  # Flag for custom ensemble
        print("Ensemble model created using weighted average")

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and select the best one"""
        print("\nEvaluating models...")
        results = {}

        for model_name, model in self.models.items():
            # Apply same preprocessing as training
            if model_name in ['ridge', 'lasso']:
                X_processed = self.scalers[model_name].transform(
                    self.imputers[model_name].transform(X_test)
                )
            else:
                X_processed = self.imputers[model_name].transform(X_test)

            # Make predictions
            y_pred = model.predict(X_processed)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results[model_name] = {
                'mae': mae,
                'rmse': rmse,
                'predictions': y_pred
            }

            print(f"{model_name:10s} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        # Evaluate ensemble
        ensemble_pred = np.zeros(len(y_test))
        for model_name in self.models.keys():
            ensemble_pred += results[model_name]['predictions']
        ensemble_pred /= len(self.models)

        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))

        results['ensemble'] = {
            'mae': ensemble_mae,
            'rmse': ensemble_rmse,
            'predictions': ensemble_pred
        }

        print(f"{'ensemble':10s} | MAE: {ensemble_mae:.4f} | RMSE: {ensemble_rmse:.4f}")

        # Select best model by MAE
        best_model = min(results.keys(), key=lambda x: results[x]['mae'])
        self.best_model_name = best_model

        print(f"\nBest model: {best_model} (MAE: {results[best_model]['mae']:.4f})")

        return results

    def create_prediction_chart(self, timestamps, y_true, y_pred, save_path='forecast_chart.png'):
        """Create and save prediction vs actual chart"""
        plt.figure(figsize=(15, 8))

        plt.plot(timestamps, y_true, label='Actual', alpha=0.7, linewidth=1)
        plt.plot(timestamps, y_pred, label='Predicted', alpha=0.8, linewidth=1)

        plt.title('Actual vs Predicted Values', fontsize=16, pad=20)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Premium', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # Improve layout
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save chart
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Prediction chart saved to {save_path}")

    def refit_best_model(self, X_full, y_full):
        """Refit the best model on full dataset"""
        print(f"\nRefitting {self.best_model_name} on full dataset...")

        if self.best_model_name == 'ensemble':
            # Refit all models for ensemble
            for model_name, model in self.models.items():
                if model_name in ['ridge', 'lasso']:
                    X_processed = self.scalers[model_name].fit_transform(
                        self.imputers[model_name].fit_transform(X_full)
                    )
                else:
                    X_processed = self.imputers[model_name].fit_transform(X_full)

                model.fit(X_processed, y_full)
        else:
            # Refit single best model
            if self.best_model_name in ['ridge', 'lasso']:
                X_processed = self.scalers[self.best_model_name].fit_transform(
                    self.imputers[self.best_model_name].fit_transform(X_full)
                )
            else:
                X_processed = self.imputers[self.best_model_name].fit_transform(X_full)

            self.models[self.best_model_name].fit(X_processed, y_full)

        print("Model refitting completed")

    def save_model(self, filepath='model.pkl'):
        """Save the complete model bundle"""
        model_bundle = {
            'best_model_name': self.best_model_name,
            'models': self.models,
            'scalers': self.scalers,
            'imputers': self.imputers,
            'feature_columns': self.feature_columns,
            'lag_set': self.lag_set,
            'max_lag': self.max_lag,
            'freq': self.freq,
            'target_col': self.target_col,
            'timestamp_col': self.timestamp_col,
            'ensemble_model': self.ensemble_model
        }

        joblib.dump(model_bundle, filepath)
        print(f"Model bundle saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Time Series Forecasting')
    parser.add_argument('csv_path', help='Path to CSV file with time series data')
    parser.add_argument('--max-lag', type=int, default=96, help='Maximum lag for features')
    parser.add_argument('--freq', type=str, help='Frequency for regularization (e.g., "H", "D")')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of hyperparameter optimization trials')
    parser.add_argument('--cv-folds', type=int, default=3, help='Number of cross-validation folds')
    parser.add_argument('--model-path', type=str, default='model.pkl', help='Path to save model')
    parser.add_argument('--forecast-path', type=str, default='forecast.csv', help='Path to save forecasts')
    parser.add_argument('--metrics-path', type=str, default='metrics.json', help='Path to save metrics')
    parser.add_argument('--chart-path', type=str, default='forecast_chart.png', help='Path to save chart')

    args = parser.parse_args()

    # Initialize forecaster
    forecaster = EnhancedTimeSeriesForecaster(
        max_lag=args.max_lag,
        freq=args.freq,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds
    )

    # Load and process data
    df = forecaster.load_data(args.csv_path)
    df = forecaster.regularize_frequency(df)
    df = forecaster.remove_outliers(df)

    # Feature engineering
    feature_df = forecaster.feature_engineering(df)

    # Prepare features and target
    feature_cols = [col for col in feature_df.columns
                   if col not in [forecaster.timestamp_col, forecaster.target_col]]

    X = feature_df[feature_cols]
    y = feature_df[forecaster.target_col]
    timestamps = feature_df[forecaster.timestamp_col]

    # Time-based split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    timestamps_test = timestamps.iloc[split_idx:]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train models
    forecaster.train_models(X_train, y_train)

    # Create ensemble
    forecaster.create_ensemble(X_train, y_train)

    # Evaluate models
    results = forecaster.evaluate_models(X_test, y_test)

    # Create prediction chart
    best_predictions = results[forecaster.best_model_name]['predictions']
    forecaster.create_prediction_chart(timestamps_test, y_test, best_predictions, args.chart_path)

    # Refit best model on full data
    forecaster.refit_best_model(X, y)

    # Save results
    forecaster.save_model(args.model_path)

    # Save forecast results
    forecast_df = pd.DataFrame({
        'timestamp': timestamps_test,
        'y_true': y_test,
        'y_pred': best_predictions
    })
    forecast_df.to_csv(args.forecast_path, index=False)
    print(f"Forecast results saved to {args.forecast_path}")

    # Save metrics
    metrics = {model: {'mae': results[model]['mae'], 'rmse': results[model]['rmse']}
              for model in results.keys()}

    with open(args.metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {args.metrics_path}")

    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()