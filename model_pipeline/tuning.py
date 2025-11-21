"""
Hyperparameter tuning for XGBoost using Optuna.

This script finds the best XGBoost parameters by testing different combinations
and evaluating them using time-series cross-validation to avoid data leakage.
"""

import optuna
import xgboost as xgb
import joblib
import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import torch
import warnings

from model_pipeline.data_prep import prepare_training_dataset

warnings.filterwarnings("ignore")

# Set to True to see detailed debugging info about data leakage
debug_mode = True

def generate_time_weights(df, time_col, min_weight=0.4):
    """
    Creates weights that give more importance to recent data.
    
    Older data gets a lower weight (starting at min_weight), and newer data
    gets progressively higher weights (up to 1.0). This helps the model
    focus on recent patterns which are more relevant for stock prediction.
    
    Args:
        df: DataFrame with time column
        time_col: Name of the time column
        min_weight: Minimum weight for oldest data (default 0.4)
    
    Returns:
        Array of weights, one per row
    """
    dates = pd.to_datetime(df[time_col])
    start_date = dates.min()
    end_date = dates.max()

    # Calculate how long the entire dataset spans
    total_duration = (end_date - start_date).total_seconds()

    # For each date, calculate how far through the timeline it is (0.0 to 1.0)
    time_progress = (dates - start_date).dt.total_seconds() / total_duration

    # Convert progress to weights: older data gets min_weight, newer gets 1.0
    weights = min_weight + (1.0 - min_weight) * time_progress

    return weights.values

def objective(trial, X, y, time_col, ratio, weights_all):
    """
    This function is called by Optuna for each trial (parameter combination).
    It trains a model, evaluates it using cross-validation, and returns the score.
    
    Args:
        trial: Optuna trial object that suggests parameter values
        X: Feature matrix
        y: Target vector
        time_col: Time column for proper time-based splitting
        ratio: Class imbalance ratio (for handling unbalanced data)
        weights_all: Time-based weights for all samples
    
    Returns:
        Average F1 score across all CV folds
    """
    # Check if we can use GPU (makes training much faster)
    try:
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    # Build parameter dictionary
    # Optuna will try different values for the 'suggest_*' parameters
    params = {
        'tree_method': 'hist',  # Fast histogram-based method
        'device': 'cuda' if has_cuda else 'cpu',
        'objective': 'binary:logistic',  # Binary classification
        'eval_metric': 'logloss',  # Metric to optimize during training
        'random_state': 42,  # For reproducibility
        'verbosity': 0,  # Suppress output
        'scale_pos_weight': ratio,  # Handle class imbalance (more 0s than 1s)
        # These are the parameters Optuna will optimize:
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # Number of trees
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),  # How fast to learn
        'max_depth': trial.suggest_int('max_depth', 3, 12),  # How deep trees can grow
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),  # Minimum samples in leaf
        'gamma': trial.suggest_float('gamma', 0.1, 5.0),  # Minimum loss reduction to split
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Fraction of samples per tree
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Fraction of features per tree
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),  # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),  # L2 regularization
    }

    # Use time-series cross-validation to avoid data leakage
    # This splits data by time periods, not randomly, so we never train on future data
    tscv = TimeSeriesSplit(n_splits=5, gap=2)  # 5 folds
    scores = []
    
    # Get all unique dates and sort them
    # We split by dates, not rows, because multiple stocks share the same dates
    # This prevents leakage when stocks are interleaved in the dataset
    unique_times = time_col.unique()
    unique_times = pd.Series(unique_times).sort_values()
    
    # Need at least 4 unique dates to do 3-fold CV (each fold needs some dates)
    if len(unique_times) < 4:
        return 0.0
    
    # Convert dates to the format TimeSeriesSplit expects (2D array)
    unique_times_array = unique_times.values.reshape(-1, 1)
    date_splits = list(tscv.split(unique_times_array))
    
    # Evaluate on each fold
    for train_date_idx, test_date_idx in date_splits:
        # Get which dates are in training and which are in testing
        train_dates = unique_times.iloc[train_date_idx]
        test_dates = unique_times.iloc[test_date_idx]
        
        # Find all rows (across all stocks) that match these dates
        train_mask = time_col.isin(train_dates).values
        test_mask = time_col.isin(test_dates).values
        
        # Split the data using these masks
        X_train_fold_raw = X[train_mask]
        X_test_fold_raw = X[test_mask]
        y_train_fold = y[train_mask]
        y_test_fold = y[test_mask]

        # Get the time weights for training samples only
        weights_train = weights_all[train_mask]
        
        # Skip this fold if it's too small (not enough data to learn/test)
        if len(X_train_fold_raw) < 10 or len(X_test_fold_raw) < 10:
            continue

        # If we fit on the whole dataset, we'd leak test set statistics into training
        scaler_fold = StandardScaler()
        X_train_fold = scaler_fold.fit_transform(X_train_fold_raw)
        X_test_fold = scaler_fold.transform(X_test_fold_raw)  # Use training stats to transform test

        # Train the model with the suggested parameters
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_fold, y_train_fold, sample_weight=weights_train)

        # Make predictions on the test fold
        preds = model.predict(X_test_fold)

        # Count how many "buy" predictions we made
        n_trades = sum(preds)

        # If we predicted "don't buy" for everything, F1 score would be undefined
        # So we give it a score of 0 instead
        if n_trades == 0:
            scores.append(0.0)
        else:
            # Calculate F1 score (balance between precision and recall)
            score = f1_score(y_test_fold, preds, zero_division=0)
            scores.append(score)

    # Return the average score across all folds
    return sum(scores) / len(scores) if scores else 0.0

if __name__ == '__main__':
    print("Loading data for tuning...")
    
    # Load the full training dataset
    # target_threshold=0.0 means any positive return counts as "up"
    df = prepare_training_dataset(limit=None, target_threshold=0.0)
    if df.empty:
        print("No data to tune. Run feature generator first.")
        exit()

    # Save the time column - we need it for proper time-based splitting
    time_col_local = df['time'].copy()
    
    # Split into features (X) and target (y)
    # Drop time and symbol_id since they're not features, and Target since it's our target
    X = df.drop(columns=['time', 'symbol_id', 'Target']).fillna(0)
    y = df['Target'].astype(int)

    # Calculate class imbalance ratio
    # If we have 10x more "down" days than "up" days, ratio = 10
    # This helps XGBoost handle the imbalance
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)

    # Generate weights that favor recent data
    # Recent patterns are more relevant for predicting tomorrow's price
    print("Generating time-decay weights...")
    time_weights = generate_time_weights(df, 'time', min_weight=0.5)

    # Debug mode shows detailed info about the data and potential issues
    if debug_mode:
        print("\n=== DEBUGGING DATA LEAKAGE ===")
        print(f"Total rows: {len(df)}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        print(f"Target percentage (class 1): {y.mean():.4f}")

        # Highly imbalanced data can cause misleadingly high scores
        # If 99% of days are "down", predicting "down" always gives 99% accuracy
        if y.mean() > 0.9 or y.mean() < 0.1:
            print(f"WARNING: Highly imbalanced target! This could explain high precision.")
            print(f"Consider adjusting target_threshold in prepare_training_dataset()")

        # Show basic stats about the dataset
        print(f"Time range: {df['time'].min()} to {df['time'].max()}")
        print(f"Number of unique dates: {df['time'].nunique()}")
        print(f"Number of unique symbols: {df['symbol_id'].nunique()}")

        # Check if any features are too correlated with the target
        # Features with >0.9 correlation might be leaking future information
        print("\nChecking for suspicious feature correlations...")
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        high_corr = correlations[correlations > 0.9]
        if len(high_corr) > 0:
            print(f"WARNING: Found {len(high_corr)} features with >0.9 correlation with target:")
            print(high_corr.head(10))
            print("These features might be leaking information!")
        else:
            print(f"Highest correlation: {correlations.max():.4f} (feature: {correlations.idxmax()})")

        # Verify that our time-based splitting works correctly
        # Make sure training dates always come before test dates
        print("\nChecking time-based split behavior...")
        unique_times = time_col_local.unique()
        unique_times = pd.Series(unique_times).sort_values()
        tscv = TimeSeriesSplit(n_splits=3)
        date_splits = list(tscv.split(unique_times.values.reshape(-1, 1)))

        for i, (train_date_idx, test_date_idx) in enumerate(date_splits):
            train_dates = unique_times.iloc[train_date_idx]
            test_dates = unique_times.iloc[test_date_idx]
            train_mask = time_col_local.isin(train_dates)
            test_mask = time_col_local.isin(test_dates)

            train_times = time_col_local[train_mask]
            test_times = time_col_local[test_mask]

            print(f"Fold {i+1}:")
            print(f"Train: {train_times.min()} to {train_times.max()} ({len(train_times)} rows)")
            print(f"Test:  {test_times.min()} to {test_times.max()} ({len(test_times)} rows)")
            
            # If the latest training date >= earliest test date, we have leakage!
            if train_times.max() >= test_times.min():
                print(f"ERROR: Time leakage detected! Train max >= Test min")
            else:
                print(f"No time leakage")

        print("=== END DEBUG ===\n")

    # Decide how many parameter combinations to try
    # Fewer trials for small datasets (faster), more for large datasets (better results)
    N_trials = 20 if len(X) < 5000 else 50
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Starting Optuna optimization for {len(X)} rows with {N_trials} trials...")
    
    # Create an Optuna study that will try to maximize F1 score
    study = optuna.create_study(direction='maximize')
    
    # Run the optimization with a progress bar
    with tqdm(total=N_trials, desc="Tuning XGBoost") as pbar:
        def tqdm_callback(study, trial):
            # Update progress bar after each trial
            pbar.update(1)
            if study.best_value:
                # Show the best F1 score found so far
                pbar.set_description(f"Best F1: {study.best_value:.4f}")

        # Start the optimization
        # For each trial, Optuna suggests parameters and calls objective() to test them
        study.optimize(
            lambda trial: objective(trial, X, y, time_col_local, ratio, time_weights),
            n_trials=N_trials,
            callbacks=[tqdm_callback]
        )

    # Save the best parameters found
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(study.best_params, f"models/best_xgboost_params.pkl")
    print("Parameters saved to models/best_xgboost_params.pkl")