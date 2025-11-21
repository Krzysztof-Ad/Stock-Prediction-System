"""
Stacking Ensemble Training

This module trains a stacking ensemble model that combines multiple base models
(XGBoost, LightGBM, CatBoost, Random Forest) using a meta-learner to make final predictions.
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, confusion_matrix, f1_score

from model_pipeline.data_prep import prepare_training_dataset
from model_pipeline.architecture import get_base_models, get_meta_learner


def generate_time_weights(df, time_col, min_weight=0.5):
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

def train_stacking_ensemble(limit=None):
    """
    Trains a stacking ensemble model that combines multiple base models.
    
    The stacking approach works like this:
    1. Train several base models (XGBoost, LightGBM, etc.) on the data
    2. Use these models to make predictions
    3. Train a meta-learner to combine these predictions into a final prediction
    
    This often works better than any single model alone.
    
    Args:
        limit: Optional limit on number of rows (useful for testing).
               Set to None to use all available data.
    """
    print('\n=== TRAINING STACKING ENSEMBLE ===')
    print("Fetching and preparing data...")
    if limit:
        print(f"Using limited dataset: {limit} rows (for testing)")
    
    # Load the training dataset
    # target_threshold=0.0 means any positive return counts as "up"
    df = prepare_training_dataset(limit=limit, target_threshold=0.0)
    if df.empty:
        print('No data available')
        return

    # Calculate class imbalance ratio
    # If we have 10x more "down" days than "up" days, ratio = 10
    # This helps models handle the imbalance
    y_full = df['Target'].astype(int)
    ratio = float(np.sum(y_full == 0)) / np.sum(y_full == 1)
    print(f"Class Imbalance Ratio: {ratio:.2f} (Scale_pos_weight)")

    # Generate weights that favor recent data
    # Recent patterns are more relevant for predicting tomorrow's price
    print("Generating time-decay weights...")
    weights_full = generate_time_weights(df, 'time', min_weight=0.5)

    # Split into features (X) and target (y)
    times = df['time'].copy()  # Keep time for verification
    X = df.drop(columns=['time', 'symbol_id', 'Target']).fillna(0)
    y = df['Target'].astype(int)
    print(f"Data loaded: {X.shape[0]} rows, {X.shape[1]} features.")

    # Split into training (85%) and test (15%) sets
    # We split by time, not randomly, to avoid data leakage
    print("Dividing dataset into training and test sets...")
    test_size = int(len(df) * 0.15)
    train_size = len(df) - test_size

    # Take first 85% for training, last 15% for testing
    # Since data is sorted by time, this ensures no future data in training
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    weights_train = weights_full[:train_size]

    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    # Verify there's no time leakage
    # Training should end before testing begins
    max_train_time = times.iloc[train_size - 1]
    min_test_time = times.iloc[train_size]
    print(f"Time Split: Train ends {max_train_time} | Test starts {min_test_time}")
    if max_train_time >= min_test_time:
        raise ValueError("CRITICAL: Time leakage detected in manual split!")

    print(f"Training: {len(X_train)} rows.")
    print(f"Testing: {len(X_test)} rows.")

    # Scale features to have mean=0 and std=1
    # This helps models train faster and perform better
    # CRITICAL: Fit scaler only on training data to avoid leakage
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Learn scaling from training data
    X_test_scaled = scaler.transform(X_test)  # Apply same scaling to test data

    # Convert back to DataFrames (easier to work with)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Get the base models (XGBoost, LightGBM, CatBoost, Random Forest)
    print("Initializing models...")
    base_models_dict = get_base_models(ratio=ratio, use_gpu=True)
    estimators_list = list(base_models_dict.items())  # Convert to list format
    
    # Get the meta-learner (combines base model predictions)
    meta_learner = get_meta_learner()

    # Use time-series cross-validation for stacking
    # This ensures the meta-learner doesn't see future data when learning
    # gap=2 means leave 2 time periods between train and test (extra safety)
    cv_strategy = TimeSeriesSplit(n_splits=5, gap=2)

    # Create the stacking ensemble
    # It will train base models, get their predictions, then train meta-learner
    stacking_model = StackingClassifier(
        estimators=estimators_list,  # The base models
        final_estimator=meta_learner,  # The model that combines predictions
        cv=cv_strategy,  # How to split data for meta-learner training
        n_jobs=1,  # Number of parallel jobs (1 = sequential)
        passthrough=False,  # Don't include original features in meta-learner
        verbose=1  # Show progress
    )

    # Train the entire stacking ensemble
    # This trains all base models and the meta-learner
    print(f"Starting training on {len(X_train)} rows with sample_weights...")
    start_time = time.time()

    stacking_model.fit(X_train_scaled, y_train, sample_weight=weights_train)

    duration = time.time() - start_time
    print(f"Training took {duration:.2f} seconds.")

    # Evaluate the model on the test set (unseen data)
    print("\n=== FINAL EVALUATION (Test Set) ===")
    y_pred = stacking_model.predict(X_test_scaled)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)  # Overall accuracy
    f1 = f1_score(y_test, y_pred, zero_division=0)  # F1 score (balance of precision/recall)

    # Calculate baseline: what if we just always predicted the majority class?
    # This is the "dumb" strategy we need to beat
    baseline_buy_hold = sum(y_test) / len(y_test)  # Percentage of "up" days
    baseline_acc = max(baseline_buy_hold, 1 - baseline_buy_hold)  # Best we can do by always guessing one class

    # Print results
    print(f"Test Period: {min_test_time} to {times.iloc[-1]}")
    print("-" * 40)
    print(f"STACKING ACCURACY: {acc * 100:.2f}%")
    print(f"MARKET BASELINE:   {baseline_acc * 100:.2f}%")
    print(f"STACKING F1-SCORE: {f1:.4f}")
    print("-" * 40)

    # Compare to baseline
    if acc > baseline_acc:
        print("RESULT: The Ensemble beats the market!")
    else:
        print("RESULT: Strategy underperforms baseline.")

    # Show detailed performance metrics
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model and scaler for later use
    print("Saving model...")
    if not os.path.exists('models'):
        os.makedirs('models')

    joblib.dump(stacking_model, 'models/stacking_model_v1.joblib')
    joblib.dump(scaler, 'models/scaler_v1.joblib')  # Need to save scaler to transform new data

    print("Done. Models saved.")

if __name__ == '__main__':
    train_stacking_ensemble(limit=None)