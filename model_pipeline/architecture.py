"""
Model Architecture Definitions

This module defines the base models and meta-learner used in the stacking ensemble.
The base models make predictions, and the meta-learner combines their predictions.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import os

def get_base_models(ratio=1.0, use_gpu=True):
    """
    Creates a dictionary of base models for the stacking ensemble.
    
    These models will make predictions independently, and then a meta-learner
    will learn how to best combine their predictions.
    
    Args:
        ratio: Class imbalance ratio (for handling unbalanced data)
        use_gpu: Whether to use GPU acceleration if available
    
    Returns:
        Dictionary of model name -> model instance
    """
    models = {}

    # Start with default XGBoost parameters
    # These are safe defaults that work well for most cases
    xgb_params = {
        'n_estimators': 1000,  # Number of trees
        'learning_rate': 0.05,  # How fast the model learns
        'max_depth': 6,  # How deep trees can grow
        'subsample': 0.8,  # Use 80% of samples per tree (prevents overfitting)
        'colsample_bytree': 0.8,  # Use 80% of features per tree
        'eval_metric': 'logloss',  # Metric to optimize
        'random_state': 42,  # For reproducibility
        'scale_pos_weight': ratio  # Handle class imbalance
    }

    # Try to load optimized parameters from hyperparameter tuning
    # If we've run tuning.py, it saves the best parameters here
    params_path = 'models/best_xgboost_params.pkl'
    if os.path.exists(params_path):
        print(f"[XGBoost] Found tuning file! Loading parameters from {params_path}...")
        best_params = joblib.load(params_path)
        
        # Remove scale_pos_weight from tuned params (we set it separately based on data)
        if 'scale_pos_weight' in best_params:
            del best_params['scale_pos_weight']
        
        # Update defaults with tuned parameters
        xgb_params.update(best_params)
    else:
        print("[XGBoost] No tuning files found. Using safe defaults.")

    # Set hardware-specific options
    xgb_params['tree_method'] = 'hist'  # Fast histogram-based method
    xgb_params['device'] = 'cuda' if use_gpu else 'cpu'

    models['xgboost'] = XGBClassifier(**xgb_params)

    # LightGBM - Another gradient boosting algorithm, often faster than XGBoost
    models['lightgbm'] = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=31,  # Number of leaves in each tree
        random_state=42,
        scale_pos_weight=ratio,  # Handle class imbalance
        device='gpu' if use_gpu else 'cpu',
        verbose=-1  # Suppress output
    )

    # CatBoost - Good at handling categorical features and overfitting
    models['catboost'] = CatBoostClassifier(
        n_estimators=600,
        learning_rate=0.04,
        depth=6,  # Tree depth
        task_type='GPU' if use_gpu else 'CPU',
        scale_pos_weight=ratio,  # Handle class imbalance
        verbose=False,  # Suppress output
        random_state=42,
        allow_writing_files=False  # Don't create temp files
    )

    # Random Forest - Different algorithm (bagging instead of boosting)
    # Having diverse algorithms helps the ensemble learn better
    models['random_forest'] = RandomForestClassifier(
        n_estimators=300,  # Number of trees
        max_depth=12,  # How deep trees can grow
        class_weight='balanced',  # Automatically handle class imbalance
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )

    return models

def get_meta_learner():
    """
    Creates the meta-learner that combines predictions from base models.
    
    The meta-learner takes the predictions from all base models as input
    and learns how to best combine them to make the final prediction.
    
    Returns:
        LogisticRegression model configured for stacking
    """
    return LogisticRegression(
        random_state=42,  # For reproducibility
        solver='liblinear',  # Fast solver for small datasets
        penalty='l1',  # L1 regularization (helps with feature selection)
        C=1.0  # Regularization strength (lower = more regularization)
    )