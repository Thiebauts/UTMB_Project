#!/usr/bin/env python3
"""
Model training module for UTMB Score prediction.

Trains an XGBoost regressor with hyperparameter optimization using RandomizedSearchCV.
Saves the best model for later use in predictions.
"""

from __future__ import annotations
from typing import Tuple, Dict, List

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from data_preparation import load_and_prepare_data, FEATURE_COLS


# Paths
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def get_hyperparameter_grid() -> dict:
    """
    Define the hyperparameter search space for XGBoost.
    
    Returns a dictionary of parameter distributions for RandomizedSearchCV.
    """
    return {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.3],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [0.1, 1, 10],
    }


def train_with_hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 50,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True
) -> Tuple[XGBRegressor, Dict]:
    """
    Train XGBoost with RandomizedSearchCV for hyperparameter optimization.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_iter: Number of random parameter combinations to try
        cv: Number of cross-validation folds
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 = all cores)
        verbose: Print progress
    
    Returns:
        best_model: The trained model with best parameters
        results: Dictionary with training results and metrics
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Hyperparameter Tuning with RandomizedSearchCV")
        print("=" * 60)
        print(f"  Iterations: {n_iter}")
        print(f"  CV Folds: {cv}")
        print(f"  Training samples: {len(X_train):,}")
    
    # Base model
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=1,  # Use 1 job per model, parallelize across CV folds
    )
    
    # Hyperparameter grid
    param_grid = get_hyperparameter_grid()
    
    # RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=2 if verbose else 0,
        return_train_score=True,
    )
    
    if verbose:
        print("\nStarting search...")
    
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    best_params = search.best_params_
    
    if verbose:
        print("\n" + "-" * 40)
        print("Best Parameters Found:")
        print("-" * 40)
        for param, value in sorted(best_params.items()):
            print(f"  {param}: {value}")
        
        # Cross-validation score (RMSE)
        best_rmse = np.sqrt(-search.best_score_)
        print(f"\nBest CV RMSE: {best_rmse:.2f}")
    
    results = {
        "best_params": best_params,
        "best_cv_score": float(-search.best_score_),  # MSE
        "best_cv_rmse": float(np.sqrt(-search.best_score_)),
        "cv_results": {
            "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
            "std_test_score": search.cv_results_["std_test_score"].tolist(),
            "mean_train_score": search.cv_results_["mean_train_score"].tolist(),
        },
        "n_iter": n_iter,
        "cv_folds": cv,
    }
    
    return best_model, results


def evaluate_model(
    model: XGBRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True
) -> dict:
    """
    Evaluate the trained model on train and test sets.
    
    Returns dictionary with all metrics.
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "train": {
            "rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "mae": float(mean_absolute_error(y_train, y_train_pred)),
            "r2": float(r2_score(y_train, y_train_pred)),
        },
        "test": {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "mae": float(mean_absolute_error(y_test, y_test_pred)),
            "r2": float(r2_score(y_test, y_test_pred)),
        }
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Model Evaluation")
        print("=" * 60)
        print(f"\nTraining Set ({len(y_train):,} samples):")
        print(f"  RMSE: {metrics['train']['rmse']:.2f}")
        print(f"  MAE:  {metrics['train']['mae']:.2f}")
        print(f"  R²:   {metrics['train']['r2']:.4f}")
        
        print(f"\nTest Set ({len(y_test):,} samples):")
        print(f"  RMSE: {metrics['test']['rmse']:.2f}")
        print(f"  MAE:  {metrics['test']['mae']:.2f}")
        print(f"  R²:   {metrics['test']['r2']:.4f}")
    
    return metrics


def save_model(
    model: XGBRegressor,
    training_results: dict,
    evaluation_metrics: dict,
    feature_names: list,
    model_name: str = "xgboost_utmb_score"
) -> Path:
    """
    Save the trained model and metadata.
    
    Saves:
    - model.joblib: The trained model
    - model_metadata.json: Training parameters and metrics
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"{model_name}.joblib"
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "created_at": timestamp,
        "feature_names": feature_names,
        "training_results": training_results,
        "evaluation_metrics": evaluation_metrics,
        "model_path": str(model_path),
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    return model_path


def train_simple_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    verbose: bool = True
) -> XGBRegressor:
    """
    Train a simple XGBoost model with default parameters (for quick testing).
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Training Simple XGBoost Model (no hyperparameter tuning)")
        print("=" * 60)
    
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
    
    model.fit(X_train, y_train)
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost model for UTMB Score prediction"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick training without hyperparameter tuning"
    )
    parser.add_argument(
        "--n-iter", type=int, default=50,
        help="Number of hyperparameter combinations to try (default: 50)"
    )
    parser.add_argument(
        "--cv", type=int, default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Load and prepare data
    data = load_and_prepare_data(random_state=args.random_state)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    
    if args.quick:
        # Quick training without hyperparameter tuning
        model = train_simple_model(X_train, y_train, random_state=args.random_state)
        training_results = {"mode": "quick", "params": model.get_params()}
    else:
        # Full training with hyperparameter optimization
        model, training_results = train_with_hyperparameter_tuning(
            X_train, y_train,
            n_iter=args.n_iter,
            cv=args.cv,
            random_state=args.random_state,
        )
    
    # Evaluate
    evaluation_metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Save
    save_model(
        model,
        training_results,
        evaluation_metrics,
        data["feature_names"]
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

