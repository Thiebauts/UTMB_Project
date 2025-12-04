#!/usr/bin/env python3
"""
Model evaluation module for UTMB Score prediction.

Generates comprehensive evaluation plots and metrics for the trained model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Check for xgboost (required for loading the model)
try:
    import xgboost
except ImportError:
    print("❌ Error: xgboost is not installed.")
    print("\nPlease install it using:")
    print("  pip install xgboost")
    print("\nOr install all requirements:")
    print("  pip install -r requirements.txt")
    import sys
    sys.exit(1)

from data_preparation import load_and_prepare_data, FEATURE_COLS


# Paths
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Plot style
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "dark": "#1B1B1E",
    "light": "#E8E8E8",
}


def load_model(model_name: str = "xgboost_utmb_score"):
    """Load the trained model and metadata."""
    model_path = MODELS_DIR / f"{model_name}.joblib"
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    metadata = None
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    
    return model, metadata


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual UTMB Scores",
    save_path: Path = None
):
    """
    Create scatter plot of predicted vs actual values with identity line.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, c=COLORS["primary"], edgecolors="none")
    
    # Identity line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            color=COLORS["secondary"], linestyle="--", linewidth=2, label="Perfect prediction")
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Add metrics text
    textstr = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.4f}"
    props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor=COLORS["dark"])
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", bbox=props)
    
    ax.set_xlabel("Actual UTMB Score", fontsize=12)
    ax.set_ylabel("Predicted UTMB Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Distribution",
    save_path: Path = None
):
    """
    Create histogram of residuals (prediction errors).
    """
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(residuals, bins=50, color=COLORS["primary"], edgecolor="white", alpha=0.8)
    ax1.axvline(0, color=COLORS["secondary"], linestyle="--", linewidth=2, label="Zero error")
    ax1.axvline(residuals.mean(), color=COLORS["accent"], linestyle="-", linewidth=2, 
                label=f"Mean: {residuals.mean():.2f}")
    ax1.set_xlabel("Residual (Predicted - Actual)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Residual Distribution", fontsize=14, fontweight="bold")
    ax1.legend()
    
    # Residuals vs Predicted
    ax2 = axes[1]
    ax2.scatter(y_pred, residuals, alpha=0.5, s=20, c=COLORS["primary"], edgecolors="none")
    ax2.axhline(0, color=COLORS["secondary"], linestyle="--", linewidth=2)
    ax2.set_xlabel("Predicted UTMB Score", fontsize=12)
    ax2.set_ylabel("Residual", fontsize=12)
    ax2.set_title("Residuals vs Predicted Values", fontsize=14, fontweight="bold")
    
    # Add ±1 std bands
    std = residuals.std()
    ax2.axhline(std, color=COLORS["accent"], linestyle=":", linewidth=1.5, alpha=0.7)
    ax2.axhline(-std, color=COLORS["accent"], linestyle=":", linewidth=1.5, alpha=0.7,
                label=f"±1 std ({std:.1f})")
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_feature_importance(
    model,
    feature_names: list,
    title: str = "Feature Importance",
    save_path: Path = None
):
    """
    Create bar plot of feature importances.
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Horizontal bar chart (better for reading labels)
    y_pos = np.arange(len(feature_names))
    sorted_importance = importance[indices]
    sorted_names = [feature_names[i] for i in indices]
    
    bars = ax.barh(y_pos, sorted_importance, color=COLORS["primary"], edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.invert_yaxis()  # Top to bottom
    
    # Add value labels
    for bar, val in zip(bars, sorted_importance):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{val:.3f}", va="center", fontsize=10)
    
    ax.set_xlabel("Importance (Gain)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_learning_curves(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    title: str = "Learning Curves",
    save_path: Path = None,
    cv: int = 5
):
    """
    Plot learning curves showing training and validation scores vs training size.
    """
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=42
    )
    
    # Convert to positive RMSE
    train_scores = -train_scores
    val_scores = -val_scores
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Training score
    ax.plot(train_sizes_abs, train_mean, "o-", color=COLORS["primary"], 
            label="Training RMSE", linewidth=2, markersize=6)
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color=COLORS["primary"])
    
    # Validation score
    ax.plot(train_sizes_abs, val_mean, "o-", color=COLORS["secondary"],
            label="Validation RMSE (CV)", linewidth=2, markersize=6)
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color=COLORS["secondary"])
    
    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_error_by_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    distances: np.ndarray,
    title: str = "Error Distribution by Distance Category",
    save_path: Path = None
):
    """
    Box plot of absolute errors grouped by distance ranges.
    """
    # Create distance categories
    bins = [0, 30, 50, 80, 120, 200, 1000]
    labels = ["<30km", "30-50km", "50-80km", "80-120km", "120-200km", ">200km"]
    
    distance_cat = pd.cut(distances, bins=bins, labels=labels, right=False)
    abs_errors = np.abs(y_pred - y_true)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        "Distance Category": distance_cat,
        "Absolute Error": abs_errors
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Box plot
    box_data = [df[df["Distance Category"] == cat]["Absolute Error"].dropna() 
                for cat in labels]
    box_data = [d for d in box_data if len(d) > 0]
    valid_labels = [labels[i] for i, d in enumerate(
        [df[df["Distance Category"] == cat]["Absolute Error"].dropna() for cat in labels]
    ) if len(d) > 0]
    
    bp = ax.boxplot(box_data, labels=valid_labels, patch_artist=True)
    
    # Style the boxes
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["primary"])
        patch.set_alpha(0.7)
    for whisker in bp["whiskers"]:
        whisker.set_color(COLORS["dark"])
    for cap in bp["caps"]:
        cap.set_color(COLORS["dark"])
    for median in bp["medians"]:
        median.set_color(COLORS["accent"])
        median.set_linewidth(2)
    
    # Add sample counts
    for i, (cat, data) in enumerate(zip(valid_labels, box_data)):
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f"n={len(data)}", 
                ha="center", fontsize=9, style="italic")
    
    ax.set_xlabel("Distance Category", fontsize=12)
    ax.set_ylabel("Absolute Error", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_score_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "UTMB Score Distribution",
    save_path: Path = None
):
    """
    Compare distribution of actual vs predicted scores.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histograms
    ax.hist(y_true, bins=50, alpha=0.6, color=COLORS["primary"], 
            label="Actual", edgecolor="white")
    ax.hist(y_pred, bins=50, alpha=0.6, color=COLORS["secondary"],
            label="Predicted", edgecolor="white")
    
    ax.set_xlabel("UTMB Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def generate_all_figures(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list,
    output_dir: Path = FIGURES_DIR
):
    """
    Generate all evaluation figures and save them.
    """
    print("\n" + "=" * 60)
    print("Generating Evaluation Figures")
    print("=" * 60)
    
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Use test set for most plots
    y_true = y_test.values
    y_pred = y_test_pred
    distances = X_test["distance_km"].values
    
    # Generate each figure
    figures = []
    
    # 1. Predicted vs Actual
    fig = plot_predicted_vs_actual(
        y_true, y_pred,
        title="Predicted vs Actual UTMB Scores (Test Set)",
        save_path=output_dir / "01_predicted_vs_actual.png"
    )
    figures.append(fig)
    
    # 2. Residuals
    fig = plot_residuals(
        y_true, y_pred,
        title="Residual Analysis",
        save_path=output_dir / "02_residuals.png"
    )
    figures.append(fig)
    
    # 3. Feature Importance
    fig = plot_feature_importance(
        model, feature_names,
        title="Feature Importance",
        save_path=output_dir / "03_feature_importance.png"
    )
    figures.append(fig)
    
    # 4. Learning Curves (use full data)
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    fig = plot_learning_curves(
        model, X_full, y_full,
        title="Learning Curves",
        save_path=output_dir / "04_learning_curves.png"
    )
    figures.append(fig)
    
    # 5. Error by Distance
    fig = plot_error_by_distance(
        y_true, y_pred, distances,
        title="Error Distribution by Distance Category (Test Set)",
        save_path=output_dir / "05_error_by_distance.png"
    )
    figures.append(fig)
    
    # 6. Score Distribution
    fig = plot_score_distribution(
        y_true, y_pred,
        title="UTMB Score Distribution: Actual vs Predicted",
        save_path=output_dir / "06_score_distribution.png"
    )
    figures.append(fig)
    
    plt.close("all")
    
    print(f"\nGenerated {len(figures)} figures in {output_dir}")
    
    return figures


def print_evaluation_summary(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
):
    """Print detailed evaluation summary."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)
    
    print("\n📊 Training Set Performance:")
    print(f"   Samples: {len(y_train):,}")
    print(f"   RMSE:    {np.sqrt(mean_squared_error(y_train, y_train_pred)):.2f}")
    print(f"   MAE:     {mean_absolute_error(y_train, y_train_pred):.2f}")
    print(f"   R²:      {r2_score(y_train, y_train_pred):.4f}")
    
    print("\n📊 Test Set Performance:")
    print(f"   Samples: {len(y_test):,}")
    print(f"   RMSE:    {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
    print(f"   MAE:     {mean_absolute_error(y_test, y_test_pred):.2f}")
    print(f"   R²:      {r2_score(y_test, y_test_pred):.4f}")
    
    # Error percentiles
    abs_errors = np.abs(y_test_pred - y_test)
    print("\n📈 Prediction Error Percentiles (Test Set):")
    for p in [50, 75, 90, 95]:
        print(f"   {p}th percentile: {np.percentile(abs_errors, p):.1f} points")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation figures for trained UTMB Score model"
    )
    parser.add_argument(
        "--model-name", type=str, default="xgboost_utmb_score",
        help="Name of the saved model (default: xgboost_utmb_score)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for data split (must match training)"
    )
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, metadata = load_model(args.model_name)
    
    if metadata:
        print(f"Model created: {metadata.get('created_at', 'unknown')}")
    
    # Load data
    data = load_and_prepare_data(random_state=args.random_state)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    
    # Print summary
    print_evaluation_summary(model, X_train, y_train, X_test, y_test)
    
    # Generate figures
    generate_all_figures(
        model,
        X_train, y_train,
        X_test, y_test,
        data["feature_names"]
    )
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

