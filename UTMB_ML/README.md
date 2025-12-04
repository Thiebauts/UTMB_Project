# UTMB Score Prediction ML Pipeline

Machine learning pipeline to predict UTMB scores based on race parameters (distance, elevation gain, and finish time).

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

## Quick Start

### 1. Train the Model

Quick training (no hyperparameter tuning):
```bash
python3 train_model.py --quick
```

Full training with hyperparameter optimization:
```bash
python3 train_model.py --n-iter 50 --cv 5
```

### 2. Evaluate the Model

Generate evaluation figures:
```bash
python3 evaluate_model.py
```

This creates 6 evaluation plots in the `figures/` directory:
- `01_predicted_vs_actual.png` - Scatter plot of predictions vs actual values
- `02_residuals.png` - Residual distribution and analysis
- `03_feature_importance.png` - Feature importance rankings
- `04_learning_curves.png` - Learning curves showing model convergence
- `05_error_by_distance.png` - Error distribution by distance category
- `06_score_distribution.png` - Distribution comparison of actual vs predicted scores

### 3. Make Predictions

Command-line prediction:
```bash
python3 predict_score.py --distance 100 --elevation 4000 --time "10:30:00"
python3 predict_score.py -d 173 -e 10000 -t "24:15:30"
python3 predict_score.py --distance 50 --elevation 2500 --time-seconds 18000
```

Interactive mode:
```bash
python3 predict_score.py --interactive
```

## Model Performance

The trained XGBoost model achieves excellent accuracy:
- **Test RMSE**: ~4.8 points
- **Test MAE**: ~2.6 points
- **Test R²**: ~0.998

Error percentiles on test set:
- 50% of predictions within **1.5 points**
- 90% of predictions within **5.0 points**
- 95% of predictions within **7.4 points**

## Project Structure

```
UTMB_ML/
├── data_preparation.py    # Load & filter data with valid UTMB scores
├── train_model.py         # Train XGBoost with hyperparameter tuning
├── evaluate_model.py     # Generate evaluation plots
├── predict_score.py       # CLI tool for predictions
├── requirements.txt       # Python dependencies
├── models/                # Saved trained models
│   ├── xgboost_utmb_score.joblib
│   └── xgboost_utmb_score_metadata.json
└── figures/              # Evaluation plots
    ├── 01_predicted_vs_actual.png
    ├── 02_residuals.png
    ├── 03_feature_importance.png
    ├── 04_learning_curves.png
    ├── 05_error_by_distance.png
    └── 06_score_distribution.png
```

## Data Source

The model is trained on data from `../UTMB_project/master_results.csv`, which contains race results from various UTMB World Series events. Only records with valid UTMB scores are used for training (~5,000 samples).

## Features

The model uses three input features:
- **distance_km**: Race distance in kilometers
- **elevation_gain**: Total elevation gain in meters
- **race_time_seconds**: Finish time in seconds

## Troubleshooting

### ModuleNotFoundError: No module named 'xgboost'

If you see this error, install xgboost:
```bash
pip install xgboost
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### Model not found

If you see "Model not found", train the model first:
```bash
python3 train_model.py --quick
```

