#!/usr/bin/env python3
"""
train_threshold_model.py
Train ML model to predict threshold from image features.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    import joblib
    import matplotlib.pyplot as plt
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn matplotlib joblib")


def train_model(training_csv, model_output='threshold_model.pkl', 
                feature_names_output='feature_names.txt'):
    """
    Train threshold prediction model.
    """
    if not HAS_SKLEARN:
        print("ERROR: scikit-learn required for training. Install with: pip install scikit-learn matplotlib joblib")
        return None, None
    
    # Load data
    df = pd.read_csv(training_csv)
    
    if len(df) < 2:
        print(f"WARNING: Only {len(df)} samples available. Need more data for reliable training.")
        print("This is just a demonstration. For production, collect more training samples.")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['threshold', 'region_name'] 
                    and not col.startswith('metric_')]
    X = df[feature_cols].values
    y = df['threshold'].values
    
    print(f"\n{'='*60}")
    print(f"Training Threshold Prediction Model")
    print(f"{'='*60}")
    print(f"Training samples: {len(df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Threshold range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Threshold mean: {y.mean():.2f}, std: {y.std():.2f}")
    print()
    
    if len(df) < 3:
        print("WARNING: Too few samples for train/test split. Using all data for training.")
        X_train, X_test = X, X
        y_train, y_test = y, y
    else:
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Scale features (some models benefit from scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if len(X_test) != len(X_train) else X_train_scaled
    
    # Train RandomForest (works well with few samples)
    print("Training RandomForest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,  # Shallow to avoid overfitting with small dataset
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print()
    
    if len(df) >= 3:
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, 
                                   cv=min(3, len(X_train)), 
                                   scoring='neg_mean_absolute_error')
        print(f"  CV MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print()
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Top 15 Most Important Features:")
        for i in range(min(15, len(feature_cols))):
            idx = indices[i]
            print(f"  {i+1:2d}. {feature_cols[idx]:30s} {importances[idx]:.4f}")
        print()
    
    # Plot predictions vs actual (if we have test set)
    if len(X_test) != len(X_train) and len(y_test) > 1:
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.6, s=100)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
            plt.xlabel('Actual Threshold')
            plt.ylabel('Predicted Threshold')
            plt.title(f'Threshold Prediction (n={len(y_test)} samples)')
            plt.legend()
            plt.tight_layout()
            plot_file = Path(training_csv).parent / 'threshold_prediction_plot.png'
            plt.savefig(plot_file, dpi=150)
            plt.close()
            print(f"Saved prediction plot to {plot_file}")
        except Exception as e:
            print(f"Could not create plot: {e}")
    
    # Save model
    model_file = Path(training_csv).parent / model_output
    joblib.dump(model, model_file)
    print(f"\nSaved model to {model_file}")
    
    # Save feature names
    feature_names_file = Path(training_csv).parent / feature_names_output
    with open(feature_names_file, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"Saved feature names to {feature_names_file}")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print("\nNOTE: With only 2 training samples, this model is for demonstration only.")
    print("For production use, collect more training data (10+ regions recommended).")
    
    return model, feature_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train threshold prediction model"
    )
    parser.add_argument(
        "--training_data",
        type=str,
        required=True,
        help="Training CSV file"
    )
    parser.add_argument(
        "--model_output",
        type=str,
        default="threshold_model.pkl",
        help="Output model file name"
    )
    parser.add_argument(
        "--feature_names",
        type=str,
        default="feature_names.txt",
        help="Feature names file name"
    )
    args = parser.parse_args()
    
    model, features = train_model(args.training_data, args.model_output, args.feature_names)

