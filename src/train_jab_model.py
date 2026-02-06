"""
Train a jab vs non-jab classifier from feature dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from src.config import (
    FEATURES_DATASET_CSV,
    MODELS_DIR,
    MODEL_FILENAME,
    TEST_SIZE,
    RANDOM_STATE,
)


def load_and_prepare_data(csv_path: str) -> tuple:
    """
    Load feature dataset and prepare X (features) and y (labels).
    
    Args:
        csv_path: Path to the features CSV file
    
    Returns:
        Tuple of (X, y) where X is feature matrix and y is label array
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Features dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        raise ValueError(f"Features dataset is empty: {csv_path}")
    
    # Identify feature columns (exclude metadata and label columns)
    exclude_cols = ["label", "source_file", "window_start_frame", "window_end_frame",
                    "window_start_time", "window_end_time"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not feature_cols:
        raise ValueError("No feature columns found in dataset")
    
    # Extract features and labels
    X = df[feature_cols].values
    y = df["label"].values
    
    # Handle NaN values (fill with median)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    
    print(f"Loaded dataset: {len(df)} samples, {len(feature_cols)} features")
    print(f"Feature columns: {feature_cols}")
    print(f"Label distribution: {np.bincount(y.astype(int))}")
    
    return X, y, feature_cols


def train_model(X_train: np.ndarray, y_train: np.ndarray, model_type: str = "random_forest") -> object:
    """
    Train a classifier model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model
    
    Returns:
        Trained model
    """
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"\nTraining {model_type}...")
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model: object, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluate model and print metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 50)
    print("Model Evaluation")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Jab", "Jab"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main(model_type: str = "random_forest") -> None:
    """
    Main training pipeline.
    
    Args:
        model_type: Type of model to train ("random_forest" or "logistic_regression")
    """
    print("=" * 50)
    print("Jab Detection Model Training")
    print("=" * 50)
    
    # Load data
    X, y, feature_cols = load_and_prepare_data(str(FEATURES_DATASET_CSV))
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Ensure balanced split
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    model = train_model(X_train, y_train, model_type=model_type)
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILENAME)
    print(f"\nModel saved to: {MODEL_FILENAME}")
    
    # Save feature names for later use
    feature_names_path = MODELS_DIR / "feature_names.txt"
    with open(feature_names_path, "w") as f:
        for name in feature_cols:
            f.write(f"{name}\n")
    print(f"Feature names saved to: {feature_names_path}")


if __name__ == "__main__":
    """
    CLI for model training.
    Usage:
        python -m src.train_jab_model
        python -m src.train_jab_model --model_type logistic_regression
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train jab detection model")
    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic_regression"],
        help="Type of model to train"
    )
    
    args = parser.parse_args()
    
    try:
        main(model_type=args.model_type)
    except Exception as e:
        print(f"Error: {e}", file=__import__("sys").stderr)
        exit(1)

