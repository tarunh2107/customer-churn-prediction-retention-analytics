"""
Data Preprocessing & Splitting Pipeline for Customer Churn Modeling.

This module prepares the training and testing datasets using leak-free practices:
- Extracts the 19 approved features via feature_selection.py.
- Preserves customer IDs aligned with splits for downstream reporting.
- Performs stratified 80/20 train/test splitting (random_state=42).
- Builds a reusable ColumnTransformer with StandardScaler for numerical features
  and OneHotEncoder(handle_unknown='ignore') for categorical features.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Import feature selection module
try:
    from src.feature_selection import select_features, load_clean_data
except ImportError:
    from feature_selection import select_features, load_clean_data


def get_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Identifies numerical and categorical column names from feature matrix X.
    """
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return numerical_features, categorical_features


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str]
) -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer configured for numerical scaling
    and categorical one-hot encoding.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor


def get_train_test_data(
    file_path: str | Path = "data/processed/telco_churn_clean.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Loads clean data, isolates approved features and target, performs stratified train/test split,
    and returns feature subsets, target subsets, customer IDs, and the un-fitted preprocessor.
    """
    df = load_clean_data(file_path)
    X, y, customer_ids, meta = select_features(df)
    
    numerical_features, categorical_features = get_feature_types(X)
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    
    # Split features, target, and customer IDs synchronously
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        customer_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return {
        "X": X,
        "y": y,
        "customer_ids": customer_ids,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "id_train": id_train,
        "id_test": id_test,
        "random_state": random_state,
        "test_size": test_size,
    }


def display_preprocessing_summary():
    """Executes dataset splitting and displays diagnostic verification."""
    data = get_train_test_data()
    
    X = data["X"]
    y = data["y"]
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    num_feats = data["numerical_features"]
    cat_feats = data["categorical_features"]
    
    print("=" * 80)
    print("MACHINE LEARNING PREPROCESSING & DATA SPLIT SUMMARY")
    print("=" * 80)
    print(f"Overall Feature Matrix (X) Shape: {X.shape}")
    print(f"Overall Target Vector  (y) Shape: {y.shape}")
    print("-" * 80)
    print(f"Numerical Features ({len(num_feats)}):")
    for f in num_feats:
        print(f"  - {f} (StandardScaler)")
    print(f"\nCategorical Features ({len(cat_feats)}):")
    for f in cat_feats:
        print(f"  - {f} (OneHotEncoder, handle_unknown='ignore')")
    print("-" * 80)
    print(f"Train/Test Split: {100 * (1 - data['test_size']):.0f}% Train / {100 * data['test_size']:.0f}% Test (Stratified, random_state={data['random_state']})")
    print(f"Training Set (X_train) Shape:     {X_train.shape}")
    print(f"Testing Set  (X_test)  Shape:     {X_test.shape}")
    print(f"Training Target (y_train) Shape:  {y_train.shape}")
    print(f"Testing Target  (y_test)  Shape:  {y_test.shape}")
    print("-" * 80)
    
    # Train distribution
    train_counts = y_train.value_counts()
    train_pcts = (y_train.value_counts(normalize=True) * 100).round(2)
    print("Training Churn Distribution (y_train):")
    print(f"  - Retained (0): {train_counts[0]:,} ({train_pcts[0]:.2f}%)")
    print(f"  - Churned  (1): {train_counts[1]:,} ({train_pcts[1]:.2f}%)")
    
    # Test distribution
    test_counts = y_test.value_counts()
    test_pcts = (y_test.value_counts(normalize=True) * 100).round(2)
    print("\nTesting Churn Distribution (y_test):")
    print(f"  - Retained (0): {test_counts[0]:,} ({test_pcts[0]:.2f}%)")
    print(f"  - Churned  (1): {test_counts[1]:,} ({test_pcts[1]:.2f}%)")
    print("=" * 80)
    print("Pipeline ready for model training without data leakage.")


if __name__ == "__main__":
    display_preprocessing_summary()
