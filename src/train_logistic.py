"""
Logistic Regression Model Training & Evaluation for Customer Churn Prediction.

This module trains a balanced Logistic Regression pipeline on the stratified training data,
evaluates performance on the test set, saves diagnostic visualizations (Confusion Matrix, ROC Curve),
and exports the serialized model pipeline.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

# Import data preprocessing pipeline
try:
    from src.preprocess import get_train_test_data
except ImportError:
    from preprocess import get_train_test_data


def build_logistic_pipeline(preprocessor, random_state: int = 42) -> Pipeline:
    """Constructs a scikit-learn Pipeline with preprocessing and balanced Logistic Regression."""
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    return pipeline


def evaluate_logistic_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_fig_dir: str | Path = "outputs/figures"
) -> dict:
    """Evaluates the fitted pipeline on test data, generates diagnostic plots, and returns metrics."""
    fig_dir = Path(output_fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Predictions & Probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    clf_report = classification_report(y_test, y_pred, digits=4)
    
    # 1. Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_display = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(
        cm_display,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Retained (0)", "Predicted Churned (1)"],
        yticklabels=["Actual Retained (0)", "Actual Churned (1)"],
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title("Logistic Regression Confusion Matrix (Test Set)", fontsize=13, pad=12, weight="bold")
    plt.tight_layout()
    cm_path = fig_dir / "logistic_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # 2. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"Logistic Regression (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Baseline (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("Logistic Regression ROC Curve (Test Set)", fontsize=13, pad=12, weight="bold")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    roc_path = fig_dir / "logistic_roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "actual_churners": int(tp + fn),
        "detected_churners": int(tp),
        "missed_churners": int(fn),
        "classification_report": clf_report,
        "cm_path": cm_path,
        "roc_path": roc_path,
    }
    return metrics


def train_and_save_logistic(
    model_save_path: str | Path = "models/logistic_regression_pipeline.joblib"
) -> tuple[Pipeline, dict]:
    """Runs data loading, pipeline assembly, training, evaluation, and serialization."""
    print("Loading preprocessed train/test data splits...")
    data = get_train_test_data()
    
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    preprocessor = data["preprocessor"]
    
    print(f"Training feature shape: {X_train.shape}, Test feature shape: {X_test.shape}")
    print("Building balanced Logistic Regression pipeline...")
    pipeline = build_logistic_pipeline(preprocessor, random_state=data["random_state"])
    
    print("Fitting pipeline exclusively on training data...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating pipeline on untouched test set...")
    metrics = evaluate_logistic_model(pipeline, X_test, y_test)
    
    # Save model
    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"Successfully saved fitted pipeline to {save_path.resolve()}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION MODEL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Accuracy:                    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision (Churn Class 1):   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall (Churn Class 1):      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-score (Churn Class 1):    {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:                     {metrics['roc_auc']:.4f}")
    print("-" * 80)
    print("Confusion Matrix:")
    print(f"  - True Negatives  (TN):     {metrics['tn']}")
    print(f"  - False Positives (FP):     {metrics['fp']}")
    print(f"  - False Negatives (FN):     {metrics['fn']}")
    print(f"  - True Positives  (TP):     {metrics['tp']}")
    print("-" * 80)
    print(f"Actual Churners in Test Set: {metrics['actual_churners']}")
    print(f"Correctly Detected Churners: {metrics['detected_churners']} ({metrics['recall']*100:.2f}%)")
    print(f"Missed Churners (FN):        {metrics['missed_churners']} ({(metrics['fn']/metrics['actual_churners'])*100:.2f}%)")
    print("=" * 80)
    print("\nClassification Report:\n", metrics["classification_report"])
    
    return pipeline, metrics


if __name__ == "__main__":
    train_and_save_logistic()
