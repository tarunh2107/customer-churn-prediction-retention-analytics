"""
Classification Threshold Analysis Module for Customer Churn Prediction.

This module evaluates the trained Logistic Regression model across varying decision thresholds (0.30 - 0.70),
analyzes the precision-recall-F1 trade-offs, quantifies error shifts (FP vs. FN),
generates diagnostic curve visualizations, and exports threshold comparison tables.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Import data preprocessing pipeline
try:
    from src.preprocess import get_train_test_data
except ImportError:
    from preprocess import get_train_test_data


def evaluate_thresholds(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list[float] = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    output_dir: str | Path = "outputs",
    fig_dir: str | Path = "outputs/figures"
) -> pd.DataFrame:
    """
    Computes classification metrics across candidate thresholds, generates trade-off charts,
    and exports summary CSV.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    figures_path = Path(fig_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        predicted_churn = int(tp + fp)
        
        rows.append({
            "Threshold": t,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "True_Negatives": int(tn),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp),
            "Predicted_Churn_Risk": predicted_churn,
            "Detection_Rate_Pct": round(rec * 100, 2),
            "Miss_Rate_Pct": round((fn / (tp + fn)) * 100, 2)
        })
        
    df_res = pd.DataFrame(rows)
    csv_path = out_path / "logistic_threshold_analysis.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"Saved threshold analysis table to {csv_path.resolve()}")
    
    # 1. Plot Threshold vs Precision, Recall, F1-Score
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(df_res["Threshold"], df_res["Precision"], marker="o", lw=2.2, color="#e67e22", label="Precision (Class 1)")
    ax.plot(df_res["Threshold"], df_res["Recall"], marker="s", lw=2.2, color="#2980b9", label="Recall (Class 1)")
    ax.plot(df_res["Threshold"], df_res["F1_Score"], marker="^", lw=2.2, color="#27ae60", label="F1-Score (Class 1)")
    ax.axvline(0.50, color="#7f8c8d", linestyle="--", lw=1.5, label="Default Threshold (0.50)")
    ax.axvline(0.60, color="#8e44ad", linestyle=":", lw=1.5, label="Peak F1 Threshold (0.60)")
    
    ax.set_title("Logistic Regression: Precision, Recall & F1 vs. Classification Threshold", fontsize=13, pad=12, weight="bold")
    ax.set_xlabel("Classification Decision Threshold", fontsize=12)
    ax.set_ylabel("Metric Score", fontsize=12)
    ax.set_ylim(0.35, 1.0)
    ax.set_xticks(thresholds)
    ax.legend(loc="lower left", fontsize=10.5)
    plt.tight_layout()
    tradeoff_plot = figures_path / "logistic_threshold_tradeoff.png"
    plt.savefig(tradeoff_plot, dpi=300)
    plt.close()
    
    # 2. Plot Threshold vs False Positives and False Negatives
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(df_res["Threshold"], df_res["False_Positives"], marker="o", lw=2.2, color="#e74c3c", label="False Positives (False Alarms)")
    ax.plot(df_res["Threshold"], df_res["False_Negatives"], marker="s", lw=2.2, color="#9b59b6", label="False Negatives (Missed Churners)")
    ax.axvline(0.50, color="#7f8c8d", linestyle="--", lw=1.5, label="Default Threshold (0.50)")
    ax.axvline(0.60, color="#8e44ad", linestyle=":", lw=1.5, label="Peak F1 Threshold (0.60)")
    
    ax.set_title("Logistic Regression: Classification Errors vs. Decision Threshold", fontsize=13, pad=12, weight="bold")
    ax.set_xlabel("Classification Decision Threshold", fontsize=12)
    ax.set_ylabel("Error Count (Instances)", fontsize=12)
    ax.set_xticks(thresholds)
    ax.legend(loc="upper center", fontsize=10.5)
    plt.tight_layout()
    errors_plot = figures_path / "logistic_threshold_errors.png"
    plt.savefig(errors_plot, dpi=300)
    plt.close()
    
    return df_res


def run_threshold_analysis():
    """Loads pipeline, executes threshold analysis, and prints findings."""
    data = get_train_test_data()
    X_test, y_test = data["X_test"], data["y_test"]
    
    pipeline = joblib.load("models/logistic_regression_pipeline.joblib")
    df_res = evaluate_thresholds(pipeline, X_test, y_test)
    
    # Key thresholds
    best_f1_row = df_res.loc[df_res["F1_Score"].idxmax()]
    max_rec_row = df_res.loc[df_res["Recall"].idxmax()]
    max_prec_row = df_res.loc[df_res["Precision"].idxmax()]
    default_row = df_res[df_res["Threshold"] == 0.50].iloc[0]
    
    print("\n" + "=" * 105)
    print("LOGISTIC REGRESSION CLASSIFICATION THRESHOLD ANALYSIS (TEST SET, N = 1,409)")
    print("=" * 105)
    print(df_res.to_string(index=False))
    print("=" * 105)
    print("\nKEY THRESHOLD BENCHMARKS:")
    print(f"  1. Default Baseline (Threshold = 0.50):")
    print(f"     Accuracy: {default_row['Accuracy']:.4f} | Precision: {default_row['Precision']:.4f} | Recall: {default_row['Recall']:.4f} | F1: {default_row['F1_Score']:.4f}")
    print(f"     Confusion: TP={default_row['True_Positives']}, FP={default_row['False_Positives']}, FN={default_row['False_Negatives']}, TN={default_row['True_Negatives']}")
    print(f"  2. Peak F1-Score Benchmark (Threshold = {best_f1_row['Threshold']:.2f}):")
    print(f"     Accuracy: {best_f1_row['Accuracy']:.4f} | Precision: {best_f1_row['Precision']:.4f} | Recall: {best_f1_row['Recall']:.4f} | F1: {best_f1_row['F1_Score']:.4f} (Max F1)")
    print(f"     Confusion: TP={best_f1_row['True_Positives']}, FP={best_f1_row['False_Positives']}, FN={best_f1_row['False_Negatives']}, TN={best_f1_row['True_Negatives']}")
    print(f"  3. Maximum Recall Benchmark (Threshold = {max_rec_row['Threshold']:.2f}):")
    print(f"     Accuracy: {max_rec_row['Accuracy']:.4f} | Precision: {max_rec_row['Precision']:.4f} | Recall: {max_rec_row['Recall']:.4f} (Max Recall) | F1: {max_rec_row['F1_Score']:.4f}")
    print(f"     Confusion: TP={max_rec_row['True_Positives']}, FP={max_rec_row['False_Positives']}, FN={max_rec_row['False_Negatives']}, TN={max_rec_row['True_Negatives']}")
    print(f"  4. Maximum Precision Benchmark (Threshold = {max_prec_row['Threshold']:.2f}):")
    print(f"     Accuracy: {max_prec_row['Accuracy']:.4f} | Precision: {max_prec_row['Precision']:.4f} (Max Precision) | Recall: {max_prec_row['Recall']:.4f} | F1: {max_prec_row['F1_Score']:.4f}")
    print(f"     Confusion: TP={max_prec_row['True_Positives']}, FP={max_prec_row['False_Positives']}, FN={max_prec_row['False_Negatives']}, TN={max_prec_row['True_Negatives']}")
    print("=" * 105)


if __name__ == "__main__":
    run_threshold_analysis()
