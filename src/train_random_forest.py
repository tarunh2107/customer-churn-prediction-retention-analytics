"""
Random Forest Model Training, Feature Importance & Evaluation for Customer Churn Prediction.

This module trains a balanced Random Forest pipeline on the stratified training data,
evaluates performance on the untouched test set, extracts transformed feature importances,
generates diagnostic visualizations, and exports the serialized model pipeline.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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


def build_random_forest_pipeline(
    preprocessor,
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = -1
) -> Pipeline:
    """Constructs a scikit-learn Pipeline with preprocessing and balanced Random Forest."""
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    return pipeline


def extract_feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extracts transformed feature names from the fitted preprocessor
    and pairs them with Random Forest Gini feature importances.
    """
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    
    # Clean feature names for readability
    clean_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]
    
    feat_imp_df = pd.DataFrame({
        "Transformed_Feature": feature_names,
        "Clean_Feature_Name": clean_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    return feat_imp_df


def evaluate_random_forest_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_fig_dir: str | Path = "outputs/figures",
    output_table_dir: str | Path = "outputs"
) -> tuple[dict, pd.DataFrame]:
    """Evaluates the fitted pipeline on test data, generates diagnostic plots, and exports feature importances."""
    fig_dir = Path(output_fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir = Path(output_table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    
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
    ax.set_title("Random Forest Confusion Matrix (Test Set)", fontsize=13, pad=12, weight="bold")
    plt.tight_layout()
    cm_path = fig_dir / "random_forest_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # 2. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(fpr, tpr, color="#27ae60", lw=2.5, label=f"Random Forest (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Baseline (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("Random Forest ROC Curve (Test Set)", fontsize=13, pad=12, weight="bold")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    roc_path = fig_dir / "random_forest_roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    # 3. Extract & Plot Feature Importances
    feat_imp_df = extract_feature_importances(model)
    imp_csv_path = table_dir / "random_forest_feature_importance.csv"
    feat_imp_df.to_csv(imp_csv_path, index=False)
    
    # Plot top 15 features
    top_15 = feat_imp_df.head(15).sort_values(by="Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(top_15["Clean_Feature_Name"], top_15["Importance"], color="#2ecc71", edgecolor="#27ae60", height=0.65)
    ax.set_title("Random Forest: Top 15 Feature Importances", fontsize=14, pad=12, weight="bold")
    ax.set_xlabel("Gini Importance Score", fontsize=12)
    ax.set_ylabel("Transformed Feature", fontsize=12)
    for idx, (val, name) in enumerate(zip(top_15["Importance"], top_15["Clean_Feature_Name"])):
        ax.annotate(f"{val:.4f}", xy=(val, idx), xytext=(5, 0), textcoords="offset points", va="center", fontsize=10, weight="bold")
    ax.set_xlim(0, max(top_15["Importance"]) * 1.18)
    plt.tight_layout()
    feat_imp_plot_path = fig_dir / "random_forest_feature_importance.png"
    plt.savefig(feat_imp_plot_path, dpi=300)
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
        "feat_imp_path": imp_csv_path,
        "feat_imp_plot_path": feat_imp_plot_path,
    }
    return metrics, feat_imp_df


def train_and_save_random_forest(
    model_save_path: str | Path = "models/random_forest_pipeline.joblib"
) -> tuple[Pipeline, dict, pd.DataFrame]:
    """Runs data loading, pipeline assembly, training, evaluation, and serialization."""
    print("Loading preprocessed train/test data splits...")
    data = get_train_test_data()
    
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    preprocessor = data["preprocessor"]
    
    print(f"Training feature shape: {X_train.shape}, Test feature shape: {X_test.shape}")
    print("Building balanced Random Forest pipeline (n_estimators=200, class_weight='balanced')...")
    pipeline = build_random_forest_pipeline(
        preprocessor,
        n_estimators=200,
        random_state=data["random_state"],
        n_jobs=-1
    )
    
    print("Fitting pipeline exclusively on training data...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating pipeline on untouched test set...")
    metrics, feat_imp_df = evaluate_random_forest_model(pipeline, X_test, y_test)
    
    # Save model
    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"Successfully saved fitted pipeline to {save_path.resolve()}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("RANDOM FOREST MODEL EVALUATION SUMMARY")
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
    print("TOP 10 TRANSFORMED FEATURES BY GINI IMPORTANCE:")
    for idx, row in feat_imp_df.head(10).iterrows():
        print(f"  {idx+1:2d}. {row['Clean_Feature_Name']:<32} (Score: {row['Importance']:.4f})")
    print("=" * 80)
    print("\nClassification Report:\n", metrics["classification_report"])
    
    return pipeline, metrics, feat_imp_df


if __name__ == "__main__":
    train_and_save_random_forest()
