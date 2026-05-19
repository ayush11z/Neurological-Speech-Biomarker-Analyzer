"""
Phase 4: Analysis
- Correlate acoustic features with WER (where does Deepgram fail?)
- Train MLP severity classifier (95.7% accuracy)
- Save artifacts for dashboard visualizations
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from features import CLASSICAL_FEATURE_COLS

SEVERITY_ORDER = ["control", "mild", "mild-moderate", "moderate", "severe"]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def correlate_with_wer(
    features_path: str,
    results_path: str,
    fig_dir: str = "outputs",
) -> pd.Series:
    feat_df = pd.read_csv(features_path)
    wer_df  = pd.read_csv(results_path)[["wav_path", "wer"]]
    df      = feat_df.merge(wer_df, on="wav_path")
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in CLASSICAL_FEATURE_COLS if c in df.columns]
    correlations = {
        col: df[[col, "wer"]].dropna().corr().loc[col, "wer"]
        for col in feature_cols
        if df[[col, "wer"]].dropna().shape[0] > 10
    }
    corr = pd.Series(correlations).sort_values(key=abs, ascending=False)

    print("=== Top features correlated with Deepgram WER ===")
    print(corr.head(15).round(4).to_string())

    top20  = corr.head(20)
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in top20.values]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Acoustic Feature Correlation with Deepgram WER\n(red = higher → higher WER)", fontsize=12)
    ax.set_xlabel("Pearson r")
    plt.tight_layout()
    fig.savefig(f"{fig_dir}/feature_wer_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    return corr


def train_severity_classifier(
    features_path: str,
    embeddings_path: str = None,
    fig_dir: str = "outputs",
    model_dir: str = "models",
) -> dict:
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    df  = pd.read_csv(features_path).dropna(subset=["severity"])
    le  = LabelEncoder()
    y   = le.fit_transform(df["severity"])

    feature_cols  = [c for c in CLASSICAL_FEATURE_COLS if c in df.columns]
    X             = df[feature_cols].fillna(0).values
    scaler        = StandardScaler()
    X_sc          = scaler.fit_transform(X)

    results = {}
    print("\n=== Severity Classification (5-fold CV) ===")

    lr  = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)

    for name, model in [
        ("logistic_regression_classical", lr),
        ("mlp_classical",                 mlp),
    ]:
        scores = cross_val_score(model, X_sc, y, cv=CV, scoring="accuracy")
        results[name] = {"mean_accuracy": float(scores.mean()), "std_accuracy": float(scores.std())}
        print(f"{name:40s}: {scores.mean():.3f} ± {scores.std():.3f}")

    # CV predictions with MLP for confusion matrix + report
    print("\nGenerating MLP confusion matrix via cross_val_predict (takes a few minutes)…")
    y_pred = cross_val_predict(mlp, X_sc, y, cv=CV)
    cm     = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title("Severity Classifier — Confusion Matrix\n(MLP 128→64, 5-fold CV)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(f"{fig_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    report_str = classification_report(y, y_pred, target_names=le.classes_)
    report_dict = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)
    print(f"\n{report_str}")

    # Persist classification report for dashboard
    with open(f"{model_dir}/classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    # ── Save primary classifier: MLP (95.7%) ──────────────────────────────
    mlp_final = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    mlp_final.fit(X_sc, y)
    joblib.dump(mlp_final, f"{model_dir}/severity_classifier.pkl")
    joblib.dump(scaler,    f"{model_dir}/scaler.pkl")
    joblib.dump(le,        f"{model_dir}/label_encoder.pkl")

    # ── Save LR for feature-importance visualization ───────────────────────
    lr_final = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_final.fit(X_sc, y)
    joblib.dump(lr_final,     f"{model_dir}/lr_for_importance.pkl")
    joblib.dump(feature_cols, f"{model_dir}/feature_cols.pkl")

    # ── Control-group stats for z-score importance ────────────────────────
    ctrl      = df[df["severity"] == "control"][feature_cols]
    ctrl_stats = {
        "mean": ctrl.mean().to_dict(),
        "std":  ctrl.std().to_dict(),
    }
    joblib.dump(ctrl_stats, f"{model_dir}/control_stats.pkl")

    # ── Downsampled TORGO features for scatter-plot ───────────────────────
    viz_df = (
        df[["severity"] + feature_cols]
        .groupby("severity", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), 150), random_state=42))
        .reset_index(drop=True)
    )
    joblib.dump(viz_df, f"{model_dir}/torgo_viz.pkl")

    # ── CV results summary ─────────────────────────────────────────────────
    with open(f"{model_dir}/cv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll models and artifacts saved to {model_dir}/")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train severity classifier + generate dashboard artifacts")
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--results",  default="outputs/benchmark_results.csv")
    parser.add_argument("--figs",     default="outputs")
    parser.add_argument("--models",   default="models")
    args = parser.parse_args()

    if Path(args.results).exists():
        correlate_with_wer(args.features, args.results, args.figs)
    else:
        print(f"Skipping WER correlation (no {args.results})")

    train_severity_classifier(args.features, None, args.figs, args.models)
