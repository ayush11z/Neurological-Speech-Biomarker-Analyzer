"""
Phase 4: Analysis
- Correlate acoustic features with WER (where does Deepgram fail?)
- Train severity classifier (logistic regression + MLP)
- Compare classical features vs wav2vec embeddings
"""
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
    fig_dir: str = "figures",
) -> pd.Series:
    """
    Find which acoustic features best predict where Deepgram fails.
    Returns a Series of Pearson correlations sorted by absolute value.
    """
    feat_df = pd.read_csv(features_path)
    wer_df = pd.read_csv(results_path)[["wav_path", "wer"]]
    df = feat_df.merge(wer_df, on="wav_path")
    Path(fig_dir).mkdir(exist_ok=True)

    feature_cols = [c for c in CLASSICAL_FEATURE_COLS if c in df.columns]
    correlations = {
        col: df[[col, "wer"]].dropna().corr().loc[col, "wer"]
        for col in feature_cols
        if df[[col, "wer"]].dropna().shape[0] > 10
    }
    corr = pd.Series(correlations).sort_values(key=abs, ascending=False)

    print("=== Top features correlated with Deepgram WER ===")
    print(corr.head(15).round(4).to_string())

    top20 = corr.head(20)
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in top20.values]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(
        "Acoustic Feature Correlation with Deepgram WER\n"
        "(red = higher feature → higher WER)",
        fontsize=12,
    )
    ax.set_xlabel("Pearson r")
    plt.tight_layout()
    fig.savefig(f"{fig_dir}/feature_wer_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {fig_dir}/feature_wer_correlation.png")

    return corr


def train_severity_classifier(
    features_path: str,
    embeddings_path: str = None,
    fig_dir: str = "figures",
    model_dir: str = "models",
) -> dict:
    """
    Train and cross-validate severity classifiers.
    Saves final logistic-regression model + scaler + label encoder for the dashboard.
    """
    Path(fig_dir).mkdir(exist_ok=True)
    Path(model_dir).mkdir(exist_ok=True)

    df = pd.read_csv(features_path).dropna(subset=["severity"])
    le = LabelEncoder()
    y = le.fit_transform(df["severity"])

    feature_cols = [c for c in CLASSICAL_FEATURE_COLS if c in df.columns]
    X_cls = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_cls_scaled = scaler.fit_transform(X_cls)

    results = {}

    def _eval(name, model, X):
        scores = cross_val_score(model, X, y, cv=CV, scoring="accuracy")
        results[name] = {"mean_accuracy": float(scores.mean()), "std_accuracy": float(scores.std())}
        print(f"{name:40s}: {scores.mean():.3f} ± {scores.std():.3f}")
        return scores

    print("\n=== Severity Classification (5-fold CV) ===")
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    _eval("logistic_regression_classical", lr, X_cls_scaled)

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    _eval("mlp_classical", mlp, X_cls_scaled)

    if embeddings_path and Path(embeddings_path).exists():
        emb = np.load(embeddings_path)
        if len(emb) == len(df):
            X_emb = StandardScaler().fit_transform(emb)
            _eval("logistic_regression_wav2vec", LogisticRegression(max_iter=1000, random_state=42), X_emb)
        else:
            print(f"Warning: embeddings length {len(emb)} != features length {len(df)}, skipping")

    # Confusion matrix from CV predictions
    y_pred = cross_val_predict(lr, X_cls_scaled, y, cv=CV)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title("Severity Classifier Confusion Matrix\n(Logistic Regression, 5-fold CV)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(f"{fig_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n{classification_report(y, y_pred, target_names=le.classes_)}")

    # Save final models
    lr_final = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_final.fit(X_cls_scaled, y)
    joblib.dump(lr_final, f"{model_dir}/severity_classifier.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    joblib.dump(le, f"{model_dir}/label_encoder.pkl")
    print(f"Models saved to {model_dir}/")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse benchmark + train classifier")
    parser.add_argument("--features", default="features.csv")
    parser.add_argument("--results", default="benchmark_results.csv")
    parser.add_argument("--embeddings", default="wav2vec_embeddings.npy")
    parser.add_argument("--figs", default="figures")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()

    if Path(args.results).exists():
        correlate_with_wer(args.features, args.results, args.figs)
    else:
        print(f"Skipping WER correlation (no {args.results})")

    train_severity_classifier(args.features, args.embeddings, args.figs, args.models)
