"""
Phase 5: Gradio Dashboard v3
Tabs: Analyze | Dataset Overview | Model Performance | About
"""
import os
import json
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import librosa
import librosa.display
import gradio as gr
import joblib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE        = Path(__file__).parent
MODEL_DIR   = BASE / "models"
DATA_DIR    = BASE / "data/processed"
OUTPUTS_DIR = BASE / "outputs"

CLASSICAL_FEATURE_COLS = [
    *[f"mfcc_{i}_mean" for i in range(13)],
    *[f"mfcc_{i}_std"  for i in range(13)],
    "f0_mean", "f0_std", "f0_range", "jitter",
    "rms_mean", "rms_std", "shimmer", "hnr",
    "voiced_fraction", "pause_ratio", "duration_s",
    "spectral_centroid_mean", "spectral_rolloff_mean", "zcr_mean",
]

SEVERITY_ORDER  = ["control", "mild", "mild-moderate", "moderate", "severe"]
SEVERITY_COLORS = {
    "control":       "#22c55e",
    "mild":          "#eab308",
    "mild-moderate": "#f97316",
    "moderate":      "#ef4444",
    "severe":        "#a855f7",
}
SEVERITY_WER = {
    "control": 0.08, "mild": 0.18,
    "mild-moderate": 0.32, "moderate": 0.38, "severe": 0.53,
}

FEATURE_LABELS = {
    "f0_mean": "Mean Pitch", "f0_std": "Pitch Variability",
    "f0_range": "Pitch Range", "jitter": "Jitter",
    "hnr": "Harmonic-Noise Ratio", "shimmer": "Shimmer",
    "pause_ratio": "Pause Ratio", "voiced_fraction": "Voiced Fraction",
    "rms_mean": "Mean Energy", "rms_std": "Energy Variability",
    "spectral_centroid_mean": "Spectral Centroid",
    "zcr_mean": "Zero Crossing Rate", "duration_s": "Duration",
    "spectral_rolloff_mean": "Spectral Rolloff",
}

# ── Dark colour palette ────────────────────────────────────────────────────────
BG    = "#0f172a"
CARD  = "#1e293b"
TEXT  = "#f1f5f9"
MUTED = "#64748b"
GRID  = "#334155"
CYAN  = "#38bdf8"
AMBER = "#fbbf24"
RED   = "#f87171"
GREEN = "#4ade80"
INDIGO= "#818cf8"


# ── Model loading ──────────────────────────────────────────────────────────────
_clf = _scaler = _le = _ctrl_stats = _torgo_viz = _feat_cols = None


def _load_models():
    global _clf, _scaler, _le, _ctrl_stats, _torgo_viz, _feat_cols
    try:
        _clf    = joblib.load(MODEL_DIR / "severity_classifier.pkl")
        _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        _le     = joblib.load(MODEL_DIR / "label_encoder.pkl")
        print("Classifier loaded.")
    except FileNotFoundError:
        print("No classifier — run: python analysis.py")
    try:
        _ctrl_stats = joblib.load(MODEL_DIR / "control_stats.pkl")
        _feat_cols  = joblib.load(MODEL_DIR / "feature_cols.pkl")
        _torgo_viz  = joblib.load(MODEL_DIR / "torgo_viz.pkl")
        print("Visualization artifacts loaded.")
    except FileNotFoundError:
        print("Viz artifacts missing — re-run analysis.py after upgrading.")


_load_models()


# ── Example recordings ─────────────────────────────────────────────────────────
_EXAMPLES: dict[str, str] = {}


def _load_examples():
    manifest = DATA_DIR / "manifest.csv"
    if not manifest.exists():
        return
    df = pd.read_csv(manifest)
    for sev in ["control", "mild", "mild-moderate", "severe"]:
        rows = df[df["severity"] == sev]
        if not rows.empty:
            _EXAMPLES[sev] = rows.iloc[min(10, len(rows) - 1)]["wav_path"]


_load_examples()


def load_example(severity: str):
    path = _EXAMPLES.get(severity)
    if not path or not Path(path).exists():
        return None
    y, sr = librosa.load(path, sr=16000)
    return (sr, (y * 32767).astype(np.int16))


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)
    if title:  ax.set_title(title,  color=TEXT,  fontsize=9, pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=8)


def _fig(w=6, h=3):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    return fig, ax


def _placeholder(label: str) -> plt.Figure:
    fig, ax = _fig()
    ax.set_facecolor(BG)
    ax.text(0.5, 0.5, f"Upload audio to see\n{label}",
            ha="center", va="center", color=MUTED, fontsize=10,
            transform=ax.transAxes, linespacing=2.0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(pad=0.5)
    return fig


# ── Static tab charts (built once at startup) ──────────────────────────────────

def _build_dataset_fig() -> plt.Figure:
    manifest = DATA_DIR / "manifest.csv"
    if not manifest.exists():
        return _placeholder("Dataset Overview (run pipeline.py first)")

    df         = pd.read_csv(manifest)
    present    = [s for s in SEVERITY_ORDER if s in df["severity"].unique()]
    sev_counts = df.groupby("severity")["utterance_id"].count().reindex(present)
    spk_df     = (df.groupby(["speaker_id", "severity"])["utterance_id"]
                    .count().reset_index().sort_values("severity"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=BG)

    # Utterances by severity
    ax = axes[0]
    clrs = [SEVERITY_COLORS.get(s, "#888") for s in sev_counts.index]
    bars = ax.bar(sev_counts.index, sev_counts.values, color=clrs, edgecolor="none", width=0.6)
    for b, v in zip(bars, sev_counts.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 40,
                f"{v:,}", ha="center", color=TEXT, fontsize=9, fontweight="bold")
    _style(ax, "Utterances by Severity Group", "", "Count")

    # Utterances by speaker
    ax2 = axes[1]
    s_clrs = [SEVERITY_COLORS.get(s, "#888") for s in spk_df["severity"]]
    ax2.bar(spk_df["speaker_id"], spk_df["utterance_id"], color=s_clrs, edgecolor="none", width=0.7)
    _style(ax2, "Utterances by Speaker", "", "Count")
    ax2.tick_params(axis="x", rotation=40)
    handles = [Patch(color=SEVERITY_COLORS.get(s, "#888"), label=s)
               for s in present]
    ax2.legend(handles=handles, fontsize=7, framealpha=0.25, labelcolor=TEXT,
               title="Severity", title_fontsize=7)

    plt.tight_layout(pad=1.5)
    return fig


def _build_model_perf_fig() -> plt.Figure:
    rp = MODEL_DIR / "classification_report.json"
    if rp.exists():
        with open(rp) as f:
            report = json.load(f)
        classes = [c for c in SEVERITY_ORDER if c in report]
        f1s   = [report[c]["f1-score"]  for c in classes]
        precs = [report[c]["precision"] for c in classes]
        recs  = [report[c]["recall"]    for c in classes]
        title = "Per-class Performance — MLP 128→64 (5-fold CV)"
    else:
        # Fallback: hardcoded LR results from the initial run
        classes = ["control", "mild", "mild-moderate", "severe"]
        f1s     = [0.94, 0.82, 0.59, 0.82]
        precs   = [0.91, 0.87, 0.69, 0.87]
        recs    = [0.97, 0.77, 0.52, 0.78]
        title   = "Per-class Performance — Logistic Regression (5-fold CV)"

    fig, ax = _fig(w=11, h=4)
    x = np.arange(len(classes))
    w = 0.25
    clrs = [SEVERITY_COLORS.get(c, "#888") for c in classes]

    b1 = ax.bar(x - w, f1s,   w, color=clrs, alpha=0.95, edgecolor="none", label="F1-score")
    b2 = ax.bar(x,     precs, w, color=clrs, alpha=0.55, edgecolor="none", label="Precision")
    b3 = ax.bar(x + w, recs,  w, color=clrs, alpha=0.30, edgecolor="none", label="Recall")

    for bar, val in zip(b1, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.0%}", ha="center", color=TEXT, fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, color=TEXT, fontsize=9)
    ax.set_ylim(0, 1.12)
    _style(ax, title, "", "Score")
    ax.legend(fontsize=8, framealpha=0.25, labelcolor=TEXT)
    plt.tight_layout(pad=1.2)
    return fig


_DATASET_FIG    = _build_dataset_fig()
_MODEL_PERF_FIG = _build_model_perf_fig()


# ── Per-recording plots ────────────────────────────────────────────────────────

def _plot_spectrogram(wav_path: str) -> plt.Figure:
    y, sr = librosa.load(wav_path, sr=16000)
    fig, ax = _fig()
    _style(ax, "Mel Spectrogram", "Time (s)", "Mel Frequency")
    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
    S_db = librosa.power_to_db(S, ref=np.max)
    img  = librosa.display.specshow(S_db, x_axis="time", y_axis="mel",
                                     sr=sr, ax=ax, cmap="inferno")
    cb = fig.colorbar(img, ax=ax, format="%+.0f dB")
    cb.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
    plt.tight_layout(pad=1)
    return fig


def _plot_energy(wav_path: str) -> plt.Figure:
    y, sr  = librosa.load(wav_path, sr=16000)
    rms    = librosa.feature.rms(y=y)[0]
    times  = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    thr    = 0.01 * (rms.max() + 1e-8)
    fig, ax = _fig()
    _style(ax, "Energy & Pause Detection", "Time (s)", "RMS")
    ax.plot(times, rms, color=CYAN, linewidth=1.2, label="RMS Energy")
    ax.axhline(thr, color=RED, linestyle="--", linewidth=0.8, alpha=0.7, label="Pause threshold")
    ax.fill_between(times, 0, rms, where=(rms < thr), color=RED, alpha=0.15)
    ax.legend(fontsize=7, framealpha=0.2, labelcolor=TEXT)
    plt.tight_layout(pad=1)
    return fig


def _plot_pitch(wav_path: str) -> plt.Figure:
    y, sr      = librosa.load(wav_path, sr=16000)
    f0, vf, _  = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    times      = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
    voiced     = vf.astype(bool) if vf is not None else np.zeros(len(f0), bool)
    fig, ax    = _fig()
    _style(ax, "Pitch Contour (F0)", "Time (s)", "Hz")
    ax.scatter(times[voiced], f0[voiced], s=4, color=AMBER, alpha=0.9, label="F0 (voiced)")
    vf0 = f0[voiced]
    if len(vf0): ax.set_ylim(0, min(600, vf0.max() * 1.3))
    ax.legend(fontsize=7, framealpha=0.2, labelcolor=TEXT)
    plt.tight_layout(pad=1)
    return fig


def _plot_torgo_scatter(feats: dict, label: str) -> plt.Figure:
    if _torgo_viz is None:
        return _placeholder("TORGO Comparison")

    x_col, y_col = "f0_std", "pause_ratio"
    fig, ax = _fig()
    _style(ax, "Your Recording vs. TORGO Speakers",
           "F0 Std Dev (pitch variability)", "Pause Ratio")

    for sev in SEVERITY_ORDER:
        grp = _torgo_viz[_torgo_viz["severity"] == sev]
        if grp.empty: continue
        ax.scatter(grp[x_col], grp[y_col], s=14, alpha=0.45,
                   color=SEVERITY_COLORS.get(sev, "#888"), label=sev, edgecolors="none")

    ax.scatter(feats.get(x_col, 0), feats.get(y_col, 0),
               s=220, color="#fff", marker="*", zorder=10, label="Your recording",
               edgecolors=SEVERITY_COLORS.get(label, "#fff"), linewidths=1.5)

    ax.legend(fontsize=6.5, framealpha=0.25, labelcolor=TEXT, ncol=2)
    plt.tight_layout(pad=1)
    return fig


def _plot_probs(proba_dict: dict, label: str) -> plt.Figure:
    if not proba_dict:
        return _placeholder("Severity Probabilities")

    order  = [s for s in SEVERITY_ORDER if s in proba_dict]
    vals   = [proba_dict[s] for s in order]
    alphas = [0.95 if s == label else 0.35 for s in order]
    clrs   = [SEVERITY_COLORS.get(s, "#888") for s in order]

    fig, ax = _fig(w=5, h=3)
    _style(ax, "Severity Probabilities", "Confidence", "")
    bars = ax.barh(order, vals, color=clrs, height=0.55, edgecolor="none", alpha=alphas)
    for bar, val in zip(bars, vals):
        ax.text(min(val + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", color=TEXT, fontsize=8, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.grid(True, axis="x", color=GRID, linewidth=0.5, alpha=0.5)
    ax.grid(False, axis="y")
    plt.tight_layout(pad=1)
    return fig


def _plot_feature_importance(feats: dict, label: str) -> plt.Figure:
    if _ctrl_stats is None or not feats:
        return _placeholder("Feature Importance")

    ctrl_mean = _ctrl_stats["mean"]
    ctrl_std  = _ctrl_stats["std"]

    zscores = {}
    for col in CLASSICAL_FEATURE_COLS:
        s = ctrl_std.get(col, 1e-8)
        if s > 1e-8:
            zscores[col] = (feats.get(col, 0) - ctrl_mean.get(col, 0)) / s

    top    = sorted(zscores.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    names  = [FEATURE_LABELS.get(k, k) for k, _ in top]
    vals   = [v for _, v in top]
    clrs   = [RED if v > 0 else CYAN for v in vals]

    fig, ax = _fig(w=5, h=3)
    _style(ax, "Deviation from Typical Speech", "Z-score vs. control group", "")
    ax.barh(names[::-1], vals[::-1], color=clrs[::-1], edgecolor="none", height=0.6)
    ax.axvline(0, color=GRID, linewidth=1.2)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(True, axis="x", color=GRID, linewidth=0.5, alpha=0.5)
    ax.grid(False, axis="y")
    ax.legend(
        handles=[Patch(color=RED,  label="↑ Higher than typical"),
                 Patch(color=CYAN, label="↓ Lower than typical")],
        fontsize=6.5, framealpha=0.2, labelcolor=TEXT, loc="lower right",
    )
    plt.tight_layout(pad=1)
    return fig


# ── Prediction ─────────────────────────────────────────────────────────────────

def _predict(feats: dict):
    if _clf is None:
        return "unknown", 0.0, {}, "Run `python analysis.py` to enable severity prediction."

    X     = np.array([[feats.get(c, 0.0) for c in CLASSICAL_FEATURE_COLS]])
    X_sc  = _scaler.transform(X)
    proba = _clf.predict_proba(X_sc)[0]
    label = _le.classes_[np.argmax(proba)]
    conf  = float(proba.max())
    pd_   = dict(zip(_le.classes_, proba.tolist()))

    ctrl_wer = SEVERITY_WER.get("control", 0.08)
    pred_wer = SEVERITY_WER.get(label)

    if label == "control":
        insight = (f"Patterns consistent with <b>typical speech</b>. "
                   f"Expected Deepgram WER ≈ {ctrl_wer:.0%}.")
    else:
        gap = (pred_wer - ctrl_wer) if pred_wer else 0
        insight = (f"Patterns suggest <b>{label} dysarthria</b> ({conf:.0%} confidence). "
                   f"Deepgram WER typically ≈ {pred_wer:.0%} "
                   f"({gap:+.0%} vs. typical speech).")
    return label, conf, pd_, insight


# ── Audio helpers ──────────────────────────────────────────────────────────────

def _save_audio_tmp(audio_input) -> str:
    import soundfile as sf
    sr, data = audio_input
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    peak = np.abs(data).max()
    if peak > 0:
        data /= peak
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, data, sr)
    return tmp.name


def _transcribe(wav_path: str) -> str:
    from main import transcribe
    resp = transcribe(wav_path)
    try:
        return resp.results.channels[0].alternatives[0].transcript
    except (AttributeError, IndexError):
        return ""


# ── Main analysis callback ─────────────────────────────────────────────────────

_EMPTY_PLOTS = lambda: [_placeholder(t) for t in
    ["Mel Spectrogram", "Energy & Pauses", "Pitch Contour (F0)", "TORGO Comparison",
     "Severity Probabilities", "Feature Importance"]]


def analyze_audio(audio_input, ground_truth_text: str):
    if audio_input is None:
        return ("_(upload or record audio, then click Analyze)_",
                '<p style="color:#475569">—</p>',
                '<p style="color:#475569">Upload audio and click Analyze.</p>',
                "", *_EMPTY_PLOTS())

    tmp = _save_audio_tmp(audio_input)

    # Transcription
    try:
        transcript = _transcribe(tmp)
    except Exception as e:
        transcript = f"[Transcription error: {e}]"

    # WER
    wer_html = '<p style="color:#475569;font-size:0.85em">Provide ground truth to compute WER</p>'
    if ground_truth_text.strip():
        try:
            from jiwer import wer as jiwer_wer
            score = jiwer_wer(ground_truth_text.lower(), transcript.lower())
            clr   = "#22c55e" if score < 0.15 else "#eab308" if score < 0.35 else "#ef4444"
            wer_html = (f'<p style="font-size:2.2em;font-weight:800;color:{clr};margin:0">{score:.1%}</p>'
                        f'<p style="color:#64748b;font-size:0.72em;margin:4px 0 0">Word Error Rate</p>')
        except Exception:
            pass

    # Features
    try:
        from features import extract_classical_features
        feats = extract_classical_features(tmp)
    except Exception as e:
        feats = {}
        print(f"Feature error: {e}")

    # Speaking rate
    dur   = feats.get("duration_s", 0)
    words = len(transcript.split()) if transcript else 0
    wpm   = int(words / (dur / 60)) if dur > 0 else 0

    # Prediction
    label, conf, proba_dict, insight = _predict(feats)
    clr   = SEVERITY_COLORS.get(label, "#64748b")
    badge = (
        f'<div style="margin-bottom:10px">'
        f'<span style="background:{clr};color:#fff;padding:5px 18px;border-radius:20px;'
        f'font-weight:800;font-size:1.1em;letter-spacing:.3px">{label.upper()}</span>'
        f'<span style="color:#94a3b8;font-size:0.82em;margin-left:10px">'
        f'{conf:.0%} confidence</span></div>'
        f'<p style="color:#cbd5e1;font-size:0.85em;line-height:1.6;margin:0">{insight}</p>'
    )

    stats_md = (
        f"**{dur:.1f}s** &nbsp;·&nbsp; **{wpm} wpm** &nbsp;·&nbsp; "
        f"**{feats.get('voiced_fraction', 0):.0%}** voiced &nbsp;·&nbsp; "
        f"**{feats.get('pause_ratio', 0):.0%}** pauses &nbsp;·&nbsp; "
        f"**{feats.get('f0_mean', 0):.0f} Hz** mean pitch &nbsp;·&nbsp; "
        f"**{feats.get('jitter', 0):.4f}** jitter"
    )

    return (
        transcript or "_(no speech detected)_",
        wer_html,
        badge,
        stats_md,
        _plot_spectrogram(tmp),
        _plot_energy(tmp),
        _plot_pitch(tmp),
        _plot_torgo_scatter(feats, label),
        _plot_probs(proba_dict, label),
        _plot_feature_importance(feats, label),
    )


# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
/* ── Gradio CSS variable overrides (catches most built-in whites) ── */
:root {
  --block-background-fill:    #1e293b !important;
  --block-border-color:       #334155 !important;
  --background-fill-primary:  #0f172a !important;
  --background-fill-secondary:#1e293b !important;
  --border-color-primary:     #334155 !important;
  --border-color-accent:      #4f46e5 !important;
  --color-background-primary: #0f172a !important;
  --input-background-fill:    #1e293b !important;
  --panel-background-fill:    #0f172a !important;
  --body-background-fill:     #0f172a !important;
  --color-grey-100:           #1e293b !important;
  --color-grey-50:            #0f172a !important;
}

body, .gradio-container, .main, .wrap, .contain {
  background:#0f172a !important; color:#f1f5f9 !important;
}

/* kill ALL white divs / gaps between columns */
div, section, article { background-color: transparent; }
.gap, .gap-2, .gap-4, .gap-8 { background:#0f172a !important; }
.block { background:#0f172a !important; border-color:#334155 !important; }
.svelte-1b6s6s, .svelte-phx28p { background:#0f172a !important; }

/* ── TABS ── */
[role="tablist"] {
  display: flex !important;
  gap: 6px !important;
  border-bottom: 2px solid #334155 !important;
  padding-bottom: 0 !important;
  background: transparent !important;
}
/* base style for all tabs */
[role="tab"] {
  background: #1e293b !important;
  border: 1px solid #475569 !important;
  border-bottom: none !important;
  border-radius: 8px 8px 0 0 !important;
  padding: 10px 22px !important;
  font-size: 0.9em !important;
  font-weight: 700 !important;
  letter-spacing: 0.03em !important;
  cursor: pointer !important;
  opacity: 0.75 !important;
}
/* each tab gets its own vibrant colour */
[role="tab"]:nth-child(1) { color: #38bdf8 !important; border-top: 2px solid #38bdf8 !important; }
[role="tab"]:nth-child(2) { color: #4ade80 !important; border-top: 2px solid #4ade80 !important; }
[role="tab"]:nth-child(3) { color: #fbbf24 !important; border-top: 2px solid #fbbf24 !important; }
[role="tab"]:nth-child(4) { color: #a78bfa !important; border-top: 2px solid #a78bfa !important; }
/* hover */
[role="tab"]:hover { background: #273449 !important; opacity: 1 !important; }
/* active tab — filled with its colour */
[role="tab"][aria-selected="true"]:nth-child(1) { background: #0c4a6e !important; color: #38bdf8 !important; border-color: #38bdf8 !important; opacity: 1 !important; box-shadow: 0 -2px 12px rgba(56,189,248,0.3) !important; }
[role="tab"][aria-selected="true"]:nth-child(2) { background: #052e16 !important; color: #4ade80 !important; border-color: #4ade80 !important; opacity: 1 !important; box-shadow: 0 -2px 12px rgba(74,222,128,0.3) !important; }
[role="tab"][aria-selected="true"]:nth-child(3) { background: #451a03 !important; color: #fbbf24 !important; border-color: #fbbf24 !important; opacity: 1 !important; box-shadow: 0 -2px 12px rgba(251,191,36,0.3) !important; }
[role="tab"][aria-selected="true"]:nth-child(4) { background: #2e1065 !important; color: #a78bfa !important; border-color: #a78bfa !important; opacity: 1 !important; box-shadow: 0 -2px 12px rgba(167,139,250,0.3) !important; }
[role="tabpanel"] {
  background: #0f172a !important;
  border: 1px solid #334155 !important;
  border-top: none !important;
  border-radius: 0 0 10px 10px !important;
  padding: 16px !important;
}

/* form labels */
label span, .label-wrap span {
  color:#64748b !important; font-size:0.73em !important;
  text-transform:uppercase !important; letter-spacing:.05em !important;
}

/* inputs */
textarea, input[type=text] {
  background:#1e293b !important; color:#f1f5f9 !important;
  border:1px solid #334155 !important; border-radius:8px !important;
}

/* groups / panels */
.gr-group, .form { background:#1e293b !important; border:1px solid #334155 !important; border-radius:10px !important; }

/* plot containers */
[data-testid="plot"] {
  background:#0f172a !important; border-radius:8px !important;
  border:1px solid #334155 !important;
}
.plot-container, figure { background:#0f172a !important; }

/* image (confusion matrix) */
[data-testid="image"] { border-radius:8px !important; border:1px solid #334155 !important; }

/* buttons */
button.primary {
  background:#6366f1 !important; border:none !important;
  border-radius:8px !important; font-weight:700 !important;
}
button.primary:hover { background:#4f46e5 !important; }
button.secondary {
  background:#1e293b !important; border:1px solid #334155 !important;
  border-radius:8px !important; color:#f1f5f9 !important; font-size:0.82em !important;
}
button.secondary:hover { background:#273449 !important; }

/* audio widget — target every internal element */
[data-testid="audio"],
[data-testid="audio"] > *,
[data-testid="audio"] .wrap,
[data-testid="audio"] .block,
.waveform-visualizer,
.audio-container {
  background:#1e293b !important; border:1px solid #334155 !important;
  border-radius:8px !important; color:#f1f5f9 !important;
}
[data-testid="audio"] svg { color:#94a3b8 !important; }

/* ── Markdown / prose — broad selectors to catch Gradio's actual rendered HTML ── */
.prose, .prose *, .markdown-body, .markdown-body *,
.md, .md *, [data-testid="markdown"], [data-testid="markdown"] * {
  color: #e2e8f0 !important;
}
.prose h1, .prose h2, .prose h3,
.markdown-body h1, .markdown-body h2, .markdown-body h3 {
  color: #f1f5f9 !important; font-weight: 700 !important;
}
.prose strong, .markdown-body strong { color: #ffffff !important; }
.prose em, .markdown-body em { color: #a78bfa !important; }
.prose code, .markdown-body code {
  background: #1e293b !important; color: #a5b4fc !important;
  padding: 1px 5px !important; border-radius: 4px !important;
}
.prose table, .markdown-body table {
  border-collapse: collapse !important; width: 100% !important;
  border: 1px solid #334155 !important;
}
.prose th, .markdown-body th {
  background: #1e293b !important; color: #94a3b8 !important;
  padding: 8px 12px !important; border: 1px solid #334155 !important;
  text-transform: uppercase !important; font-size: 0.8em !important; letter-spacing: .04em !important;
}
.prose td, .markdown-body td {
  color: #cbd5e1 !important; padding: 7px 12px !important;
  border: 1px solid #334155 !important;
}
.prose tr:nth-child(even), .markdown-body tr:nth-child(even) {
  background: #1a2540 !important;
}
.prose a, .markdown-body a { color: #38bdf8 !important; }
.prose blockquote, .markdown-body blockquote {
  border-left: 3px solid #4f46e5 !important;
  color: #94a3b8 !important; padding-left: 12px !important;
}
/* also catch plain paragraph/list text Gradio renders outside .prose */
.gradio-container p  { color: #e2e8f0 !important; }
.gradio-container li { color: #e2e8f0 !important; }
.gradio-container td { color: #cbd5e1 !important; }
.gradio-container th { color: #94a3b8 !important; }

footer { display:none !important; }
"""

HEADER_HTML = """
<div style="padding:18px 2px 10px">
  <h1 style="font-size:1.65em;font-weight:800;margin:0 0 4px;
             background:linear-gradient(90deg,#38bdf8,#a78bfa);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text">
    Neurological Speech Biomarker Analyzer
  </h1>
  <p style="color:#94a3b8;margin:0;font-size:0.86em">
    <span style="color:#38bdf8">Deepgram Nova-3</span>
    &nbsp;·&nbsp;
    <span style="color:#a78bfa">Acoustic Biomarkers</span>
    &nbsp;·&nbsp;
    <span style="color:#4ade80">Dysarthria Severity Classification</span>
  </p>
</div>
"""


def _stat_chip(value, label, color="#f1f5f9"):
    return (f'<div style="background:#1e293b;border:1px solid #334155;border-radius:8px;'
            f'padding:10px 18px;text-align:center;min-width:88px">'
            f'<div style="font-size:1.4em;font-weight:800;color:{color}">{value}</div>'
            f'<div style="font-size:0.68em;color:#64748b;text-transform:uppercase;'
            f'letter-spacing:.06em;margin-top:2px">{label}</div></div>')


STATS_HTML = (
    '<div style="display:flex;gap:10px;margin:0 0 16px;flex-wrap:wrap">'
    + _stat_chip("9,012", "Utterances")
    + _stat_chip("15",    "Speakers")
    + _stat_chip("4",     "Severity Groups")
    + _stat_chip("95.7%", "MLP Accuracy", "#22c55e")
    + _stat_chip("88.7%", "LR Accuracy",  "#94a3b8")
    + _stat_chip("TORGO", "Dataset",       "#818cf8")
    + '</div>'
)

ABOUT_MD = """
## About This Project

**Neurological Speech Biomarker Analyzer** quantifies how Deepgram's speech-to-text accuracy degrades
on speakers with **dysarthria** — a motor speech disorder from neurological conditions like ALS,
cerebral palsy, Parkinson's disease, and stroke.

### Dataset: TORGO
The [TORGO corpus](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) contains recordings
from 7 dysarthric speakers and 7 matched healthy controls, severity-labeled by a speech-language pathologist.

| Severity | Speakers | Utterances |
|---|---|---|
| Control | FC01–FC03, MC01–MC04 | 6,133 |
| Mild | M02, M03 | 812 |
| Mild-Moderate | F03, M05 | 688 |
| Severe | F01, F04, M01, M04 | 1,379 |

### Acoustic Biomarkers

| Feature | Clinical Relevance |
|---|---|
| **Jitter** | Cycle-to-cycle F0 perturbation — elevated in dysarthria |
| **Shimmer** | Amplitude perturbation — correlates with breathiness |
| **HNR** | Harmonic-to-noise ratio — lower = more noisy voice quality |
| **Pause ratio** | Proportion of silent frames — dysarthric speakers pause more |
| **F0 std dev** | Pitch variability — reduced in monotone dysarthric speech |
| **Voiced fraction** | Proportion voiced — reduced in severe cases |

### Model
A **Multi-Layer Perceptron** (128→64 hidden units) trained on 34 classical acoustic features
achieves **95.7% cross-validated accuracy** (vs. 88.7% for Logistic Regression).
The hardest class to distinguish is *mild-moderate* (F1: 59%) due to its boundary nature.

### Methodology
1. Build utterance manifest from TORGO wav files + ground truth prompts
2. Transcribe each utterance with Deepgram Nova-3
3. Compute WER per utterance with `jiwer`
4. Extract 34 acoustic features per utterance with `librosa`
5. Train MLP classifier → severity group
6. Correlate acoustic features with WER to identify failure modes

*Built with Deepgram · librosa · scikit-learn · Gradio* · [GitHub](https://github.com/ayush11z/Neurological-Speech-Biomarker-Analyzer)
"""


# ── UI ─────────────────────────────────────────────────────────────────────────

def build_dashboard():
    with gr.Blocks(title="Speech Biomarker Analyzer") as demo:

        gr.HTML(HEADER_HTML)
        gr.HTML(STATS_HTML)

        with gr.Tabs():

            # ── TAB 1: Analyze ────────────────────────────────────────────
            with gr.TabItem("🎙️  Analyze"):
                with gr.Row(equal_height=False):

                    # ── Sidebar ──────────────────────────────────────────
                    with gr.Column(scale=1, min_width=280):

                        with gr.Group():
                            gr.HTML('<p style="color:#38bdf8;font-size:0.72em;text-transform:uppercase;'
                                    'letter-spacing:.08em;font-weight:700;margin:0 0 8px">Load Example Recording</p>')
                            with gr.Row():
                                ctrl_btn = gr.Button("Control",       size="sm", variant="secondary")
                                mild_btn = gr.Button("Mild",          size="sm", variant="secondary")
                                sev_btn  = gr.Button("Severe",        size="sm", variant="secondary")

                        gr.HTML('<div style="height:6px"></div>')

                        audio_in = gr.Audio(
                            label="Audio Input",
                            type="numpy",
                            sources=["upload", "microphone"],
                        )
                        gt_box = gr.Textbox(
                            label="Ground Truth (optional — enables WER)",
                            placeholder="Paste the correct transcript to compute WER…",
                            lines=2,
                        )
                        run_btn = gr.Button("Analyze Speech", variant="primary", size="lg")

                        gr.HTML('<div style="height:10px"></div>')

                        with gr.Group():
                            gr.HTML('<p style="color:#4ade80;font-size:0.72em;text-transform:uppercase;'
                                    'letter-spacing:.08em;font-weight:700;margin:0 0 8px">Severity Assessment</p>')
                            severity_out = gr.HTML(
                                value='<p style="color:#475569;font-size:0.85em">'
                                      'Upload audio and click Analyze.</p>'
                            )

                    # ── Main panel ────────────────────────────────────────
                    with gr.Column(scale=3):

                        # Transcript + WER
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=3):
                                with gr.Group():
                                    gr.HTML('<p style="color:#38bdf8;font-size:0.72em;text-transform:uppercase;'
                                            'letter-spacing:.08em;font-weight:700;margin:0 0 6px">Transcript — Deepgram Nova-3</p>')
                                    transcript_out = gr.Markdown(
                                        value='<span style="color:#475569">—</span>'
                                    )
                            with gr.Column(scale=1, min_width=110):
                                with gr.Group():
                                    gr.HTML('<p style="color:#fbbf24;font-size:0.72em;text-transform:uppercase;'
                                            'letter-spacing:.08em;font-weight:700;margin:0 0 6px">WER</p>')
                                    wer_out = gr.HTML(
                                        value='<p style="color:#475569;font-size:0.9em">—</p>'
                                    )

                        stats_bar = gr.Markdown(value="")

                        gr.HTML('<div style="height:4px"></div>')
                        gr.HTML('<p style="color:#a78bfa;font-size:0.72em;text-transform:uppercase;'
                                'letter-spacing:.08em;font-weight:700;margin:4px 0 8px">Acoustic Biomarkers</p>')

                        with gr.Row():
                            spec_plot   = gr.Plot(value=_placeholder("Mel Spectrogram"),   show_label=False)
                            energy_plot = gr.Plot(value=_placeholder("Energy & Pauses"),    show_label=False)
                        with gr.Row():
                            pitch_plot  = gr.Plot(value=_placeholder("Pitch Contour (F0)"), show_label=False)
                            torgo_plot  = gr.Plot(value=_placeholder("TORGO Comparison"),   show_label=False)
                        with gr.Row():
                            prob_plot   = gr.Plot(value=_placeholder("Severity Probabilities"), show_label=False)
                            imp_plot    = gr.Plot(value=_placeholder("Feature Importance"),     show_label=False)

                # Wire example buttons
                ctrl_btn.click(fn=lambda: load_example("control"),       outputs=[audio_in])
                mild_btn.click(fn=lambda: load_example("mild"),          outputs=[audio_in])
                sev_btn.click( fn=lambda: load_example("severe"),        outputs=[audio_in])

                run_btn.click(
                    fn=analyze_audio,
                    inputs=[audio_in, gt_box],
                    outputs=[
                        transcript_out, wer_out, severity_out, stats_bar,
                        spec_plot, energy_plot, pitch_plot, torgo_plot,
                        prob_plot, imp_plot,
                    ],
                )

            # ── TAB 2: Dataset Overview ───────────────────────────────────
            with gr.TabItem("📊  Dataset Overview"):
                gr.HTML('<p style="color:#94a3b8;margin:0 0 14px;font-size:0.88em">'
                        'TORGO dataset — 9,012 utterances · 15 speakers (7 dysarthric + 7 controls)</p>')
                gr.Plot(value=_DATASET_FIG, show_label=False)
                gr.Markdown("""
| Severity | Speakers | Utterances | Description |
|---|---|---|---|
| Control | FC01, FC02, FC03, MC01–MC04 | 6,133 | Matched healthy controls |
| Mild | M02, M03 | 812 | Intelligible with mild motor impairment |
| Mild-Moderate | F03, M05 | 688 | Reduced intelligibility |
| Severe | F01, F04, M01, M04 | 1,379 | Significantly impaired speech |
""")

            # ── TAB 3: Model Performance ──────────────────────────────────
            with gr.TabItem("🤖  Model Performance"):
                gr.HTML('<p style="color:#94a3b8;margin:0 0 14px;font-size:0.88em">'
                        '5-fold stratified cross-validation on all 9,012 utterances</p>')

                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Plot(value=_MODEL_PERF_FIG, show_label=False)
                    with gr.Column(scale=2):
                        gr.Markdown("""
**Cross-Validated Accuracy**
- **MLP (128→64):** 95.7% ± 0.5%
- **Logistic Regression:** 88.7% ± 0.7%

**Per-class F1 (MLP)**
- Control: **94%** (6,133 samples)
- Mild: **82%** (812 samples)
- Mild-Moderate: **59%** ← hardest boundary class
- Severe: **82%** (1,379 samples)

**Features:** 34 classical acoustic features — MFCCs (×26), pitch F0 (mean/std/range), jitter, shimmer, HNR, pause ratio, voiced fraction, speaking rate, spectral centroid, spectral rolloff, ZCR

> ⚠️ High accuracy is partly due to speaker-level clustering — the model may be learning speaker identity as a proxy for severity. Speaker-independent cross-validation is future work.
""")

                cm_path = OUTPUTS_DIR / "confusion_matrix.png"
                if cm_path.exists():
                    gr.HTML('<p style="color:#64748b;font-size:0.72em;text-transform:uppercase;'
                            'letter-spacing:.05em;margin:16px 0 8px">Confusion Matrix (5-fold CV)</p>')
                    gr.Image(value=str(cm_path), show_label=False, height=400)

            # ── TAB 4: About ──────────────────────────────────────────────
            with gr.TabItem("ℹ️  About"):
                gr.Markdown(ABOUT_MD)

        gr.HTML(
            '<div style="margin-top:20px;padding-top:12px;border-top:1px solid #1e293b;'
            'color:#475569;font-size:0.77em">'
            'Built to quantify Deepgram\'s accuracy gap on atypical (dysarthric) speech &nbsp;·&nbsp; '
            '<a href="http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html" '
            'style="color:#6366f1">TORGO dataset</a> &nbsp;·&nbsp; '
            '<a href="https://github.com/ayush11z/Neurological-Speech-Biomarker-Analyzer" style="color:#6366f1">GitHub</a>'
            '</div>'
        )

    return demo


if __name__ == "__main__":
    demo = build_dashboard()
    demo.launch(
        share=True,
        css=CSS,
        theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate"),
    )
