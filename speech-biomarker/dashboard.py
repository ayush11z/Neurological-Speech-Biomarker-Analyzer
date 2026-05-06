"""
Phase 5: Gradio Dashboard
Real-time speech analysis: Deepgram transcript + acoustic biomarkers + severity prediction.
"""
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import gradio as gr
import joblib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = Path(__file__).parent / "models"
CLASSICAL_FEATURE_COLS = [
    *[f"mfcc_{i}_mean" for i in range(13)],
    *[f"mfcc_{i}_std" for i in range(13)],
    "f0_mean", "f0_std", "f0_range", "jitter",
    "rms_mean", "rms_std", "shimmer", "hnr",
    "voiced_fraction", "pause_ratio", "duration_s",
    "spectral_centroid_mean", "spectral_rolloff_mean", "zcr_mean",
]

# Typical WER values per severity from the benchmark (update after running benchmark.py)
SEVERITY_WER_BASELINE = {
    "control": 0.08,
    "mild": 0.18,
    "mild-moderate": 0.32,
    "moderate": 0.38,
    "severe": 0.53,
}

_classifier = _scaler = _label_encoder = None


def _load_models():
    global _classifier, _scaler, _label_encoder
    try:
        _classifier = joblib.load(MODEL_DIR / "severity_classifier.pkl")
        _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        _label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
        print("Severity models loaded.")
    except FileNotFoundError:
        print("No trained models found — run analysis.py first. Severity prediction disabled.")


_load_models()


# ---------- Helpers ----------

def _save_audio_tmp(audio_input) -> str:
    """Convert Gradio audio tuple to a temp WAV file and return its path."""
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


def _predict_severity(features: dict) -> str:
    if _classifier is None:
        return "Severity model not loaded — run `python analysis.py` first."

    X = np.array([[features.get(col, 0.0) for col in CLASSICAL_FEATURE_COLS]])
    X_scaled = _scaler.transform(X)
    proba = _classifier.predict_proba(X_scaled)[0]
    pred_label = _label_encoder.classes_[np.argmax(proba)]
    confidence = float(proba.max())

    prob_lines = "\n".join(
        f"- **{lbl}**: {p:.0%}"
        for lbl, p in sorted(zip(_label_encoder.classes_, proba), key=lambda x: -x[1])
    )

    control_wer = SEVERITY_WER_BASELINE.get("control", 0.08)
    pred_wer = SEVERITY_WER_BASELINE.get(pred_label, "N/A")

    if pred_label == "control":
        summary = (
            f"This speech shows markers consistent with **typical speech** (control group).\n"
            f"Deepgram WER on similar speech: ~{pred_wer:.0%}"
        )
    else:
        summary = (
            f"This speech shows markers consistent with **{pred_label} dysarthria** "
            f"(confidence: {confidence:.0%}).\n"
            f"Deepgram WER on similar speech: ~{pred_wer:.0%} "
            f"vs ~{control_wer:.0%} on typical speech."
        )

    return f"{summary}\n\n**Severity probabilities:**\n{prob_lines}"


# ---------- Plot generators ----------

def _plot_mfcc_energy(wav_path: str) -> plt.Figure:
    y, sr = librosa.load(wav_path, sr=16000)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img = librosa.display.specshow(mfccs, x_axis="time", sr=sr, ax=axes[0])
    axes[0].set_title("MFCC Heatmap")
    axes[0].set_ylabel("MFCC Coefficient")
    fig.colorbar(img, ax=axes[0])

    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    threshold = 0.01 * (rms.max() + 1e-8)
    axes[1].plot(times, rms, color="steelblue", linewidth=1.2, label="RMS Energy")
    axes[1].axhline(threshold, color="red", linestyle="--", alpha=0.6, label="Pause threshold")
    axes[1].fill_between(times, 0, rms, where=(rms < threshold),
                          color="red", alpha=0.25, label="Pause regions")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("RMS Energy")
    axes[1].set_title("Energy & Pause Detection")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    return fig


def _plot_pitch(wav_path: str) -> plt.Figure:
    y, sr = librosa.load(wav_path, sr=16000)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)

    voiced_mask = voiced_flag.astype(bool) if voiced_flag is not None else np.zeros(len(f0), bool)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(times[voiced_mask], f0[voiced_mask], s=5, color="darkorange", label="F0 (voiced)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Pitch Contour (F0)")
    ax.legend(fontsize=8)
    voiced_f0 = f0[voiced_mask]
    if len(voiced_f0) > 0:
        ax.set_ylim(0, min(600, voiced_f0.max() * 1.25))
    plt.tight_layout()
    return fig


# ---------- Main analysis function ----------

def analyze_audio(audio_input, ground_truth_text: str):
    if audio_input is None:
        return "Upload or record audio first.", None, None, "—"

    tmp_path = _save_audio_tmp(audio_input)

    # Transcription
    try:
        transcript = _transcribe(tmp_path)
    except Exception as e:
        transcript = f"[Transcription error: {e}]"

    wer_line = ""
    if ground_truth_text.strip():
        from jiwer import wer as compute_wer
        score = compute_wer(ground_truth_text.lower(), transcript.lower())
        wer_line = f"\n\n**WER: {score:.1%}**"

    transcript_md = f"**Deepgram Nova-3 Transcript:**\n\n> {transcript}{wer_line}"

    # Feature extraction
    try:
        from features import extract_classical_features
        feats = extract_classical_features(tmp_path)
    except Exception as e:
        feats = {}
        print(f"Feature extraction error: {e}")

    mfcc_fig = _plot_mfcc_energy(tmp_path)
    pitch_fig = _plot_pitch(tmp_path)
    severity_md = _predict_severity(feats)

    return transcript_md, mfcc_fig, pitch_fig, severity_md


# ---------- UI ----------

def build_dashboard():
    with gr.Blocks(title="Neurological Speech Biomarker Analyzer") as demo:
        gr.Markdown("""
# Neurological Speech Biomarker Analyzer
**Deepgram + Acoustic Biomarkers for Dysarthria Assessment**

Upload or record an audio sample to see:
- Deepgram Nova-3 transcription (with optional WER if you provide the correct text)
- Acoustic biomarker plots: pitch contour, MFCC heatmap, energy & pause detection
- ML-based severity prediction with confidence breakdown
""")
        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(label="Input Audio", type="numpy", sources=["upload", "microphone"])
                gt_box = gr.Textbox(
                    label="Ground Truth (optional — enables WER)",
                    placeholder="Type the correct transcript to compute WER...",
                    lines=2,
                )
                run_btn = gr.Button("Analyze", variant="primary", size="lg")

            with gr.Column(scale=1):
                transcript_out = gr.Markdown(label="Transcription")
                severity_out = gr.Markdown(label="Severity Prediction")

        with gr.Row():
            mfcc_plot = gr.Plot(label="MFCC Heatmap & Energy / Pauses")
            pitch_plot = gr.Plot(label="Pitch Contour (F0)")

        run_btn.click(
            fn=analyze_audio,
            inputs=[audio_in, gt_box],
            outputs=[transcript_out, mfcc_plot, pitch_plot, severity_out],
        )

        gr.Markdown("""
---
Built to quantify Deepgram's accuracy gap on atypical (dysarthric) speech using the
[TORGO dataset](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html).
See the [GitHub repo](https://github.com/ayush11z/speech-biomarker) for methodology and benchmark results.
""")

    return demo


if __name__ == "__main__":
    demo = build_dashboard()
    demo.launch(share=True, theme=gr.themes.Soft())
