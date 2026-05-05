"""
Phase 2: Deepgram Benchmark
Transcribe every utterance in the manifest with Deepgram Nova-3,
compute per-utterance WER, and generate severity-group summaries.
"""
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions
from jiwer import wer as jiwer_wer
from tqdm import tqdm

load_dotenv()

SEVERITY_ORDER = ["control", "mild", "mild-moderate", "moderate", "severe"]


def _transcribe(client: DeepgramClient, wav_path: str, model: str = "nova-3") -> str:
    with open(wav_path, "rb") as f:
        buf = f.read()
    options = PrerecordedOptions(model=model, smart_format=True, language="en")
    resp = client.listen.rest.v("1").transcribe_file({"buffer": buf}, options)
    try:
        return resp.results.channels[0].alternatives[0].transcript
    except (AttributeError, IndexError):
        return ""


def run_benchmark(
    manifest_path: str,
    out_path: str = "benchmark_results.csv",
    model: str = "nova-3",
    delay: float = 0.4,
    limit: int = None,
) -> pd.DataFrame:
    """
    Transcribe each utterance and record WER.
    Saves incrementally so you can resume if interrupted.
    """
    df = pd.read_csv(manifest_path)
    if limit:
        df = df.head(limit)

    # Resume support
    if Path(out_path).exists():
        done = pd.read_csv(out_path)
        done_paths = set(done["wav_path"])
        df = df[~df["wav_path"].isin(done_paths)]
        print(f"Resuming: {len(done_paths)} done, {len(df)} remaining")
    else:
        done = pd.DataFrame()

    if df.empty:
        print("All utterances already processed.")
        return pd.read_csv(out_path)

    client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Benchmarking"):
        try:
            hypothesis = _transcribe(client, row["wav_path"], model)
            utterance_wer = float(jiwer_wer(row["ground_truth"].lower(), hypothesis.lower()))
        except Exception as e:
            print(f"\nError on {row['wav_path']}: {e}")
            hypothesis, utterance_wer = "", 1.0

        rows.append({**row.to_dict(), "hypothesis": hypothesis, "wer": utterance_wer})
        time.sleep(delay)

    new_df = pd.DataFrame(rows)
    all_results = pd.concat([done, new_df], ignore_index=True) if len(done) else new_df
    all_results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    return all_results


def analyze_results(results_path: str, fig_dir: str = "figures") -> pd.DataFrame:
    """Print WER summary table and save plots."""
    df = pd.read_csv(results_path)
    Path(fig_dir).mkdir(exist_ok=True)

    present_order = [s for s in SEVERITY_ORDER if s in df["severity"].values]

    summary = (
        df.groupby("severity")["wer"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"mean": "mean_wer", "median": "median_wer",
                         "std": "std_wer", "count": "n_utterances"})
        .reindex(present_order)
        .round(4)
    )

    print("\n=== WER by Severity Group (Deepgram Nova-3) ===")
    print(summary.to_string())

    palette = sns.color_palette("RdYlGn_r", len(present_order))

    # --- Bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = axes[0].bar(present_order, summary["mean_wer"], color=palette)
    axes[0].set_title("Mean WER by Severity Group\n(Deepgram Nova-3)", fontsize=13)
    axes[0].set_xlabel("Severity")
    axes[0].set_ylabel("Word Error Rate")
    axes[0].set_ylim(0, 1.05)
    if "control" in summary.index:
        axes[0].axhline(summary.loc["control", "mean_wer"], color="green",
                        linestyle="--", alpha=0.7, label="Control baseline")
        axes[0].legend()
    for bar, val in zip(bars, summary["mean_wer"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.0%}", ha="center", va="bottom", fontsize=9)

    # --- Box plot ---
    sns.boxplot(data=df[df["severity"].isin(present_order)],
                x="severity", y="wer", order=present_order,
                palette="RdYlGn_r", ax=axes[1])
    axes[1].set_title("WER Distribution by Severity\n(Deepgram Nova-3)", fontsize=13)
    axes[1].set_xlabel("Severity")
    axes[1].set_ylabel("Word Error Rate")

    plt.tight_layout()
    fig.savefig(f"{fig_dir}/wer_by_severity.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Per-speaker bar chart ---
    speaker_wer = (
        df.groupby(["speaker_id", "severity"])["wer"]
        .mean()
        .reset_index()
        .sort_values("wer")
    )
    color_map = {
        "control": "#2ecc71", "mild": "#f1c40f",
        "mild-moderate": "#e67e22", "moderate": "#e74c3c", "severe": "#8e44ad",
    }
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.bar(
        speaker_wer["speaker_id"], speaker_wer["wer"],
        color=[color_map.get(s, "gray") for s in speaker_wer["severity"]],
    )
    ax2.set_title("Mean WER by Speaker (Deepgram Nova-3)")
    ax2.set_xlabel("Speaker ID")
    ax2.set_ylabel("Mean WER")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()]
    ax2.legend(handles, color_map.keys(), title="Severity")
    plt.tight_layout()
    fig2.savefig(f"{fig_dir}/wer_by_speaker.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nFigures saved to {fig_dir}/")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Deepgram on TORGO")
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--out", default="benchmark_results.csv")
    parser.add_argument("--model", default="nova-3")
    parser.add_argument("--delay", type=float, default=0.4, help="Seconds between API calls")
    parser.add_argument("--limit", type=int, default=None, help="Cap utterances (for testing)")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if not args.analyze_only:
        run_benchmark(args.manifest, args.out, args.model, args.delay, args.limit)
    analyze_results(args.out)
