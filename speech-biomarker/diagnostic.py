"""
Quick diagnostic: build manifest, extract features from sample utterances,
and show per-group stats.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
import numpy as np
import pandas as pd

TORGO_ROOT = Path(__file__).parent.parent  # .../deepgram/
SAMPLES_PER_GROUP = 5  # utterances per severity group for the summary

# ── 1. Manifest ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. BUILDING MANIFEST")
print("=" * 60, flush=True)

from pipeline import build_manifest
df = build_manifest(str(TORGO_ROOT))

print(f"\nTotal utterances: {len(df)}")
print(f"Speakers found:   {sorted(df['speaker_id'].unique().tolist())}")
print("\nUtterances per speaker/severity:")
print(df.groupby(["speaker_id", "severity"])["utterance_id"].count().to_string())

# ── 2. Feature extraction sample ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. FEATURE EXTRACTION (3 samples per severity)")
print("=" * 60, flush=True)

from features import extract_classical_features

sample_rows = (
    df.groupby("severity", group_keys=False)
    .apply(lambda g: g.sample(min(3, len(g)), random_state=42), include_groups=False)
    .reset_index(drop=True)
)
# Re-attach severity column lost by include_groups=False
sample_rows = df.loc[sample_rows.index]

for _, row in sample_rows.iterrows():
    try:
        feats = extract_classical_features(row["wav_path"])
        print(f"  OK  {row['speaker_id']}/{row['utterance_id']} ({row['severity']})  "
              f"dur={feats['duration_s']:.2f}s  f0={feats['f0_mean']:.1f}Hz  "
              f"pause={feats['pause_ratio']:.2f}  hnr={feats['hnr']:.1f}dB",
              flush=True)
    except Exception as e:
        print(f"  ERR {row['speaker_id']}/{row['utterance_id']}: {e}", flush=True)

# ── 3. Group-level acoustic summary (sampled) ─────────────────────────────────
print("\n" + "=" * 60)
print(f"3. ACOUSTIC SUMMARY BY SEVERITY ({SAMPLES_PER_GROUP} samples/group)")
print("=" * 60, flush=True)

key_cols = ["f0_mean", "f0_std", "jitter", "shimmer", "hnr", "pause_ratio",
            "voiced_fraction", "duration_s"]

summary_rows = (
    df.groupby("severity", group_keys=False)
    .apply(lambda g: g.sample(min(SAMPLES_PER_GROUP, len(g)), random_state=0))
)

all_feats = []
for _, row in summary_rows.iterrows():
    try:
        feats = extract_classical_features(row["wav_path"])
        feats["severity"] = row["severity"]
        feats["speaker_id"] = row["speaker_id"]
        all_feats.append({k: feats[k] for k in key_cols + ["severity", "speaker_id"]})
        print(f"  extracted {row['speaker_id']}/{row['utterance_id']}", flush=True)
    except Exception as e:
        print(f"  failed  {row['speaker_id']}/{row['utterance_id']}: {e}", flush=True)

feat_df = pd.DataFrame(all_feats)
print("\nMean values per severity group:")
print(feat_df.groupby("severity")[key_cols].mean().round(3).to_string())

os.makedirs(Path(__file__).parent / "data", exist_ok=True)
feat_df.to_csv(Path(__file__).parent / "data" / "feature_summary.csv", index=False)
print(f"\nSample feature summary saved to data/feature_summary.csv")
