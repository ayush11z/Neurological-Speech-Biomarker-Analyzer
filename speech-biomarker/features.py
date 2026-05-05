"""
Phase 3: Acoustic Feature Extraction
Classical features via librosa + deep embeddings via wav2vec2.
"""
import numpy as np
import pandas as pd
import librosa
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

CLASSICAL_FEATURE_COLS = [
    *[f"mfcc_{i}_mean" for i in range(13)],
    *[f"mfcc_{i}_std" for i in range(13)],
    "f0_mean", "f0_std", "f0_range", "jitter",
    "rms_mean", "rms_std", "shimmer", "hnr",
    "voiced_fraction", "pause_ratio", "duration_s",
    "spectral_centroid_mean", "spectral_rolloff_mean", "zcr_mean",
]


def extract_classical_features(wav_path: str, sr: int = 16000) -> dict:
    """Return a flat dict of scalar acoustic features for one utterance."""
    y, _ = librosa.load(wav_path, sr=sr)
    feats = {}

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
        feats[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))

    # Pitch (F0) via pyin
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    voiced_f0 = f0[voiced_flag.astype(bool)] if voiced_flag is not None else np.array([])
    feats["f0_mean"] = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    feats["f0_std"] = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    feats["f0_range"] = float(np.ptp(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    # Jitter: average absolute F0 cycle-to-cycle variation / mean F0
    if len(voiced_f0) > 1:
        feats["jitter"] = float(np.mean(np.abs(np.diff(voiced_f0))) / (np.mean(voiced_f0) + 1e-8))
    else:
        feats["jitter"] = 0.0

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"] = float(np.std(rms))

    # Shimmer proxy: amplitude variation relative to mean amplitude
    feats["shimmer"] = float(np.std(rms) / (np.mean(rms) + 1e-8))

    # HNR: harmonic / residual energy ratio
    harmonic = librosa.effects.harmonic(y)
    noise = y - harmonic
    feats["hnr"] = float(
        10 * np.log10(np.mean(harmonic ** 2) / (np.mean(noise ** 2) + 1e-8) + 1e-8)
    )

    # Voiced fraction and pause ratio
    total_frames = len(f0) if f0 is not None else 1
    voiced_frames = int(np.sum(voiced_flag.astype(bool))) if voiced_flag is not None else 0
    feats["voiced_fraction"] = float(voiced_frames / (total_frames + 1e-8))

    silence_threshold = 0.01 * (np.max(rms) + 1e-8)
    feats["pause_ratio"] = float(np.sum(rms < silence_threshold) / (len(rms) + 1e-8))

    feats["duration_s"] = float(len(y) / sr)

    # Spectral features
    feats["spectral_centroid_mean"] = float(
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    )
    feats["spectral_rolloff_mean"] = float(
        np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
    )
    feats["zcr_mean"] = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

    return feats


class Wav2VecExtractor:
    """Mean-pool the final hidden states of facebook/wav2vec2-base."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        print(f"Loading {model_name}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print(f"wav2vec2 on {self.device}")

    @torch.no_grad()
    def extract(self, wav_path: str, sr: int = 16000) -> np.ndarray:
        """Returns a 768-dim float32 embedding."""
        y, _ = librosa.load(wav_path, sr=sr)
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def extract_all_features(
    manifest_path: str,
    out_path: str = "features.csv",
    embeddings_path: str = "wav2vec_embeddings.npy",
    include_wav2vec: bool = True,
) -> pd.DataFrame:
    """
    Extract features for every utterance in the manifest.
    Saves incrementally so interruptions are resumable.
    """
    df = pd.read_csv(manifest_path)

    # Resume support
    if Path(out_path).exists():
        done = pd.read_csv(out_path)
        done_paths = set(done["wav_path"])
        remaining = df[~df["wav_path"].isin(done_paths)]
        print(f"Resuming: {len(done_paths)} done, {len(remaining)} remaining")
    else:
        done = pd.DataFrame()
        remaining = df

    if remaining.empty:
        print("All features already extracted.")
        return pd.read_csv(out_path)

    wav2vec = Wav2VecExtractor() if include_wav2vec else None
    classical_rows, embeddings = [], []

    for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Extracting features"):
        try:
            feats = extract_classical_features(row["wav_path"])
        except Exception as e:
            print(f"\nClassical features failed ({row['wav_path']}): {e}")
            feats = {col: 0.0 for col in CLASSICAL_FEATURE_COLS}

        classical_rows.append({**row.to_dict(), **feats})

        if wav2vec:
            try:
                emb = wav2vec.extract(row["wav_path"])
            except Exception as e:
                print(f"\nwav2vec failed ({row['wav_path']}): {e}")
                emb = np.zeros(768, dtype=np.float32)
            embeddings.append(emb)

    new_df = pd.DataFrame(classical_rows)
    all_df = pd.concat([done, new_df], ignore_index=True) if len(done) else new_df
    all_df.to_csv(out_path, index=False)
    print(f"Features saved to {out_path}")

    if include_wav2vec and embeddings:
        new_emb = np.array(embeddings, dtype=np.float32)
        if Path(embeddings_path).exists():
            existing = np.load(embeddings_path)
            new_emb = np.vstack([existing, new_emb])
        np.save(embeddings_path, new_emb)
        print(f"Embeddings saved: {new_emb.shape}")

    return all_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract acoustic features from TORGO")
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--out", default="features.csv")
    parser.add_argument("--embeddings", default="wav2vec_embeddings.npy")
    parser.add_argument("--no-wav2vec", action="store_true")
    args = parser.parse_args()

    extract_all_features(args.manifest, args.out, args.embeddings,
                         include_wav2vec=not args.no_wav2vec)
