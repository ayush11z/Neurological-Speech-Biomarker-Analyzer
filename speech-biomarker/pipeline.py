"""
Phase 1: Data Pipeline
Build a clean manifest CSV from the TORGO dataset.
"""
import pandas as pd
from pathlib import Path

# From TORGO paper Table 2 — severity per dysarthric speaker
SEVERITY_MAP = {
    "F01": "severe", "F03": "mild-moderate", "F04": "severe",
    "FC01": "control", "FC02": "control", "FC03": "control",
    "M01": "severe", "M02": "mild", "M03": "mild",
    "M04": "severe", "M05": "mild-moderate",
    "MC01": "control", "MC02": "control", "MC03": "control", "MC04": "control",
}

SEX_MAP = {
    "F01": "F", "F03": "F", "F04": "F",
    "FC01": "F", "FC02": "F", "FC03": "F",
    "M01": "M", "M02": "M", "M03": "M", "M04": "M", "M05": "M",
    "MC01": "M", "MC02": "M", "MC03": "M", "MC04": "M",
}


def _load_errors(torgo_root: Path) -> set:
    """Return set of flagged utterance keys (speaker/session/utterance_id)."""
    errors = set()
    errors_path = torgo_root / "ERRORS.xls"
    if not errors_path.exists():
        return errors
    try:
        import xlrd
        wb = xlrd.open_workbook(str(errors_path))
        ws = wb.sheet_by_index(0)
        for row in range(ws.nrows):
            val = str(ws.cell_value(row, 0)).strip()
            if val:
                errors.add(val)
    except ImportError:
        print("Warning: xlrd not installed, skipping ERRORS.xls filtering")
    return errors


def build_manifest(torgo_root: str, mic: str = "arrayMic") -> pd.DataFrame:
    """
    Walk the TORGO directory tree and build a clean utterance manifest.

    Returns a DataFrame with columns:
        speaker_id, severity, sex, is_dysarthric, session,
        utterance_id, wav_path, ground_truth
    """
    root = Path(torgo_root)
    errors = _load_errors(root)
    records = []

    # Support both flat (torgo_root/F01) and nested (torgo_root/F/F01) layouts
    candidates = []
    for entry in sorted(root.iterdir()):
        if entry.name in SEVERITY_MAP:
            candidates.append(entry)
        elif entry.is_dir():
            for sub in sorted(entry.iterdir()):
                if sub.name in SEVERITY_MAP:
                    candidates.append(sub)

    for speaker_dir in candidates:
        speaker_id = speaker_dir.name
        if speaker_id not in SEVERITY_MAP:
            continue

        severity = SEVERITY_MAP[speaker_id]
        sex = SEX_MAP[speaker_id]

        for session_dir in sorted(speaker_dir.iterdir()):
            if not session_dir.is_dir() or not session_dir.name.startswith("Session"):
                continue

            wav_dir = session_dir / f"wav_{mic}"
            prompt_dir = session_dir / "prompts"
            if not wav_dir.exists() or not prompt_dir.exists():
                continue

            for wav_path in sorted(wav_dir.glob("*.wav")):
                utterance_id = wav_path.stem
                error_key = f"{speaker_id}/{session_dir.name}/{utterance_id}"
                if error_key in errors:
                    continue

                prompt_path = prompt_dir / f"{utterance_id}.txt"
                if not prompt_path.exists():
                    continue

                ground_truth = prompt_path.read_text(encoding="utf-8", errors="ignore").strip()
                # Skip non-word vocalizations
                if not ground_truth or ground_truth.lower() in ("yyy", "xxx", "---"):
                    continue

                records.append({
                    "speaker_id": speaker_id,
                    "severity": severity,
                    "sex": sex,
                    "is_dysarthric": severity != "control",
                    "session": session_dir.name,
                    "utterance_id": utterance_id,
                    "wav_path": str(wav_path.resolve()),
                    "ground_truth": ground_truth,
                })

    df = pd.DataFrame(records)
    print(f"Manifest: {len(df)} utterances | {df['speaker_id'].nunique()} speakers")
    print(df.groupby(["severity", "sex"])["utterance_id"].count().to_string())
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build TORGO manifest CSV")
    parser.add_argument("torgo_root", help="Path to TORGO dataset root directory")
    parser.add_argument("--mic", default="arrayMic", choices=["arrayMic", "headMic"])
    parser.add_argument("--out", default="manifest.csv")
    args = parser.parse_args()

    df = build_manifest(args.torgo_root, args.mic)
    df.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")
