"""
Orchestrate all five phases end-to-end.

Usage:
    python run_all.py --torgo /path/to/TORGO
    python run_all.py --torgo /path/to/TORGO --limit 50  # quick smoke test
    python run_all.py --skip-benchmark                   # features + analysis only
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Run full speech-biomarker pipeline")
parser.add_argument("--torgo", required=True, help="Path to TORGO dataset root")
parser.add_argument("--mic", default="arrayMic", choices=["arrayMic", "headMic"])
parser.add_argument("--manifest", default="manifest.csv")
parser.add_argument("--benchmark-out", default="benchmark_results.csv")
parser.add_argument("--features-out", default="features.csv")
parser.add_argument("--embeddings-out", default="wav2vec_embeddings.npy")
parser.add_argument("--limit", type=int, default=None, help="Cap utterances for testing")
parser.add_argument("--delay", type=float, default=0.4)
parser.add_argument("--no-wav2vec", action="store_true")
parser.add_argument("--skip-benchmark", action="store_true")
parser.add_argument("--figs", default="figures")
parser.add_argument("--models", default="models")
args = parser.parse_args()

print("=" * 60)
print("Phase 1: Building manifest from TORGO dataset")
print("=" * 60)
from pipeline import build_manifest
df = build_manifest(args.torgo, args.mic)
df.to_csv(args.manifest, index=False)
print(f"Saved {args.manifest}")

if not args.skip_benchmark:
    print("\n" + "=" * 60)
    print("Phase 2: Deepgram benchmark")
    print("=" * 60)
    from benchmark import run_benchmark, analyze_results
    run_benchmark(args.manifest, args.benchmark_out, delay=args.delay, limit=args.limit)
    analyze_results(args.benchmark_out, args.figs)
else:
    print("\nSkipping Phase 2 (--skip-benchmark)")

print("\n" + "=" * 60)
print("Phase 3: Acoustic feature extraction")
print("=" * 60)
from features import extract_all_features
extract_all_features(
    args.manifest, args.features_out, args.embeddings_out,
    include_wav2vec=not args.no_wav2vec,
)

print("\n" + "=" * 60)
print("Phase 4: Analysis & classifier training")
print("=" * 60)
from analysis import correlate_with_wer, train_severity_classifier
if Path(args.benchmark_out).exists():
    correlate_with_wer(args.features_out, args.benchmark_out, args.figs)
train_severity_classifier(
    args.features_out,
    args.embeddings_out if not args.no_wav2vec else None,
    args.figs, args.models,
)

print("\n" + "=" * 60)
print("All phases complete. Launch the dashboard with:")
print("  python dashboard.py")
print("=" * 60)
