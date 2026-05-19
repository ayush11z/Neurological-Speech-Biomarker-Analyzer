# Neurological Speech Biomarker Analyzer

A research tool that quantifies how **Deepgram Nova-3** speech-to-text accuracy degrades on speakers with **dysarthria** — a motor speech disorder caused by neurological conditions such as ALS, cerebral palsy, Parkinson's disease, and stroke.

The project combines Deepgram benchmarking, classical acoustic feature extraction, and an MLP severity classifier into an interactive Gradio dashboard.

---

## Demo

![Dashboard](outputs/wer_by_severity.png)

**Live features:**
- Upload or record audio → get a Deepgram transcript, WER, severity prediction, and 6 acoustic plots
- Load example recordings from all 15 TORGO speakers across 4 severity levels
- Explore dataset statistics and model performance in dedicated tabs

---

## Key Results

| Model | CV Accuracy |
|---|---|
| MLP (128→64) | **95.7% ± 0.5%** |
| Logistic Regression | 88.7% ± 0.7% |

| Severity | Deepgram WER (approx.) |
|---|---|
| Control | ~8% |
| Mild | ~18% |
| Mild-Moderate | ~32% |
| Severe | ~53% |

---

## Dataset: TORGO

The [TORGO corpus](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) contains recordings from 7 dysarthric speakers and 7 matched healthy controls, severity-labeled by a speech-language pathologist.

| Severity | Speakers | Utterances |
|---|---|---|
| Control | FC01–FC03, MC01–MC04 | 6,133 |
| Mild | M02, M03 | 812 |
| Mild-Moderate | F03, M05 | 688 |
| Severe | F01, F04, M01, M04 | 1,379 |
| **Total** | **15** | **9,012** |

---

## Architecture

```
TORGO Dataset
     │
     ▼
Phase 1 ── pipeline.py      Build utterance manifest CSV
     │
     ▼
Phase 2 ── benchmark.py     Transcribe with Deepgram Nova-3, compute WER
     │
     ▼
Phase 3 ── features.py      Extract 34 acoustic features via librosa
     │
     ▼
Phase 4 ── analysis.py      Train MLP + LR classifiers, save artifacts
     │
     ▼
Phase 5 ── dashboard.py     Gradio interactive dashboard
```

---

## Acoustic Features (34 total)

| Feature Group | Features |
|---|---|
| MFCCs | 13 mean + 13 std (26 features) |
| Pitch (F0) | Mean, std dev, range |
| Voice quality | Jitter, shimmer, HNR |
| Temporal | Voiced fraction, pause ratio, duration |
| Spectral | Centroid, rolloff, zero-crossing rate |

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/ayush11z/Neurological-Speech-Biomarker-Analyzer.git
cd Neurological-Speech-Biomarker-Analyzer
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your Deepgram API key

```bash
cp .env.example .env
# then edit .env and add:
# DEEPGRAM_API_KEY=your_key_here
```

Get a free API key at [deepgram.com](https://deepgram.com).

### 3. Download TORGO

Request access at the [TORGO project page](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html). The expected directory structure is:

```
TORGO/
├── F/
│   ├── F01/Session1/wav_arrayMic/
│   └── ...
├── FC/
├── M/
└── MC/
```

---

## Running the Pipeline

### Full pipeline (all phases)

```bash
python run_all.py --torgo /path/to/TORGO
```

### Quick smoke test (50 utterances only)

```bash
python run_all.py --torgo /path/to/TORGO --limit 50
```

### Skip Deepgram benchmarking (features + classifier only)

```bash
python run_all.py --torgo /path/to/TORGO --skip-benchmark
```

### Launch the dashboard only (after pipeline has run)

```bash
python dashboard.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser. A public share link is also printed to the terminal.

---

## Project Structure

```
speech-biomarker/
├── pipeline.py          # Phase 1: build manifest CSV from TORGO
├── benchmark.py         # Phase 2: Deepgram transcription + WER
├── features.py          # Phase 3: acoustic feature extraction (librosa)
├── analysis.py          # Phase 4: classifier training + artifacts
├── dashboard.py         # Phase 5: Gradio dashboard
├── run_all.py           # Orchestrate all phases end-to-end
├── main.py              # Deepgram client wrapper
├── data/processed/      # manifest.csv, features.csv (generated)
├── outputs/             # benchmark_results.csv, plots (generated)
└── models/              # trained classifier artifacts (generated)
```

---

## Methodology

1. Walk the TORGO directory tree → build a manifest of (speaker, severity, wav path, ground truth)
2. Transcribe each utterance with Deepgram Nova-3 and compute WER against the ground truth prompt
3. Extract 34 acoustic features per utterance with `librosa`
4. Train a 5-fold cross-validated MLP (128→64) and Logistic Regression on the features
5. Correlate acoustic features with WER to identify which features predict Deepgram failure
6. Serve results in an interactive Gradio dashboard

---

## Limitations

- **Speaker identity proxy**: The classifier achieves high accuracy partly because MFCCs encode speaker identity, which correlates with fixed severity labels. Speaker-independent cross-validation is future work.
- **Approximate voice quality features**: Shimmer and HNR are computed via librosa approximations rather than pitch-synchronous analysis (e.g., Praat). They serve as relative indicators within this dataset rather than clinical measurements.
- **Closed-world model**: The classifier was trained on 15 fixed TORGO speakers. Generalization to unseen dysarthric speakers requires further validation.

---

## Tech Stack

- [Deepgram Nova-3](https://deepgram.com) — speech-to-text
- [librosa](https://librosa.org) — acoustic feature extraction
- [scikit-learn](https://scikit-learn.org) — MLP + Logistic Regression
- [Gradio](https://gradio.app) — interactive dashboard
- [jiwer](https://github.com/jitsi/jiwer) — Word Error Rate computation

---

## License

MIT
