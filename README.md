# CS2 Data Pipeline

Automated pipeline that detects personal kill moments from CS2 (Counter-Strike 2) gameplay clips and extracts frame sequences for each kill. Paste any video URL (Medal.tv, YouTube, Twitch, Kick, etc.) and go.

## Features

- **ML Kill Sound Detection** — GradientBoosting classifier trained on 35-dim audio features (MFCC, spectral) replaces hand-tuned thresholds
- **Auto-Retrain** — Model automatically retrains after each new video, improving over time
- **Kill Feed Verification** — Three-signal visual verification eliminates false positives
- **Frame Sequence Extraction** — Saves 26 frames per kill (2s before, kill moment, 0.5s after)
- **NCC Fallback** — Original audio fingerprinting system kicks in when no trained model exists
- **Mass Training Data Collection** — Automated Medal.tv API search to collect hundreds of labeled training videos
- **Confidence-Based Verification** — High-confidence ML detections bypass visual verification for speed
- **Web Interface** — Flask-based web UI with real-time progress tracking
- **Metadata Generation** — JSON metadata for each processing session

## Installation

```bash
pip install -r requirements.txt
```

**System Requirements:**
- Python 3.10+
- FFmpeg (must be in PATH)

**Python Dependencies:**
- opencv-python — video frame processing
- numpy — numerical operations
- flask — web interface
- yt-dlp — video download (supports Medal.tv, YouTube, Twitch, Kick, and 1000+ sites)
- scikit-learn — ML kill sound classifier
- scipy — signal processing and peak finding

## Usage

```bash
python app.py
```

Open `http://localhost:5000` in your browser. Paste any CS2 gameplay video URL and click "Process".

### Training the Model

The model auto-trains after each processed video. To manually train or retrain:

```bash
python train_classifier.py
```

This reads kill timestamps from `metadata/*.json`, extracts audio features from the corresponding videos, and trains the classifier. The model is saved to `models/kill_classifier.pkl`.

### Mass Training Data Collection

Automatically search Medal.tv for CS2 kill clips and collect training data:

```bash
python collect_training_data.py                # default: 10 videos per query
python collect_training_data.py --per-query 100 # more data
```

Searches "cs2 1k" through "cs2 5k" + "cs2 ace", uses title kill count as ground truth, and retrains the model with collected data.

## Output

Each kill gets its own folder with a frame sequence:

```
clips/session_20260214_123456/
├── kill_001/
│   ├── before_20.jpg      ← 2.0s before kill
│   ├── before_19.jpg      ← 1.9s before
│   ├── ...
│   ├── before_01.jpg      ← 0.1s before
│   ├── kill.jpg           ← exact kill moment
│   ├── after_01.jpg       ← 0.1s after
│   ├── ...
│   └── after_05.jpg       ← 0.5s after
├── kill_002/
│   └── ...
└── kill_005/
    └── ...
```

## Technical Details

### Detection Pipeline

```
Video → FFmpeg → WAV (22050Hz mono)
  → Sliding window (250ms, 50ms hop)
    → Feature extraction (35-dim) per window
      → GradientBoosting.predict_proba()
        → Probability curve
          → scipy.signal.find_peaks (natural deduplication)
            → Kill feed verification (3-signal visual check)
              → Final kill timestamps
```

### ML Classifier (Primary)

A `GradientBoostingClassifier` trained on a 35-dimensional feature vector per audio window:

| Index | Feature | Purpose |
|-------|---------|---------|
| 0-12 | 13 MFCC | Timbral character of kill sound |
| 13-25 | 13 delta-MFCC | Onset dynamics (sudden start) |
| 26 | Spectral centroid | Kill sound sits at 2000-3500 Hz |
| 27 | Spectral bandwidth | Narrow band = tonal sound |
| 28 | Spectral rolloff | Energy distribution |
| 29 | Spectral flatness | Tonal (low) vs noise (high) |
| 30 | Zero crossing rate | Sound character |
| 31 | RMS energy | Volume level |
| 32 | Band energy ratio | 1800-4500 Hz band energy / total |
| 33 | Spectral flux | Sudden change (onset) |
| 34 | Peak frequency | Dominant frequency |

Peak finding with `scipy.signal.find_peaks` provides natural deduplication — no cooldown or echo suppression needed.

### Data Augmentation

Training data is augmented 16x per kill sample:
- Time shift (±15ms, ±30ms)
- Volume scaling (0.7x, 0.85x, 1.15x)
- Noise injection (SNR 10/15/20 dB)
- Pitch shift (±2% resampling)
- Bandpass variation (wide/narrow)
- Random jitter

### Kill Feed Verification

Three complementary signals verify each ML candidate by analyzing the kill feed ROI (top-right corner):

| Signal | What it detects | Key conditions |
|--------|----------------|----------------|
| **Backward delta** | Dark ratio increased vs 2s before | `peak_dark < 0.96` filters pitch-black scenes |
| **Forward delta** | Dark ratio increased after audio event | `self_dark < 0.50`, `peak_dark_fwd < 0.96` |
| **Forward text** | New bright text appeared in dark ROI within 1s | `base_white < 0.001`, `delta_fwd >= 0` |

Any single signal confirming is sufficient.

### NCC Fallback

When no trained model exists (`models/kill_classifier.pkl` not found), the system falls back to the original detection method:
- Two-pass spectral flux + NCC audio fingerprinting
- Auto-calibrating reference discovery via pairwise cross-correlation
- Near-miss promotion for borderline candidates with strong visual evidence

### Auto-Retrain

After each video is processed:
1. Pipeline checks if new sessions exist since last training
2. If yes, retrains the model in a background thread with all available data
3. Reloads the updated model for the next video

More videos processed = better model accuracy.

## Project Structure

```
├── app.py                     # Flask web server
├── pipeline.py                # CS2DataPipeline class (ML + NCC fallback)
├── audio_classifier.py        # Feature extraction + classifier wrapper
├── train_classifier.py        # Model training script
├── collect_training_data.py   # Mass training data collector (Medal.tv API)
├── requirements.txt           # Python dependencies
├── models/
│   ├── kill_classifier.pkl    # Trained model (gitignored)
│   └── training_meta.json     # Training metadata
├── templates/
│   └── index.html         # Web UI
├── static/
│   └── style.css          # Styling
├── downloads/             # Downloaded videos (runtime)
├── clips/                 # Extracted kill frames (runtime)
└── metadata/              # JSON session metadata (runtime)
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/process` | POST | Start processing (body: `{"url": "..."}`) |
| `/api/status/<id>` | GET | Poll processing status |
| `/api/sessions` | GET | List previous sessions |
| `/clips/<path>` | GET | Serve extracted frames |

## License

MIT
