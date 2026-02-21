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

**Step 1: Generate synthetic training data from clean reference sounds:**

```bash
# Put CS2 kill sound WAV files in reference_sounds/
# Put at least one CS2 gameplay video in downloads/
python generate_synthetic_data.py
python generate_synthetic_data.py --n-positive 1000 --n-negative 3000  # more data
```

This mixes clean kill sounds into real gameplay audio at various volumes, creating perfectly labeled training data.

**Step 2: Train the classifier:**

```bash
python train_classifier.py
```

Training uses synthetic data (primary) and real metadata from processed sessions (secondary). The model is saved to `models/kill_classifier.pkl`.

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

### Synthetic Training Data

Instead of noisy auto-labeled data, training uses **clean reference sounds** mixed with real gameplay audio:

1. **Reference sounds** — Clean kill sound WAVs extracted from CS2 game files (`reference_sounds/`)
2. **Background audio** — Real CS2 gameplay audio from downloaded videos
3. **Mixing** — Kill sounds injected at random positions with varying SNR (3-20 dB)
4. **Labels** — Perfect: we know exactly where each kill sound was placed

This gives unlimited, perfectly labeled training data without manual annotation.

### Data Augmentation

Each positive sample is augmented 8-16x:
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
├── app.py                        # Flask web server
├── pipeline.py                   # CS2DataPipeline class (ML + NCC fallback)
├── audio_classifier.py           # Feature extraction + classifier wrapper
├── train_classifier.py           # Model training script
├── generate_synthetic_data.py    # Synthetic training data from reference sounds
├── collect_training_data.py      # Mass training data collector (Medal.tv API)
├── requirements.txt              # Python dependencies
├── reference_sounds/             # Clean kill sound WAVs from game files
├── models/
│   ├── kill_classifier.pkl       # Trained model (gitignored)
│   └── training_meta.json        # Training metadata
├── templates/
│   └── index.html                # Web UI
├── static/
│   └── style.css                 # Styling
├── downloads/                    # Downloaded videos (runtime)
├── clips/                        # Extracted kill frames (runtime)
└── metadata/                     # JSON session metadata (runtime)
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
