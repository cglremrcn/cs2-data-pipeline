# CS2 Data Pipeline

Automated pipeline that detects personal kill moments from CS2 (Counter-Strike 2) gameplay clips and extracts frame sequences for each kill.

## Features

- **Automatic Video Download** — Downloads videos from Medal.tv links via yt-dlp
- **Personal Kill Detection** — Detects only the player's own kills using audio fingerprinting + kill feed verification
- **Frame Sequence Extraction** — Saves 26 frames per kill (2s before → kill moment → 0.5s after) with no overlap between adjacent kills
- **Metadata Generation** — JSON metadata for each processing session
- **Web Interface** — Flask-based web UI with real-time progress tracking

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- FFmpeg (must be in PATH)
- yt-dlp

## Usage

```bash
python app.py
```

Open `http://localhost:5000` in your browser. Paste a Medal.tv URL and click "Baslat" — processing starts immediately, no setup required.

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

- Frames are sampled every 0.1 seconds (every 6th frame at 60fps)
- Before/after windows are clamped at adjacent kill boundaries — frames from one kill never cross into another

## Technical Details

### Kill Detection Algorithm

**Step 1: Auto-calibrating audio fingerprinting** (personal kills only)
- Two-pass approach: spectral flux onset detection → NCC fingerprint matching
- Pass 1: Detects all audio onset candidates in the 1800-4500 Hz band using spectral flux with a low threshold (intentional over-detection)
- Auto-calibration: Extracts 250ms audio snippets from top candidates, computes pairwise Normalized Cross-Correlation (NCC) to discover the repeating kill sound pattern — no manual template needed
- Pass 2: Scores all candidates against the discovered reference, keeps matches with NCC ≥ 0.28
- Time-domain NCC with ±30ms shift tolerance via FFT cross-correlation

**Step 2: Kill feed verification** (eliminates false positives)
- CS2 kill feed entries have dark semi-transparent background bars
- Each audio candidate is verified by checking dark pixel ratio (brightness < 60) in the kill feed ROI at multiple time offsets
- Real kills show dark_ratio > 0.25, false positives show ~0.0
- Only candidates with confirmed kill feed presence are kept

**Cooldown:** 1.0 second between detections (allows multikills while preventing duplicates)

### Why This Approach Works

The kill confirmation sound in CS2 is always the same audio effect, so it forms a tight cluster among varied game sounds (gunshots, grenades, footsteps). Audio fingerprinting isolates the player's kills specifically — unlike visual-only methods that detect ALL players' kills in the kill feed.

Kill feed verification then removes false positives where a similar sound occurred but no kill feed entry is visible, providing a two-layer detection system with high precision.

## Project Structure

```
├── app.py              # Flask web server
├── pipeline.py         # CS2DataPipeline class
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Web UI
├── static/
│   └── style.css       # Styling
├── downloads/          # Downloaded videos (runtime)
├── clips/              # Extracted kill frames (runtime)
└── metadata/           # JSON session metadata (runtime)
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
