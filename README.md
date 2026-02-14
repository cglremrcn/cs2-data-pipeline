# CS2 Data Pipeline

Automated pipeline that detects personal kill moments from CS2 (Counter-Strike 2) gameplay clips and extracts frame sequences for each kill. No manual setup required — paste a Medal.tv URL and go.

## Features

- **Automatic Video Download** — Downloads videos from Medal.tv links via yt-dlp
- **Personal Kill Detection** — Detects only the player's own kills using audio fingerprinting + three-signal kill feed verification
- **Frame Sequence Extraction** — Saves 26 frames per kill (2s before, kill moment, 0.5s after) with no overlap between adjacent kills
- **Near-Miss Promotion** — Audio candidates just below NCC threshold get a second chance if strong visual evidence exists
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

Open `http://localhost:5000` in your browser. Paste a Medal.tv URL and click "Start" — processing begins immediately, no setup required.

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
- Pass 1: Detects all audio onset candidates in the 1800–4500 Hz band using spectral flux with a low threshold (intentional over-detection)
- Auto-calibration: Extracts 250ms audio snippets from top candidates, computes pairwise Normalized Cross-Correlation (NCC) to discover the repeating kill sound pattern — no manual template needed
- Pass 2: Scores all candidates against the discovered reference, keeps matches with NCC >= 0.28. Candidates with NCC between 0.168–0.28 are kept as near-misses for potential promotion
- Time-domain NCC with +/-30ms shift tolerance via FFT cross-correlation

**Step 2: Kill feed verification** (eliminates false positives)

Three complementary signals verify each audio candidate by analyzing the kill feed ROI (top-right corner of the screen):

| Signal | What it detects | Key conditions |
|--------|----------------|----------------|
| **Backward delta** | Dark_ratio increased vs 2s before (bright → dark overlay) | `peak_dark < 0.96` filters pitch-black tunnels |
| **Forward delta** | Dark_ratio increased AFTER audio event (scene transition + kill) | `self_dark < 0.50`, `peak_dark_fwd < 0.96` |
| **Forward text** | New bright text appeared in dark ROI within 1s after audio | `base_white < 0.001`, `delta_fwd >= 0` |

Any single signal confirming is sufficient. Adaptive thresholds (`0.05 + baseline * 0.10`) prevent false triggers in dark environments.

**Step 2b: Near-miss promotion**
- Audio candidates just below the NCC threshold (0.168–0.28) are re-evaluated with kill feed verification
- Must be >5s from any already-confirmed kill to avoid duplicates
- Catches kills where the audio fingerprint was slightly distorted but the visual evidence is clear

**Cooldown:** 2.0 seconds between final detections (allows multikills while merging duplicate detections of the same kill)

### Why This Approach Works

The kill confirmation sound in CS2 is always the same audio effect, so it forms a tight cluster among varied game sounds (gunshots, grenades, footsteps). Audio fingerprinting isolates the player's kills specifically — unlike visual-only methods that detect ALL players' kills in the kill feed.

Kill feed verification then removes false positives using three independent visual signals. The combination of audio + visual provides high precision without requiring any manual calibration.

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
