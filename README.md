# CS2 Data Pipeline

Automated data processing pipeline that detects kill moments from CS2 (Counter-Strike 2) clips and extracts labeled video segments.

## Features

- **Automatic Video Download** — Downloads videos from Medal.tv links (yt-dlp)
- **Personal Kill Detection** — Detects the player's own kills via audio analysis of the kill confirmation sound (falls back to visual detection)
- **Smart Clip Extraction** — Cuts 3s before and 3s after each kill with FFmpeg (lossless, audio preserved)
- **Metadata Generation** — Detailed JSON metadata for each processing session
- **Web Interface** — User-friendly Flask-based web UI

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

## Technical Details

### Kill Detection Algorithm

**Primary: Auto-calibrating audio fingerprinting** (personal kills only)
- Two-pass approach: spectral flux candidates → audio fingerprint matching
- Pass 1: Detects all audio onset candidates in the 1800-4500 Hz band (over-detects intentionally)
- Auto-calibration: Extracts audio snippets from top candidates, computes pairwise Normalized Cross-Correlation (NCC), identifies the repeating kill sound pattern automatically
- Pass 2: Scores all candidates against the discovered reference — only matching sounds (NCC ≥ 0.3) are kept
- No manual template needed — the kill sound is discovered from the clip itself

**Kill feed verification** (eliminates false positives)
- CS2 kill feed entries have dark semi-transparent background bars
- Each audio candidate is verified by checking if dark bars are visible in the kill feed ROI
- Dark pixel ratio (brightness < 60) measured at multiple time offsets around each candidate
- Real kills show dark_ratio > 0.25, false positives (empty ROI) show ~0.0
- Only candidates with confirmed kill feed presence are kept

**Fallback: Visual frame differencing** (all kill feed activity)
- Used when audio is unavailable (no audio track, extraction failure)
- Monitors kill feed ROI for pixel changes
- Filters camera movement by comparing ROI vs global frame changes

**Cooldown:** 1.0 second (allows rapid multikills while preventing duplicate detections)

### Clip Extraction

- Cuts in seconds using FFmpeg `-c copy` (no re-encoding)
- Audio is preserved, zero quality loss
- Boundary clamping for kills near video start/end

## Project Structure

```
├── app.py              # Flask web server
├── pipeline.py         # CS2DataPipeline class
├── templates/          # HTML templates
├── static/             # CSS files
├── downloads/          # Downloaded videos
├── clips/              # Extracted clips
└── metadata/           # JSON metadata
```

## License

MIT
