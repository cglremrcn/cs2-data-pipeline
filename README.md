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

**Primary: Audio analysis** (personal kills only)
- Extracts audio from video via FFmpeg
- Computes spectral flux in the kill confirmation sound frequency band (1800-4500 Hz)
- Adaptive threshold (median + 2.5x standard deviation) detects kill "ding" sounds
- Only the local player's kill confirmation sound is analyzed — teammate/enemy kills are ignored

**Fallback: Visual frame differencing** (all kill feed activity)
- Used when audio is unavailable (no audio track, extraction failure)
- Monitors kill feed ROI (top-right 30%) for pixel changes
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
