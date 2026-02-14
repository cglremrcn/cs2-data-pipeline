# CS2 Data Pipeline

Automated data processing pipeline that detects kill moments from CS2 (Counter-Strike 2) clips and extracts labeled video segments.

## Features

- **Automatic Video Download** — Downloads videos from Medal.tv links (yt-dlp)
- **Kill Moment Detection** — Detects kill moments from the kill feed using frame difference analysis (no template needed)
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

- **Method:** Frame difference analysis (`cv2.absdiff`)
- **ROI:** Kill feed region (top-right 30% of the screen)
- **Filtering:** ROI changes are compared against global frame changes to filter camera movement and scene transitions
- **Rule:** `roi_change >= threshold AND roi_change > global_change * 2`
- **Performance:** Every 6th frame is processed (60fps → effective 10fps)
- **Cooldown:** 4 seconds (prevents duplicate detections of the same kill)

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
