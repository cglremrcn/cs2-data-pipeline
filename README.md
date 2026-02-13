# CS2 Data Pipeline

Automated data processing pipeline that detects kill moments from CS2 (Counter-Strike 2) clips and extracts labeled video segments.

## Features

- **Automatic Video Download** — Downloads videos from Medal.tv links (yt-dlp)
- **Kill Moment Detection** — Detects kill moments from the kill feed using OpenCV template matching
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

Open `http://localhost:5000` in your browser.

### First Run

1. Click "Create Template" in the web interface
2. Enter a Medal.tv clip URL that contains a kill
3. Select and crop the kill icon from one of the extracted frames
4. You're ready — paste any URL and start processing automatically

## Technical Details

### Kill Detection Algorithm

- **Method:** `cv2.matchTemplate` (TM_CCOEFF_NORMED)
- **ROI:** Kill feed region (top-right 30% of the screen)
- **Multi-scale:** Template is searched at 6 different scales
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
├── kill_templates/     # Template images
├── downloads/          # Downloaded videos
├── clips/              # Extracted clips
└── metadata/           # JSON metadata
```

## License

MIT
