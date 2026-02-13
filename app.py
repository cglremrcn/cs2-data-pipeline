"""
CS2 Data Pipeline - Flask Web Interface
"""

import json
import threading
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory

from pipeline import CS2DataPipeline

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
pipeline = CS2DataPipeline(base_dir=BASE_DIR)

# Store pipeline status per session
pipeline_status = {}
pipeline_results = {}


@app.route("/")
def index():
    """Main page."""
    # Check if template exists
    template_exists = (BASE_DIR / pipeline.config["templates_dir"] / pipeline.config["template_file"]).exists()
    return render_template("index.html", template_exists=template_exists)


@app.route("/api/process", methods=["POST"])
def process_video():
    """Start pipeline processing for a given URL."""
    data = request.get_json()
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "URL gerekli"}), 400

    # Check template
    template_path = BASE_DIR / pipeline.config["templates_dir"] / pipeline.config["template_file"]
    if not template_path.exists():
        return jsonify({"error": "Template bulunamadi. Once template olusturun."}), 400

    # Generate session id
    from datetime import datetime
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    pipeline_status[session_id] = {"phase": "queued", "message": "Kuyrukta..."}
    pipeline_results[session_id] = None

    def run_pipeline():
        def progress_cb(phase, message):
            pipeline_status[session_id] = {"phase": phase, "message": message}

        try:
            result = pipeline.run(url, progress_callback=progress_cb)
            pipeline_results[session_id] = result
        except Exception as e:
            pipeline_status[session_id] = {"phase": "error", "message": str(e)}
            pipeline_results[session_id] = {"status": "error", "error": str(e)}

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    return jsonify({"session_id": session_id, "status": "started"})


@app.route("/api/status/<session_id>")
def get_status(session_id):
    """Get current pipeline status."""
    status = pipeline_status.get(session_id)
    if not status:
        return jsonify({"error": "Session bulunamadi"}), 404

    result = pipeline_results.get(session_id)
    return jsonify({
        "status": status,
        "result": result,
        "done": result is not None,
    })


@app.route("/api/template/create", methods=["POST"])
def create_template():
    """Extract frames from a video for template creation."""
    data = request.get_json()
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "URL gerekli"}), 400

    try:
        video_path = pipeline.download_video(url)
        frames = pipeline.extract_frames_for_template(video_path, count=15)
        return jsonify({
            "status": "ok",
            "frames": [str(Path(f).relative_to(BASE_DIR)) for f in frames],
            "video_path": str(video_path.relative_to(BASE_DIR)),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/template/save", methods=["POST"])
def save_template():
    """Save a cropped region as the kill template."""
    data = request.get_json()
    frame_path = data.get("frame_path", "")
    crop = data.get("crop", {})

    x = int(crop.get("x", 0))
    y = int(crop.get("y", 0))
    w = int(crop.get("w", 0))
    h = int(crop.get("h", 0))

    if not frame_path or w == 0 or h == 0:
        return jsonify({"error": "Gecersiz parametreler"}), 400

    try:
        import cv2
        full_path = BASE_DIR / frame_path
        frame = cv2.imread(str(full_path))
        if frame is None:
            return jsonify({"error": "Kare okunamadi"}), 400

        cropped = frame[y:y + h, x:x + w]
        template_dir = BASE_DIR / pipeline.config["templates_dir"]
        template_dir.mkdir(parents=True, exist_ok=True)
        template_path = template_dir / pipeline.config["template_file"]
        cv2.imwrite(str(template_path), cropped)

        # Reload template
        pipeline._load_template()

        return jsonify({"status": "ok", "size": f"{w}x{h}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sessions")
def list_sessions():
    """List all previous sessions from metadata files."""
    meta_dir = BASE_DIR / "metadata"
    sessions = []
    if meta_dir.exists():
        for f in sorted(meta_dir.glob("*.json"), reverse=True):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    sessions.append({
                        "session_id": data.get("session_id", f.stem),
                        "created_at": data.get("created_at", ""),
                        "kills": data.get("summary", {}).get("total_kills_detected", 0),
                        "clips": data.get("summary", {}).get("total_clips_created", 0),
                        "url": data.get("source", {}).get("url", ""),
                    })
            except (json.JSONDecodeError, KeyError):
                continue
    return jsonify(sessions)


@app.route("/frames/<path:filename>")
def serve_frame(filename):
    """Serve extracted frame images."""
    return send_from_directory(str(BASE_DIR / "frames"), filename)


@app.route("/clips/<path:filename>")
def serve_clip(filename):
    """Serve extracted clips."""
    return send_from_directory(str(BASE_DIR / "clips"), filename)


if __name__ == "__main__":
    print("CS2 Data Pipeline - Web Interface")
    print("http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
