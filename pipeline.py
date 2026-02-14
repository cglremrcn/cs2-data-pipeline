"""
CS2 Data Pipeline - Kill moment detection and clip extraction.
Processes Medal.tv clips to detect kill moments and extract labeled clips.
"""

import subprocess
import json
import logging
import time
import re
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    # Video Download
    "download_dir": "downloads",
    "yt_dlp_format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]",

    # ROI - CS2 kill feed (top-right corner)
    "roi_x_start_ratio": 0.70,
    "roi_x_end_ratio": 1.0,
    "roi_y_start_ratio": 0.0,
    "roi_y_end_ratio": 0.185,

    # Detection
    "frame_skip": 6,
    "match_method": cv2.TM_CCOEFF_NORMED,
    "confidence_threshold": 0.70,
    "cooldown_seconds": 4.0,
    "use_grayscale": True,
    "scales": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],

    # Clip Extraction
    "clip_pre_seconds": 3.0,
    "clip_post_seconds": 3.0,
    "clips_dir": "clips",

    # Metadata
    "metadata_dir": "metadata",

    # Template
    "templates_dir": "kill_templates",
    "template_file": "kill_icon.png",
}


class CS2DataPipeline:
    """End-to-end pipeline for CS2 kill moment detection and clip extraction."""

    def __init__(self, base_dir=None, config=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._setup_directories()
        self._setup_logging()
        self.template = None

    def _setup_directories(self):
        """Create required output directories."""
        for key in ["download_dir", "clips_dir", "metadata_dir", "templates_dir"]:
            dir_path = self.base_dir / self.config[key]
            dir_path.mkdir(parents=True, exist_ok=True)

        (self.base_dir / "frames").mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Configure logging with file and console handlers."""
        log_file = self.base_dir / "pipeline.log"
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s",
                                    datefmt="%H:%M:%S")
            fh.setFormatter(fmt)
            ch.setFormatter(fmt)
            logger.addHandler(fh)
            logger.addHandler(ch)

    def _load_template(self):
        """Load template image for matching."""
        template_path = self.base_dir / self.config["templates_dir"] / self.config["template_file"]
        if not template_path.exists():
            logger.warning(f"Template not found: {template_path}")
            return None

        if self.config["use_grayscale"]:
            self.template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        else:
            self.template = cv2.imread(str(template_path))

        if self.template is None:
            logger.error(f"Failed to load template: {template_path}")
            return None

        logger.info(f"Template loaded: {template_path} ({self.template.shape[1]}x{self.template.shape[0]})")
        return self.template

    # =========================================================================
    # Phase 1: Video Download
    # =========================================================================

    def download_video(self, url, progress_callback=None):
        """
        Download video from Medal.tv using yt-dlp.
        Returns path to downloaded file.
        """
        if progress_callback:
            progress_callback("downloading", "Video indiriliyor...")

        logger.info(f"Downloading: {url}")

        output_dir = self.base_dir / self.config["download_dir"]
        output_template = str(output_dir / "%(id)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "-f", self.config["yt_dlp_format"],
            "-o", output_template,
            "--no-playlist",
            "--merge-output-format", "mp4",
            "--print", "after_move:filepath",
            url
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                check=True, timeout=180
            )

            # yt-dlp --print after_move:filepath gives us the final path
            downloaded_path = None
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line and Path(line).exists():
                    downloaded_path = Path(line)
                    break

            if not downloaded_path:
                # Fallback: find the most recent mp4 in download dir
                mp4_files = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
                if mp4_files:
                    downloaded_path = mp4_files[-1]

            if not downloaded_path or not downloaded_path.exists():
                raise FileNotFoundError("Could not find downloaded video file")

            logger.info(f"Downloaded: {downloaded_path.name}")
            return downloaded_path

        except subprocess.CalledProcessError as e:
            logger.error(f"yt-dlp failed: {e.stderr[:500]}")
            raise RuntimeError(f"Video download failed: {e.stderr[:200]}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Video download timed out (180s)")

    # =========================================================================
    # Phase 2: Kill Detection
    # =========================================================================

    def detect_kills(self, video_path, progress_callback=None):
        """
        Scan video frames for kill feed matches using template matching.
        Returns list of detection dicts with timestamp, confidence, etc.
        """
        if progress_callback:
            progress_callback("detecting", "Kill anlari tespit ediliyor...")

        if self.template is None:
            self._load_template()
        if self.template is None:
            raise FileNotFoundError(
                "Template bulunamadi. Lutfen kill_templates/kill_icon.png "
                "dosyasini olusturun (Web arayuzunden 'Template Olustur' butonunu kullanin)."
            )

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Video acilamadi: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(f"Video: {width}x{height}, {fps:.1f}fps, {duration:.1f}s, {total_frames} frames")

        # Calculate ROI pixel coordinates
        roi_x1 = int(width * self.config["roi_x_start_ratio"])
        roi_x2 = int(width * self.config["roi_x_end_ratio"])
        roi_y1 = int(height * self.config["roi_y_start_ratio"])
        roi_y2 = int(height * self.config["roi_y_end_ratio"])
        logger.info(f"ROI: ({roi_x1},{roi_y1}) -> ({roi_x2},{roi_y2})")

        raw_detections = []
        frame_skip = self.config["frame_skip"]
        frame_number = 0
        analyzed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_skip == 0:
                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

                if self.config["use_grayscale"]:
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                confidence, location, scale = self._match_template_multiscale(roi)

                if confidence >= self.config["confidence_threshold"]:
                    timestamp = frame_number / fps
                    raw_detections.append({
                        "timestamp": round(timestamp, 2),
                        "frame_number": frame_number,
                        "confidence": round(confidence, 4),
                        "matched_scale": scale,
                        "matched_location": list(location) if location else None,
                    })

                analyzed += 1
                if analyzed % 100 == 0:
                    pct = int((frame_number / total_frames) * 100)
                    logger.info(f"Analiz: {pct}% ({analyzed} kare islendi)")

            frame_number += 1

        cap.release()

        # Apply cooldown filter
        detections = self._apply_cooldown(raw_detections)

        logger.info(f"Analiz tamamlandi: {analyzed} kare islendi, "
                     f"{len(raw_detections)} ham eslesme, {len(detections)} kill tespit edildi")

        return detections

    def _match_template_multiscale(self, roi):
        """Run template matching across multiple scales. Returns (confidence, location, scale)."""
        best_confidence = -1
        best_location = None
        best_scale = 1.0

        template = self.template
        roi_h, roi_w = roi.shape[:2]

        for scale in self.config["scales"]:
            new_w = int(template.shape[1] * scale)
            new_h = int(template.shape[0] * scale)

            if new_w >= roi_w or new_h >= roi_h or new_w < 5 or new_h < 5:
                continue

            scaled_template = cv2.resize(template, (new_w, new_h))
            result = cv2.matchTemplate(roi, scaled_template, self.config["match_method"])
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_confidence:
                best_confidence = max_val
                best_location = max_loc
                best_scale = scale

        return best_confidence, best_location, best_scale

    def _apply_cooldown(self, detections):
        """Filter detections that are too close together, keeping the highest confidence one."""
        if not detections:
            return []

        cooldown = self.config["cooldown_seconds"]
        filtered = [detections[0]]

        for det in detections[1:]:
            if det["timestamp"] - filtered[-1]["timestamp"] >= cooldown:
                filtered.append(det)
            elif det["confidence"] > filtered[-1]["confidence"]:
                filtered[-1] = det

        return filtered

    # =========================================================================
    # Phase 3: Clip Extraction
    # =========================================================================

    def cut_clips(self, video_path, detections, session_id, progress_callback=None):
        """
        Cut 6-second clips around each kill moment using FFmpeg.
        Returns list of created clip paths.
        """
        if progress_callback:
            progress_callback("cutting", "Klipler kesiliyor...")

        video_path = Path(video_path)
        clip_dir = self.base_dir / self.config["clips_dir"] / session_id
        clip_dir.mkdir(parents=True, exist_ok=True)

        # Get video duration for boundary clamping
        duration = self._get_video_duration(video_path)
        clip_paths = []

        for i, det in enumerate(detections, 1):
            start = max(0, det["timestamp"] - self.config["clip_pre_seconds"])
            end = min(duration, det["timestamp"] + self.config["clip_post_seconds"])
            clip_name = f"cs2_clip_{i:03d}.mp4"
            output_path = clip_dir / clip_name

            success = self._ffmpeg_cut(video_path, start, end, output_path)
            if success:
                clip_paths.append(output_path)
                det["clip_file"] = str(output_path.relative_to(self.base_dir))
                det["clip_start"] = round(start, 2)
                det["clip_end"] = round(end, 2)
                logger.info(f"Klip kesildi: {clip_name} ({start:.1f}s - {end:.1f}s)")
            else:
                logger.warning(f"Klip kesilemedi: {clip_name}")

        logger.info(f"Toplam {len(clip_paths)}/{len(detections)} klip olusturuldu")
        return clip_paths

    def _ffmpeg_cut(self, video_path, start, end, output_path):
        """Execute a single FFmpeg cut with -c copy (no re-encoding)."""
        duration = end - start
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{start:.3f}",
            "-i", str(video_path),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"FFmpeg error: {e.stderr[:200]}")
            return False
        except subprocess.TimeoutExpired:
            logger.warning(f"FFmpeg timed out for {output_path}")
            return False

    def _get_video_duration(self, video_path):
        """Get video duration in seconds using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames / fps if fps > 0 else 0

    # =========================================================================
    # Phase 4: Metadata Generation
    # =========================================================================

    def generate_metadata(self, video_path, url, detections, clip_paths, session_id):
        """Generate JSON metadata file for the session."""
        video_path = Path(video_path)
        meta_dir = self.base_dir / self.config["metadata_dir"]
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / f"{session_id}.json"

        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        file_size = video_path.stat().st_size if video_path.exists() else 0

        metadata = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "pipeline_version": "1.0.0",
            "source": {
                "url": url,
                "platform": "medal.tv",
                "downloaded_file": str(video_path.relative_to(self.base_dir)),
                "duration_seconds": round(duration, 2),
                "resolution": f"{width}x{height}",
                "fps": round(fps, 1),
                "file_size_bytes": file_size,
            },
            "detection_config": {
                "template_file": self.config["template_file"],
                "roi": {
                    "x_start_ratio": self.config["roi_x_start_ratio"],
                    "x_end_ratio": self.config["roi_x_end_ratio"],
                    "y_start_ratio": self.config["roi_y_start_ratio"],
                    "y_end_ratio": self.config["roi_y_end_ratio"],
                },
                "confidence_threshold": self.config["confidence_threshold"],
                "cooldown_seconds": self.config["cooldown_seconds"],
                "frame_skip": self.config["frame_skip"],
                "scales": self.config["scales"],
            },
            "detections": [],
            "annotations": [],
            "summary": {
                "total_kills_detected": len(detections),
                "total_clips_created": len(clip_paths),
                "average_confidence": 0,
            },
        }

        total_conf = 0
        for i, det in enumerate(detections, 1):
            metadata["detections"].append({
                "kill_index": i,
                "timestamp_seconds": det["timestamp"],
                "frame_number": det["frame_number"],
                "confidence": det["confidence"],
                "matched_scale": det["matched_scale"],
                "clip_file": det.get("clip_file", ""),
                "clip_start": det.get("clip_start", 0),
                "clip_end": det.get("clip_end", 0),
            })

            metadata["annotations"].append({
                "clip_path": det.get("clip_file", ""),
                "label": "cs2_kill_moment",
                "kill_timestamp_in_clip": self.config["clip_pre_seconds"],
                "confidence": det["confidence"],
                "source_game": "counter_strike_2",
            })

            total_conf += det["confidence"]

        if detections:
            metadata["summary"]["average_confidence"] = round(total_conf / len(detections), 4)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata kaydedildi: {meta_path}")
        return meta_path

    # =========================================================================
    # Template Creation Helper
    # =========================================================================

    def extract_frames_for_template(self, video_path, count=10):
        """Extract evenly-spaced frames from video for template creation."""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            raise RuntimeError("Video bos veya acilamadi")

        frames_dir = self.base_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Clean old frames
        for old in frames_dir.glob("frame_*.png"):
            old.unlink()

        step = max(1, total_frames // count)
        saved = []

        for i in range(count):
            frame_idx = i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Draw ROI overlay
            h, w = frame.shape[:2]
            roi_x1 = int(w * self.config["roi_x_start_ratio"])
            roi_y2 = int(h * self.config["roi_y_end_ratio"])
            cv2.rectangle(frame, (roi_x1, 0), (w, roi_y2), (0, 255, 0), 2)
            cv2.putText(frame, "KILL FEED ROI", (roi_x1 + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            path = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(path), frame)
            saved.append(str(path))

        cap.release()
        logger.info(f"{len(saved)} kare cikarildi: {frames_dir}")
        return saved

    def create_template_interactive(self, video_path):
        """
        Launch OpenCV window for interactive template creation.
        User navigates frames and crops the kill icon.
        Returns path to saved template.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            raise RuntimeError("Video bos veya acilamadi")

        window_name = "Template Olustur | A/D: Gezin | SPACE: Sec | Q: Cik"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        frame_idx = 0
        template_path = None

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            h, w = display.shape[:2]
            roi_x1 = int(w * self.config["roi_x_start_ratio"])
            roi_y2 = int(h * self.config["roi_y_end_ratio"])

            # Draw ROI
            cv2.rectangle(display, (roi_x1, 0), (w, roi_y2), (0, 255, 0), 2)
            cv2.putText(display, f"Frame {frame_idx}/{total_frames} | ROI: yesil alan",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, "A/D: Gezin | SPACE: Kill ikonunu sec | Q: Cik",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('d') or key == 83:  # Right
                frame_idx = min(frame_idx + 30, total_frames - 1)
            elif key == ord('a') or key == 81:  # Left
                frame_idx = max(frame_idx - 30, 0)
            elif key == ord('w'):  # Small step right
                frame_idx = min(frame_idx + 5, total_frames - 1)
            elif key == ord('s'):  # Small step left
                frame_idx = max(frame_idx - 5, 0)
            elif key == ord(' ') or key == 13:  # SPACE or ENTER
                cv2.destroyAllWindows()
                roi = cv2.selectROI("Kill ikonunu secin (mouse ile cizin)", frame)
                if roi[2] > 0 and roi[3] > 0:
                    x, y, w_roi, h_roi = [int(v) for v in roi]
                    cropped = frame[y:y + h_roi, x:x + w_roi]

                    template_dir = self.base_dir / self.config["templates_dir"]
                    template_dir.mkdir(parents=True, exist_ok=True)
                    template_path = template_dir / self.config["template_file"]
                    cv2.imwrite(str(template_path), cropped)
                    logger.info(f"Template kaydedildi: {template_path} ({w_roi}x{h_roi})")
                    break
                else:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 1280, 720)

        cap.release()
        cv2.destroyAllWindows()
        return str(template_path) if template_path else None

    # =========================================================================
    # Orchestrator
    # =========================================================================

    def run(self, url, progress_callback=None):
        """
        Full pipeline: download -> detect -> cut -> metadata.
        Returns summary dict.
        """
        start_time = time.time()
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info("=" * 60)
        logger.info(f"Pipeline baslatildi: {session_id}")
        logger.info(f"URL: {url}")
        logger.info("=" * 60)

        result = {
            "session_id": session_id,
            "status": "started",
            "url": url,
            "video_path": None,
            "kills_detected": 0,
            "clips_created": 0,
            "clip_paths": [],
            "metadata_path": None,
            "error": None,
            "processing_time": 0,
        }

        try:
            # Phase 1: Download
            logger.info("[1/4] Video indiriliyor...")
            video_path = self.download_video(url, progress_callback)
            result["video_path"] = str(video_path)

            # Check template
            template_path = self.base_dir / self.config["templates_dir"] / self.config["template_file"]
            if not template_path.exists():
                logger.info("Template bulunamadi - interaktif olusturma baslatiliyor...")
                if progress_callback:
                    progress_callback("template", "Template olusturma gerekiyor...")
                created = self.create_template_interactive(video_path)
                if not created:
                    raise RuntimeError("Template olusturulamadi. Pipeline durduruluyor.")

            self._load_template()

            # Phase 2: Detect
            logger.info("[2/4] Kill anlari tespit ediliyor...")
            detections = self.detect_kills(video_path, progress_callback)
            result["kills_detected"] = len(detections)

            if not detections:
                logger.warning("Hicbir kill tespit edilemedi.")
                if progress_callback:
                    progress_callback("done", "Kill tespit edilemedi.")
                result["status"] = "no_kills"
            else:
                # Phase 3: Cut clips
                logger.info(f"[3/4] {len(detections)} klip kesiliyor...")
                clip_paths = self.cut_clips(video_path, detections, session_id, progress_callback)
                result["clips_created"] = len(clip_paths)
                result["clip_paths"] = [str(p.relative_to(self.base_dir)).replace("\\", "/") for p in clip_paths]

                # Phase 4: Metadata
                logger.info("[4/4] Metadata olusturuluyor...")
                if progress_callback:
                    progress_callback("metadata", "Metadata olusturuluyor...")
                meta_path = self.generate_metadata(
                    video_path, url, detections, clip_paths, session_id
                )
                result["metadata_path"] = str(meta_path.relative_to(self.base_dir)).replace("\\", "/")
                result["status"] = "completed"

            elapsed = time.time() - start_time
            result["processing_time"] = round(elapsed, 2)

            logger.info("=" * 60)
            logger.info(f"Pipeline tamamlandi! ({elapsed:.1f}s)")
            logger.info(f"  Kill tespit: {result['kills_detected']}")
            logger.info(f"  Klip olusturuldu: {result['clips_created']}")
            logger.info("=" * 60)

            if progress_callback:
                progress_callback("done", f"Tamamlandi! {result['kills_detected']} kill, "
                                          f"{result['clips_created']} klip.")

        except Exception as e:
            logger.error(f"Pipeline hatasi: {e}", exc_info=True)
            result["status"] = "error"
            result["error"] = str(e)
            result["processing_time"] = round(time.time() - start_time, 2)
            if progress_callback:
                progress_callback("error", str(e))

        return result


if __name__ == "__main__":
    pipeline = CS2DataPipeline()
    url = input("Medal.tv URL: ").strip()
    if url:
        result = pipeline.run(url)
        print(f"\nSonuc: {json.dumps(result, indent=2, ensure_ascii=False)}")
    else:
        print("URL girilmedi.")
