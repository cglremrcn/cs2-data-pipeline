"""
CS2 Data Pipeline - Kill moment detection and clip extraction.
Processes Medal.tv clips to detect kill moments and extract labeled clips.
"""

import subprocess
import json
import logging
import time
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

    # Detection (frame differencing)
    "frame_skip": 6,
    "diff_threshold": 0.03,
    "diff_noise_threshold": 30,
    "global_scale_factor": 0.25,
    "cooldown_seconds": 4.0,

    # Clip Extraction
    "clip_pre_seconds": 3.0,
    "clip_post_seconds": 3.0,
    "clips_dir": "clips",

    # Metadata
    "metadata_dir": "metadata",

}


class CS2DataPipeline:
    """End-to-end pipeline for CS2 kill moment detection and clip extraction."""

    def __init__(self, base_dir=None, config=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._setup_directories()
        self._setup_logging()

    def _setup_directories(self):
        """Create required output directories."""
        for key in ["download_dir", "clips_dir", "metadata_dir"]:
            dir_path = self.base_dir / self.config[key]
            dir_path.mkdir(parents=True, exist_ok=True)

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
        Detect kill moments via frame differencing in the kill feed ROI.
        Compares ROI pixel changes against global frame changes to filter
        camera movement and scene transitions.
        Returns list of detection dicts with timestamp, confidence, etc.
        """
        if progress_callback:
            progress_callback("detecting", "Kill anlari tespit ediliyor...")

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

        diff_threshold = self.config["diff_threshold"]
        noise_threshold = self.config["diff_noise_threshold"]
        global_scale = self.config["global_scale_factor"]

        raw_detections = []
        frame_skip = self.config["frame_skip"]
        frame_number = 0
        analyzed = 0
        prev_roi_gray = None
        prev_global_gray = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_skip == 0:
                # Extract ROI and convert to grayscale
                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Downscale full frame for global change measurement
                small = cv2.resize(frame, (0, 0), fx=global_scale, fy=global_scale)
                global_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                if prev_roi_gray is not None:
                    # ROI difference
                    roi_diff = cv2.absdiff(roi_gray, prev_roi_gray)
                    roi_diff[roi_diff < noise_threshold] = 0
                    roi_change = np.count_nonzero(roi_diff) / roi_diff.size

                    # Global difference
                    global_diff = cv2.absdiff(global_gray, prev_global_gray)
                    global_diff[global_diff < noise_threshold] = 0
                    global_change = np.count_nonzero(global_diff) / global_diff.size

                    # Kill detected: ROI change above threshold AND significantly
                    # more than global change (filters camera movement)
                    if roi_change >= diff_threshold and roi_change > global_change * 2:
                        timestamp = frame_number / fps
                        confidence = min(roi_change / diff_threshold, 1.0)
                        raw_detections.append({
                            "timestamp": round(timestamp, 2),
                            "frame_number": frame_number,
                            "confidence": round(confidence, 4),
                            "roi_change": round(roi_change, 6),
                            "global_change": round(global_change, 6),
                        })

                prev_roi_gray = roi_gray
                prev_global_gray = global_gray

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
            "pipeline_version": "2.0.0",
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
                "method": "frame_difference",
                "roi": {
                    "x_start_ratio": self.config["roi_x_start_ratio"],
                    "x_end_ratio": self.config["roi_x_end_ratio"],
                    "y_start_ratio": self.config["roi_y_start_ratio"],
                    "y_end_ratio": self.config["roi_y_end_ratio"],
                },
                "diff_threshold": self.config["diff_threshold"],
                "diff_noise_threshold": self.config["diff_noise_threshold"],
                "cooldown_seconds": self.config["cooldown_seconds"],
                "frame_skip": self.config["frame_skip"],
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
                "roi_change": det.get("roi_change", 0),
                "global_change": det.get("global_change", 0),
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
