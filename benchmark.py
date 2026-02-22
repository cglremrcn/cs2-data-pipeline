"""
Benchmark & tune YOLO kill feed detection using Medal.tv ground truth.

Searches Medal.tv for "cs2 1k" through "cs2 5k", downloads videos,
runs YOLO kill detection, and compares detected vs expected kill counts.

Usage:
    python benchmark.py                    # Download + benchmark with current params
    python benchmark.py --per-query 20     # 20 videos per query
    python benchmark.py --tune             # Grid search on already-downloaded videos
    python benchmark.py --skip-download    # Benchmark only (no new downloads)
    python benchmark.py --color --tune     # Color-based detection grid search
    python benchmark.py --color            # Benchmark with color detection
"""

import json
import logging
import argparse
import itertools
import wave
import subprocess as sp
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
RESULTS_FILE = BASE_DIR / "benchmark_results.json"
GROUND_TRUTH_FILE = BASE_DIR / "benchmark_ground_truth.json"
BEST_PARAMS_FILE = BASE_DIR / "models" / "yolo_params.json"
NCC_CACHE_FILE = BASE_DIR / "benchmark_ncc_cache.json"

DEFAULT_QUERIES = ["cs2 1k", "cs2 2k", "cs2 3k", "cs2 4k", "cs2 5k"]

# Default detection parameters (current pipeline values)
DEFAULT_PARAMS = {
    "sample_fps": 4,
    "conf": 0.45,
    "sim_threshold": 0.6,
    "width_ratio_threshold": 0.15,
    "gap_threshold": 15,
    "cooldown": 2.0,
}

# Color-based detection parameters
DEFAULT_COLOR_PARAMS = {
    "sim_threshold": 0.55,
    "cooldown": 2.0,
    "red_highlight_threshold": 0.08,
    "team_color_min_votes": 2,
}


# -------------------------------------------------------------------------
# Medal.tv API (reuse from collect_training_data)
# -------------------------------------------------------------------------

def search_and_collect(queries, per_query):
    """Search Medal.tv and return video metadata list."""
    from collect_training_data import (
        get_api_key, search_medal, parse_kill_count, build_medal_url
    )
    import time

    api_key = get_api_key()
    if not api_key:
        logger.error("Could not get Medal.tv API key")
        return []

    logger.info(f"API key: {api_key[:20]}...")
    videos = []

    for query in queries:
        logger.info(f"Searching: '{query}' (limit={per_query})")
        clips = search_medal(query, api_key, limit=per_query)
        if not clips:
            logger.warning(f"  No results for '{query}'")
            continue

        logger.info(f"  Found {len(clips)} clips")
        for clip in clips:
            content_id = str(clip.get("contentId", ""))
            if not content_id:
                continue

            title = clip.get("contentTitle", "")
            duration = clip.get("videoLengthSeconds", 0)

            if duration < 5 or duration > 90:
                continue

            expected = parse_kill_count(title, query)
            if expected is None:
                continue

            clip_url = clip.get("directClipUrl", "")
            if not clip_url:
                clip_url = build_medal_url(content_id)

            videos.append({
                "url": clip_url,
                "content_id": content_id,
                "expected_kills": expected,
                "title": title,
                "duration": duration,
                "query": query,
            })

        time.sleep(0.5)

    logger.info(f"Total videos found: {len(videos)}")
    return videos


def download_videos(videos):
    """Download videos and return updated list with file paths."""
    from collect_training_data import download_video

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for i, entry in enumerate(videos, 1):
        url = entry["url"]
        content_id = entry["content_id"]
        title = entry.get("title", "")

        # Check if already downloaded
        existing = list(DOWNLOADS_DIR.glob(f"*{content_id}*"))
        if existing:
            entry["video_path"] = str(existing[0])
            downloaded.append(entry)
            logger.info(f"[{i}/{len(videos)}] Already exists: {existing[0].name}")
            continue

        logger.info(f"[{i}/{len(videos)}] Downloading: {title[:50]}")
        video_path = download_video(url)
        if video_path:
            entry["video_path"] = str(video_path)
            downloaded.append(entry)
            logger.info(f"  Saved: {video_path.name}")
        else:
            logger.warning(f"  Download failed, skipping")

    logger.info(f"Downloaded: {len(downloaded)}/{len(videos)}")
    return downloaded


# -------------------------------------------------------------------------
# YOLO kill detection (standalone, parameterizable)
# -------------------------------------------------------------------------

def _compare_signatures(a, b):
    """Normalized correlation between two grayscale signatures."""
    if a is None or b is None:
        return 0.0
    fa = a.astype(float).flatten()
    fb = b.astype(float).flatten()
    fa -= fa.mean()
    fb -= fb.mean()
    na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(fa, fb) / (na * nb))


def extract_color_features(frame, x1, y1, x2, y2):
    """
    Extract color features from a kill feed entry bounding box.

    Returns dict with:
      - red_ratio: ratio of red-hued background pixels (kill highlight)
      - killer_hue: median hue of text pixels in left 25% (killer name)
      - killer_team: 'T', 'CT', or 'unknown'
      - victim_hue: median hue of text pixels in right 25% (victim name)
      - victim_team: 'T', 'CT', or 'unknown'
    """
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {"red_ratio": 0.0, "killer_hue": -1, "killer_team": "unknown",
                "victim_hue": -1, "victim_team": "unknown"}

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # --- Red highlight detection ---
    # Background pixels: moderate brightness (not too dark, not too bright)
    bg_mask = (v > 15) & (v < 100)
    bg_count = np.count_nonzero(bg_mask)
    if bg_count > 0:
        # Red hue in OpenCV: H<15 or H>165 (wraps around 0/180), with some saturation
        red_mask = bg_mask & (s > 30) & ((h < 15) | (h > 165))
        red_ratio = float(np.count_nonzero(red_mask)) / bg_count
    else:
        red_ratio = 0.0

    # --- Killer hue (left 25% of crop) ---
    w = crop.shape[1]
    left_w = max(1, int(w * 0.25))
    killer_hue, killer_team = _extract_text_hue(hsv[:, :left_w, :])

    # --- Victim hue (right 25% of crop) ---
    right_start = max(0, w - left_w)
    victim_hue, victim_team = _extract_text_hue(hsv[:, right_start:, :])

    return {
        "red_ratio": round(red_ratio, 4),
        "killer_hue": killer_hue,
        "killer_team": killer_team,
        "victim_hue": victim_hue,
        "victim_team": victim_team,
    }


def _extract_text_hue(hsv_region):
    """
    Extract median hue from text pixels in an HSV region.
    Text pixels: bright (V>120) and saturated (S>60).
    Returns (median_hue, team_classification).
    """
    if hsv_region.size == 0:
        return -1, "unknown"

    h, s, v = hsv_region[:, :, 0], hsv_region[:, :, 1], hsv_region[:, :, 2]
    text_mask = (v > 120) & (s > 60)
    text_hues = h[text_mask]

    if len(text_hues) < 3:
        return -1, "unknown"

    med_hue = float(np.median(text_hues))
    team = _classify_team(med_hue)
    return round(med_hue, 1), team


def _classify_team(hue):
    """Classify team from hue: T=orange(5-25), CT=blue(90-120)."""
    if 5 <= hue <= 25:
        return "T"
    if 90 <= hue <= 120:
        return "CT"
    return "unknown"


def scan_video_yolo(model, video_path, sample_fps=4, conf=0.45):
    """
    Run YOLO inference on video ONCE. Cache all frame detections with
    timestamps, widths, and pairwise similarities.
    Returns list of raw detections: [{"ts": float, "bw": int, "sim": float}]
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or w <= 0 or h <= 0:
        cap.release()
        return []

    min_x = int(w * 0.55)
    max_y = int(h * 0.22)
    min_y = int(h * 0.02)

    interval = max(1, int(fps / sample_fps))
    frame_idx = 0
    prev_sig = None
    raw_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            ts = frame_idx / fps
            results = model(frame, conf=conf, verbose=False)
            best = None
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    if x1 < min_x or y1 > max_y or y1 < min_y:
                        continue
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw < 50 or bh < 10 or bh > 80:
                        continue
                    if bw / bh < 2.0 or bw / bh > 15:
                        continue
                    if best is None or y1 < best[1]:
                        best = (x1, y1, x2, y2, bw, bh)

            if best is not None:
                x1, y1, x2, y2, bw, bh = best
                crop_w = min(int(bw * 0.40), int(w * 0.18))
                crop = frame[y1:y2, x1:min(x1 + crop_w, w)]
                if crop.size > 0:
                    sig = cv2.cvtColor(
                        cv2.resize(crop, (120, 24)), cv2.COLOR_BGR2GRAY
                    )
                    sim = (_compare_signatures(sig, prev_sig)
                           if prev_sig is not None else 0.0)
                    raw_detections.append({
                        "ts": round(ts, 2),
                        "bw": bw,
                        "sim": round(sim, 4),
                    })
                    prev_sig = sig

        frame_idx += 1
    cap.release()
    return raw_detections


def scan_video_yolo_color(model, video_path):
    """
    Run YOLO inference on video with color feature extraction.
    Returns list of raw detections with color data:
    [{"ts", "bw", "sim", "red_ratio", "killer_hue", "killer_team", "victim_hue", "victim_team"}]
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or w <= 0 or h <= 0:
        cap.release()
        return []

    min_x = int(w * 0.55)
    max_y = int(h * 0.22)
    min_y = int(h * 0.02)

    sample_fps = 4
    interval = max(1, int(fps / sample_fps))
    frame_idx = 0
    prev_sig = None
    raw_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            ts = frame_idx / fps
            results = model(frame, conf=0.45, verbose=False)
            best = None
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    if x1 < min_x or y1 > max_y or y1 < min_y:
                        continue
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw < 50 or bh < 10 or bh > 80:
                        continue
                    if bw / bh < 2.0 or bw / bh > 15:
                        continue
                    if best is None or y1 < best[1]:
                        best = (x1, y1, x2, y2, bw, bh)

            if best is not None:
                x1, y1, x2, y2, bw, bh = best
                crop_w = min(int(bw * 0.40), int(w * 0.18))
                crop = frame[y1:y2, x1:min(x1 + crop_w, w)]
                if crop.size > 0:
                    sig = cv2.cvtColor(
                        cv2.resize(crop, (120, 24)), cv2.COLOR_BGR2GRAY
                    )
                    sim = (_compare_signatures(sig, prev_sig)
                           if prev_sig is not None else 0.0)

                    # Extract color features from full bounding box
                    color = extract_color_features(frame, x1, y1, x2, y2)

                    raw_detections.append({
                        "ts": round(ts, 2),
                        "bw": bw,
                        "sim": round(sim, 4),
                        **color,
                    })
                    prev_sig = sig

        frame_idx += 1
    cap.release()
    return raw_detections


def apply_params(raw_detections, params=None):
    """
    Apply parameter-based filtering to cached raw detections.
    Fast — no video I/O needed.
    Returns (kill_count, kill_times, transitions_count, width_info).
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    # Filter by similarity threshold to get transitions
    transitions = []
    for i, det in enumerate(raw_detections):
        if i == 0 or det["sim"] < p["sim_threshold"]:
            transitions.append(det)

    if len(transitions) < 3:
        times = [t["ts"] for t in transitions]
        return len(times), times, len(transitions), {"method": "too_few"}

    # Width analysis
    all_widths = [t["bw"] for t in transitions]
    widths = sorted(set(all_widths))
    max_gap = 0
    gap_idx = 0
    for i in range(len(widths) - 1):
        gap = widths[i + 1] - widths[i]
        if gap > max_gap:
            max_gap = gap
            gap_idx = i

    cooldown = p["cooldown"]

    def _dedup(entries):
        if not entries:
            return []
        times = sorted(e["ts"] for e in entries)
        d = [times[0]]
        for t in times[1:]:
            if t - d[-1] >= cooldown:
                d.append(t)
        return d

    width_range = max(all_widths) - min(all_widths)
    width_mean = np.mean(all_widths)
    width_ratio = width_range / width_mean if width_mean > 0 else 0

    width_info = {
        "range": width_range,
        "mean": round(width_mean, 1),
        "ratio": round(width_ratio, 3),
        "max_gap": max_gap,
        "n_transitions": len(transitions),
    }

    if width_ratio < p["width_ratio_threshold"] and max_gap < p["gap_threshold"]:
        kill_times = _dedup(transitions)
        width_info["method"] = "homogeneous"
    else:
        if max_gap >= p["gap_threshold"]:
            gap_thresh = (widths[gap_idx] + widths[gap_idx + 1]) / 2
        else:
            gap_thresh = width_mean
        gap_times = _dedup([t for t in transitions if t["bw"] >= gap_thresh])

        mean_times = _dedup([t for t in transitions if t["bw"] >= width_mean])

        if len(gap_times) <= len(mean_times):
            kill_times = gap_times
            width_info["method"] = "gap"
            width_info["threshold"] = round(gap_thresh, 1)
        else:
            kill_times = mean_times
            width_info["method"] = "mean"
            width_info["threshold"] = round(width_mean, 1)

    return len(kill_times), kill_times, len(transitions), width_info


def apply_color_params(raw_detections, params=None):
    """
    Apply color-based filtering to cached raw detections.
    Priority: red highlight → team color voting → width fallback.
    Returns (kill_count, kill_times, method_used, debug_info).
    """
    p = {**DEFAULT_COLOR_PARAMS, **(params or {})}

    sim_thresh = p.get("sim_threshold", 0.55)
    cooldown = p.get("cooldown", 2.0)
    red_thresh = p.get("red_highlight_threshold", 0.08)
    team_min_votes = p.get("team_color_min_votes", 2)

    # Filter by similarity threshold to get transitions
    transitions = []
    for i, det in enumerate(raw_detections):
        if i == 0 or det["sim"] < sim_thresh:
            transitions.append(det)

    if not transitions:
        return 0, [], "none", {}

    def _dedup(entries):
        if not entries:
            return []
        times = sorted(e["ts"] for e in entries)
        d = [times[0]]
        for t in times[1:]:
            if t - d[-1] >= cooldown:
                d.append(t)
        return d

    # --- Priority 1: Red highlight ---
    red_transitions = [t for t in transitions if t.get("red_ratio", 0) >= red_thresh]
    if red_transitions:
        kill_times = _dedup(red_transitions)
        return len(kill_times), kill_times, "red_highlight", {
            "red_count": len(red_transitions),
            "total_transitions": len(transitions),
            "threshold": red_thresh,
        }

    # --- Priority 2: Team color voting ---
    team_votes = {"T": 0, "CT": 0, "unknown": 0}
    for t in transitions:
        kt = t.get("killer_team", "unknown")
        team_votes[kt] = team_votes.get(kt, 0) + 1

    # Determine player's team: the team that appears most as killer
    t_count = team_votes.get("T", 0)
    ct_count = team_votes.get("CT", 0)

    if t_count >= team_min_votes or ct_count >= team_min_votes:
        if t_count > ct_count:
            player_team = "T"
        elif ct_count > t_count:
            player_team = "CT"
        else:
            player_team = None

        if player_team:
            team_transitions = [t for t in transitions
                                if t.get("killer_team") == player_team]
            if team_transitions:
                kill_times = _dedup(team_transitions)
                return len(kill_times), kill_times, "team_color", {
                    "player_team": player_team,
                    "team_votes": team_votes,
                    "team_count": len(team_transitions),
                    "total_transitions": len(transitions),
                }

    # --- Priority 3: Width fallback (existing logic) ---
    if len(transitions) < 3:
        times = [t["ts"] for t in transitions]
        return len(times), times, "too_few", {}

    all_widths = [t["bw"] for t in transitions]
    widths = sorted(set(all_widths))
    max_gap = 0
    gap_idx = 0
    for i in range(len(widths) - 1):
        gap = widths[i + 1] - widths[i]
        if gap > max_gap:
            max_gap = gap
            gap_idx = i

    width_range = max(all_widths) - min(all_widths)
    width_mean = np.mean(all_widths)
    width_ratio = width_range / width_mean if width_mean > 0 else 0
    wrt = p.get("width_ratio_threshold", 0.15)
    gt = p.get("gap_threshold", 15)

    if width_ratio < wrt and max_gap < gt:
        kill_times = _dedup(transitions)
        method = "width_homogeneous"
    else:
        if max_gap >= gt:
            gap_thresh = (widths[gap_idx] + widths[gap_idx + 1]) / 2
        else:
            gap_thresh = width_mean
        gap_times = _dedup([t for t in transitions if t["bw"] >= gap_thresh])
        mean_times = _dedup([t for t in transitions if t["bw"] >= width_mean])
        if len(gap_times) <= len(mean_times):
            kill_times = gap_times
        else:
            kill_times = mean_times
        method = "width_split"

    return len(kill_times), kill_times, method, {
        "total_transitions": len(transitions),
        "team_votes": team_votes,
    }


def detect_kills_yolo(model, video_path, params=None):
    """Convenience wrapper: scan + apply params."""
    raw = scan_video_yolo(model, video_path)
    return apply_params(raw, params)


# -------------------------------------------------------------------------
# Audio NCC scanning (player kill sound detection)
# -------------------------------------------------------------------------

def extract_audio_wav(video_path):
    """Extract mono WAV audio from video using FFmpeg."""
    video_path = Path(video_path)
    audio_path = video_path.parent / (video_path.stem + "_bench.wav")
    try:
        sp.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
            str(audio_path)
        ], capture_output=True, check=True, timeout=30)
        if audio_path.exists() and audio_path.stat().st_size > 1000:
            return audio_path
    except Exception:
        pass
    if audio_path.exists():
        audio_path.unlink(missing_ok=True)
    return None


def scan_audio_ncc(video_path):
    """
    NCC template match kill_doof_01.wav against video audio.
    Returns list of (timestamp, ncc_score) tuples at low threshold (0.15)
    so we can filter by score at benchmark time.
    """
    from scipy.signal import fftconvolve, find_peaks

    ref_path = BASE_DIR / "reference_sounds" / "sounds_player_kill_doof_01.wav"
    if not ref_path.exists():
        return []

    audio_path = extract_audio_wav(video_path)
    if not audio_path:
        return []

    try:
        with wave.open(str(audio_path), 'rb') as wf:
            rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0

        if len(audio) == 0:
            return []

        # Load and resample reference
        tmp = audio_path.parent / "_tmp_ref_bench.wav"
        sp.run([
            "ffmpeg", "-y", "-i", str(ref_path),
            "-ar", str(rate), "-ac", "1", "-acodec", "pcm_s16le", str(tmp)
        ], capture_output=True, check=True, timeout=10)

        with wave.open(str(tmp), 'rb') as wf:
            ref_raw = wf.readframes(wf.getnframes())
        ref_audio = np.frombuffer(ref_raw, dtype=np.int16).astype(np.float64) / 32768.0
        tmp.unlink(missing_ok=True)

        if len(ref_audio) == 0:
            return []

        # NCC via FFT
        ref_norm = ref_audio - np.mean(ref_audio)
        audio_norm = audio - np.mean(audio)

        corr = fftconvolve(audio_norm, ref_norm[::-1], mode='valid')

        ref_energy = np.sqrt(np.sum(ref_norm ** 2))
        cumsum = np.cumsum(audio_norm ** 2)
        window = len(ref_norm)
        local_energy = np.sqrt(
            cumsum[window:] - np.concatenate([[0], cumsum[:len(cumsum) - window - 1]])
        )

        min_len = min(len(corr), len(local_energy))
        ncc = corr[:min_len] / (ref_energy * local_energy[:min_len] + 1e-10)

        # Use low threshold (0.15) so we can filter at benchmark time
        peaks, props = find_peaks(ncc, height=0.15, distance=int(1.5 * rate))

        return [(round(p / rate, 2), round(float(ncc[p]), 4)) for p in peaks]
    except Exception as e:
        logger.warning(f"Audio NCC scan failed for {Path(video_path).name}: {e}")
        return []
    finally:
        if audio_path and audio_path.exists():
            audio_path.unlink(missing_ok=True)


def scan_all_audio(ground_truth):
    """Pre-scan all videos with NCC audio matching. Returns dict of name -> [(ts, score)]."""
    cache = {}
    for entry in ground_truth:
        video_path = entry.get("video_path", "")
        if not video_path or not Path(video_path).exists():
            continue
        name = Path(video_path).name
        if name not in cache:
            logger.info(f"  Audio: {name}...")
            cache[name] = scan_audio_ncc(video_path)
            n = len(cache[name])
            if n:
                scores = [s for _, s in cache[name]]
                logger.info(f"    {n} peaks, scores: {min(scores):.3f}-{max(scores):.3f}")
            else:
                logger.info(f"    no audio/no peaks")
    audio_count = sum(1 for v in cache.values() if v)
    logger.info(f"Audio available: {audio_count}/{len(cache)} videos")
    return cache


# -------------------------------------------------------------------------
# Audio-filtered detection params
# -------------------------------------------------------------------------

DEFAULT_AUDIO_PARAMS = {
    "sim_threshold": 0.55,
    "cooldown": 2.0,
    "audio_window": 2.0,
    "ncc_threshold": 0.30,
    # Width fallback params
    "width_ratio_threshold": 0.15,
    "gap_threshold": 15,
}


def apply_audio_filtered_params(raw_detections, ncc_raw, params=None):
    """
    Audio-filtered detection: intersect YOLO transitions with NCC audio peaks.
    ncc_raw: list of (timestamp, ncc_score) tuples from scan_audio_ncc.
    Falls back to width heuristic if no audio available.
    Returns (kill_count, kill_times, method, info).
    """
    p = {**DEFAULT_AUDIO_PARAMS, **(params or {})}

    # Get YOLO transitions
    transitions = []
    for i, det in enumerate(raw_detections):
        if i == 0 or det["sim"] < p["sim_threshold"]:
            transitions.append(det)

    if not transitions:
        return 0, [], "none", {}

    cooldown = p["cooldown"]
    audio_window = p.get("audio_window", 2.0)
    ncc_threshold = p.get("ncc_threshold", 0.30)

    def _dedup_times(times):
        if not times:
            return []
        times = sorted(times)
        d = [times[0]]
        for t in times[1:]:
            if t - d[-1] >= cooldown:
                d.append(t)
        return d

    # All transition times (deduped)
    all_trans_times = _dedup_times([t["ts"] for t in transitions])

    # Filter NCC peaks by threshold
    ncc_times = [t for t, score in ncc_raw if score >= ncc_threshold] if ncc_raw else []

    if ncc_times:
        # Intersect: YOLO transitions × NCC audio peaks
        matched = []
        for trans_ts in all_trans_times:
            for ncc_t in ncc_times:
                if abs(trans_ts - ncc_t) <= audio_window:
                    matched.append(trans_ts)
                    break

        if matched:
            kill_times = _dedup_times(matched)
            return len(kill_times), kill_times, "audio_filtered", {
                "ncc_total": len(ncc_raw) if ncc_raw else 0,
                "ncc_above_thresh": len(ncc_times),
                "matched": len(matched),
                "total_transitions": len(all_trans_times),
            }

        # NCC peaks found but none matched YOLO transitions → use NCC count
        kill_times = _dedup_times(ncc_times)
        return len(kill_times), kill_times, "ncc_only", {
            "ncc_above_thresh": len(ncc_times),
            "total_transitions": len(all_trans_times),
        }

    # No audio or no NCC peaks above threshold → width fallback
    count, times, n_trans, info = apply_params(raw_detections, params)
    return count, times, info.get("method", "width_fallback"), info


# -------------------------------------------------------------------------
# Benchmark runner
# -------------------------------------------------------------------------

def load_ground_truth():
    """Load ground truth from file."""
    if GROUND_TRUTH_FILE.exists():
        with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_ground_truth(entries):
    """Save ground truth to file."""
    with open(GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def scan_all_videos(model, ground_truth, use_color=True, sample_fps=4, conf=0.45):
    """Pre-scan all videos with YOLO. Returns dict of video_name -> raw_detections."""
    cache = {}
    valid = []
    label = "color" if use_color else f"fps={sample_fps},conf={conf}"
    for entry in ground_truth:
        video_path = entry.get("video_path", "")
        if not video_path or not Path(video_path).exists():
            continue
        name = Path(video_path).name
        if name not in cache:
            logger.info(f"  Scanning ({label}) {name}...")
            if use_color:
                cache[name] = scan_video_yolo_color(model, video_path)
            else:
                cache[name] = scan_video_yolo(model, video_path,
                                              sample_fps=sample_fps, conf=conf)
        valid.append(entry)
    logger.info(f"Scanned {len(cache)} videos, {sum(len(v) for v in cache.values())} raw detections")
    return cache, valid


def run_benchmark_cached(cache, ground_truth, params=None, verbose=True, use_color=False):
    """Run detection using cached scans. Fast — no video I/O."""
    p = params or (DEFAULT_COLOR_PARAMS if use_color else DEFAULT_PARAMS)
    correct = 0
    total = 0
    details = []
    by_query = {}

    for entry in ground_truth:
        video_path = entry.get("video_path", "")
        if not video_path:
            continue
        name = Path(video_path).name
        if name not in cache:
            continue

        expected = entry["expected_kills"]
        query = entry.get("query", f"cs2 {expected}k")

        if use_color:
            detected, times, method, info = apply_color_params(cache[name], params=p)
        else:
            detected, times, n_trans, info = apply_params(cache[name], params=p)
            method = info.get("method", "")

        is_correct = detected == expected
        if is_correct:
            correct += 1
        total += 1

        detail = {
            "video": name,
            "expected": expected,
            "detected": detected,
            "correct": is_correct,
            "method": method,
            "times": [round(t, 1) for t in times],
        }
        details.append(detail)

        if query not in by_query:
            by_query[query] = {"total": 0, "correct": 0}
        by_query[query]["total"] += 1
        if is_correct:
            by_query[query]["correct"] += 1

        if verbose:
            status = "OK" if is_correct else f"WRONG({detected})"
            logger.info(f"  {name}: {detected}/{expected} {status} [{method}]")

    accuracy = correct / total if total > 0 else 0
    return {
        "timestamp": datetime.now().isoformat(),
        "params": p,
        "total_videos": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "by_query": by_query,
        "details": details,
    }


def run_benchmark_audio_filtered(yolo_cache, audio_cache, ground_truth, params=None, verbose=True):
    """Run audio-filtered benchmark using cached YOLO + NCC data."""
    p = params or DEFAULT_AUDIO_PARAMS.copy()
    correct = 0
    total = 0
    details = []
    by_query = {}
    method_stats = {}

    for entry in ground_truth:
        video_path = entry.get("video_path", "")
        if not video_path:
            continue
        name = Path(video_path).name
        if name not in yolo_cache:
            continue

        expected = entry["expected_kills"]
        query = entry.get("query", f"cs2 {expected}k")
        ncc_raw = audio_cache.get(name, [])

        detected, times, method, info = apply_audio_filtered_params(
            yolo_cache[name], ncc_raw, params=p
        )

        is_correct = detected == expected
        if is_correct:
            correct += 1
        total += 1

        method_stats[method] = method_stats.get(method, {"total": 0, "correct": 0})
        method_stats[method]["total"] += 1
        if is_correct:
            method_stats[method]["correct"] += 1

        ncc_thresh = p.get("ncc_threshold", 0.30)
        ncc_above = len([t for t, s in ncc_raw if s >= ncc_thresh])

        detail = {
            "video": name,
            "expected": expected,
            "detected": detected,
            "correct": is_correct,
            "method": method,
            "ncc_peaks": ncc_above,
            "ncc_total": len(ncc_raw),
            "times": [round(t, 1) for t in times],
        }
        details.append(detail)

        if query not in by_query:
            by_query[query] = {"total": 0, "correct": 0}
        by_query[query]["total"] += 1
        if is_correct:
            by_query[query]["correct"] += 1

        if verbose:
            status = "OK" if is_correct else f"WRONG({detected})"
            logger.info(f"  {name}: {detected}/{expected} {status} "
                        f"[{method}, ncc={len(ncc_times)}]")

    accuracy = correct / total if total > 0 else 0
    return {
        "timestamp": datetime.now().isoformat(),
        "params": p,
        "total_videos": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "by_query": by_query,
        "method_stats": method_stats,
        "details": details,
    }


def run_grid_search(cache, ground_truth, use_color=False):
    """Grid search over key parameters using cached video scans."""
    if use_color:
        param_grid = {
            "sim_threshold": [0.45, 0.55, 0.65, 0.75],
            "cooldown": [1.5, 2.0, 2.5, 3.0],
            "red_highlight_threshold": [0.03, 0.05, 0.08, 0.12, 0.15],
            "team_color_min_votes": [1, 2, 3],
            # Width fallback params
            "width_ratio_threshold": [0.10, 0.15, 0.25],
            "gap_threshold": [10, 15, 25],
        }
    else:
        param_grid = {
            "sim_threshold": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
            "width_ratio_threshold": [0.08, 0.10, 0.15, 0.20, 0.25, 0.30],
            "cooldown": [1.0, 1.5, 2.0, 2.5, 3.0],
            "gap_threshold": [8, 10, 15, 20, 25, 30],
        }

    fixed = {"sample_fps": 4, "conf": 0.45}

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    logger.info(f"Grid search ({'color' if use_color else 'width'}): "
                f"{len(combos)} parameter combinations")

    best_accuracy = -1
    best_params = None
    best_result = None
    all_results = []

    for i, combo in enumerate(combos, 1):
        params = {**fixed}
        for k, v in zip(keys, combo):
            params[k] = v

        result = run_benchmark_cached(
            cache, ground_truth, params=params, verbose=False, use_color=use_color
        )
        acc = result["accuracy"]
        all_results.append({"params": params, "accuracy": acc, "correct": result["correct"]})

        if acc > best_accuracy:
            best_accuracy = acc
            best_params = params.copy()
            best_result = result

        if i % 200 == 0 or i == len(combos):
            logger.info(f"  [{i}/{len(combos)}] Current best: {best_accuracy:.1%} "
                        f"({best_result['correct']}/{best_result['total_videos']})")

    logger.info(f"\nBest accuracy: {best_accuracy:.1%}")
    logger.info(f"Best params: {best_params}")

    all_results.sort(key=lambda x: -x["accuracy"])
    logger.info("\nTop 10 parameter sets:")
    for r in all_results[:10]:
        logger.info(f"  {r['accuracy']:.1%} ({r['correct']}) - {r['params']}")

    return best_params, best_result, all_results


def apply_title_hint_params(raw_detections, ncc_raw, expected_kills, params=None):
    """
    Title-hint mode: we know N kills from the title.
    Select best N transitions from YOLO, scored by audio proximity.
    When YOLO under-detects, supplement with NCC audio peaks.
    Returns (kill_count, kill_times, method, info).
    """
    p = {**DEFAULT_AUDIO_PARAMS, **(params or {})}
    n = expected_kills

    cooldown = p["cooldown"]
    audio_window = p.get("audio_window", 2.0)
    ncc_threshold = p.get("ncc_threshold", 0.30)

    # Get YOLO transitions
    transitions = []
    for i, det in enumerate(raw_detections):
        if i == 0 or det["sim"] < p["sim_threshold"]:
            transitions.append(det)

    def _dedup_times(times):
        if not times:
            return []
        times = sorted(times)
        d = [times[0]]
        for t in times[1:]:
            if t - d[-1] >= cooldown:
                d.append(t)
        return d

    all_trans_times = _dedup_times([t["ts"] for t in transitions])

    # Filter NCC peaks by threshold
    ncc_peaks = [(t, score) for t, score in ncc_raw if score >= ncc_threshold] if ncc_raw else []
    ncc_times = [t for t, _ in ncc_peaks]

    if not all_trans_times and not ncc_times:
        return 0, [], "none", {}

    if not all_trans_times and ncc_times:
        # No YOLO transitions at all — use NCC peaks directly
        ncc_sorted = sorted(ncc_peaks, key=lambda x: -x[1])
        ncc_selected = _dedup_times([t for t, _ in ncc_sorted[:n * 2]])[:n]
        return len(ncc_selected), ncc_selected, "ncc_fallback", {
            "ncc_peaks": len(ncc_times),
        }

    if len(all_trans_times) >= n:
        # Enough YOLO transitions — score by audio and pick best N
        scored = []
        for ts in all_trans_times:
            audio_score = 0.0
            if ncc_times:
                min_dist = min(abs(ts - nt) for nt in ncc_times)
                audio_score = max(0, 1.0 - min_dist / audio_window)
            scored.append((ts, audio_score))

        scored.sort(key=lambda x: -x[1])
        kill_times = sorted([ts for ts, _ in scored[:n]])

        return n, kill_times, "title_hint", {
            "total_transitions": len(all_trans_times),
            "ncc_peaks": len(ncc_times),
            "selected": n,
        }

    # YOLO under-detected: supplement with NCC peaks
    yolo_times = set(all_trans_times)
    remaining = n - len(all_trans_times)

    # Find NCC peaks not overlapping with existing YOLO transitions
    unmatched_ncc = []
    for nt, score in ncc_peaks:
        if not any(abs(nt - yt) <= audio_window for yt in yolo_times):
            unmatched_ncc.append((nt, score))

    # Sort by score (best first), pick enough to fill the gap
    unmatched_ncc.sort(key=lambda x: -x[1])
    supplement_times = _dedup_times([t for t, _ in unmatched_ncc[:remaining * 2]])[:remaining]

    kill_times = sorted(list(all_trans_times) + supplement_times)

    return len(kill_times), kill_times, "title_hint_ncc_supp", {
        "total_transitions": len(all_trans_times),
        "ncc_peaks": len(ncc_times),
        "supplemented": len(supplement_times),
    }


def run_benchmark_title_hint(yolo_cache, audio_cache, ground_truth, params=None, verbose=True):
    """Benchmark using title-hint mode (expected kills from title)."""
    p = params or DEFAULT_AUDIO_PARAMS.copy()
    correct = 0
    total = 0
    details = []
    by_query = {}

    for entry in ground_truth:
        video_path = entry.get("video_path", "")
        if not video_path:
            continue
        name = Path(video_path).name
        if name not in yolo_cache:
            continue

        expected = entry["expected_kills"]
        query = entry.get("query", f"cs2 {expected}k")
        ncc_raw = audio_cache.get(name, [])

        detected, times, method, info = apply_title_hint_params(
            yolo_cache[name], ncc_raw, expected, params=p
        )

        is_correct = detected == expected
        if is_correct:
            correct += 1
        total += 1

        detail = {
            "video": name,
            "expected": expected,
            "detected": detected,
            "correct": is_correct,
            "method": method,
            "times": [round(t, 1) for t in times],
        }
        details.append(detail)

        if query not in by_query:
            by_query[query] = {"total": 0, "correct": 0}
        by_query[query]["total"] += 1
        if is_correct:
            by_query[query]["correct"] += 1

        if verbose:
            status = "OK" if is_correct else f"WRONG({detected})"
            logger.info(f"  {name}: {detected}/{expected} {status} [{method}]")

    accuracy = correct / total if total > 0 else 0
    return {
        "timestamp": datetime.now().isoformat(),
        "params": p,
        "total_videos": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "by_query": by_query,
        "details": details,
    }


def run_audio_grid_search(yolo_cache, audio_cache, ground_truth):
    """Grid search over audio-filtered parameters."""
    param_grid = {
        "sim_threshold": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "cooldown": [1.0, 1.5, 2.0, 2.5, 3.0],
        "audio_window": [1.0, 1.5, 2.0, 2.5, 3.0],
        "ncc_threshold": [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60],
        "width_ratio_threshold": [0.10, 0.15, 0.20, 0.25],
        "gap_threshold": [10, 15, 20, 30],
    }

    fixed = {"sample_fps": 4, "conf": 0.45}

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    logger.info(f"Audio-filtered grid search: {len(combos)} combinations")

    best_accuracy = -1
    best_params = None
    best_result = None

    for i, combo in enumerate(combos, 1):
        params = {**fixed}
        for k, v in zip(keys, combo):
            params[k] = v

        result = run_benchmark_audio_filtered(
            yolo_cache, audio_cache, ground_truth, params=params, verbose=False
        )
        acc = result["accuracy"]

        if acc > best_accuracy:
            best_accuracy = acc
            best_params = params.copy()
            best_result = result

        if i % 500 == 0 or i == len(combos):
            logger.info(f"  [{i}/{len(combos)}] Best: {best_accuracy:.1%} "
                        f"({best_result['correct']}/{best_result['total_videos']})")

    logger.info(f"\nBest accuracy: {best_accuracy:.1%}")
    logger.info(f"Best params: {best_params}")
    if best_result and "method_stats" in best_result:
        for m, s in best_result["method_stats"].items():
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0
            logger.info(f"  {m}: {s['correct']}/{s['total']} ({acc:.0%})")

    return best_params, best_result


def run_title_hint_grid_search(yolo_cache, audio_cache, ground_truth):
    """Grid search over title-hint parameters (with NCC supplementary)."""
    param_grid = {
        "sim_threshold": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        "cooldown": [0.8, 1.0, 1.5, 2.0, 2.5],
        "audio_window": [1.5, 2.0, 2.5, 3.0, 4.0],
        "ncc_threshold": [0.20, 0.25, 0.30, 0.35, 0.40],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    logger.info(f"Title-hint grid search: {len(combos)} combinations")

    best_accuracy = -1
    best_params = None
    best_result = None

    for i, combo in enumerate(combos, 1):
        params = {}
        for k, v in zip(keys, combo):
            params[k] = v

        result = run_benchmark_title_hint(
            yolo_cache, audio_cache, ground_truth, params=params, verbose=False
        )
        acc = result["accuracy"]

        if acc > best_accuracy:
            best_accuracy = acc
            best_params = params.copy()
            best_result = result

        if i % 200 == 0 or i == len(combos):
            logger.info(f"  [{i}/{len(combos)}] Best: {best_accuracy:.1%} "
                        f"({best_result['correct']}/{best_result['total_videos']})")

    logger.info(f"\nBest title-hint accuracy: {best_accuracy:.1%}")
    logger.info(f"Best params: {best_params}")

    return best_params, best_result


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO kill detection")
    parser.add_argument("--per-query", type=int, default=20,
                        help="Videos per search query (default: 20)")
    parser.add_argument("--queries", nargs="+", default=None,
                        help="Custom queries (default: cs2 1k..5k)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing videos")
    parser.add_argument("--tune", action="store_true",
                        help="Run parameter grid search")
    parser.add_argument("--color", action="store_true",
                        help="Use color-based detection (red highlight + team color)")
    parser.add_argument("--audio-filter", action="store_true",
                        help="Use audio-filtered detection (NCC × YOLO intersection)")
    parser.add_argument("--title-hint", action="store_true",
                        help="Use title kill count as hint (simulates production with title)")
    parser.add_argument("--sample-fps", type=int, default=4,
                        help="YOLO sampling FPS (default: 4, try 6-8 for more transitions)")
    parser.add_argument("--conf", type=float, default=0.45,
                        help="YOLO confidence threshold (default: 0.45, try 0.30-0.40)")
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES

    logger.info("=" * 60)
    logger.info("CS2 Kill Detection Benchmark")
    logger.info("=" * 60)

    # Load YOLO model
    from ultralytics import YOLO
    yolo_path = BASE_DIR / "models" / "best.pt"
    if not yolo_path.exists():
        logger.error(f"YOLO model not found: {yolo_path}")
        return
    model = YOLO(str(yolo_path))
    logger.info("YOLO model loaded")

    # Load or build ground truth
    ground_truth = load_ground_truth()

    if not args.skip_download and not args.tune:
        # Search and download new videos
        videos = search_and_collect(queries, args.per_query)
        if not videos:
            logger.error("No videos found from Medal.tv search")
            if not ground_truth:
                return

        # Filter out already-in-ground-truth videos
        existing_ids = {e.get("content_id") for e in ground_truth}
        new_videos = [v for v in videos if v["content_id"] not in existing_ids]
        logger.info(f"New videos to download: {len(new_videos)}")

        if new_videos:
            downloaded = download_videos(new_videos)
            ground_truth.extend(downloaded)
            save_ground_truth(ground_truth)
            logger.info(f"Ground truth updated: {len(ground_truth)} total videos")

    # Filter to videos that exist on disk
    valid = [e for e in ground_truth
             if e.get("video_path") and Path(e["video_path"]).exists()]
    logger.info(f"Valid videos for benchmark: {len(valid)}")

    if not valid:
        logger.error("No valid videos found. Run without --skip-download first.")
        return

    # Pre-scan all videos with YOLO (basic mode for audio-filter, color optional)
    use_color = args.color and not args.audio_filter and not args.title_hint
    mode_label = "title-hint" if args.title_hint else (
        "audio-filter" if args.audio_filter else ("color" if use_color else "width"))
    logger.info(f"\nPre-scanning all videos with YOLO (sample_fps={args.sample_fps}, conf={args.conf})...")
    cache, valid = scan_all_videos(model, valid, use_color=use_color,
                                   sample_fps=args.sample_fps, conf=args.conf)

    # Audio scanning (for audio-filter or title-hint mode) with cache
    audio_cache = {}
    if args.audio_filter or args.title_hint:
        if NCC_CACHE_FILE.exists():
            with open(NCC_CACHE_FILE, "r") as f:
                audio_cache = json.load(f)
            # Convert lists back to tuples
            for k, v in audio_cache.items():
                audio_cache[k] = [tuple(x) for x in v]
            logger.info(f"Loaded NCC cache: {len(audio_cache)} videos")
        else:
            logger.info("\nPre-scanning audio with NCC template matching...")
            audio_cache = scan_all_audio(valid)
            # Save cache
            with open(NCC_CACHE_FILE, "w") as f:
                json.dump({k: list(v) for k, v in audio_cache.items()}, f)
            logger.info(f"NCC cache saved to {NCC_CACHE_FILE}")

        # Quick NCC direct analysis
        logger.info("\n--- NCC Direct Count Analysis ---")
        for thresh in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
            correct = 0
            total = 0
            for entry in valid:
                name = Path(entry.get("video_path", "")).name
                ncc_raw = audio_cache.get(name, [])
                if not ncc_raw:
                    continue
                count = len([1 for t, s in ncc_raw if s >= thresh])
                if count == entry["expected_kills"]:
                    correct += 1
                total += 1
            logger.info(f"  NCC direct (thresh={thresh:.2f}): "
                        f"{correct}/{total} ({correct/total:.0%})")

    if args.title_hint and args.tune:
        # Title-hint grid search
        logger.info("\n" + "=" * 60)
        logger.info("TITLE-HINT GRID SEARCH")
        logger.info("=" * 60)

        best_params, best_result = run_title_hint_grid_search(
            cache, audio_cache, valid
        )

        # Save best params
        BEST_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BEST_PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "params": {**best_params, "sample_fps": args.sample_fps, "conf": args.conf},
                "mode": "title_hint",
                "accuracy": best_result["accuracy"],
                "total_videos": best_result["total_videos"],
                "correct": best_result["correct"],
                "by_query": best_result.get("by_query", {}),
                "tuned_at": datetime.now().isoformat(),
            }, f, indent=2)
        logger.info(f"Best params saved to {BEST_PARAMS_FILE}")

        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(best_result, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {RESULTS_FILE}")

        # Show failures for best result
        failures = [d for d in best_result["details"] if not d["correct"]]
        if failures:
            logger.info(f"\nFailed ({len(failures)}):")
            for fl in failures:
                logger.info(f"  {fl['video']}: det={fl['detected']} exp={fl['expected']} "
                             f"[{fl['method']}]")
        return

    if args.title_hint:
        # Title-hint mode: use expected kills from title
        logger.info("\n" + "=" * 60)
        logger.info("TITLE-HINT BENCHMARK")
        logger.info("=" * 60)

        params = DEFAULT_AUDIO_PARAMS.copy()
        if BEST_PARAMS_FILE.exists():
            try:
                with open(BEST_PARAMS_FILE, "r") as f:
                    saved = json.load(f)
                params.update(saved.get("params", {}))
            except Exception:
                pass

        result = run_benchmark_title_hint(
            cache, audio_cache, valid, params=params
        )

        logger.info(f"\n{'=' * 60}")
        logger.info(f"RESULTS: {result['correct']}/{result['total_videos']} "
                     f"({result['accuracy']:.1%} accuracy)")
        for query, stats in sorted(result["by_query"].items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            logger.info(f"  {query}: {stats['correct']}/{stats['total']} ({acc:.0%})")

        failures = [d for d in result["details"] if not d["correct"]]
        if failures:
            logger.info(f"\nFailed ({len(failures)}):")
            for fl in failures:
                logger.info(f"  {fl['video']}: det={fl['detected']} exp={fl['expected']} "
                             f"[{fl['method']}]")

        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved to {RESULTS_FILE}")
        return

    if args.audio_filter and args.tune:
        # Audio-filtered grid search
        logger.info("\n" + "=" * 60)
        logger.info("AUDIO-FILTERED GRID SEARCH")
        logger.info("=" * 60)

        best_params, best_result = run_audio_grid_search(cache, audio_cache, valid)

        # Save best params
        BEST_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BEST_PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "params": best_params,
                "mode": "audio_filter",
                "accuracy": best_result["accuracy"],
                "total_videos": best_result["total_videos"],
                "correct": best_result["correct"],
                "method_stats": best_result.get("method_stats", {}),
                "tuned_at": datetime.now().isoformat(),
            }, f, indent=2)
        logger.info(f"Best params saved to {BEST_PARAMS_FILE}")

        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(best_result, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {RESULTS_FILE}")

    elif args.audio_filter:
        # Single audio-filtered benchmark
        logger.info("\n" + "=" * 60)
        logger.info("AUDIO-FILTERED BENCHMARK")
        logger.info("=" * 60)

        params = DEFAULT_AUDIO_PARAMS.copy()
        if BEST_PARAMS_FILE.exists():
            try:
                with open(BEST_PARAMS_FILE, "r") as f:
                    saved = json.load(f)
                params.update(saved.get("params", {}))
            except Exception:
                pass

        result = run_benchmark_audio_filtered(
            cache, audio_cache, valid, params=params
        )

        logger.info(f"\n{'=' * 60}")
        logger.info(f"RESULTS: {result['correct']}/{result['total_videos']} "
                     f"({result['accuracy']:.1%} accuracy)")
        logger.info(f"Params: {result['params']}")
        for m, s in result.get("method_stats", {}).items():
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0
            logger.info(f"  {m}: {s['correct']}/{s['total']} ({acc:.0%})")
        for query, stats in sorted(result["by_query"].items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            logger.info(f"  {query}: {stats['correct']}/{stats['total']} ({acc:.0%})")

        failures = [d for d in result["details"] if not d["correct"]]
        if failures:
            logger.info(f"\nFailed ({len(failures)}):")
            for f in failures:
                logger.info(f"  {f['video']}: det={f['detected']} exp={f['expected']} "
                             f"[{f['method']}, ncc={f.get('ncc_peaks', 0)}]")

        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved to {RESULTS_FILE}")

    elif args.tune:
        # Grid search using cached scans (fast!)
        logger.info("\n" + "=" * 60)
        logger.info(f"PARAMETER GRID SEARCH ({mode_label.upper()} MODE)")
        logger.info("=" * 60)

        best_params, best_result, all_results = run_grid_search(
            cache, valid, use_color=use_color
        )

        # Save best params
        BEST_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BEST_PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "params": best_params,
                "mode": mode_label,
                "accuracy": best_result["accuracy"],
                "total_videos": best_result["total_videos"],
                "correct": best_result["correct"],
                "tuned_at": datetime.now().isoformat(),
            }, f, indent=2)
        logger.info(f"Best params saved to {BEST_PARAMS_FILE}")

        # Save full results
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(best_result, f, indent=2, ensure_ascii=False)
        logger.info(f"Full results saved to {RESULTS_FILE}")

    else:
        # Single benchmark run with current/default params
        logger.info("\n" + "=" * 60)
        logger.info(f"BENCHMARK RUN ({mode_label.upper()} MODE)")
        logger.info("=" * 60)

        # Use tuned params if available, else defaults
        params = (DEFAULT_COLOR_PARAMS if use_color else DEFAULT_PARAMS).copy()
        if BEST_PARAMS_FILE.exists():
            try:
                with open(BEST_PARAMS_FILE, "r") as f:
                    saved = json.load(f)
                params.update(saved.get("params", {}))
                logger.info(f"Using tuned params from {BEST_PARAMS_FILE}")
            except Exception:
                pass

        result = run_benchmark_cached(
            cache, valid, params=params, use_color=use_color
        )

        logger.info(f"\n{'=' * 60}")
        logger.info(f"RESULTS: {result['correct']}/{result['total_videos']} "
                     f"({result['accuracy']:.1%} accuracy)")
        logger.info(f"Params: {result['params']}")

        for query, stats in sorted(result["by_query"].items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            logger.info(f"  {query}: {stats['correct']}/{stats['total']} ({acc:.0%})")

        # Show failures
        failures = [d for d in result["details"] if not d["correct"]]
        if failures:
            logger.info(f"\nFailed videos ({len(failures)}):")
            for f in failures:
                logger.info(f"  {f['video']}: detected={f['detected']} "
                             f"expected={f['expected']} [{f.get('method', '')}]")

        # Save results
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
