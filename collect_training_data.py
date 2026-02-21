"""
Automatically collect training data from Medal.tv for the CS2 kill sound classifier.

Searches Medal.tv API for "cs2 1k", "cs2 2k", ... "cs2 5k" clips, downloads them,
uses the video title's kill count as ground truth, extracts audio features, and
retrains the model.

Usage:
    python collect_training_data.py              # default: 10 videos per query
    python collect_training_data.py --per-query 20
    python collect_training_data.py --queries "cs2 3k" "cs2 ace"
"""

import re
import json
import wave
import time
import logging
import argparse
import subprocess
from pathlib import Path
from urllib.parse import quote

import numpy as np

try:
    import requests
except ImportError:
    requests = None

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
RATE = 22050
WINDOW_MS = 250
WINDOW_SAMPLES = int(WINDOW_MS / 1000.0 * RATE)
HOP_MS = 50
HOP_SAMPLES = int(HOP_MS / 1000.0 * RATE)
NEG_MIN_DISTANCE_S = 1.5

MEDAL_API_BASE = "https://developers.medal.tv/v1"

# Kill count patterns in video titles
KILL_PATTERNS = {
    "1k": 1, "2k": 2, "3k": 3, "4k": 4, "5k": 5,
    "ace": 5,
    "1 kill": 1, "2 kill": 2, "3 kill": 3, "4 kill": 4, "5 kill": 5,
    "1 kills": 1, "2 kills": 2, "3 kills": 3, "4 kills": 4, "5 kills": 5,
}

# Default search queries
DEFAULT_QUERIES = ["cs2 1k", "cs2 2k", "cs2 3k", "cs2 4k", "cs2 5k", "cs2 ace"]

# State file to track processed videos
STATE_FILE = BASE_DIR / "training_data" / "processed_videos.json"


def get_api_key():
    """Get or generate a Medal.tv public API key."""
    key_path = BASE_DIR / "training_data" / "medal_api_key.txt"

    # Check saved key
    if key_path.exists():
        key = key_path.read_text().strip()
        if key:
            return key

    # Generate new key
    if requests is None:
        logger.info("Generating API key with subprocess (requests not installed)...")
        try:
            result = subprocess.run(
                ["curl", "-s", f"{MEDAL_API_BASE}/generate_public_key"],
                capture_output=True, text=True, timeout=15,
            )
            # Response is just the key text
            key = result.stdout.strip()
            # Try to parse as JSON if wrapped
            try:
                data = json.loads(key)
                key = data.get("key", data.get("apiKey", key))
            except (json.JSONDecodeError, TypeError):
                pass
        except Exception as e:
            logger.error(f"Failed to generate API key: {e}")
            return None
    else:
        try:
            resp = requests.get(f"{MEDAL_API_BASE}/generate_public_key", timeout=15)
            resp.raise_for_status()
            try:
                data = resp.json()
                key = data.get("key", data.get("apiKey", resp.text.strip()))
            except (json.JSONDecodeError, ValueError):
                key = resp.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate API key: {e}")
            return None

    if key:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(key)
        logger.info(f"API key saved to {key_path}")
    return key


def search_medal(query, api_key, limit=10, offset=0):
    """Search Medal.tv API for clips. Returns list of clip dicts."""
    url = f"{MEDAL_API_BASE}/search?text={quote(query)}&limit={limit}&offset={offset}"
    headers = {"Authorization": api_key}

    try:
        if requests:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        else:
            result = subprocess.run(
                ["curl", "-s", "-H", f"Authorization: {api_key}", url],
                capture_output=True, text=True, timeout=15,
            )
            data = json.loads(result.stdout)

        clips = data.get("contentObjects", [])
        return clips

    except Exception as e:
        logger.error(f"Medal API search failed for '{query}': {e}")
        return []


def parse_kill_count(title, query):
    """
    Extract expected kill count from video title or search query.
    Returns int or None if can't determine.
    """
    title_lower = title.lower()

    # Check title for kill patterns
    for pattern, count in KILL_PATTERNS.items():
        if pattern in title_lower:
            return count

    # Fallback: extract from query
    query_lower = query.lower()
    for pattern, count in KILL_PATTERNS.items():
        if pattern in query_lower:
            return count

    return None


def load_processed_videos():
    """Load set of already processed video content IDs."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return set()


def save_processed_videos(processed):
    """Save set of processed video content IDs."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(list(processed), f)


def build_medal_url(content_id):
    """Build a Medal.tv clip URL from content ID."""
    return f"https://medal.tv/clips/{content_id}"


def download_video(url):
    """Download a video from Medal.tv using yt-dlp. Returns path or None."""
    output_dir = BASE_DIR / "downloads"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]",
        "-o", output_template,
        "--no-playlist",
        "--merge-output-format", "mp4",
        "--print", "after_move:filepath",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=180)
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line and Path(line).exists():
                return Path(line)

        # Fallback: most recent mp4
        mp4_files = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if mp4_files:
            return mp4_files[-1]
    except subprocess.CalledProcessError as e:
        logger.error(f"yt-dlp failed: {e.stderr[:300]}")
    except subprocess.TimeoutExpired:
        logger.error(f"Download timed out")

    return None


def extract_audio(video_path):
    """Extract mono WAV audio from video. Returns numpy array or None."""
    audio_path = video_path.parent / (video_path.stem + "_collect_audio.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(RATE),
        "-ac", "1",
        str(audio_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        with wave.open(str(audio_path), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        return audio
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return None
    finally:
        if audio_path.exists():
            audio_path.unlink(missing_ok=True)


def find_top_peaks_ml(audio, n_peaks):
    """Find top-N peaks using the ML classifier if available."""
    try:
        from audio_classifier import KillSoundClassifier, extract_features_with_context
        from scipy.signal import find_peaks

        model_path = BASE_DIR / "models" / "kill_classifier.pkl"
        clf = KillSoundClassifier(str(model_path))
        if not clf.is_ready:
            return None

        windows = []
        timestamps = []
        for start in range(0, len(audio) - WINDOW_SAMPLES, HOP_SAMPLES):
            windows.append(audio[start:start + WINDOW_SAMPLES])
            timestamps.append((start + WINDOW_SAMPLES / 2) / RATE)

        if not windows:
            return None

        features = []
        for i in range(len(windows)):
            feat = extract_features_with_context(windows, RATE, i)
            features.append(feat)
        features = np.array(features)

        proba = clf.predict_proba(features)

        peaks, _ = find_peaks(proba, height=0.15, distance=10, prominence=0.10)
        if len(peaks) == 0:
            return None

        peak_probs = [(peaks[i], float(proba[peaks[i]])) for i in range(len(peaks))]
        peak_probs.sort(key=lambda x: -x[1])

        top = peak_probs[:n_peaks]
        result = [(timestamps[idx], prob) for idx, prob in top]
        result.sort(key=lambda x: x[0])
        return result

    except Exception as e:
        logger.info(f"ML peak detection not available: {e}")
        return None


def find_top_peaks_spectral(audio, n_peaks):
    """Find top-N peaks using spectral flux (fallback when no ML model)."""
    from scipy.signal import find_peaks

    win_size = int(50 / 1000 * RATE)
    hop_size = int(10 / 1000 * RATE)
    freq_low, freq_high = 1800, 4500

    hann = np.hanning(win_size)
    freqs = np.fft.rfftfreq(win_size, 1.0 / RATE)
    band_mask = (freqs >= freq_low) & (freqs <= freq_high)

    prev_band = None
    flux_list = []
    time_list = []

    for start in range(0, len(audio) - win_size, hop_size):
        chunk = audio[start:start + win_size] * hann
        spectrum = np.abs(np.fft.rfft(chunk))
        band = spectrum[band_mask]

        if prev_band is not None and len(band) == len(prev_band):
            flux = np.sum(np.maximum(band - prev_band, 0))
            flux_list.append(flux)
            time_list.append(start / RATE)
        prev_band = band.copy()

    if not flux_list:
        return None

    flux_arr = np.array(flux_list)
    min_dist = int(0.5 * RATE / hop_size)
    peaks, _ = find_peaks(flux_arr, distance=min_dist, prominence=np.std(flux_arr) * 0.5)

    if len(peaks) == 0:
        return None

    peak_fluxes = [(peaks[i], float(flux_arr[peaks[i]])) for i in range(len(peaks))]
    peak_fluxes.sort(key=lambda x: -x[1])

    top = peak_fluxes[:n_peaks]
    result = [(time_list[idx], flux) for idx, flux in top]
    result.sort(key=lambda x: x[0])
    return result


def extract_samples(audio, kill_timestamps, n_kills):
    """Extract positive and negative feature samples from audio."""
    from audio_classifier import extract_features, extract_features_with_context, augment_sample

    total_samples = len(audio)
    total_duration = total_samples / RATE

    pos_features = []
    neg_features = []

    # --- Positive samples ---
    for ts, _ in kill_timestamps:
        center = int(ts * RATE)
        start = center - WINDOW_SAMPLES // 2
        end = start + WINDOW_SAMPLES

        if start < 0 or end > total_samples:
            continue

        window = audio[start:end]

        context_windows = []
        for offset in [-1, 0, 1]:
            ctx_start = start + offset * HOP_SAMPLES
            ctx_end = ctx_start + WINDOW_SAMPLES
            if 0 <= ctx_start and ctx_end <= total_samples:
                context_windows.append(audio[ctx_start:ctx_end])
            else:
                context_windows.append(window)

        feat = extract_features_with_context(context_windows, RATE, 1)
        pos_features.append(feat)

        augmented = augment_sample(window, RATE)
        for aug_window, _ in augmented:
            aug_feat = extract_features(aug_window, RATE)
            pos_features.append(aug_feat)

    # --- Negative samples ---
    kill_times = [ts for ts, _ in kill_timestamps]
    n_neg = max(30, n_kills * 30)
    neg_count = 0
    attempts = 0

    while neg_count < n_neg and attempts < n_neg * 10:
        attempts += 1
        t = np.random.uniform(0.5, total_duration - 0.5)

        if any(abs(t - kt) < NEG_MIN_DISTANCE_S for kt in kill_times):
            continue

        center = int(t * RATE)
        start = center - WINDOW_SAMPLES // 2
        end = start + WINDOW_SAMPLES

        if start < 0 or end > total_samples:
            continue

        window = audio[start:end]
        if np.sqrt(np.mean(window ** 2)) < 0.005:
            continue

        context_windows = []
        for offset in [-1, 0, 1]:
            ctx_start = start + offset * HOP_SAMPLES
            ctx_end = ctx_start + WINDOW_SAMPLES
            if 0 <= ctx_start and ctx_end <= total_samples:
                context_windows.append(audio[ctx_start:ctx_end])
            else:
                context_windows.append(window)

        feat = extract_features_with_context(context_windows, RATE, 1)
        neg_features.append(feat)
        neg_count += 1

    logger.info(f"  Positives: {len(pos_features)}, Negatives: {len(neg_features)}")
    return pos_features, neg_features


def save_collected_data(all_pos, all_neg):
    """Save collected features to training_data/features.npz (append if exists)."""
    data_dir = BASE_DIR / "training_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / "features.npz"

    new_X = np.array(all_pos + all_neg)
    new_y = np.array([1] * len(all_pos) + [0] * len(all_neg))

    if npz_path.exists():
        try:
            existing = np.load(npz_path)
            old_X = existing["X"]
            old_y = existing["y"]
            new_X = np.vstack([old_X, new_X])
            new_y = np.concatenate([old_y, new_y])
            logger.info(f"Appending to existing data ({len(old_X)} -> {len(new_X)} samples)")
        except Exception as e:
            logger.warning(f"Could not load existing data, overwriting: {e}")

    np.savez(npz_path, X=new_X, y=new_y)
    logger.info(f"Saved {len(new_X)} samples to {npz_path}")
    logger.info(f"  Positives: {int(np.sum(new_y == 1))}, Negatives: {int(np.sum(new_y == 0))}")
    return npz_path


def retrain_model():
    """Retrain the classifier using all available data."""
    try:
        from train_classifier import bootstrap_labels, extract_training_data, train_model
        import pickle
        from datetime import datetime

        logger.info("Retraining model with all available data...")

        labels = bootstrap_labels(BASE_DIR)

        if labels:
            X_meta, y_meta = extract_training_data(labels, BASE_DIR)
        else:
            X_meta, y_meta = np.array([]), np.array([])

        npz_path = BASE_DIR / "training_data" / "features.npz"
        if npz_path.exists():
            collected = np.load(npz_path)
            X_coll = collected["X"]
            y_coll = collected["y"]
            logger.info(f"Collected data: {len(X_coll)} samples")
        else:
            X_coll, y_coll = np.array([]), np.array([])

        parts_X = [x for x in [X_meta, X_coll] if len(x) > 0]
        parts_y = [y for y in [y_meta, y_coll] if len(y) > 0]

        if not parts_X:
            logger.error("No training data available")
            return

        X = np.vstack(parts_X)
        y = np.concatenate(parts_y)

        logger.info(f"Combined dataset: {len(X)} samples "
                     f"({int(np.sum(y == 1))} pos, {int(np.sum(y == 0))} neg)")

        model, cv_scores = train_model(X, y)

        models_dir = BASE_DIR / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "kill_classifier.pkl"

        session_ids = [s["session_id"] for s in labels] if labels else []
        model_data = {
            "model": model,
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X),
            "n_positive": int(np.sum(y == 1)),
            "n_negative": int(np.sum(y == 0)),
            "cv_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_std": round(float(cv_scores.std()), 4),
            "feature_dim": X.shape[1],
            "sessions_used": session_ids,
            "includes_collected_data": len(X_coll) > 0,
            "collected_samples": len(X_coll),
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        meta = {k: v for k, v in model_data.items() if k != "model"}
        with open(models_dir / "training_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Model saved! F1={model_data['cv_f1_mean']:.4f}, "
                     f"{model_data['n_samples']} samples")

    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)


def collect_from_api(queries, per_query):
    """Search Medal.tv API and return list of {url, expected_kills, title}."""
    api_key = get_api_key()
    if not api_key:
        logger.error("Could not get Medal.tv API key")
        return []

    logger.info(f"Medal.tv API key: {api_key[:20]}...")

    processed = load_processed_videos()
    logger.info(f"Already processed: {len(processed)} videos")

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
            if not content_id or content_id in processed:
                continue

            title = clip.get("contentTitle", "")
            duration = clip.get("videoLengthSeconds", 0)

            # Skip very short or very long clips
            if duration < 3 or duration > 120:
                continue

            expected = parse_kill_count(title, query)
            if expected is None:
                continue

            # Build URL - prefer directClipUrl if available
            clip_url = clip.get("directClipUrl", "")
            if not clip_url:
                clip_url = build_medal_url(content_id)

            videos.append({
                "url": clip_url,
                "content_id": content_id,
                "expected_kills": expected,
                "title": title,
                "duration": duration,
            })

        # Rate limiting
        time.sleep(0.5)

    logger.info(f"Total new videos to process: {len(videos)}")
    return videos


def collect_from_json():
    """Load videos from training_videos.json (manual mode)."""
    path = BASE_DIR / "training_videos.json"
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        videos = json.load(f)

    valid = []
    for v in videos:
        if "url" not in v or "expected_kills" not in v:
            continue
        if v["expected_kills"] < 1:
            continue
        valid.append(v)

    if valid:
        logger.info(f"Loaded {len(valid)} videos from training_videos.json")
    return valid


def process_videos(videos):
    """Download and process a list of videos, return (all_pos, all_neg, processed_ids)."""
    all_pos = []
    all_neg = []
    new_processed = []
    completed = 0

    for i, entry in enumerate(videos, 1):
        url = entry["url"]
        expected = entry["expected_kills"]
        title = entry.get("title", url)
        content_id = entry.get("content_id", "")

        logger.info(f"\n[{i}/{len(videos)}] \"{title}\" (expected: {expected} kills)")
        logger.info(f"  URL: {url}")

        # Download
        video_path = download_video(url)
        if not video_path:
            logger.error(f"  Download failed, skipping")
            continue

        logger.info(f"  Downloaded: {video_path.name}")

        # Extract audio
        audio = extract_audio(video_path)
        if audio is None:
            logger.error(f"  Audio extraction failed, skipping")
            continue

        duration = len(audio) / RATE
        logger.info(f"  Audio: {duration:.1f}s")

        # Find top-N peaks
        peaks = find_top_peaks_ml(audio, expected)
        if peaks:
            logger.info(f"  ML peaks: {[(f'{t:.2f}s', f'{p:.4f}') for t, p in peaks]}")
        else:
            peaks = find_top_peaks_spectral(audio, expected)
            if peaks:
                logger.info(f"  Spectral peaks: {[(f'{t:.2f}s', f'{p:.1f}') for t, p in peaks]}")
            else:
                logger.warning(f"  No peaks found, skipping")
                continue

        # Extract samples
        pos, neg = extract_samples(audio, peaks, expected)
        all_pos.extend(pos)
        all_neg.extend(neg)
        completed += 1

        if content_id:
            new_processed.append(content_id)

        # Clean up downloaded video to save disk space
        try:
            video_path.unlink(missing_ok=True)
            logger.info(f"  Cleaned up: {video_path.name}")
        except OSError:
            pass

        logger.info(f"  Running total: {len(all_pos)} pos, {len(all_neg)} neg")

    logger.info(f"\nProcessed: {completed}/{len(videos)} videos")
    return all_pos, all_neg, new_processed


def main():
    parser = argparse.ArgumentParser(description="Collect CS2 training data from Medal.tv")
    parser.add_argument("--per-query", type=int, default=10,
                        help="Number of videos per search query (default: 10)")
    parser.add_argument("--queries", nargs="+", default=None,
                        help="Custom search queries (default: cs2 1k..5k, cs2 ace)")
    parser.add_argument("--manual", action="store_true",
                        help="Only use training_videos.json (skip API search)")
    parser.add_argument("--skip-retrain", action="store_true",
                        help="Skip model retraining after collection")
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES

    logger.info("=" * 60)
    logger.info("CS2 Training Data Collector")
    logger.info("=" * 60)

    # Collect video list
    videos = []

    if not args.manual:
        api_videos = collect_from_api(queries, args.per_query)
        videos.extend(api_videos)

    # Also include manual entries if present
    json_videos = collect_from_json()
    videos.extend(json_videos)

    if not videos:
        logger.error("No videos to process. Check API connectivity or add URLs to training_videos.json")
        return

    logger.info(f"\nTotal videos to process: {len(videos)}")

    # Process
    all_pos, all_neg, new_processed = process_videos(videos)

    if not all_pos:
        logger.error("No positive samples collected")
        return

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Total: {len(all_pos)} positive, {len(all_neg)} negative samples")

    # Save
    save_collected_data(all_pos, all_neg)

    # Update processed state
    if new_processed:
        processed = load_processed_videos()
        processed.update(new_processed)
        save_processed_videos(processed)
        logger.info(f"Updated processed list: {len(processed)} total videos")

    # Retrain
    if not args.skip_retrain:
        retrain_model()

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
