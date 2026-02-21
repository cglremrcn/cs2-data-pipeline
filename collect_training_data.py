"""
Collect training data from Medal.tv videos for the CS2 kill sound classifier.

Reads training_videos.json with URL + expected_kills pairs, downloads each video,
extracts audio, finds top-N peaks (N = expected kills) as positive samples, and
collects negative samples from non-kill regions. Saves features to
training_data/features.npz and retrains the model.

Usage:
    1. Search Medal.tv for "cs2 3k", "cs2 5k", etc.
    2. Add URLs to training_videos.json:
       [{"url": "https://medal.tv/.../abc", "expected_kills": 3}, ...]
    3. Run: python collect_training_data.py
"""

import json
import wave
import logging
import subprocess
from pathlib import Path

import numpy as np

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


def load_video_list():
    """Load training_videos.json."""
    path = BASE_DIR / "training_videos.json"
    if not path.exists():
        logger.error(
            "training_videos.json not found. Create it with:\n"
            '[{"url": "https://medal.tv/.../abc", "expected_kills": 3}]'
        )
        return []

    with open(path, "r", encoding="utf-8") as f:
        videos = json.load(f)

    valid = []
    for v in videos:
        if "url" not in v or "expected_kills" not in v:
            logger.warning(f"Skipping invalid entry: {v}")
            continue
        if v["expected_kills"] < 1:
            logger.warning(f"Skipping entry with 0 kills: {v['url']}")
            continue
        valid.append(v)

    logger.info(f"Loaded {len(valid)} videos from training_videos.json")
    return valid


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
        logger.error(f"yt-dlp failed for {url}: {e.stderr[:300]}")
    except subprocess.TimeoutExpired:
        logger.error(f"Download timed out for {url}")

    return None


def extract_audio(video_path):
    """Extract mono WAV audio from video. Returns (samples, rate) or None."""
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

        # Sliding window
        windows = []
        timestamps = []
        for start in range(0, len(audio) - WINDOW_SAMPLES, HOP_SAMPLES):
            windows.append(audio[start:start + WINDOW_SAMPLES])
            timestamps.append((start + WINDOW_SAMPLES / 2) / RATE)

        if not windows:
            return None

        # Extract features
        features = []
        for i in range(len(windows)):
            feat = extract_features_with_context(windows, RATE, i)
            features.append(feat)
        features = np.array(features)

        # Predict
        proba = clf.predict_proba(features)

        # Find peaks
        peaks, props = find_peaks(proba, height=0.15, distance=10, prominence=0.10)
        if len(peaks) == 0:
            return None

        # Sort by probability and take top-N
        peak_probs = [(peaks[i], float(proba[peaks[i]])) for i in range(len(peaks))]
        peak_probs.sort(key=lambda x: -x[1])

        top = peak_probs[:n_peaks]
        result = [(timestamps[idx], prob) for idx, prob in top]
        result.sort(key=lambda x: x[0])  # sort by time

        return result

    except (ImportError, Exception) as e:
        logger.info(f"ML peak detection not available: {e}")
        return None


def find_top_peaks_spectral(audio, n_peaks):
    """Find top-N peaks using spectral flux (fallback when no ML model)."""
    from scipy.signal import find_peaks

    win_size = int(50 / 1000 * RATE)  # 50ms
    hop_size = int(10 / 1000 * RATE)  # 10ms
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

    # Find peaks
    min_dist = int(0.5 * RATE / hop_size)
    peaks, props = find_peaks(flux_arr, distance=min_dist, prominence=np.std(flux_arr) * 0.5)

    if len(peaks) == 0:
        return None

    # Sort by flux and take top-N
    peak_fluxes = [(peaks[i], float(flux_arr[peaks[i]])) for i in range(len(peaks))]
    peak_fluxes.sort(key=lambda x: -x[1])

    top = peak_fluxes[:n_peaks]
    result = [(time_list[idx], flux) for idx, flux in top]
    result.sort(key=lambda x: x[0])

    return result


def extract_samples(audio, kill_timestamps, n_kills):
    """
    Extract positive and negative feature samples from audio.

    Positive: windows centered on kill timestamps + augmented copies.
    Negative: random windows far from any kill.
    """
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

        # Context windows for delta-MFCC
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

        # Augmented copies
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

        # Skip silent windows
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

    # Append to existing data if present
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

        # Get metadata-based labels
        labels = bootstrap_labels(BASE_DIR)

        # Extract metadata-based features
        if labels:
            X_meta, y_meta = extract_training_data(labels, BASE_DIR)
        else:
            X_meta, y_meta = np.array([]), np.array([])

        # Load collected data
        npz_path = BASE_DIR / "training_data" / "features.npz"
        if npz_path.exists():
            collected = np.load(npz_path)
            X_coll = collected["X"]
            y_coll = collected["y"]
            logger.info(f"Collected data: {len(X_coll)} samples")
        else:
            X_coll, y_coll = np.array([]), np.array([])

        # Combine
        parts_X = [x for x in [X_meta, X_coll] if len(x) > 0]
        parts_y = [y for y in [y_meta, y_coll] if len(y) > 0]

        if not parts_X:
            logger.error("No training data available")
            return

        X = np.vstack(parts_X)
        y = np.concatenate(parts_y)

        logger.info(f"Combined dataset: {len(X)} samples "
                     f"({int(np.sum(y == 1))} pos, {int(np.sum(y == 0))} neg)")

        # Train
        model, cv_scores = train_model(X, y)

        # Save
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

        # Save training metadata
        meta = {k: v for k, v in model_data.items() if k != "model"}
        with open(models_dir / "training_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Model saved! F1={model_data['cv_f1_mean']:.4f}, "
                     f"{model_data['n_samples']} samples")

    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)


def main():
    logger.info("=" * 60)
    logger.info("CS2 Training Data Collector")
    logger.info("=" * 60)

    videos = load_video_list()
    if not videos:
        return

    all_pos = []
    all_neg = []
    processed = 0

    for i, entry in enumerate(videos, 1):
        url = entry["url"]
        expected = entry["expected_kills"]

        logger.info(f"\n[{i}/{len(videos)}] {url} (expected: {expected} kills)")

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

        # Find top-N peaks (ML first, spectral flux fallback)
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
        processed += 1

        logger.info(f"  Done! Running total: {len(all_pos)} pos, {len(all_neg)} neg")

    if not all_pos:
        logger.error("No positive samples collected")
        return

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Collection complete: {processed}/{len(videos)} videos processed")
    logger.info(f"Total: {len(all_pos)} positive, {len(all_neg)} negative samples")

    # Save collected data
    save_collected_data(all_pos, all_neg)

    # Retrain model
    retrain_model()

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
