"""
Train the ML kill sound classifier for CS2.

Reads metadata/*.json files to get kill timestamps, extracts audio features
from the corresponding video files, and trains a GradientBoostingClassifier.

Usage:
    python train_classifier.py
"""

import json
import wave
import pickle
import logging
import subprocess
from pathlib import Path
from datetime import datetime

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
KILL_HALF_WINDOW_S = 0.15  # ±150ms around kill timestamp = positive region
NEG_MIN_DISTANCE_S = 1.5   # negative samples must be >1.5s from any kill


def bootstrap_labels(base_dir):
    """
    Read metadata/*.json files and extract kill timestamps per video.

    Returns list of dicts:
      {"video_path": Path, "kill_timestamps": [float, ...], "duration": float}
    """
    meta_dir = base_dir / "metadata"
    if not meta_dir.exists():
        logger.error("No metadata/ directory found")
        return []

    labels = []
    for meta_file in sorted(meta_dir.glob("*.json")):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping {meta_file.name}: {e}")
            continue

        video_rel = meta.get("source", {}).get("downloaded_file", "")
        if not video_rel:
            continue

        video_path = base_dir / video_rel.replace("\\", "/")
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            continue

        timestamps = [
            d["timestamp_seconds"]
            for d in meta.get("detections", [])
            if "timestamp_seconds" in d
        ]

        if not timestamps:
            continue

        duration = meta.get("source", {}).get("duration_seconds", 0)
        labels.append({
            "video_path": video_path,
            "kill_timestamps": timestamps,
            "duration": duration,
            "session_id": meta.get("session_id", meta_file.stem),
        })
        logger.info(f"  {meta_file.name}: {len(timestamps)} kills, "
                     f"video={video_path.name}")

    logger.info(f"Found {len(labels)} sessions with {sum(len(l['kill_timestamps']) for l in labels)} total kills")
    return labels


def extract_audio(video_path):
    """Extract mono WAV audio from video, return (samples, rate) or None."""
    audio_path = video_path.parent / (video_path.stem + "_train_audio.wav")
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
        logger.error(f"Audio extraction failed for {video_path.name}: {e}")
        return None
    finally:
        if audio_path.exists():
            audio_path.unlink(missing_ok=True)


def extract_training_data(labels, base_dir):
    """
    Extract feature matrix and label array from labeled sessions.

    For each session:
    - Positive: windows centered on kill timestamps + augmented copies
    - Negative: random windows far from any kill

    Returns (X, y) numpy arrays.
    """
    from audio_classifier import extract_features, extract_features_with_context, augment_sample

    all_features = []
    all_labels = []

    for session in labels:
        video_path = session["video_path"]
        kill_times = session["kill_timestamps"]
        duration = session["duration"]

        logger.info(f"Processing {video_path.name} ({len(kill_times)} kills)...")

        audio = extract_audio(video_path)
        if audio is None:
            continue

        total_samples = len(audio)
        total_duration = total_samples / RATE

        # --- Positive samples ---
        for kt in kill_times:
            center_sample = int(kt * RATE)
            start = center_sample - WINDOW_SAMPLES // 2
            end = start + WINDOW_SAMPLES

            if start < 0 or end > total_samples:
                continue

            window = audio[start:end]

            # Get context windows for delta-MFCC
            context_windows = []
            for offset in [-1, 0, 1]:
                ctx_start = start + offset * HOP_SAMPLES
                ctx_end = ctx_start + WINDOW_SAMPLES
                if 0 <= ctx_start and ctx_end <= total_samples:
                    context_windows.append(audio[ctx_start:ctx_end])
                else:
                    context_windows.append(window)

            feat = extract_features_with_context(context_windows, RATE, 1)
            all_features.append(feat)
            all_labels.append(1)

            # Augmented copies
            augmented = augment_sample(window, RATE)
            for aug_window, desc in augmented:
                aug_feat = extract_features(aug_window, RATE)
                all_features.append(aug_feat)
                all_labels.append(1)

        # --- Negative samples ---
        # Sample ~30 negatives per kill from non-kill regions
        n_neg = max(30, int(len(kill_times) * 30))
        neg_count = 0
        attempts = 0
        max_attempts = n_neg * 10

        while neg_count < n_neg and attempts < max_attempts:
            attempts += 1
            # Random timestamp
            t = np.random.uniform(0.5, total_duration - 0.5)

            # Check distance from all kills
            if any(abs(t - kt) < NEG_MIN_DISTANCE_S for kt in kill_times):
                continue

            center_sample = int(t * RATE)
            start = center_sample - WINDOW_SAMPLES // 2
            end = start + WINDOW_SAMPLES

            if start < 0 or end > total_samples:
                continue

            window = audio[start:end]

            # Skip silent windows
            if np.sqrt(np.mean(window ** 2)) < 0.005:
                continue

            # Get context windows
            context_windows = []
            for offset in [-1, 0, 1]:
                ctx_start = start + offset * HOP_SAMPLES
                ctx_end = ctx_start + WINDOW_SAMPLES
                if 0 <= ctx_start and ctx_end <= total_samples:
                    context_windows.append(audio[ctx_start:ctx_end])
                else:
                    context_windows.append(window)

            feat = extract_features_with_context(context_windows, RATE, 1)
            all_features.append(feat)
            all_labels.append(0)
            neg_count += 1

        logger.info(f"  Positives: {len(kill_times)} raw + "
                     f"{len(kill_times) * 16} augmented = "
                     f"{len(kill_times) * 17}, Negatives: {neg_count}")

    X = np.array(all_features)
    y = np.array(all_labels)
    logger.info(f"Total dataset: {len(X)} samples ({np.sum(y == 1)} pos, {np.sum(y == 0)} neg)")
    return X, y


def train_model(X, y):
    """
    Train GradientBoostingClassifier with cross-validation.

    Returns (model, cv_scores).
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    logger.info("Training GradientBoostingClassifier...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )),
    ])

    # Cross-validation
    n_splits = min(5, max(2, int(np.sum(y == 1) / 3)))
    logger.info(f"Running {n_splits}-fold cross-validation...")

    cv_scores = cross_val_score(pipeline, X, y, cv=n_splits, scoring="f1")
    logger.info(f"CV F1 scores: {cv_scores}")
    logger.info(f"Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train final model on all data
    pipeline.fit(X, y)
    logger.info("Final model trained on all data")

    return pipeline, cv_scores


def main():
    """Full training pipeline.

    Data sources (in priority order):
    1. Synthetic data from reference sounds (training_data/synthetic_features.npz)
    2. Real labeled data from metadata/*.json sessions
    Either or both can be used.
    """
    logger.info("=" * 60)
    logger.info("CS2 Kill Sound Classifier - Training")
    logger.info("=" * 60)

    X_parts = []
    y_parts = []
    data_sources = []

    # Source 1: Synthetic data from reference sounds (preferred)
    synth_path = BASE_DIR / "training_data" / "synthetic_features.npz"
    if synth_path.exists():
        logger.info("\n[1/4] Loading synthetic training data...")
        try:
            synth = np.load(synth_path)
            X_synth, y_synth = synth["X"], synth["y"]
            X_parts.append(X_synth)
            y_parts.append(y_synth)
            logger.info(f"  Synthetic: {len(X_synth)} samples "
                        f"({int(np.sum(y_synth == 1))} pos, "
                        f"{int(np.sum(y_synth == 0))} neg)")
            data_sources.append("synthetic")
        except Exception as e:
            logger.warning(f"  Failed to load synthetic data: {e}")
    else:
        logger.info("\n[1/4] No synthetic data found (training_data/synthetic_features.npz)")
        logger.info("  Run 'python generate_synthetic_data.py' to generate it")

    # Source 2: Real labeled data from metadata sessions
    logger.info("\n[2/4] Reading labels from metadata...")
    labels = bootstrap_labels(BASE_DIR)
    if labels:
        logger.info("Extracting features from real sessions...")
        X_real, y_real = extract_training_data(labels, BASE_DIR)
        if len(X_real) > 0:
            X_parts.append(X_real)
            y_parts.append(y_real)
            data_sources.append("metadata")
    else:
        logger.info("  No labeled metadata sessions available")

    # Combine all data
    if not X_parts:
        logger.error("No training data from any source!")
        logger.error("  - Run 'python generate_synthetic_data.py' for synthetic data")
        logger.error("  - Or process videos through the pipeline for real data")
        return

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    logger.info(f"\nCombined dataset: {len(X)} samples "
                f"({int(np.sum(y == 1))} pos, {int(np.sum(y == 0))} neg)")
    logger.info(f"Data sources: {', '.join(data_sources)}")

    # Step 3: Train model
    logger.info("\n[3/4] Training model...")
    model, cv_scores = train_model(X, y)

    # Step 4: Save model
    logger.info("\n[4/4] Saving model...")
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "kill_classifier.pkl"
    model_data = {
        "model": model,
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(X),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std": round(float(cv_scores.std()), 4),
        "feature_dim": X.shape[1],
        "data_sources": data_sources,
        "sessions_used": [s["session_id"] for s in labels] if labels else [],
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    logger.info(f"Model saved: {model_path}")

    # Save training metadata (for git tracking)
    meta_path = models_dir / "training_meta.json"
    training_meta = {k: v for k, v in model_data.items() if k != "model"}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2, ensure_ascii=False)
    logger.info(f"Training metadata saved: {meta_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"  Sources: {', '.join(data_sources)}")
    logger.info(f"  Samples: {len(X)} ({model_data['n_positive']} pos, {model_data['n_negative']} neg)")
    logger.info(f"  CV F1: {model_data['cv_f1_mean']:.4f} (+/- {model_data['cv_f1_std']:.4f})")
    logger.info(f"  Model: {model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
