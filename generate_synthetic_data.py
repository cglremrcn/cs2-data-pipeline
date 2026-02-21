"""
Generate synthetic training data from clean reference kill sounds.

Approach:
1. Load clean kill sound WAVs from reference_sounds/
2. Extract background audio from real CS2 gameplay videos
3. Mix kill sounds into background at random positions with varying SNR
4. Extract features using the same pipeline as inference
5. Save perfectly labeled training data

Usage:
    python generate_synthetic_data.py
    python generate_synthetic_data.py --background path/to/video.mp4
    python generate_synthetic_data.py --n-positive 1000 --n-negative 3000
"""

import argparse
import json
import wave
import logging
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np

from audio_classifier import (
    extract_features,
    extract_features_with_context,
    augment_sample,
)

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

# Curated kill sound files with sampling weights.
# These are the actual headshot/kill sounds from CS2 game files (player folder).
# noarmor variants are the most common (most kills are headshots without helmet).
KILL_SOUND_FILES = {
    "sounds_player_headshot_noarmor_01.wav": 0.11,
    "sounds_player_headshot_noarmor_02.wav": 0.11,
    "sounds_player_headshot_noarmor_03.wav": 0.11,
    "sounds_player_headshot_noarmor_04.wav": 0.11,
    "sounds_player_headshot_noarmor_05.wav": 0.11,
    "sounds_player_headshot_armor_e1.wav": 0.08,        # Headshot with armor (dink)
    "sounds_player_headshot_armor_flesh.wav": 0.11,     # Headshot armor + flesh
    "sounds_player_bodyshot_kill_01.wav": 0.11,         # Bodyshot kill
    "sounds_player_kill_doof_01.wav": 0.15,             # Kill confirmation sound (Reddit verified)
}


def _load_audio_file(path):
    """Load any audio file as mono float64 at 22050 Hz."""
    tmp_path = path.parent / f"_tmp_{path.stem}_conv.wav"
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-ar", str(RATE), "-ac", "1",
            "-acodec", "pcm_s16le", str(tmp_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        with wave.open(str(tmp_path), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    except Exception as e:
        logger.warning(f"Failed to load {path.name}: {e}")
        return None
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def load_reference_sounds(ref_dir):
    """Load curated kill sound WAVs from reference_sounds/ directory.

    Only loads files listed in KILL_SOUND_FILES with their sampling weights.
    """
    ref_dir = Path(ref_dir)
    sounds = []

    for filename, weight in KILL_SOUND_FILES.items():
        wav_file = ref_dir / filename
        if not wav_file.exists():
            logger.warning(f"  Missing: {filename}")
            continue

        audio = _load_audio_file(wav_file)
        if audio is not None and len(audio) > 0:
            sounds.append({"name": wav_file.stem, "audio": audio, "weight": weight})
            logger.info(f"  Loaded: {wav_file.name} "
                        f"({len(audio)} samples, {len(audio)/RATE:.3f}s, "
                        f"weight={weight:.0%})")

    return sounds


def extract_background_audio(video_path):
    """Extract full audio from a gameplay video."""
    logger.info(f"  Extracting audio from {video_path.name}...")
    audio = _load_audio_file(video_path)
    if audio is not None:
        logger.info(f"    -> {len(audio)/RATE:.1f}s of background audio")
    return audio


def find_background_videos():
    """Find CS2 gameplay videos in downloads/."""
    videos = []
    dl_dir = BASE_DIR / "downloads"
    if dl_dir.exists():
        for ext in ["*.mp4", "*.mkv", "*.avi", "*.webm"]:
            videos.extend(dl_dir.glob(ext))
    return sorted(videos)


def mix_kill_into_background(background, kill_sound, position, snr_db):
    """Mix a kill sound into background audio at given position and SNR.

    Returns (mixed_audio, kill_center_sample).
    """
    result = background.copy()
    kill_len = len(kill_sound)

    if position + kill_len > len(result):
        return result, -1

    # Calculate gain for desired SNR
    bg_segment = result[position:position + kill_len]
    bg_power = np.mean(bg_segment ** 2) + 1e-10
    kill_power = np.mean(kill_sound ** 2) + 1e-10

    desired_kill_power = bg_power * (10 ** (snr_db / 10))
    gain = np.sqrt(desired_kill_power / kill_power)

    result[position:position + kill_len] += kill_sound * gain
    result = np.clip(result, -1.0, 1.0)

    return result, position + kill_len // 2


def generate_dataset(reference_sounds, background_audios,
                     n_positive=500, n_negative=1500,
                     snr_range=(3, 20), augment_per_positive=8):
    """Generate synthetic training dataset.

    Positive: kill sound mixed into random background position at random SNR.
    Negative: random positions in raw background audio.
    """
    all_features = []
    all_labels = []

    logger.info(f"Target: {n_positive} positive + {n_negative} negative samples")

    # Build weighted sampling probabilities
    weights = np.array([s["weight"] for s in reference_sounds])
    weights = weights / weights.sum()

    # --- Positive samples ---
    for i in range(n_positive):
        bg = background_audios[np.random.randint(len(background_audios))]
        idx = np.random.choice(len(reference_sounds), p=weights)
        kill = reference_sounds[idx]["audio"]

        snr = np.random.uniform(*snr_range)

        margin = WINDOW_SAMPLES + len(kill)
        if len(bg) < margin * 2:
            continue

        position = np.random.randint(margin, len(bg) - margin)
        mixed, kill_center = mix_kill_into_background(bg, kill, position, snr)
        if kill_center < 0:
            continue

        # Extract feature window centered on kill
        start = kill_center - WINDOW_SAMPLES // 2
        end = start + WINDOW_SAMPLES
        if start < 0 or end > len(mixed):
            continue

        # Context windows for delta-MFCC
        context_windows = []
        for offset in [-1, 0, 1]:
            ctx_start = start + offset * HOP_SAMPLES
            ctx_end = ctx_start + WINDOW_SAMPLES
            if 0 <= ctx_start and ctx_end <= len(mixed):
                context_windows.append(mixed[ctx_start:ctx_end])
            else:
                context_windows.append(mixed[start:end])

        feat = extract_features_with_context(context_windows, RATE, 1)
        all_features.append(feat)
        all_labels.append(1)

        # Augmented copies
        window = mixed[start:end]
        augmented = augment_sample(window, RATE)
        for aug_window, desc in augmented[:augment_per_positive]:
            aug_feat = extract_features(aug_window, RATE)
            all_features.append(aug_feat)
            all_labels.append(1)

        if (i + 1) % 100 == 0:
            logger.info(f"  Positive: {i+1}/{n_positive}")

    # --- Negative samples ---
    neg_count = 0
    attempts = 0
    max_attempts = n_negative * 5

    while neg_count < n_negative and attempts < max_attempts:
        attempts += 1
        bg = background_audios[np.random.randint(len(background_audios))]

        margin = WINDOW_SAMPLES * 2
        if len(bg) < margin * 2:
            continue

        pos = np.random.randint(margin, len(bg) - margin)
        start = pos
        end = start + WINDOW_SAMPLES

        window = bg[start:end]

        # Skip silent windows
        if np.sqrt(np.mean(window ** 2)) < 0.005:
            continue

        # Context windows
        context_windows = []
        for offset in [-1, 0, 1]:
            ctx_start = start + offset * HOP_SAMPLES
            ctx_end = ctx_start + WINDOW_SAMPLES
            if 0 <= ctx_start and ctx_end <= len(bg):
                context_windows.append(bg[ctx_start:ctx_end])
            else:
                context_windows.append(window)

        feat = extract_features_with_context(context_windows, RATE, 1)
        all_features.append(feat)
        all_labels.append(0)
        neg_count += 1

        if neg_count % 500 == 0:
            logger.info(f"  Negative: {neg_count}/{n_negative}")

    X = np.array(all_features)
    y = np.array(all_labels)

    actual_pos = int(np.sum(y == 1))
    actual_neg = int(np.sum(y == 0))
    logger.info(f"Dataset: {len(X)} samples ({actual_pos} pos, {actual_neg} neg)")

    return X, y


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--background", type=str, nargs="*",
                        help="Path(s) to background video/audio files")
    parser.add_argument("--n-positive", type=int, default=500,
                        help="Number of positive base samples (default: 500)")
    parser.add_argument("--n-negative", type=int, default=1500,
                        help="Number of negative samples (default: 1500)")
    parser.add_argument("--augment", type=int, default=8,
                        help="Augmented copies per positive (default: 8)")
    parser.add_argument("--snr-min", type=float, default=3,
                        help="Minimum SNR in dB (default: 3)")
    parser.add_argument("--snr-max", type=float, default=20,
                        help="Maximum SNR in dB (default: 20)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Synthetic Training Data Generator")
    logger.info("=" * 60)

    # Step 1: Load reference sounds
    ref_dir = BASE_DIR / "reference_sounds"
    logger.info("\n[1/4] Loading reference kill sounds...")
    if not ref_dir.exists() or not list(ref_dir.glob("*.wav")):
        logger.error(f"No WAV files found in {ref_dir}/")
        logger.error("Put CS2 kill sound WAV files in reference_sounds/")
        return

    reference_sounds = load_reference_sounds(ref_dir)
    if not reference_sounds:
        logger.error("No valid reference sounds loaded")
        return
    logger.info(f"Loaded {len(reference_sounds)} reference sounds")

    # Step 2: Get background audio
    logger.info("\n[2/4] Loading background audio...")
    background_audios = []

    if args.background:
        video_paths = [Path(p) for p in args.background]
    else:
        video_paths = find_background_videos()

    if not video_paths:
        logger.error("No background videos found in downloads/")
        logger.error("Use --background <video_path> to specify a CS2 gameplay video")
        return

    for vp in video_paths[:5]:  # Cap at 5 to keep generation fast
        audio = extract_background_audio(vp)
        if audio is not None and len(audio) > RATE * 10:  # At least 10 seconds
            background_audios.append(audio)

    if not background_audios:
        logger.error("No usable background audio extracted")
        return

    total_bg_s = sum(len(a) / RATE for a in background_audios)
    logger.info(f"Background: {len(background_audios)} sources, "
                f"{total_bg_s:.0f}s total")

    # Step 3: Generate dataset
    logger.info("\n[3/4] Generating synthetic dataset...")
    X, y = generate_dataset(
        reference_sounds, background_audios,
        n_positive=args.n_positive,
        n_negative=args.n_negative,
        snr_range=(args.snr_min, args.snr_max),
        augment_per_positive=args.augment,
    )

    if len(X) == 0:
        logger.error("No training data generated")
        return

    # Step 4: Save
    logger.info("\n[4/4] Saving...")
    out_dir = BASE_DIR / "training_data"
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / "synthetic_features.npz"
    np.savez_compressed(out_path, X=X, y=y)
    logger.info(f"Saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    meta = {
        "generated_at": datetime.now().isoformat(),
        "n_samples": len(X),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "reference_sounds": [s["name"] for s in reference_sounds],
        "n_background_videos": len(background_audios),
        "background_seconds": round(total_bg_s, 1),
        "snr_range": [args.snr_min, args.snr_max],
        "augment_per_positive": args.augment,
    }
    meta_path = out_dir / "synthetic_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info(f"  Samples: {len(X)} ({meta['n_positive']} pos, "
                f"{meta['n_negative']} neg)")
    logger.info(f"  Output: {out_path}")
    logger.info("=" * 60)
    logger.info("\nNext: run 'python train_classifier.py' to train the model")


if __name__ == "__main__":
    main()
