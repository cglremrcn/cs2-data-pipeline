"""
CS2 Data Pipeline - Kill moment detection and clip extraction.
Processes Medal.tv clips to detect kill moments and extract labeled clips.
"""

import subprocess
import json
import logging
import time
import wave
import threading
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
    "cooldown_seconds": 1.0,

    # Audio kill sound detection
    "kill_sound_freq_low": 1800,
    "kill_sound_freq_high": 4500,
    "audio_window_ms": 50,
    "audio_hop_ms": 10,

    # Audio fingerprinting (auto-calibration)
    "fingerprint_snippet_ms": 250,
    "fingerprint_ncc_threshold": 0.28,
    "fingerprint_top_n": 15,

    # ML classifier
    "ml_model_path": "models/kill_classifier.pkl",
    "ml_window_ms": 250,
    "ml_hop_ms": 50,
    "ml_peak_height": 0.25,
    "ml_peak_distance": 10,       # 10 hops = 0.5s
    "ml_peak_prominence": 0.15,

    # Output
    "clips_dir": "clips",

    # Metadata
    "metadata_dir": "metadata",
}


class CS2DataPipeline:
    """End-to-end pipeline for CS2 kill moment detection and clip extraction."""

    def __init__(self, base_dir=None, config=None):
        self.base_dir = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parent
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._setup_directories()
        self._setup_logging()
        self._load_ml_model()

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

    def _load_ml_model(self):
        """Load the ML kill sound classifier if available."""
        self.ml_classifier = None
        model_path = self.base_dir / self.config["ml_model_path"]
        try:
            from audio_classifier import KillSoundClassifier
            clf = KillSoundClassifier(str(model_path))
            if clf.is_ready:
                self.ml_classifier = clf
                logger.info("ML classifier loaded — will use ML detection path")
            else:
                logger.info("ML model not found — will use NCC fallback")
        except ImportError:
            logger.info("audio_classifier module not available — using NCC fallback")
        except Exception as e:
            logger.warning(f"ML model load failed: {e} — using NCC fallback")

    # =========================================================================
    # Phase 1: Video Download
    # =========================================================================

    def download_video(self, url, progress_callback=None):
        """
        Download video from Medal.tv using yt-dlp.
        Returns path to downloaded file.
        """
        if progress_callback:
            progress_callback("downloading", "Downloading video...")

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
        Detect personal kills using ML classifier (preferred) or NCC fallback.

        ML path:  sliding window → predict_proba → find_peaks → kill feed verify
        NCC path: spectral flux → NCC fingerprint → kill feed verify → dedup
        """
        if progress_callback:
            progress_callback("detecting", "Detecting kill moments...")

        video_path = Path(video_path)

        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        logger.info(f"Video: {width}x{height}, {fps:.1f}fps, {duration:.1f}s, {total_frames} frames")

        # --- ML path ---
        if self.ml_classifier is not None:
            logger.info("Using ML classifier for kill detection")
            audio_path = self._extract_audio(video_path)
            if audio_path:
                try:
                    ml_dets = self._detect_kills_ml(audio_path, fps)
                    logger.info(f"ML detection: {len(ml_dets)} candidates")
                except Exception as e:
                    logger.warning(f"ML detection failed: {e}, falling back to NCC")
                    ml_dets = []
                finally:
                    audio_path.unlink(missing_ok=True)

                if ml_dets:
                    # High-confidence ML detections bypass kill feed verification
                    high_conf = [d for d in ml_dets if d["confidence"] >= 0.9]
                    low_conf = [d for d in ml_dets if d["confidence"] < 0.9]

                    if high_conf:
                        for d in high_conf:
                            d["detection_method"] = "ml_classifier_high"
                        logger.info(f"ML high-confidence: {len(high_conf)} kills "
                                    f"(bypass verification)")

                    # Low-confidence detections still go through kill feed verification
                    verified_low = []
                    if low_conf:
                        verified_low = self._verify_kill_feed(
                            video_path, low_conf, fps, width, height
                        )
                        logger.info(f"Kill feed verification: "
                                    f"{len(verified_low)}/{len(low_conf)} low-conf confirmed")

                    all_dets = high_conf + verified_low
                    all_dets.sort(key=lambda x: x["timestamp"])

                    logger.info(f"Detection complete (ML): {len(all_dets)} kills")
                    return all_dets
            else:
                logger.warning("Audio extraction failed, falling back to NCC")

        # --- NCC fallback path ---
        logger.info("Using NCC fingerprint fallback for kill detection")
        return self._detect_kills_ncc(video_path, fps, width, height, total_frames, duration)

    def _detect_kills_ncc(self, video_path, fps, width, height, total_frames, duration):
        """Original NCC-based kill detection (fallback when ML model unavailable)."""
        audio_dets = []
        near_misses = []
        audio_path = self._extract_audio(video_path)
        if audio_path:
            try:
                audio_dets, near_misses = self._detect_kill_sounds(audio_path, fps)
                logger.info(f"Audio fingerprint: {len(audio_dets)} candidates, "
                            f"{len(near_misses)} near-misses")
            except Exception as e:
                logger.warning(f"Audio analysis failed: {e}")
            finally:
                audio_path.unlink(missing_ok=True)

        if not audio_dets:
            logger.info("No audio detections found")
            return []

        # Step 2: Verify audio candidates — is kill feed visible?
        verified = self._verify_kill_feed(video_path, audio_dets, fps, width, height)
        logger.info(f"Kill feed verification: {len(verified)}/{len(audio_dets)} confirmed")

        # Step 2b: Promote near-misses with strong kill feed evidence.
        if near_misses and verified:
            ver_times = {v["timestamp"] for v in verified}
            distant = [
                nm for nm in near_misses
                if all(abs(nm["timestamp"] - vt) > 5.0 for vt in ver_times)
            ]
            if distant:
                promoted = self._verify_kill_feed(
                    video_path, distant, fps, width, height
                )
                all_times = set(ver_times)
                final_promoted = []
                for p in sorted(promoted, key=lambda x: -x["confidence"]):
                    if all(abs(p["timestamp"] - t) > 5.0 for t in all_times):
                        final_promoted.append(p)
                        all_times.add(p["timestamp"])
                if final_promoted:
                    logger.info(f"Near-miss promotion: {len(final_promoted)} promoted")
                    verified.extend(final_promoted)
                    verified.sort(key=lambda x: x["timestamp"])

        # Merge duplicate detections using ROI comparison
        detections = self._deduplicate_detections(
            video_path, verified, fps, width, height
        )

        logger.info(f"Detection complete (NCC): {len(detections)} kills")
        return detections

    # -------------------------------------------------------------------------
    # ML kill detection
    # -------------------------------------------------------------------------

    def _detect_kills_ml(self, audio_path, fps):
        """
        ML-based kill detection using sliding window + classifier + peak finding.

        Flow: WAV → sliding windows → feature extraction → predict_proba
              → probability curve → scipy.signal.find_peaks → kill timestamps
        """
        from scipy.signal import find_peaks
        from audio_classifier import extract_features, extract_features_with_context

        with wave.open(str(audio_path), 'rb') as wf:
            rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        audio /= 32768.0

        if len(audio) == 0:
            return []

        window_samples = int(self.config["ml_window_ms"] / 1000.0 * rate)
        hop_samples = int(self.config["ml_hop_ms"] / 1000.0 * rate)

        # Sliding window extraction
        windows = []
        timestamps = []
        for start in range(0, len(audio) - window_samples, hop_samples):
            windows.append(audio[start:start + window_samples])
            timestamps.append((start + window_samples / 2) / rate)

        if not windows:
            return []

        logger.info(f"ML: {len(windows)} windows ({self.config['ml_window_ms']}ms, "
                     f"{self.config['ml_hop_ms']}ms hop)")

        # Extract features with delta-MFCC context
        features = []
        for i in range(len(windows)):
            feat = extract_features_with_context(windows, rate, i)
            features.append(feat)

        features = np.array(features)

        # Predict probabilities
        proba = self.ml_classifier.predict_proba(features)

        # Find peaks in probability curve
        peaks, properties = find_peaks(
            proba,
            height=self.config["ml_peak_height"],
            distance=self.config["ml_peak_distance"],
            prominence=self.config["ml_peak_prominence"],
        )

        detections = []
        for peak_idx in peaks:
            ts = timestamps[peak_idx]
            prob = float(proba[peak_idx])
            detections.append({
                "timestamp": round(ts, 2),
                "frame_number": int(ts * fps),
                "confidence": round(prob, 4),
                "ml_probability": round(prob, 4),
                "detection_method": "ml_classifier",
            })
            logger.info(f"  ML peak: t={ts:.2f}s, prob={prob:.4f}")

        return detections

    # -------------------------------------------------------------------------
    # Audio kill detection (NCC fallback)
    # -------------------------------------------------------------------------

    def _extract_audio(self, video_path):
        """Extract mono WAV audio from video using FFmpeg."""
        audio_path = video_path.parent / (video_path.stem + "_audio.wav")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "22050",
            "-ac", "1",
            str(audio_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=30)
            if audio_path.exists() and audio_path.stat().st_size > 1000:
                logger.info(f"Audio extracted: {audio_path.name}")
                return audio_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Audio extraction failed: {e}")

        # Cleanup on failure
        if audio_path.exists():
            audio_path.unlink(missing_ok=True)
        return None

    def _detect_kill_sounds(self, audio_path, fps):
        """
        Auto-calibrating kill sound detection via audio fingerprinting.

        Two-pass approach:
        1. Spectral flux (low threshold) → many candidates (intentionally over-detect)
        2. Extract audio snippets from top candidates, compute pairwise NCC
           to find the repeating kill sound pattern (auto-calibration)
        3. Score ALL candidates against the discovered reference → keep matches

        The kill "ding" is always the same sound effect, so it forms a cluster
        of similar-sounding events. Other sounds (gunshots, dinks, etc.) are
        varied and don't cluster.
        """
        with wave.open(str(audio_path), 'rb') as wf:
            rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        audio /= 32768.0

        if len(audio) == 0:
            return [], []

        # Bandpass filter full audio to kill sound frequency range
        filtered = self._bandpass_filter(audio, rate)

        # ---- Pass 1: Spectral flux candidates (intentionally over-detect) ----
        candidates = self._spectral_flux_candidates(audio, rate, fps)
        if not candidates:
            return [], []

        logger.info(f"Pass 1: {len(candidates)} spectral flux candidates")

        # If very few candidates, skip fingerprinting
        if len(candidates) <= 2:
            return candidates, []

        # ---- Auto-calibrate: find kill sound reference via pairwise NCC ----
        snippet_ms = self.config["fingerprint_snippet_ms"]
        snippet_samples = int(snippet_ms / 1000 * rate)
        ncc_threshold = self.config["fingerprint_ncc_threshold"]
        top_n = min(len(candidates), self.config["fingerprint_top_n"])

        # Take top N candidates by flux strength
        sorted_cands = sorted(candidates, key=lambda x: x["audio_flux"], reverse=True)
        top_cands = sorted_cands[:top_n]

        # Extract snippets from bandpass-filtered audio
        snippets = [
            self._extract_snippet(filtered, rate, c["timestamp"], snippet_samples)
            for c in top_cands
        ]

        # Compute pairwise NCC matrix
        n = len(snippets)
        ncc_matrix = np.zeros((n, n))
        for i in range(n):
            ncc_matrix[i][i] = 1.0
            for j in range(i + 1, n):
                ncc = self._compute_ncc(snippets[i], snippets[j])
                ncc_matrix[i][j] = ncc
                ncc_matrix[j][i] = ncc

        # Find the snippet with highest average similarity to all others
        avg_sim = ncc_matrix.mean(axis=1)
        ref_idx = int(np.argmax(avg_sim))
        ref_snippet = snippets[ref_idx]
        ref_avg = float(avg_sim[ref_idx])

        logger.info(f"Reference: idx={ref_idx}, t={top_cands[ref_idx]['timestamp']:.2f}s, "
                    f"avg_ncc={ref_avg:.3f}")

        # Count how many top candidates match the reference well
        cluster_count = sum(1 for i in range(n) if ncc_matrix[ref_idx][i] >= ncc_threshold)
        logger.info(f"Cluster: {cluster_count}/{top_n} similar candidates")

        # If no clear pattern found, fall back to top candidates by flux
        # (apply cooldown first so they're spread across time)
        if ref_avg < 0.10 or cluster_count < 2:
            logger.warning("No kill sound pattern found, using flux candidates")
            cooled = self._apply_cooldown(
                sorted(candidates, key=lambda x: x["timestamp"])
            )
            cooled.sort(key=lambda x: x["audio_flux"], reverse=True)
            return cooled[:8], []

        # ---- Pass 2: Score ALL candidates against single best reference ----
        final = []
        near_miss = []
        for cand in candidates:
            snippet = self._extract_snippet(filtered, rate, cand["timestamp"], snippet_samples)
            ncc = self._compute_ncc(ref_snippet, snippet)
            if ncc >= ncc_threshold:
                cand["confidence"] = round(float(ncc), 4)
                cand["ncc_score"] = round(float(ncc), 4)
                final.append(cand)
            elif ncc >= ncc_threshold * 0.6:
                cand["confidence"] = round(float(ncc), 4)
                cand["ncc_score"] = round(float(ncc), 4)
                near_miss.append(cand)

        logger.info(f"Pass 2: {len(final)}/{len(candidates)} candidates confirmed (NCC >= {ncc_threshold})")
        for f in sorted(final, key=lambda x: x["timestamp"]):
            logger.info(f"  -> t={f['timestamp']:.2f}s, NCC={f['ncc_score']:.3f}, flux={f['audio_flux']:.2f}")
        if near_miss:
            nm_str = ", ".join(f"t={c['timestamp']:.2f}s NCC={c['ncc_score']:.3f}" for c in near_miss)
            logger.info(f"  Near-miss ({len(near_miss)}): {nm_str}")

        return sorted(final, key=lambda x: x["timestamp"]), sorted(near_miss, key=lambda x: x["timestamp"])

    def _bandpass_filter(self, audio, rate):
        """FFT-based bandpass filter to kill sound frequency range."""
        freq_low = self.config["kill_sound_freq_low"]
        freq_high = self.config["kill_sound_freq_high"]

        n = len(audio)
        fft_data = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(n, 1.0 / rate)

        # Zero out frequencies outside the kill sound band
        mask = (freqs >= freq_low) & (freqs <= freq_high)
        fft_data[~mask] = 0

        return np.fft.irfft(fft_data, n)

    def _spectral_flux_candidates(self, audio, rate, fps):
        """
        Pass 1: Spectral flux with low threshold to find all possible candidates.
        Uses a lower multiplier (2.0) than final detection to intentionally over-detect.
        """
        win_size = int(self.config["audio_window_ms"] / 1000 * rate)
        hop_size = int(self.config["audio_hop_ms"] / 1000 * rate)
        freq_low = self.config["kill_sound_freq_low"]
        freq_high = self.config["kill_sound_freq_high"]

        hann = np.hanning(win_size)
        freqs = np.fft.rfftfreq(win_size, 1.0 / rate)
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
                time_list.append(start / rate)

            prev_band = band.copy()

        if not flux_list:
            return []

        flux_arr = np.array(flux_list)
        median_flux = np.median(flux_arr)
        std_flux = np.std(flux_arr)

        # Lower multiplier (2.0) for over-detection in Pass 1
        threshold = median_flux + 2.0 * std_flux

        logger.info(f"Pass 1: median={median_flux:.2f}, std={std_flux:.2f}, "
                    f"threshold={threshold:.2f}")

        # Peak detection with 0.3s minimum distance
        min_dist = int(0.3 * rate / hop_size)
        candidates = []
        last_idx = -min_dist

        for i in range(len(flux_arr)):
            if flux_arr[i] > threshold and (i - last_idx) >= min_dist:
                ts = time_list[i]
                candidates.append({
                    "timestamp": round(ts, 2),
                    "frame_number": int(ts * fps),
                    "confidence": 0.5,
                    "audio_flux": round(float(flux_arr[i]), 4),
                    "detection_method": "audio",
                })
                last_idx = i

        return candidates

    def _extract_snippet(self, filtered_audio, rate, timestamp, snippet_samples):
        """Extract an audio snippet centered on the given timestamp."""
        center = int(timestamp * rate)
        half = snippet_samples // 2
        start = max(0, center - half)
        end = min(len(filtered_audio), start + snippet_samples)
        start = max(0, end - snippet_samples)

        snippet = filtered_audio[start:end]

        # Pad with zeros if near audio boundaries
        if len(snippet) < snippet_samples:
            padded = np.zeros(snippet_samples)
            padded[:len(snippet)] = snippet
            snippet = padded

        return snippet

    def _compute_ncc(self, a, b):
        """
        Compute NCC with ±30ms shift tolerance via FFT cross-correlation.
        Time-domain NCC is discriminating in narrow bands (unlike magnitude
        spectrum NCC which is too permissive).
        """
        a = a - np.mean(a)
        b = b - np.mean(b)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        n = len(a)
        max_shift = int(0.030 * 22050)  # ±30ms at 22050Hz

        fft_size = 1
        while fft_size < 2 * n:
            fft_size *= 2

        fa = np.fft.rfft(a, fft_size)
        fb = np.fft.rfft(b, fft_size)
        xcorr = np.fft.irfft(fa * np.conj(fb), fft_size)

        # Only check shifts within ±max_shift
        valid = np.concatenate([
            xcorr[:min(max_shift + 1, len(xcorr))],
            xcorr[-min(max_shift, len(xcorr)):]
        ])

        return float(np.max(valid) / (norm_a * norm_b))

    # -------------------------------------------------------------------------
    # Kill feed verification (dark bar detection)
    # -------------------------------------------------------------------------

    def _verify_kill_feed(self, video_path, audio_dets, fps, width, height):
        """
        Verify audio candidates by detecting NEW kill feed entries in the ROI.

        Three complementary signals (any one confirms):

        1. Backward delta: dark_ratio increased vs 2s before (baseline).
           Catches kills where scene transitions from bright to dark overlay.

        2. Forward delta: dark_ratio increased AFTER the audio event (self-
           baseline at candidate time vs peak at +0.5 to +2.0s). Catches kills
           where the player moved (e.g. exited a tunnel) right before getting
           the kill.

        3. Forward text: bright text pixels appeared in a dark ROI within 1s
           after the audio event, AND the baseline had no text. Catches kills
           in dark environments where dark_ratio doesn't change but kill feed
           text becomes visible.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Cannot open video for kill feed verification")
            return audio_dets

        roi_x1 = int(width * self.config["roi_x_start_ratio"])
        roi_x2 = int(width * self.config["roi_x_end_ratio"])
        roi_y1 = int(height * self.config["roi_y_start_ratio"])
        roi_y2 = int(height * self.config["roi_y_end_ratio"])

        def analyze_roi(timestamp):
            """Get (dark_ratio, white_ratio) for the ROI at given timestamp."""
            frame_num = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                return 0.0, 0.0
            gray = cv2.cvtColor(
                frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY
            )
            dark = np.count_nonzero(gray < 60) / gray.size
            white = np.count_nonzero(gray > 180) / gray.size
            return dark, white

        verified = []

        for det in audio_dets:
            ts = det["timestamp"]

            # Baseline: 2s before candidate
            base_dark, base_white = analyze_roi(max(0, ts - 2.0))

            # Metrics at candidate time
            self_dark, self_white = analyze_roi(ts)

            # Forward metrics: check 0.5s to 2.0s after candidate
            fwd_darks = []
            fwd_whites = []
            for off in [0.5, 1.0, 1.5, 2.0]:
                d, w = analyze_roi(ts + off)
                fwd_darks.append(d)
                fwd_whites.append(w)

            peak_dark = max(self_dark, *fwd_darks)
            # For text, check a tighter window (0.5-1.0s) to avoid other kills
            peak_white_tight = max(fwd_whites[0], fwd_whites[1])  # +0.5, +1.0
            peak_dark_fwd = max(fwd_darks)

            # --- Signal 1: Backward delta ---
            # Dark_ratio increased vs 2s-before baseline (bright → dark).
            # Exclude peak_dark > 0.96 (pitch-black tunnels/rooms, not kill feed).
            delta_back = peak_dark - base_dark
            thresh_back = 0.05 + base_dark * 0.10
            sig1 = (
                peak_dark < 0.96
                and (
                    (base_dark < 0.20 and peak_dark > 0.25)
                    or delta_back > thresh_back
                )
            )

            # --- Signal 2: Forward delta ---
            # Dark_ratio increased AFTER audio event (self-baseline).
            # Only fires if scene was reasonably bright at audio time and
            # became dark enough (kill feed overlay appeared).
            delta_fwd = peak_dark_fwd - self_dark
            thresh_fwd = 0.05 + self_dark * 0.10
            sig2 = (
                self_dark < 0.50
                and peak_dark_fwd > 0.25
                and peak_dark_fwd < 0.96   # exclude pitch-black scene transitions
                and delta_fwd > thresh_fwd
            )

            # --- Signal 3: Forward text in dark scene ---
            # New bright text appeared within 1s after audio event.
            # Only fires if baseline was dark with NO text (kill feed was empty).
            # Tighter thresholds avoid noise from dark scene pixel fluctuations.
            text_delta = peak_white_tight - self_white
            sig3 = (
                base_white < 0.001       # baseline had no kill feed text
                and text_delta > 0.002   # meaningful new text appeared
                and peak_white_tight > 0.003  # enough text to be kill feed
                and peak_dark > 0.25     # ROI has dark kill feed bars
                and delta_fwd >= 0       # kill feed should not make ROI brighter
            )

            confirmed = sig1 or sig2 or sig3
            methods = []
            if sig1:
                methods.append("back_delta")
            if sig2:
                methods.append("fwd_delta")
            if sig3:
                methods.append("fwd_text")

            if confirmed:
                det["dark_ratio"] = round(peak_dark, 4)
                det["dark_ratio_delta"] = round(delta_back, 4)
                det["fwd_white_05"] = round(fwd_whites[0], 6)
                det["detection_method"] = "audio+killfeed"
                verified.append(det)
                logger.info(f"  Confirmed: t={ts:.2f}s, dark={peak_dark:.3f}, "
                            f"d_back={delta_back:+.3f}, d_fwd={delta_fwd:+.3f}, "
                            f"fw05={fwd_whites[0]:.4f}, "
                            f"method={'+'.join(methods)}")
            else:
                logger.info(f"  Rejected: t={ts:.2f}s, dark={peak_dark:.3f}, "
                            f"d_back={delta_back:+.3f}, d_fwd={delta_fwd:+.3f}, "
                            f"base_w={base_white:.4f}")

        cap.release()
        return verified

    # -------------------------------------------------------------------------
    # Common
    # -------------------------------------------------------------------------

    @staticmethod
    def _detect_platform(url):
        """Detect video platform from URL."""
        url_lower = url.lower()
        if "medal.tv" in url_lower:
            return "medal.tv"
        if "youtube.com" in url_lower or "youtu.be" in url_lower:
            return "youtube"
        if "twitch.tv" in url_lower:
            return "twitch"
        if "kick.com" in url_lower:
            return "kick"
        if "twitter.com" in url_lower or "x.com" in url_lower:
            return "twitter"
        return "unknown"

    def _apply_cooldown(self, detections, cooldown=None):
        """Filter detections that are too close together, keeping the highest confidence one."""
        if not detections:
            return []

        if cooldown is None:
            cooldown = self.config["cooldown_seconds"]
        filtered = [detections[0]]

        for det in detections[1:]:
            if det["timestamp"] - filtered[-1]["timestamp"] >= cooldown:
                filtered.append(det)
            elif det["confidence"] > filtered[-1]["confidence"]:
                filtered[-1] = det

        return filtered

    def _deduplicate_detections(self, video_path, detections, fps, width, height):
        """Merge echo detections using kill feed ROI frame comparison.

        Three zones based on time gap between consecutive detections:
        - gap >= cooldown: always separate kills
        - gap < min_gap (snippet length): always same sound event
        - ambiguous range: compare kill feed ROI dark-pixel content to decide

        For the ambiguous range, reads two ROI frames (+0.5s offset) and
        computes the mean pixel difference within the combined dark mask
        (kill feed overlay area).  A large diff means a new kill feed entry
        appeared → separate kill.  A small diff means the same entry is
        still visible → echo / duplicate.
        """
        if len(detections) < 2:
            return detections

        cooldown = self.config["cooldown_seconds"]
        min_gap = self.config["fingerprint_snippet_ms"] / 1000.0

        # Quick check: any pairs in the ambiguous range?
        needs_video = any(
            min_gap <= detections[i + 1]["timestamp"] - detections[i]["timestamp"] < cooldown
            for i in range(len(detections) - 1)
        )

        if not needs_video:
            return self._apply_cooldown(detections)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Cannot open video for deduplication, using simple cooldown")
            return self._apply_cooldown(detections)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        roi_x1 = int(width * self.config["roi_x_start_ratio"])
        roi_x2 = int(width * self.config["roi_x_end_ratio"])
        roi_y1 = int(height * self.config["roi_y_start_ratio"])
        roi_y2 = int(height * self.config["roi_y_end_ratio"])

        def read_roi_gray(ts):
            fn = min(int((ts + 0.5) * fps), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                return None
            return cv2.cvtColor(
                frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY
            )

        def kill_feed_changed(roi_a, roi_b):
            if roi_a is None or roi_b is None:
                return False
            # Compare only within dark regions (kill feed overlay bars).
            dark_mask = (roi_a < 60) | (roi_b < 60)
            if np.sum(dark_mask) < roi_a.size * 0.01:
                return False
            diff = np.abs(roi_a.astype(np.int16) - roi_b.astype(np.int16))
            score = np.mean(diff[dark_mask]) / 255.0
            return score > 0.20

        filtered = [detections[0]]
        prev_roi = read_roi_gray(detections[0]["timestamp"])

        for det in detections[1:]:
            gap = det["timestamp"] - filtered[-1]["timestamp"]

            if gap >= cooldown:
                filtered.append(det)
                prev_roi = read_roi_gray(det["timestamp"])
            elif gap < min_gap:
                if det["confidence"] > filtered[-1]["confidence"]:
                    filtered[-1] = det
                    prev_roi = read_roi_gray(det["timestamp"])
            else:
                curr_roi = read_roi_gray(det["timestamp"])
                if kill_feed_changed(prev_roi, curr_roi):
                    logger.info(
                        f"  Kill feed split: {filtered[-1]['timestamp']:.2f}s "
                        f"-> {det['timestamp']:.2f}s"
                    )
                    filtered.append(det)
                    prev_roi = curr_roi
                elif det["confidence"] > filtered[-1]["confidence"]:
                    filtered[-1] = det
                    prev_roi = read_roi_gray(det["timestamp"])

        cap.release()
        return filtered

    # =========================================================================
    # Phase 3: Kill Frame Extraction
    # =========================================================================

    def save_kill_frames(self, video_path, detections, session_id, progress_callback=None):
        """
        Extract frame sequence for each kill:
          - 20 frames before (2.0s, sampled every 0.1s)
          - kill frame
          - 5 frames after (0.5s, sampled every 0.1s)

        Folder structure:
          clips/<session>/kill_001/
            before_20.jpg ... before_01.jpg
            kill.jpg
            after_01.jpg ... after_05.jpg

        Returns list of kill directory paths.
        """
        if progress_callback:
            progress_callback("cutting", "Saving kill frames...")

        video_path = Path(video_path)
        session_dir = self.base_dir / self.config["clips_dir"] / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = int(round(fps * 0.1))  # 1 frame every 0.1s (6 frames at 60fps)
        kill_dirs = []

        # Calculate boundaries so kills don't overlap each other's frames
        kill_timestamps = [det["timestamp"] for det in detections]

        for i, det in enumerate(detections):
            kill_frame_num = int(det["timestamp"] * fps)
            kill_dir = session_dir / f"kill_{i + 1:03d}"
            kill_dir.mkdir(parents=True, exist_ok=True)

            # Clamp before/after to not cross into adjacent kills
            prev_kill_frame = int(kill_timestamps[i - 1] * fps) if i > 0 else -1
            next_kill_frame = int(kill_timestamps[i + 1] * fps) if i < len(detections) - 1 else total_frames

            # Build frame list: up to 20 before, kill, up to 5 after
            frames_to_save = []
            for b in range(20, 0, -1):
                fn = kill_frame_num - b * sample_interval
                if fn <= prev_kill_frame:
                    continue  # Would cross into previous kill
                frames_to_save.append((fn, f"before_{b:02d}.jpg"))
            frames_to_save.append((kill_frame_num, "kill.jpg"))
            for a in range(1, 6):
                fn = kill_frame_num + a * sample_interval
                if fn >= next_kill_frame:
                    break  # Would cross into next kill
                frames_to_save.append((fn, f"after_{a:02d}.jpg"))

            saved = 0
            for fn, name in frames_to_save:
                if fn < 0 or fn >= total_frames:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                ret, frame = cap.read()
                if not ret:
                    continue

                out_path = str(kill_dir / name)
                ok = cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    # Fallback for paths with non-ASCII characters (Windows)
                    success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if success:
                        Path(out_path).write_bytes(buf.tobytes())
                        ok = True
                if ok:
                    saved += 1

            kill_dirs.append(kill_dir)
            det["frame_dir"] = str(kill_dir.relative_to(self.base_dir))
            det["frame_file"] = str((kill_dir / "kill.jpg").relative_to(self.base_dir))
            logger.info(f"Kill {i}: {saved} frames saved (t={det['timestamp']:.2f}s)")

        cap.release()
        logger.info(f"Total {len(kill_dirs)} kills, {len(kill_dirs) * 10} frames")
        return kill_dirs

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
            "pipeline_version": "4.0.0",
            "source": {
                "url": url,
                "platform": self._detect_platform(url),
                "downloaded_file": str(video_path.relative_to(self.base_dir)),
                "duration_seconds": round(duration, 2),
                "resolution": f"{width}x{height}",
                "fps": round(fps, 1),
                "file_size_bytes": file_size,
            },
            "detection_config": {
                "method": "audio_fingerprint+killfeed_verification",
                "roi": {
                    "x_start_ratio": self.config["roi_x_start_ratio"],
                    "x_end_ratio": self.config["roi_x_end_ratio"],
                    "y_start_ratio": self.config["roi_y_start_ratio"],
                    "y_end_ratio": self.config["roi_y_end_ratio"],
                },
                "cooldown_seconds": self.config["cooldown_seconds"],
                "kill_sound_freq_range": [
                    self.config["kill_sound_freq_low"],
                    self.config["kill_sound_freq_high"],
                ],
                "fingerprint_ncc_threshold": self.config["fingerprint_ncc_threshold"],
            },
            "detections": [],
            "annotations": [],
            "summary": {
                "total_kills_detected": len(detections),
                "total_frames_saved": len(clip_paths),
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
                "ncc_score": det.get("ncc_score"),
                "dark_ratio": det.get("dark_ratio"),
                "frame_dir": det.get("frame_dir", ""),
            })

            metadata["annotations"].append({
                "frame_path": det.get("frame_file", ""),
                "label": "cs2_kill_moment",
                "confidence": det["confidence"],
                "source_game": "counter_strike_2",
            })

            total_conf += det["confidence"]

        if detections:
            metadata["summary"]["average_confidence"] = round(total_conf / len(detections), 4)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata saved: {meta_path}")
        return meta_path

    # =========================================================================
    # Auto-retrain
    # =========================================================================

    def _should_retrain(self):
        """Check if there are new sessions since last training."""
        meta_path = self.base_dir / "models" / "training_meta.json"
        if not meta_path.exists():
            return True

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                training_meta = json.load(f)
            trained_sessions = set(training_meta.get("sessions_used", []))
        except (json.JSONDecodeError, OSError):
            return True

        # Check current metadata sessions
        meta_dir = self.base_dir / "metadata"
        current_sessions = set()
        for mf in meta_dir.glob("*.json"):
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("detections"):
                    current_sessions.add(meta.get("session_id", mf.stem))
            except (json.JSONDecodeError, OSError):
                continue

        new_sessions = current_sessions - trained_sessions
        if new_sessions:
            logger.info(f"New sessions since last training: {len(new_sessions)}")
            return True
        return False

    def _auto_retrain(self):
        """Retrain the ML model in a background thread if new data is available."""
        if not self._should_retrain():
            logger.info("Model is up to date, skipping retrain")
            return

        def retrain_worker():
            try:
                logger.info("Auto-retrain started in background...")
                from train_classifier import bootstrap_labels, extract_training_data, train_model
                import pickle

                labels = bootstrap_labels(self.base_dir)
                if not labels:
                    return

                X, y = extract_training_data(labels, self.base_dir)
                if len(X) == 0:
                    return

                model, cv_scores = train_model(X, y)

                # Save model
                models_dir = self.base_dir / "models"
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
                    "sessions_used": [s["session_id"] for s in labels],
                }

                with open(model_path, "wb") as f:
                    pickle.dump(model_data, f)

                # Save training metadata
                training_meta = {k: v for k, v in model_data.items() if k != "model"}
                with open(models_dir / "training_meta.json", "w", encoding="utf-8") as f:
                    json.dump(training_meta, f, indent=2, ensure_ascii=False)

                # Reload model for next run
                self._load_ml_model()
                logger.info(f"Auto-retrain complete! F1={model_data['cv_f1_mean']:.4f}, "
                            f"{model_data['n_samples']} samples")

            except Exception as e:
                logger.warning(f"Auto-retrain failed: {e}")

        thread = threading.Thread(target=retrain_worker, daemon=True)
        thread.start()

    # =========================================================================
    # Orchestrator
    # =========================================================================

    def run(self, url, progress_callback=None):
        """
        Full pipeline: download -> detect -> save frames -> metadata.
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
            logger.info("[1/4] Downloading video...")
            video_path = self.download_video(url, progress_callback)
            result["video_path"] = str(video_path)

            # Phase 2: Detect
            logger.info("[2/4] Detecting kill moments...")
            detections = self.detect_kills(video_path, progress_callback)
            result["kills_detected"] = len(detections)

            if not detections:
                logger.warning("No kills detected.")
                if progress_callback:
                    progress_callback("done", "No kills detected.")
                result["status"] = "no_kills"
            else:
                # Phase 3: Save kill frames
                logger.info(f"[3/4] Saving frames for {len(detections)} kills...")
                kill_dirs = self.save_kill_frames(video_path, detections, session_id, progress_callback)
                result["clips_created"] = len(kill_dirs)
                result["clip_paths"] = [
                    str((d / "kill.jpg").relative_to(self.base_dir)).replace("\\", "/")
                    for d in kill_dirs
                ]

                # Phase 4: Metadata
                logger.info("[4/4] Generating metadata...")
                if progress_callback:
                    progress_callback("metadata", "Generating metadata...")
                meta_path = self.generate_metadata(
                    video_path, url, detections, kill_dirs, session_id
                )
                result["metadata_path"] = str(meta_path.relative_to(self.base_dir)).replace("\\", "/")
                result["status"] = "completed"

            elapsed = time.time() - start_time
            result["processing_time"] = round(elapsed, 2)

            logger.info("=" * 60)
            logger.info(f"Pipeline complete! ({elapsed:.1f}s)")
            logger.info(f"  Kills detected: {result['kills_detected']}")
            logger.info(f"  Frames saved: {result['clips_created']}")
            logger.info("=" * 60)

            if progress_callback:
                progress_callback("done", f"Done! {result['kills_detected']} kills, "
                                          f"{result['clips_created']} frames.")

            # Auto-retrain with new data
            if result["kills_detected"] > 0:
                self._auto_retrain()

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
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
