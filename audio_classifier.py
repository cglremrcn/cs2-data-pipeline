"""
ML-based kill sound classifier for CS2.

Extracts a 35-dimensional feature vector from audio windows and uses a
GradientBoostingClassifier to predict kill sound probability.

Feature vector:
  0-12   13 MFCC coefficients (timbral character)
  13-25  13 delta-MFCC (onset dynamics)
  26     Spectral centroid
  27     Spectral bandwidth
  28     Spectral rolloff (85%)
  29     Spectral flatness
  30     Zero crossing rate
  31     RMS energy
  32     Band energy ratio (1800-4500 Hz / total)
  33     Spectral flux
  34     Peak frequency
"""

import pickle
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

N_MFCC = 13
N_MELS = 40
N_FEATURES = 35


def _mel_filterbank(n_mels, n_fft_bins, rate, f_min=0.0, f_max=None):
    """Build a mel-scale triangular filterbank matrix (n_mels x n_fft_bins)."""
    if f_max is None:
        f_max = rate / 2.0

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft_bins * 2) * hz_points / rate).astype(int)

    filters = np.zeros((n_mels, n_fft_bins))
    for m in range(n_mels):
        left = bin_points[m]
        center = bin_points[m + 1]
        right = bin_points[m + 2]

        for k in range(left, center):
            if center != left:
                filters[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right != center:
                filters[m, k] = (right - k) / (right - center)

    return filters


def extract_features(window, rate):
    """
    Extract a 35-dimensional feature vector from an audio window.

    Parameters
    ----------
    window : np.ndarray
        Audio samples (float64, mono).
    rate : int
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        35-dimensional feature vector.
    """
    n = len(window)
    if n == 0:
        return np.zeros(N_FEATURES)

    # Apply Hann window
    windowed = window * np.hanning(n)

    # FFT
    n_fft = max(512, 1 << (n - 1).bit_length())  # next power of 2, min 512
    spectrum = np.abs(np.fft.rfft(windowed, n_fft))
    power_spectrum = spectrum ** 2
    n_fft_bins = len(spectrum)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / rate)

    # --- MFCCs (0-12) ---
    mel_fb = _mel_filterbank(N_MELS, n_fft_bins, rate)
    mel_energies = mel_fb @ power_spectrum
    mel_energies = np.maximum(mel_energies, 1e-10)
    log_mel = np.log(mel_energies)

    # DCT-II to get MFCCs
    n_mel = len(log_mel)
    dct_matrix = np.zeros((N_MFCC, n_mel))
    for k in range(N_MFCC):
        for i in range(n_mel):
            dct_matrix[k, i] = np.cos(np.pi * k * (2 * i + 1) / (2 * n_mel))
    mfcc = dct_matrix @ log_mel

    # --- Delta MFCCs (13-25) ---
    # For a single window we approximate delta as zero; the caller (sliding
    # window loop) should replace these with real deltas when available.
    delta_mfcc = np.zeros(N_MFCC)

    # --- Spectral features (26-34) ---
    total_energy = np.sum(power_spectrum) + 1e-10

    # Spectral centroid (26)
    spectral_centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)

    # Spectral bandwidth (27)
    spectral_bandwidth = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * spectrum) / (np.sum(spectrum) + 1e-10)
    )

    # Spectral rolloff at 85% (28)
    cumsum = np.cumsum(spectrum)
    rolloff_thresh = 0.85 * cumsum[-1]
    rolloff_idx = np.searchsorted(cumsum, rolloff_thresh)
    spectral_rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    # Spectral flatness (29)
    geo_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
    arith_mean = np.mean(spectrum) + 1e-10
    spectral_flatness = geo_mean / arith_mean

    # Zero crossing rate (30)
    zcr = np.sum(np.abs(np.diff(np.sign(window)))) / (2.0 * n)

    # RMS energy (31)
    rms = np.sqrt(np.mean(window ** 2))

    # Band energy ratio: 1800-4500 Hz (32)
    band_mask = (freqs >= 1800) & (freqs <= 4500)
    band_energy_ratio = np.sum(power_spectrum[band_mask]) / total_energy

    # Spectral flux (33) — for single window, use total energy as proxy
    spectral_flux = np.sum(spectrum) / n_fft_bins

    # Peak frequency (34)
    peak_freq = freqs[np.argmax(spectrum)]

    features = np.concatenate([
        mfcc,                           # 0-12
        delta_mfcc,                     # 13-25
        [spectral_centroid],            # 26
        [spectral_bandwidth],           # 27
        [spectral_rolloff],             # 28
        [spectral_flatness],            # 29
        [zcr],                          # 30
        [rms],                          # 31
        [band_energy_ratio],            # 32
        [spectral_flux],                # 33
        [peak_freq],                    # 34
    ])

    return features


def extract_features_with_context(windows, rate, center_idx):
    """
    Extract features for window at center_idx, including real delta-MFCCs
    computed from neighboring windows.

    Parameters
    ----------
    windows : list of np.ndarray
        List of audio windows.
    rate : int
        Sample rate.
    center_idx : int
        Index of the center window.

    Returns
    -------
    np.ndarray
        35-dimensional feature vector with real delta-MFCCs.
    """
    feat = extract_features(windows[center_idx], rate)

    # Compute delta-MFCCs from neighbors
    if len(windows) >= 3 and 0 < center_idx < len(windows) - 1:
        prev_feat = extract_features(windows[center_idx - 1], rate)
        next_feat = extract_features(windows[center_idx + 1], rate)
        feat[N_MFCC:2 * N_MFCC] = (next_feat[:N_MFCC] - prev_feat[:N_MFCC]) / 2.0
    elif center_idx > 0:
        prev_feat = extract_features(windows[center_idx - 1], rate)
        feat[N_MFCC:2 * N_MFCC] = feat[:N_MFCC] - prev_feat[:N_MFCC]
    elif center_idx < len(windows) - 1:
        next_feat = extract_features(windows[center_idx + 1], rate)
        feat[N_MFCC:2 * N_MFCC] = next_feat[:N_MFCC] - feat[:N_MFCC]

    return feat


def augment_sample(window, rate):
    """
    Generate augmented copies of an audio window.

    Returns list of (augmented_window, description) tuples.
    Total: 16 augmented copies.
    """
    augmented = []

    # Time shift: ±15ms, ±30ms (4 variants)
    for shift_ms in [-30, -15, 15, 30]:
        shift_samples = int(shift_ms / 1000.0 * rate)
        shifted = np.roll(window, shift_samples)
        if shift_samples > 0:
            shifted[:shift_samples] = 0
        elif shift_samples < 0:
            shifted[shift_samples:] = 0
        augmented.append((shifted, f"shift_{shift_ms}ms"))

    # Volume: 0.7x, 0.85x, 1.15x (3 variants)
    for gain in [0.7, 0.85, 1.15]:
        augmented.append((window * gain, f"vol_{gain}"))

    # Noise: SNR 10/15/20 dB (3 variants)
    rms = np.sqrt(np.mean(window ** 2)) + 1e-10
    for snr_db in [10, 15, 20]:
        noise_rms = rms / (10 ** (snr_db / 20.0))
        noise = np.random.randn(len(window)) * noise_rms
        augmented.append((window + noise, f"noise_snr{snr_db}"))

    # Pitch shift via resampling: ±2% (2 variants)
    for factor in [0.98, 1.02]:
        n_out = int(len(window) / factor)
        indices = np.linspace(0, len(window) - 1, n_out)
        resampled = np.interp(indices, np.arange(len(window)), window)
        # Pad or trim to original length
        result = np.zeros_like(window)
        copy_len = min(len(resampled), len(window))
        result[:copy_len] = resampled[:copy_len]
        augmented.append((result, f"pitch_{factor}"))

    # Bandpass variation: different frequency bands (2 variants)
    n = len(window)
    fft_data = np.fft.rfft(window)
    freqs = np.fft.rfftfreq(n, 1.0 / rate)

    for (f_lo, f_hi, desc) in [(1500, 5000, "wide"), (2000, 4000, "narrow")]:
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        filtered = fft_data.copy()
        # Attenuate (not zero) outside band for more natural augmentation
        filtered[~mask] *= 0.1
        augmented.append((np.fft.irfft(filtered, n), f"band_{desc}"))

    # Original with slight random variation (2 variants)
    for i in range(2):
        jitter = 1.0 + np.random.randn(len(window)) * 0.005
        augmented.append((window * jitter, f"jitter_{i}"))

    return augmented


class KillSoundClassifier:
    """Wrapper around the trained kill sound classifier model."""

    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        if model_path:
            self.load(model_path)

    def load(self, model_path):
        """Load a trained model from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return False

        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.model_path = str(model_path)
            logger.info(f"ML model loaded: {model_path.name} "
                        f"(trained {data.get('trained_at', 'unknown')})")
            return True
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            self.model = None
            return False

    @property
    def is_ready(self):
        """Check if the model is loaded and ready for inference."""
        return self.model is not None

    def predict_proba(self, features):
        """
        Predict kill probability for one or more feature vectors.

        Parameters
        ----------
        features : np.ndarray
            Shape (n_features,) for single or (n_samples, n_features) for batch.

        Returns
        -------
        np.ndarray
            Kill probability for each sample.
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # predict_proba returns [[p_neg, p_pos], ...]
        proba = self.model.predict_proba(features)
        return proba[:, 1]  # positive class probability
