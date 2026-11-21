"""Shared feature extraction. DO NOT change this per model: every model must use exactly these features.

Same math as the team's features.py, with one fix: the conversion from raw counts to m/s^2 is read
from each session's .json (counts_per_g), so the +/-8 g recordings (4096 counts/g) are scaled correctly.

Per 1-second window (1000 samples at 1 kHz, non-overlapping) of the acceleration magnitude
sqrt(ax^2 + ay^2 + az^2) in m/s^2:
    rms, peak, std, kurtosis (excess), crest_factor (peak/rms),
    band_5_25hz, band_25_75hz, band_75_200hz, band_200_500hz
    (band = mean over FFT bins in [lo, hi) Hz of (2*|FFT(x - mean)|/N)^2)
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(ROOT), "data")     # the repository's data/ folder
CACHE = os.path.join(ROOT, "results", "features_cache.csv")

FS = 1000.0
WINDOW_SIZE = 1000
FREQ_BANDS = [(5, 25), (25, 75), (75, 200), (200, 500)]
FEATURES = ["rms", "peak", "std", "kurtosis", "crest_factor",
            "band_5_25hz", "band_25_75hz", "band_75_200hz", "band_200_500hz"]
TRAIN_LABELS = ["normal", "imbalance", "loose_mount", "blocked_airflow"]
HELDOUT = "heldout"


def extract_window_features(win_ms2):
    """9 features for one window of acceleration magnitude in m/s^2 (identical to features.py)."""
    rms = float(np.sqrt(np.mean(win_ms2 ** 2)))
    peak = float(np.max(np.abs(win_ms2)))
    std = float(np.std(win_ms2))
    kurt = float(kurtosis(win_ms2, fisher=True))
    crest = float(peak / rms) if rms > 0 else 0.0
    win_ac = win_ms2 - np.mean(win_ms2)
    fft_mag = np.abs(np.fft.rfft(win_ac)) * 2.0 / len(win_ac)
    freqs = np.fft.rfftfreq(len(win_ac), d=1.0 / FS)
    bands = [float(np.mean(fft_mag[(freqs >= lo) & (freqs < hi)] ** 2)) for lo, hi in FREQ_BANDS]
    return [rms, peak, std, kurt, crest] + bands


def session_magnitude(csv_path):
    """Acceleration magnitude (m/s^2) of one recorded session, scaled with its own counts_per_g."""
    meta = json.load(open(csv_path[:-4] + ".json"))
    counts_per_g = meta.get("counts_per_g", 16384)
    c = pd.read_csv(csv_path).dropna()[["ax", "ay", "az"]].to_numpy(dtype=float)
    return np.sqrt(((c * 9.80665 / counts_per_g) ** 2).sum(axis=1)), meta


def build_features(data_dir=DATA_DIR):
    rows = []
    for label in TRAIN_LABELS + [HELDOUT]:
        files = sorted(glob.glob(os.path.join(data_dir, f"{label}_*.csv")))
        if not files:
            raise FileNotFoundError(f"No {label}_*.csv files in {data_dir}")
        for idx, f in enumerate(files):                     # idx = session number in recording order
            mag, meta = session_magnitude(f)
            for w in range(len(mag) // WINDOW_SIZE):
                feats = extract_window_features(mag[w * WINDOW_SIZE:(w + 1) * WINDOW_SIZE])
                rows.append(dict(zip(FEATURES, feats), label=label, session=os.path.basename(f),
                                 session_idx=idx, window=w, rpm=meta.get("mean_fan_rpm")))
    return pd.DataFrame(rows)


def load_features(refresh=False):
    """Per-window feature table (cached in results/features_cache.csv).
    Columns: the 9 FEATURES, label, session, session_idx (0-4), window, rpm."""
    if os.path.exists(CACHE) and not refresh:
        newest_data = max(os.path.getmtime(p) for p in glob.glob(os.path.join(DATA_DIR, "*.csv")))
        if os.path.getmtime(CACHE) > newest_data:
            return pd.read_csv(CACHE)
    df = build_features()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df
