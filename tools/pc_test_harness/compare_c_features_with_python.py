"""Check the ESP32's C feature code (fan_features.h) and logistic regression against Python, on a PC.

Build the harness once (any C++ compiler; MinGW shown):
    g++ -O2 -static -I firmware/fault_detector tools/pc_test_harness/test_features.cpp -o test_features.exe
Then, from the repository root:
    python tools/pc_test_harness/compare_c_features_with_python.py test_features.exe

Feeds every recorded 1-s window to the C code and compares its 9 features and fault probability with
the Python reference (same math as the shared features.py).
"""

import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEADER = os.path.join(ROOT, "firmware", "fault_detector", "logreg_model.h")
FS = 1000.0
FREQ_BANDS = [(5, 25), (25, 75), (75, 200), (200, 500)]
NAMES = ["rms", "peak", "std", "kurtosis", "crest", "band_5_25", "band_25_75", "band_75_200", "band_200_500"]


def extract_window_features(win_ms2):
    rms = float(np.sqrt(np.mean(win_ms2 ** 2)))
    peak = float(np.max(np.abs(win_ms2)))
    win_ac = win_ms2 - np.mean(win_ms2)
    fft_mag = np.abs(np.fft.rfft(win_ac)) * 2.0 / len(win_ac)
    freqs = np.fft.rfftfreq(len(win_ac), d=1.0 / FS)
    bands = [float(np.mean(fft_mag[(freqs >= lo) & (freqs < hi)] ** 2)) for lo, hi in FREQ_BANDS]
    return [rms, peak, float(np.std(win_ms2)), float(kurtosis(win_ms2, fisher=True)),
            peak / rms if rms > 0 else 0.0] + bands


def main():
    exe = sys.argv[1]
    text = open(HEADER).read()
    arr = lambda n: np.array([float(v.strip().rstrip("f")) for v in
                              re.search(n + r"\[N_FEATURES\] = \{(.*?)\}", text, re.S).group(1).split(",")])
    mean, scale, coef = arr("LR_MEAN"), arr("LR_SCALE"), arr("LR_COEF")
    b = float(re.search(r"LR_INTERCEPT = ([^f;]+)f", text).group(1))

    raw, ref = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "*.csv"))):
        cpg = json.load(open(f[:-4] + ".json")).get("counts_per_g", 16384)
        if cpg != 4096:
            continue
        c = pd.read_csv(f).dropna()[["ax", "ay", "az"]].to_numpy()
        mag = np.sqrt(((c * 9.80665 / 4096.0) ** 2).sum(1))
        for w in range(len(c) // 1000):
            raw.append(c[w * 1000:(w + 1) * 1000].astype(np.int16))
            ref.append(extract_window_features(mag[w * 1000:(w + 1) * 1000]))
    ref = np.array(ref)
    p_ref = 1 / (1 + np.exp(-(b + ((ref - mean) / scale) @ coef)))
    out = subprocess.run([exe], input=np.concatenate(raw).tobytes(), capture_output=True).stdout.decode()
    esp = np.array([[float(v) for v in line.split(",")] for line in out.strip().splitlines()])
    print("windows:", len(ref))
    print("largest feature error (standard deviations):",
          dict(zip(NAMES, (np.abs(esp[:, :9] - ref) / scale).max(0).round(5))))
    print("largest probability difference:", np.abs(esp[:, 9] - p_ref).max())
    print("prediction agreement:", ((esp[:, 9] >= 0.5) == (p_ref >= 0.5)).mean() * 100, "%")


if __name__ == "__main__":
    main()
