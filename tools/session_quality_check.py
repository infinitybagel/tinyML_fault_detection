"""Quick quality check of recorded sessions: fan speed, vibration per axis, gravity, clipping, FFT bands.

    python tools/session_quality_check.py            (checks data/)
    python tools/session_quality_check.py path/to/folder

Used after each recording batch to catch problems early (clipping, bumps, a shifted sensor, RPM drift).
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data")
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        meta = json.load(open(f[:-4] + ".json"))
        cpg = float(meta.get("counts_per_g", 16384))
        df = pd.read_csv(f)
        raw = df[["ax", "ay", "az"]].to_numpy()
        a = raw / cpg
        t = ((df.t_us - df.t_us.iloc[0]) & 0xFFFFFFFF).to_numpy() / 1e6
        ac = a - a.mean(0)
        mag = np.linalg.norm(a, axis=1)
        n = len(mag) // 1000
        w = mag[:n * 1000].reshape(n, 1000)
        w = w - w.mean(1, keepdims=True)
        amp = (np.abs(np.fft.rfft(w, axis=1)) * 2 / 1000).mean(0)
        spin = int(round(meta["mean_fan_rpm"] / 60)) if meta.get("mean_fan_rpm") else 28
        band = lambda lo, hi: round(float(np.sqrt((amp[lo:hi] ** 2).sum()) * 1000), 1)
        per_s = pd.Series(np.linalg.norm(ac, axis=1)).groupby(t.astype(int)).max()
        rows.append(dict(session=os.path.basename(f)[:-4], rpm=meta.get("mean_fan_rpm"),
                         std_xyz_mg=(ac.std(0) * 1000).round(1).tolist(), gravity_g=a.mean(0).round(3).tolist(),
                         clipped=int((np.abs(raw) >= 32767).sum()), spin_mg=round(float(amp[spin - 2:spin + 3].max() * 1000), 1),
                         b75_200_mg=band(75, 200), b200_500_mg=band(200, 500),
                         bumps_at_s=per_s[per_s > per_s.median() * 3].index.tolist()))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
