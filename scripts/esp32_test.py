"""Test the logistic regression running ON the ESP32 (sketch: firmware/fault_detector/fault_detector.ino).

Run from the experiment folder:  python scripts/esp32_test.py ...

Two modes:

  verify  Sends recorded windows from data/ to the ESP32. The board computes the features and the
          prediction in C; this script computes them in Python (same math as features.py) and
          compares. This is the "accuracy on the ESP32 vs. in Python" check from the outline.
          It also reports the on-board feature and inference times.

            python esp32_test.py verify --port COM3
            python esp32_test.py verify --port COM3 --windows 59        (every window of each file)

          By default it uses the shared test split (last session of each condition) plus every
          held-out session, 20 windows from each.

  live    Classifies the fan in real time for a set time and summarizes the result.

            python esp32_test.py live --port COM3 --seconds 30 --expect normal
            python esp32_test.py live --port COM3 --seconds 30 --expect fault

          Add --save CONDITION (and optionally --notes "...") to keep the run in results/live_runs/:

            python esp32_test.py live --port COM3 --seconds 30 --expect normal --save normal

Close the Arduino Serial Monitor first, or the port will be busy.
"""

import argparse
import glob
import json
import os
import re
import struct
import time
from datetime import datetime

import numpy as np
import pandas as pd
import serial
from scipy.stats import kurtosis

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # the experiment folder
HEADER = os.path.join(ROOT, "firmware", "fault_detector", "logreg_model.h")
FEATURES = ["rms", "peak", "std", "kurtosis", "crest_factor",
            "band_5_25hz", "band_25_75hz", "band_75_200hz", "band_200_500hz"]
TRAIN_LABELS = ["normal", "imbalance", "loose_mount", "blocked_airflow"]
WINDOW = 1000


# ---- Python reference: same math as features.py ---------------------------------------------
FS = 1000.0
FREQ_BANDS = [(5, 25), (25, 75), (75, 200), (200, 500)]


def extract_window_features(win_ms2):
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


def load_model(path=HEADER):
    text = open(path).read()

    def arr(name):
        body = re.search(name + r"\[N_FEATURES\] = \{(.*?)\}", text, re.S).group(1)
        return np.array([float(v.strip().rstrip("f")) for v in body.split(",")])

    intercept = float(re.search(r"LR_INTERCEPT = ([^f;]+)f", text).group(1))
    return arr("LR_MEAN"), arr("LR_SCALE"), arr("LR_COEF"), intercept


def predict_proba(x, model):
    mean, scale, coef, intercept = model
    return 1.0 / (1.0 + np.exp(-(intercept + ((np.asarray(x) - mean) / scale) @ coef)))


# ---- Serial helpers -------------------------------------------------------------------------
def open_board(port, baud):
    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = port, baud, 3
    ser.dtr = False
    ser.rts = False
    ser.open()
    time.sleep(2)
    ser.write(b"x")
    time.sleep(0.3)
    ser.reset_input_buffer()
    ser.write(b"i")
    time.sleep(0.5)
    while ser.in_waiting:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if line:
            print(line)
    return ser


def read_line_starting(ser, prefix, timeout=5):
    t_end = time.time() + timeout
    while time.time() < t_end:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if line.startswith(prefix):
            return line
        if line.startswith("# ERROR"):
            raise RuntimeError(line)
    raise TimeoutError(f"No '{prefix}' reply from the board. Is fault_detector.ino uploaded?")


def send_window(ser, win, tries=5):
    """Send one window for the 'v' command; resend if a byte was lost on the way.
    Returns (reply fields after 'V', number of resends)."""
    payload = struct.pack(f"<{WINDOW * 3}h", *win.ravel())
    checksum = sum(payload)
    for attempt in range(tries):
        ser.write(b"v" + payload)
        try:
            parts = read_line_starting(ser, "V,", timeout=3).split(",")[1:]
            if len(parts) == 13 and int(parts[12]) == checksum:
                return parts, attempt
        except (RuntimeError, TimeoutError):
            pass
        time.sleep(0.6)                 # let the board give up on a short window
        ser.reset_input_buffer()
    raise RuntimeError("The board kept receiving a damaged window. Try unplugging and reconnecting it.")


# ---- verify ---------------------------------------------------------------------------------
def default_files(data_dir):
    files = []
    for label in TRAIN_LABELS:
        sessions = sorted(glob.glob(os.path.join(data_dir, f"{label}_*.csv")))
        if sessions:
            files.append(sessions[-1])           # shared test split: last session per condition
    files += sorted(glob.glob(os.path.join(data_dir, "heldout_*.csv")))
    return files


def verify(args):
    model = load_model()
    files = args.files or default_files(args.data)
    ser = open_board(args.port, args.baud)
    rows, total_retries = [], 0
    for f in files:
        meta_path = f[:-4] + ".json"
        counts_per_g = json.load(open(meta_path)).get("counts_per_g", 16384) if os.path.exists(meta_path) else 4096
        if counts_per_g != 4096:
            print(f"Skipping {os.path.basename(f)}: recorded at {counts_per_g} counts per g, the board uses 4096.")
            continue
        label = os.path.basename(f).rsplit("_", 2)[0]
        raw = pd.read_csv(f).dropna()[["ax", "ay", "az"]].to_numpy()
        n_win = min(args.windows, len(raw) // WINDOW)
        print(f"{os.path.basename(f)}: sending {n_win} windows ...")
        for w in range(n_win):
            win = raw[w * WINDOW:(w + 1) * WINDOW].astype(np.int16)
            parts, retries = send_window(ser, win)
            total_retries += retries
            esp_feats = np.array([float(v) for v in parts[:9]])
            esp_p, feat_us, infer_us = float(parts[9]), int(parts[10]), int(parts[11])

            mag = np.sqrt(((win.astype(float) * 9.80665 / 4096.0) ** 2).sum(axis=1))
            py_feats = np.array(extract_window_features(mag))
            py_p = float(predict_proba(py_feats, model))
            rows.append({"file": os.path.basename(f), "label": label, "window": w,
                         "esp_p": esp_p, "py_p": py_p, "feat_us": feat_us, "infer_us": infer_us,
                         **{f"zerr_{n}": abs(e - p) / s for n, e, p, s in zip(FEATURES, esp_feats, py_feats, model[1])}})
    ser.close()
    if not rows:
        print("Nothing was tested.")
        return

    df = pd.DataFrame(rows)
    df["esp_pred"] = (df.esp_p >= 0.5).astype(int)
    df["py_pred"] = (df.py_p >= 0.5).astype(int)
    df["truth"] = (df.label != "normal").astype(int)

    print("\n=== ESP32 vs Python ===")
    print(f"Windows tested:            {len(df)}  (resent after a transfer error: {total_retries})")
    print(f"Prediction agreement:      {(df.esp_pred == df.py_pred).mean() * 100:.2f}%")
    print(f"Max probability diff:      {(df.esp_p - df.py_p).abs().max():.2e}")
    worst = df[[c for c in df.columns if c.startswith("zerr_")]].max()
    print(f"Largest feature error:     {worst.max():.2e} standard deviations ({worst.idxmax()[5:]})")
    print("\n=== ESP32 accuracy (fault = anything but normal) ===")
    print(df.groupby("label").apply(lambda d: f"{(d.esp_pred == d.truth).mean() * 100:.1f}% of {len(d)} windows",
                                    include_groups=False).to_string())
    print("\n=== On-board timing (ESP32) ===")
    print(f"Feature extraction:  mean {df.feat_us.mean() / 1000:.2f} ms  (max {df.feat_us.max() / 1000:.2f} ms)")
    print(f"Inference (model):   mean {df.infer_us.mean():.1f} us  (max {df.infer_us.max()} us)")

    out = os.path.join(ROOT, "results", "esp32_verify_results.csv")
    df.to_csv(out, index=False)
    print(f"\nPer-window results saved to {out}")


# ---- live -----------------------------------------------------------------------------------
def live(args):
    if args.save:
        args.features = True                     # saved runs always include the features
    started = datetime.now()
    ser = open_board(args.port, args.baud)
    ser.write(b"L" if args.features else b"l")
    rows, alerts, feats = [], 0, []
    t_end = time.time() + args.seconds
    print("window  pred    p_fault  rpm    feature_ms  inference_us")
    while time.time() < t_end:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if line.startswith("W,"):
            _, w, pred, p, rpm, feat_us, infer_us = line.split(",")
            rows.append({"window": int(w), "pred": int(pred), "p_fault": float(p), "rpm": float(rpm),
                         "feature_us": int(feat_us), "inference_us": int(infer_us), "event": ""})
            print(f"{int(w):6d}  {'FAULT ' if pred == '1' else 'normal'}  {float(p):7.4f}  {float(rpm):5.0f}  "
                  f"{int(feat_us) / 1000:10.2f}  {int(infer_us):12d}")
        elif line.startswith("F,") and rows and FEATURES[0] not in rows[-1]:
            # only attach to a window that has no features yet; if a W line was lost, its
            # F line must not overwrite the previous window's features
            vals = [float(v) for v in line.split(",")[1:]]
            feats.append(vals)
            rows[-1].update(zip(FEATURES, vals))
        elif line.startswith("# ALERT") or line.startswith("# clear"):
            alerts += line.startswith("# ALERT")
            if rows:
                rows[-1]["event"] = "ALERT" if line.startswith("# ALERT") else "clear"
            print(line)
    ser.write(b"x")
    if args.features and rows and FEATURES[0] not in rows[-1]:
        t_more = time.time() + 1.0               # the last window's feature line may still be on its way
        while time.time() < t_more:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("F,"):
                vals = [float(v) for v in line.split(",")[1:]]
                feats.append(vals)
                rows[-1].update(zip(FEATURES, vals))
                break
    ser.close()
    if args.features:
        rows = [r for r in rows if FEATURES[0] in r]   # keep only windows that have their features
    if not rows:
        print("No windows received. Is fault_detector.ino uploaded?")
        return
    df = pd.DataFrame(rows)
    pct_fault = df.pred.mean() * 100
    print(f"\n{len(df)} windows: {pct_fault:.1f}% flagged as fault, {alerts} alert(s)")
    print(f"Feature extraction mean {df.feature_us.mean() / 1000:.2f} ms, "
          f"inference mean {df.inference_us.mean():.1f} us")
    correct = None
    if args.expect:
        want = 1 if args.expect == "fault" else 0
        correct = (df.pred == want).mean() * 100
        print(f"Correct for '{args.expect}': {correct:.1f}%")
    table, scores = (explain(np.array(feats), args.data) if feats else (None, None))

    if args.save:
        out_dir = os.path.join(ROOT, "results", "live_runs")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, f"live_{args.save}_{started.strftime('%Y%m%d_%H%M%S')}")
        df.to_csv(base + ".csv", index=False)
        if table is not None:
            table.to_csv(base + "_vs_recorded_normal.csv", index_label="feature")
        summary = {
            "condition": args.save,
            "expect": args.expect,
            "notes": args.notes,
            "started": started.isoformat(timespec="seconds"),
            "seconds": args.seconds,
            "windows": len(df),
            "percent_flagged_fault": round(pct_fault, 2),
            "percent_correct": None if correct is None else round(correct, 2),
            "alerts": alerts,
            "mean_rpm": round(df.rpm[df.rpm > 0].mean(), 1) if (df.rpm > 0).any() else 0,
            "feature_ms_mean": round(df.feature_us.mean() / 1000, 3),
            "inference_us_mean": round(df.inference_us.mean(), 2),
            "inference_us_max": int(df.inference_us.max()),
            "median_score_live": scores and round(scores[0], 3),
            "median_score_recorded_normal": scores and round(scores[1], 3),
            "model_header": os.path.relpath(HEADER, ROOT),
        }
        with open(base + "_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved {base}.csv (+ _summary.json"
              f"{', _vs_recorded_normal.csv' if table is not None else ''})")


def explain(live_feats, data_dir):
    """Show how far the live features are from the recorded normal sessions, and which features
    push the score toward 'fault' (score >= 0 means fault)."""
    mean, scale, coef, intercept = load_model()
    normal = []
    for f in sorted(glob.glob(os.path.join(data_dir, "normal_*.csv"))):
        c = pd.read_csv(f).dropna()[["ax", "ay", "az"]].to_numpy()
        mag = np.sqrt(((c * 9.80665 / 4096.0) ** 2).sum(axis=1))
        normal += [extract_window_features(mag[w * WINDOW:(w + 1) * WINDOW]) for w in range(len(mag) // WINDOW)]
    normal = np.array(normal)
    z_live = (live_feats - mean) / scale
    z_norm = (normal - mean) / scale
    table = pd.DataFrame({
        "live_median": np.median(live_feats, axis=0),
        "recorded_normal_median": np.median(normal, axis=0),
        "shift_in_normal_spreads": (np.median(live_feats, axis=0) - np.median(normal, axis=0)) / normal.std(axis=0),
        "score_push_live": np.median(z_live * coef, axis=0),
        "score_push_recorded_normal": np.median(z_norm * coef, axis=0),
    }, index=FEATURES)
    pd.set_option("display.width", 200)
    print("\n=== Live features vs recorded normal (score >= 0 means fault) ===")
    print(table.to_string(float_format=lambda v: f"{v:.4g}"))
    scores = (float(np.median(intercept + z_live @ coef)), float(np.median(intercept + z_norm @ coef)))
    print(f"\nMedian score: live {scores[0]:.2f}, recorded normal {scores[1]:.2f}")
    return table, scores


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--port", required=True)
    v.add_argument("--data", default=os.path.join(ROOT, "data"))
    v.add_argument("--files", nargs="*")
    v.add_argument("--windows", type=int, default=20)
    l = sub.add_parser("live")
    l.add_argument("--port", required=True)
    l.add_argument("--seconds", type=float, default=30)
    l.add_argument("--expect", choices=["normal", "fault"])
    l.add_argument("--features", action="store_true",
                   help="also get each window's features and compare them with the recorded normal data")
    l.add_argument("--data", default=os.path.join(ROOT, "data"))
    l.add_argument("--save", metavar="CONDITION",
                   help="save this run to results/live_runs/ under a name such as normal or imbalance (includes features)")
    l.add_argument("--notes", default="", help="saved with the run, e.g. 'twist tie, blade 1'")
    for s in (v, l):
        s.add_argument("--baud", type=int, default=921600)
    args = p.parse_args()
    verify(args) if args.mode == "verify" else live(args)


if __name__ == "__main__":
    main()
