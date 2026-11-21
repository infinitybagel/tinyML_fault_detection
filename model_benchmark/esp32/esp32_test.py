"""Test ANY model running on the ESP32 (sketch: esp32/fault_detector, with your model in model.h).

Run from the model_benchmark folder. Close the Arduino Serial Monitor first.

  verify  Sends recorded windows to the board. Checks that the board's 9 features match the shared Python
          features, and that its predictions match your Python model (results/<model>_predictions.csv,
          written by `python models/<model>.py`). Reports on-board timing.

            python esp32/esp32_test.py verify --port COM3 --model linear_svm

          Default windows: the test session of each condition + all held-out sessions (the ones in
          your predictions file), 59 per file. --windows N sends fewer.

  live    Classifies the running fan; --save keeps the run (with --model: in results/<model>/live_runs/).

            python esp32/esp32_test.py live --port COM3 --seconds 30 --expect normal --save linear_svm_normal --model linear_svm
"""

import argparse
import json
import os
import struct
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from benchmark.features import DATA_DIR, FEATURES, WINDOW_SIZE, extract_window_features   # noqa: E402

RESULTS = os.path.join(ROOT, "results")
RUNS = os.path.join(HERE, "runs")


# ---- serial helpers --------------------------------------------------------------------------
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
    raise TimeoutError(f"No '{prefix}' reply from the board. Is esp32/fault_detector uploaded?")


def send_window(ser, win, tries=5):
    """Send one window for 'v'; resend if a byte was lost (checked with the board's checksum)."""
    payload = struct.pack(f"<{WINDOW_SIZE * 3}h", *win.ravel())
    checksum = sum(payload)
    for attempt in range(tries):
        ser.write(b"v" + payload)
        try:
            parts = read_line_starting(ser, "V,", timeout=3).split(",")[1:]
            if len(parts) == 13 and int(parts[12]) == checksum:
                return parts, attempt
        except (RuntimeError, TimeoutError):
            pass
        time.sleep(0.6)
        ser.reset_input_buffer()
    raise RuntimeError("The board kept receiving a damaged window. Unplug and reconnect it.")


# ---- verify ----------------------------------------------------------------------------------
def verify(args):
    model_dir = os.path.join(RESULTS, args.model)
    ref_path = os.path.join(model_dir, f"{args.model}_predictions.csv")
    if not os.path.exists(ref_path):
        raise SystemExit(f"{ref_path} not found. Run  python models/{args.model}.py  first.")
    ref = pd.read_csv(ref_path).set_index(["session", "window"])
    files = sorted(ref.index.get_level_values(0).unique())
    ser = open_board(args.port, args.baud)
    rows, retries = [], 0
    for fname in files:
        csv_path = os.path.join(DATA_DIR, fname)
        meta = json.load(open(csv_path[:-4] + ".json"))
        if meta.get("counts_per_g", 16384) != 4096:
            print(f"Skipping {fname}: not recorded at +/-8 g.")
            continue
        raw = pd.read_csv(csv_path).dropna()[["ax", "ay", "az"]].to_numpy()
        n_win = min(args.windows, len(raw) // WINDOW_SIZE)
        print(f"{fname}: sending {n_win} windows ...")
        for w in range(n_win):
            win = raw[w * WINDOW_SIZE:(w + 1) * WINDOW_SIZE].astype(np.int16)
            parts, r = send_window(ser, win)
            retries += r
            esp_feats = np.array([float(v) for v in parts[:9]])
            esp_score, feat_us, infer_us = float(parts[9]), int(parts[10]), int(parts[11])
            mag = np.sqrt(((win.astype(float) * 9.80665 / 4096.0) ** 2).sum(axis=1))
            py_feats = np.array(extract_window_features(mag))
            rel = np.abs(esp_feats - py_feats) / np.maximum(np.abs(py_feats), 1e-9)
            row = {"session": fname, "window": w, "esp_score": esp_score, "feat_us": feat_us,
                   "infer_us": infer_us, "feature_rel_err": float(np.median(rel))}
            if (fname, w) in ref.index:
                rr = ref.loc[(fname, w)]
                row.update(py_pred=int(rr["pred"]), py_score=float(rr["fault_score"]), truth=int(rr["truth"]))
            rows.append(row)
    ser.close()

    df = pd.DataFrame(rows)
    df["esp_pred"] = (df.esp_score >= args.threshold).astype(int)
    out = os.path.join(model_dir, f"{args.model}_esp32_verify.csv")
    df.to_csv(out, index=False)
    matched = df.dropna(subset=["py_pred"])
    agree = (matched.esp_pred == matched.py_pred).mean()
    print(f"\n=== {args.model}: ESP32 vs Python ===")
    print(f"Windows tested:         {len(df)}  (resent after a transfer error: {retries})")
    print(f"Prediction agreement:   {agree * 100:.2f}%  ({int((matched.esp_pred == matched.py_pred).sum())}/{len(matched)})")
    print(f"Feature match:          median relative error {df.feature_rel_err.median():.1e}")
    print(f"ESP32 accuracy:         {(matched.esp_pred == matched.truth).mean() * 100:.1f}%")
    print(f"Feature extraction:     {df.feat_us.mean() / 1000:.2f} ms per window")
    print(f"Inference (your model): {df.infer_us.mean():.1f} us per window (max {df.infer_us.max()} us)")
    print(f"\nPut these in DEVICE in models/{args.model}.py:  esp32_agreement = {agree:.4f},  "
          f"inference_us = {df.infer_us.mean():.1f}")
    print(f"Per-window results: {out}")


# ---- live ------------------------------------------------------------------------------------
def live(args):
    started = datetime.now()
    ser = open_board(args.port, args.baud)
    ser.write(b"L" if args.save else b"l")
    rows, alerts = [], 0
    t_end = time.time() + args.seconds
    print("window  pred    score    rpm    feature_ms  inference_us")
    while time.time() < t_end:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if line.startswith("W,"):
            _, w, pred, p, rpm, feat_us, infer_us = line.split(",")
            rows.append({"window": int(w), "pred": int(pred), "score": float(p), "rpm": float(rpm),
                         "feature_us": int(feat_us), "inference_us": int(infer_us), "event": ""})
            print(f"{int(w):6d}  {'FAULT ' if pred == '1' else 'normal'}  {float(p):7.4f}  {float(rpm):5.0f}  "
                  f"{int(feat_us) / 1000:10.2f}  {int(infer_us):12d}")
        elif line.startswith("F,") and rows and FEATURES[0] not in rows[-1]:
            rows[-1].update(zip(FEATURES, [float(v) for v in line.split(",")[1:]]))
        elif line.startswith("# ALERT") or line.startswith("# clear"):
            alerts += line.startswith("# ALERT")
            if rows:
                rows[-1]["event"] = "ALERT" if line.startswith("# ALERT") else "clear"
            print(line)
    ser.write(b"x")
    ser.close()
    if not rows:
        print("No windows received. Is esp32/fault_detector uploaded?")
        return
    df = pd.DataFrame(rows)
    print(f"\n{len(df)} windows: {df.pred.mean() * 100:.1f}% flagged as fault, {alerts} alert(s)")
    print(f"Feature extraction mean {df.feature_us.mean() / 1000:.2f} ms, inference mean {df.inference_us.mean():.1f} us")
    correct = None
    if args.expect:
        correct = (df.pred == (1 if args.expect == "fault" else 0)).mean() * 100
        print(f"Correct for '{args.expect}': {correct:.1f}%")
    if args.save:
        runs = os.path.join(RESULTS, args.model, "live_runs") if args.model else RUNS
        os.makedirs(runs, exist_ok=True)
        base = os.path.join(runs, f"live_{args.save}_{started.strftime('%Y%m%d_%H%M%S')}")
        df.to_csv(base + ".csv", index=False)
        json.dump({"name": args.save, "expect": args.expect, "notes": args.notes,
                   "started": started.isoformat(timespec="seconds"), "windows": len(df),
                   "percent_flagged_fault": round(df.pred.mean() * 100, 2),
                   "percent_correct": None if correct is None else round(correct, 2), "alerts": alerts,
                   "inference_us_mean": round(df.inference_us.mean(), 2)},
                  open(base + "_summary.json", "w"), indent=2)
        print(f"Saved {base}.csv and _summary.json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--port", required=True)
    v.add_argument("--model", required=True, help="name used in results/, e.g. linear_svm")
    v.add_argument("--windows", type=int, default=59)
    v.add_argument("--threshold", type=float, default=0.5, help="must equal MODEL_THRESHOLD in model.h")
    l = sub.add_parser("live")
    l.add_argument("--port", required=True)
    l.add_argument("--seconds", type=float, default=30)
    l.add_argument("--expect", choices=["normal", "fault"])
    l.add_argument("--save", metavar="NAME", help="keep the run (includes features)")
    l.add_argument("--model", help="save into results/<model>/live_runs/ instead of esp32/runs/")
    l.add_argument("--notes", default="")
    for s in (v, l):
        s.add_argument("--baud", type=int, default=921600)
    args = p.parse_args()
    verify(args) if args.mode == "verify" else live(args)


if __name__ == "__main__":
    main()
