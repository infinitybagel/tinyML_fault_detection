"""Record one labeled vibration session from the ESP32 to a CSV file.

Usage examples (from the experiment folder; sessions are saved to its data/ folder):
    python scripts/record_session.py --port COM3 --label normal --seconds 60
    python scripts/record_session.py --port COM3 --label imbalance --seconds 60 --notes "twist tie, blade 1"

Find your port with:  python -m serial.tools.list_ports
Close the Arduino Serial Monitor first, or the port will be busy.

Output (in the experiment's data/ folder, or --out):
    <label>_<date>_<time>.csv   columns: t_us, ax, ay, az, rpm
                                (ax/ay/az are raw counts; divide by COUNTS_PER_G for g.
                                 rpm is the latest fan reading at that sample; the ESP32
                                 reports it once per second, so it changes once per second)
    <label>_<date>_<time>.json  session info: sample rate, gaps, fan RPM, notes
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

import serial

LABELS = ["normal", "imbalance", "loose_mount", "blocked_airflow", "heldout"]

# Must match the sketch (vibration_logger.ino): ACCEL_RANGE_G / COUNTS_PER_G / FAN_DUTY
ACCEL_RANGE_G = 8
COUNTS_PER_G = 4096
FAN_DUTY = 204   # out of 255 (80%)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--label", required=True, choices=LABELS)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--notes", default="")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    started = datetime.now()
    base = os.path.join(args.out, f"{args.label}_{started.strftime('%Y%m%d_%H%M%S')}")

    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = args.port, args.baud, 1
    ser.dtr = False   # avoid holding the ESP32 in reset
    ser.rts = False
    ser.open()
    time.sleep(2)     # give the board time in case opening the port reset it
    ser.reset_input_buffer()

    rows, rpms, bad = [], [], 0
    current_rpm = None   # latest RPM reported by the ESP32 (None until the first report)
    print(f"Recording '{args.label}' for {args.seconds:.0f} s ...")
    ser.write(b"s")
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if not line:
            continue
        if line.startswith("#"):
            if line.startswith("# rpm,"):
                try:
                    current_rpm = round(float(line.split(",")[1]))
                    rpms.append(current_rpm)
                    print(f"  fan RPM: {current_rpm}   ", end="\r")
                except ValueError:
                    bad += 1
            continue
        parts = line.split(",")
        if len(parts) != 4:
            bad += 1
            continue
        try:
            rows.append([int(x) for x in parts] + [current_rpm])
        except ValueError:
            bad += 1
    print()
    ser.write(b"x")
    ser.close()

    if len(rows) < 2:
        print("No data received. Check the port, the baud rate, and that the sketch is uploaded.")
        return

    # The first RPM report arrives ~1 s after recording starts. Fill the samples before it
    # with that first reading so every row has an RPM value.
    if rpms:
        for r in rows:
            if r[4] is not None:
                break
            r[4] = rpms[0]
    else:
        print("  Note: no RPM readings received (is TACH_PIN set to -1 in the sketch?). The rpm column is empty.")

    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_us", "ax", "ay", "az", "rpm"])
        w.writerows(["" if v is None else v for v in r] for r in rows)

    # Timing check
    dts = [(rows[i][0] - rows[i - 1][0]) & 0xFFFFFFFF for i in range(1, len(rows))]
    duration_s = sum(dts) / 1e6
    rate = (len(rows) - 1) / duration_s if duration_s > 0 else 0
    gaps = sum(1 for d in dts if d > 1500)
    rpm_steady = rpms[2:] if len(rpms) > 3 else rpms   # skip the first readings
    mean_rpm = round(sum(rpm_steady) / len(rpm_steady)) if rpm_steady else None
    clipped = sum(1 for r in rows for v in r[1:4] if abs(v) >= 32767)   # sensor maxed out

    info = {
        "label": args.label,
        "started": started.isoformat(timespec="seconds"),
        "samples": len(rows),
        "duration_s": round(duration_s, 2),
        "sample_rate_hz": round(rate, 1),
        "gaps_over_1_5ms": gaps,
        "bad_lines": bad,
        "mean_fan_rpm": mean_rpm,
        "min_fan_rpm": min(rpm_steady) if rpm_steady else None,
        "max_fan_rpm": max(rpm_steady) if rpm_steady else None,
        "accel_range": f"+/-{ACCEL_RANGE_G} g ({COUNTS_PER_G} counts per g)",
        "counts_per_g": COUNTS_PER_G,
        "clipped_values": clipped,
        "fan_duty": f"{FAN_DUTY}/255",
        "notes": args.notes,
    }
    with open(base + ".json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Saved {base}.csv")
    print(f"  {len(rows)} samples, {rate:.0f} Hz, {gaps} gaps, {bad} bad lines, fan RPM: {mean_rpm}")
    if rate < 950 or gaps > len(rows) * 0.01:
        print("  Warning: sample rate is low or has many gaps. Check wiring and the baud rate.")
    if clipped:
        print(f"  Warning: {clipped} readings hit the +/-{ACCEL_RANGE_G} g limit (clipped). "
              "Use a larger range or a gentler fault, and re-record.")


if __name__ == "__main__":
    main()
