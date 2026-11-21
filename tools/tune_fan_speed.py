"""Find the PWM duty that runs the fan at a target speed (used to restore ~1,690 RPM on 2026-11-20).

Upload firmware/fan_tuner first, close the Serial Monitor, then:
    python tools/tune_fan_speed.py --port COM3 --target 1689

Does a coarse binary search (6 readings per duty), then measures the two closest duties for 15 s each.
Results are saved to results/fan_speed_tuning/.
"""

import argparse
import json
import os
import statistics
import time

import serial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rpm_at(ser, duty, n, settle=6):
    ser.write(f"D{duty}\n".encode())
    t0, vals = time.time(), []
    while len(vals) < n:
        line = ser.readline().decode(errors="ignore").strip()
        if line.startswith("R,") and int(line.split(",")[1]) == duty and time.time() - t0 > settle:
            vals.append(float(line.split(",")[2]))
    return vals


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--target", type=float, default=1689, help="RPM to match (default: median of the recordings)")
    p.add_argument("--low", type=int, default=100)
    p.add_argument("--high", type=int, default=204)
    args = p.parse_args()

    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = args.port, 921600, 2
    ser.dtr = False
    ser.rts = False
    ser.open()
    time.sleep(2)
    ser.reset_input_buffer()

    coarse = {}
    lo, hi = args.low, args.high
    coarse[hi] = statistics.mean(rpm_at(ser, hi, 6))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        coarse[mid] = statistics.mean(rpm_at(ser, mid, 6))
        print(f"duty {mid:3d}: {coarse[mid]:6.0f} RPM", flush=True)
        if coarse[mid] > args.target:
            hi = mid
        else:
            lo = mid

    fine = {}
    for d in sorted({lo, hi}):
        v = rpm_at(ser, d, 15)
        fine[d] = {"median": statistics.median(v), "mean": statistics.mean(v), "min": min(v), "max": max(v)}
        print(f"duty {d}: median {fine[d]['median']:.0f}, mean {fine[d]['mean']:.0f}, range {min(v):.0f}-{max(v):.0f}")
    ser.close()

    best = min(fine, key=lambda d: abs(fine[d]["median"] - args.target))
    print(f"\nBest duty: {best}/255 (set FAN_DUTY in the sketch you run)")
    out = os.path.join(ROOT, "results", "fan_speed_tuning")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"tuning_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump({"target_rpm": args.target, "best_duty": best, "coarse": coarse, "fine": fine}, f, indent=1)


if __name__ == "__main__":
    main()
