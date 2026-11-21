"""Wait until the fan runs steadily near normal speed before a live test (any fault_detector sketch).

    python tools/wait_for_fan_speed.py --port COM3

Exits when --needed consecutive 1-s readings fall within [--low, --high] RPM, or after --timeout seconds.
"""

import argparse
import sys
import time

import serial


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--low", type=float, default=1500)
    p.add_argument("--high", type=float, default=1900)
    p.add_argument("--needed", type=int, default=8)
    p.add_argument("--timeout", type=float, default=300)
    args = p.parse_args()

    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = args.port, 921600, 3
    ser.dtr = False
    ser.rts = False
    ser.open()
    time.sleep(2)
    ser.write(b"x")
    time.sleep(0.3)
    ser.reset_input_buffer()
    ser.write(b"l")
    t_end, streak = time.time() + args.timeout, 0
    try:
        while time.time() < t_end:
            line = ser.readline().decode(errors="ignore").strip()
            if not line.startswith("W,"):
                continue
            rpm = float(line.split(",")[4])
            streak = streak + 1 if args.low <= rpm <= args.high else 0
            print(f"rpm {rpm:.0f}  (steady {streak}/{args.needed})", flush=True)
            if streak >= args.needed:
                print("FAN READY")
                return 0
        print("TIMED OUT: fan not at speed")
        return 1
    finally:
        ser.write(b"x")
        ser.close()


if __name__ == "__main__":
    sys.exit(main())
