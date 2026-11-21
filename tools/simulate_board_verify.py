"""Run model_benchmark/esp32/esp32_test.py verify against a SIMULATED board (no ESP32 needed).

The "board" is a PC build of the same C code (fan_features.h + a model.h), so this checks the full
C pipeline against the Python model. Build the simulator for a model's sketch folder, e.g.:
    g++ -O2 -static -std=gnu++17 -I tools/arduino_shim -I model_benchmark/esp32/fault_detector_linear_svm \
        tools/board_simulator.cpp -o sim_svm.exe
Then, from the repository root:
    python tools/simulate_board_verify.py sim_svm.exe linear_svm 0

Arguments: simulator exe, model name (as in model_benchmark/results/), threshold (MODEL_THRESHOLD in model.h).
Timing numbers printed by verify are placeholders; only agreement and accuracy are meaningful here.
"""

import os
import runpy
import subprocess
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE, MODEL, THRESHOLD = sys.argv[1], sys.argv[2], sys.argv[3]


class FakeSerial:
    """Answers the sketch's 'v' command by running the simulator on the window it is sent."""

    def __init__(self):
        self.out, self.port = [], None

    def open(self): pass
    def close(self): pass
    def reset_input_buffer(self): pass

    @property
    def in_waiting(self):
        return 0

    def write(self, b):
        if b[:1] == b"v" and len(b) == 6001:
            res = subprocess.run([EXE], input=b[1:], capture_output=True).stdout.decode().strip().split(",")
            self.out.append(("V," + ",".join(res[:9]) + f",{res[9]},0,0,{sum(b[1:])}\n").encode())

    def readline(self):
        return self.out.pop(0) if self.out else b""


serial = types.ModuleType("serial")
serial.Serial = FakeSerial
sys.modules["serial"] = serial
time.sleep = lambda s: None
sys.argv = ["esp32_test.py", "verify", "--port", "SIM", "--model", MODEL, "--threshold", THRESHOLD]
runpy.run_path(os.path.join(ROOT, "model_benchmark", "esp32", "esp32_test.py"), run_name="__main__")
