# TinyML fault detection on an ESP32: which model catches the fault?

An MPU6050 accelerometer on a 12 V PC fan (a stand-in for transformer cooling fans), an ESP32 that
classifies every 1-second vibration window on the device, and a comparison of four machine learning
models: **logistic regression, linear SVM, isolation forest and decision tree**.

Florida Atlantic University, November 2026: Irtiza Khan, Aaron Arnold, Devon Biscardi, Angelina Guasti.

**Paper:** [Which Model Catches the Fault? Comparing Machine Learning Models for TinyML Anomaly Detection in
Predictive Maintenance of Power Equipment](paper/TinyML_Fan_Fault_Detection.pdf)

## Results at a glance

| Model | F1 (shared test split) | False alarms | Unseen fault (blade rub) caught | Size on ESP32 | Inference |
|---|---|---|---|---|---|
| Logistic regression | 1.000 | 0% | 100% | 112 B | 10–18 µs |
| Linear SVM | 1.000 | 0% | 100% | 112 B | 5.8 µs |
| Isolation forest | 0.983 | 10.2% | 100% | 13,256 B | 234 µs |
| Decision tree | 1.000 | 0% | 100% | 43 B | 1.2 µs |

All four models gave identical predictions on the ESP32 and in Python (531/531 recorded windows).
Computing the 9 features takes ~40 ms per window, far more than any model. See
`model_benchmark/results/comparison/` for the full table and charts.

## Repository layout

```
data/                 25 recorded sessions (5 each: normal, imbalance, loose_mount, blocked_airflow, heldout)
                      CSV: t_us, ax, ay, az (raw counts, ±8 g = 4096 counts/g), rpm · JSON: session info
firmware/
  vibration_logger/   ESP32 sketch: streams the accelerometer at 1 kHz + fan RPM over USB
  fault_detector/     ESP32 sketch: 9 features in C + the logistic regression, verify and live modes
  fan_tuner/          ESP32 sketch: set the fan's PWM duty over serial and read back RPM
scripts/
  record_session.py   record one labelled 60-s session into data/
  esp32_test.py       verify the board against Python on recorded windows; live tests on the fan
  make_figures.py     result figures (figures/slides, or PAPER=1 for figures/paper)
notebooks/
  logistic_regression.ipynb   full logistic regression analysis with saved outputs
  build_notebook.py           rebuilds and runs the notebook
model/                exported logistic regression (logreg_model.h) and its results row
results/              ESP32 verification (531 windows) and live runs for the logistic regression
figures/              figures used in the slides and the paper
model_benchmark/      the shared benchmark: same features, split and metrics for all four models,
                      one ESP32 sketch per model, results per model, comparison charts (own README)
paper/                the paper: TinyML_Fan_Fault_Detection.pdf (IEEE format)
tools/                fan-speed tuning, PC test harness for the C features, board simulator, checks
```

## Quick start

```
pip install -r requirements.txt
python model_benchmark/models/logistic_regression.py     # train + evaluate one model
python model_benchmark/compare.py                        # comparison table and charts for all models
```

Hardware runs (Arduino IDE or arduino-cli, board "ESP32 Dev Module", esp32 core 4.0.0):

```
python scripts/record_session.py --port COM3 --label normal --seconds 60
python scripts/esp32_test.py verify --port COM3
python scripts/esp32_test.py live --port COM3 --seconds 30 --expect normal --save normal
python model_benchmark/esp32/esp32_test.py verify --port COM3 --model linear_svm --threshold 0
```

## Hardware and wiring

ESP32 DevKit (ESP-WROOM-32), MPU6050 on I²C (SDA GPIO 21, SCL GPIO 22, 400 kHz), 12 V 4-pin fan
(PWM GPIO 19 through 1 kΩ at 25 kHz, tach GPIO 18 with a 10 kΩ pull-up to 3.3 V), common ground.
Recordings: ±8 g, ~184 Hz digital low-pass filter, 1 kHz, fan duty 204/255 (~1,690 RPM, 28 Hz spin).
Full wiring notes are at the top of `firmware/vibration_logger/vibration_logger.ino`.

## Method in one paragraph

Each session is cut into non-overlapping 1-s windows of the acceleration magnitude; each window gives 9
features (RMS, peak, std, kurtosis, crest factor and FFT energy in 5–25, 25–75, 75–200 and 200–500 Hz).
Sessions 1–4 of each training condition train the models and session 5 tests them, so no test session is
ever seen in training; the five blade-rub sessions are never trained on. A 5-round rotation of the test
session checks robustness. Models are converted to C and checked on the ESP32 against Python.

## Known limitations

- One small fan, lab-made faults, recorded on one day with one mounting.
- The shared `rms` and `peak` features include gravity, so they partly measure sensor tilt; live tests on a
  re-mounted sensor showed every model is sensitive to this.
- The MPU6050's ~184 Hz filter weakens the 200–500 Hz band the linear models rely on.
