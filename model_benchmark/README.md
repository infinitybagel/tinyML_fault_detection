# Model benchmark: fan-fault detection on the ESP32

One folder so all four models are measured **the same way**: same data, same features, same split, same
metrics, same ESP32 sketch. The logistic regression is finished and is the worked example; the other three
models have starting templates that already run.

| Model | Owner | File | Status |
|---|---|---|---|
| Logistic regression | Aaron | `models/logistic_regression.py` | done (results in `results/`) |
| Linear SVM | Iritza | `models/linear_svm.py` | template: tune it, then fill in `DEVICE` |
| Isolation forest | Angelina | `models/isolation_forest.py` | template: tune it, then fill in `DEVICE` |
| Decision tree | Devon | `models/decision_tree.py` | template: tune it, then fill in `DEVICE` |

## Quick start

```
cd model_benchmark
pip install -r requirements.txt
python models/logistic_regression.py      # the reference: prints F1 1.000, 5-round 0.993 ± 0.016, ...
python models/linear_svm.py               # your model (each file runs on its own)
python compare.py                         # comparison table + charts for every model that has results
```

The first run extracts the features from the repository's `data/` folder (about 30 s) and caches them in `results/features_cache.csv`.

## What's in the folder

```
model_benchmark/
├── README.md
├── requirements.txt
├── benchmark/                SHARED CODE: don't edit per model
│   ├── features.py           the 9 shared features (same math as features.py, scaled per session's JSON)
│   └── evaluate.py           the shared split and all metrics; evaluate() writes your results
├── models/                   one file per model: this is where you work
├── results/<model>/         <model>.json, _predictions.csv, _confusion.png, ESP32 verify, logs, live_runs/
│   └── comparison/           made by compare.py: table (CSV/Markdown), F1, held-out recall, F1 vs. size
├── compare.py
└── esp32/
    ├── fault_detector/       ESP32 sketch for ANY model: put yours in model.h
    └── esp32_test.py         verify (board vs. Python) and live tests
```

## The data

| Label | How it was made | Sessions | Used for |
|---|---|---|---|
| `normal` | fan running as is, all four corners taped | 5 | training + testing |
| `imbalance` | twist tie (~4 cm) on one blade, ~1 cm from the tip | 5 | training + testing |
| `loose_mount` | bottom-left corner untaped | 5 | training + testing |
| `blocked_airflow` | index card over half of the intake | 5 | training + testing |
| `heldout` | ~1 mm of index card touching the blade tips (rub) | 5 | **testing only, never trained on** |

Each session: 60 s at 1 kHz, raw counts at ±8 g (4096 counts per g; the JSON says `counts_per_g`),
fan duty 204/255 (~1,690 RPM, spin 28 Hz), plus fan RPM per row.

## The rules (so the comparison is fair)

1. **Same features.** Use `benchmark/features.py` as is: 1-s non-overlapping windows of the acceleration
   magnitude → `rms, peak, std, kurtosis, crest_factor, band_5_25hz, band_25_75hz, band_75_200hz,
   band_200_500hz`. Your model may transform them internally (scaling, logs) as part of its pipeline.
2. **Same split.** `evaluate()` does it for you:
   - shared split: train on sessions 1–4 of each condition, test on session 5 (the paper's main numbers);
   - 5 rounds: each session is the test session once (mean ± std, for robustness);
   - held-out fault: all 5 `heldout` sessions, never trained on.
3. **Tune on training sessions only.** Wrap your estimator in `tuned(estimator, grid, scoring)`: a
   `GridSearchCV` that splits by session. Never pick settings by looking at test or held-out scores.
4. **Supervised vs. unsupervised.** Supervised models (LR, SVM, tree) see all four training conditions,
   label 1 = any fault. The isolation forest is `kind="unsupervised"`: it is trained on **normal windows only**
   and must follow scikit-learn's outlier convention (`predict` → −1 for an anomaly).
5. **Fault score.** For ROC-AUC the benchmark uses `predict_proba[:, 1]`, else `decision_function`
   (unsupervised: `−decision_function`). Predictions use the model's own `predict`.

## What `evaluate()` reports

- Shared split: precision, recall, F1, false alarm rate (normal windows flagged), ROC-AUC, confusion matrix,
  held-out recall
- 5 rounds: mean and standard deviation of all of the above
- Which fault (supervised, if you pass `make_multiclass_model`): macro F1, confusion matrix, and how the
  held-out rub gets classified
- `device`: your ESP32 numbers (you fill these in, see below)

Everything is saved to `results/<model>/<model>.json`; `compare.py` turns all of them into the paper's Table II
and the outline's charts (F1 per model, held-out recall per model, F1 vs. model size).

## Your workflow

1. Open your file in `models/`, run it once as is to see the baseline.
2. Change the search space / settings (marked `TODO`), re-run. Keep `make_model()` returning a **new,
   unfitted** estimator each call.
3. Convert the trained model to C (micromlgen, emlearn, or by hand) and put it in
   `esp32/fault_detector/model.h` (see below). Upload with the Arduino IDE (board: ESP32 Dev Module).
4. Check the board against Python and measure timing:
   ```
   python esp32/esp32_test.py verify --port COM3 --model linear_svm
   ```
   Put the printed `esp32_agreement` and `inference_us` into `DEVICE` in your model file, plus
   `param_bytes` (your model's parameters), and `flash_kb` / `ram_kb` (Arduino's "Sketch uses … / Global
   variables use …" minus the logging sketch's numbers: 326,671 bytes flash and 25,284 bytes RAM).
5. Re-run your model file (so `DEVICE` is saved), then `python compare.py`.
6. Optional live tests on the fan:
   ```
   python esp32/esp32_test.py live --port COM3 --seconds 30 --expect normal --save linear_svm_normal --model linear_svm
   ```

## The ESP32 sketch (`esp32/fault_detector`)

Same wiring and sensor settings as the logger (±8 g, 184 Hz filter, fan duty 204/255). It computes the 9
features in C (`fan_features.h`, checked against Python on all 1,476 windows) and calls **one function
from `model.h`**:

```c
#define MODEL_NAME        "linear SVM"
#define MODEL_THRESHOLD   0.5f                 // score >= this means FAULT
#define MODEL_PARAM_BYTES (...)                // bytes of your model's parameters
static inline float model_fault_score(const float *x) { ... }   // x = the 9 features, in order
```

`model.h` currently holds the logistic regression as the example. Keep a copy of it, then swap in yours.
If your Python pipeline standardizes features, do the same in C (the LR example shows how: subtract the
mean, divide by the scale). Serial commands: `v` verify, `l` live, `L` live with features, `x` stop,
`i` info. Close the Serial Monitor before running `esp32_test.py`.

## Things to know

- **Fan speed is not a shortcut.** All conditions run at ~1,690 RPM (an earlier recording set didn't, because
  of a bad ground; it was re-recorded). A classifier on RPM alone scores F1 ≈ 0.67.
- **Scores near 1.000 are expected** on this data (one fan, one day, one test session per condition).
  The 5-round numbers, the held-out fault and the live tests are where models differ.
- **Live tests are sensitive to mounting.** If a normal test flags faults, re-tape all four corners first.
- The `rms` and `peak` features include gravity (they're computed before the mean is removed).
- The held-out rub resembles the loose mount and blocked airflow in the spectrum, so supervised models catch
  it more easily than a truly new fault would be caught. Say so when you compare against the isolation forest.
