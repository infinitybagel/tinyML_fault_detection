"""Logistic regression (Aaron Arnold). Reference example: this is the finished model.

    python models/logistic_regression.py

Reproduces the results in the paper and notebook: shared-split F1 1.000, 5-round F1 0.993 ± 0.016,
held-out recall 1.000, multiclass macro F1 0.983. Copy this file's structure for your own model.
"""

import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark import evaluate, tuned

C_GRID = {"logisticregression__C": np.logspace(-3, 3, 7)}


def make_model():
    """Normal vs. fault. Tuned by session-grouped CV on the training sessions only."""
    pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=10000))
    return tuned(pipe, C_GRID, scoring="f1")


def make_multiclass_model():
    """Which of the 4 training conditions."""
    pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=10000))
    return tuned(pipe, C_GRID, scoring="f1_macro")


# Measured on the ESP32 (fault_detector sketch). See esp32/README in the main README.
DEVICE = {
    "param_bytes": 112,          # 9 means + 9 scales + 9 weights + 1 intercept, float32
    "inference_us": 18.1,        # mean in live mode (verify mode: ~10 us)
    "feature_ms": 42.8,          # shared by every model
    "flash_kb": 8.7,             # vs. the logging sketch
    "ram_kb": 22.0,              # vs. the logging sketch (16 KB are the shared feature buffers)
    "esp32_agreement": 1.0,      # 531 / 531 recorded windows matched Python
}

if __name__ == "__main__":
    evaluate("logistic_regression", make_model, kind="supervised",
             make_multiclass_model=make_multiclass_model, device=DEVICE, owner="Aaron Arnold",
             notes="StandardScaler + LogisticRegression(class_weight='balanced'), C tuned by GroupKFold")
