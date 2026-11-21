"""Linear SVM (Iritza Khan). STARTING TEMPLATE: runs as is with basic settings; tune it and make it yours.

    python models/linear_svm.py

TODO (Iritza):
  - choose the hyperparameters to search (C, loss, ...) in C_GRID
  - after converting to C (e.g. micromlgen or emlearn) and testing on the ESP32, fill in DEVICE
  - write your section in the paper (Sections IV-C-2 and VI)
"""

import os
import sys

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark import evaluate, tuned

C_GRID = {"linearsvc__C": np.logspace(-3, 2, 6)}           # TODO: your search space


def make_model():
    # LinearSVC has no predict_proba; the benchmark uses decision_function as the fault score (ROC-AUC).
    pipe = make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", max_iter=20000))
    return tuned(pipe, C_GRID, scoring="f1")


def make_multiclass_model():
    pipe = make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", max_iter=20000))
    return tuned(pipe, C_GRID, scoring="f1_macro")


DEVICE = {                       # TODO: fill in after the ESP32 tests (leave None until measured)
    "param_bytes": None,
    "inference_us": None,
    "feature_ms": 42.8,          # shared feature extraction, same for every model
    "flash_kb": None,
    "ram_kb": None,
    "esp32_agreement": None,     # fraction of windows where the ESP32 matched Python (esp32_test.py verify)
}

if __name__ == "__main__":
    evaluate("linear_svm", make_model, kind="supervised", make_multiclass_model=make_multiclass_model,
             device=DEVICE, owner="Iritza Khan", notes="TODO: describe your final settings")
