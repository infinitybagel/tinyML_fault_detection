"""Isolation forest (Angelina Guasti). STARTING TEMPLATE: runs as is with basic settings; tune it and make it yours.

    python models/isolation_forest.py

This model is UNSUPERVISED: the benchmark trains it on NORMAL windows of the training sessions only and
never shows it a fault. It must follow scikit-learn's outlier convention (IsolationForest does):
predict() returns -1 for an anomaly, decision_function() is higher for more normal windows.

TODO (Angelina):
  - choose n_estimators, max_samples, contamination (the threshold). There are no fault labels to tune on;
    the outline's fair options are a fixed contamination, or a threshold set from normal training data only.
  - after exporting the trees to C and testing on the ESP32, fill in DEVICE
  - write your section in the paper (Sections IV-C-3 and VI; the held-out fault test is your key result)
"""

import os
import sys

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark import evaluate


def make_model():
    return make_pipeline(StandardScaler(),
                         IsolationForest(n_estimators=100, contamination="auto", random_state=0))   # TODO


DEVICE = {                       # TODO: fill in after the ESP32 tests (leave None until measured)
    "param_bytes": None,
    "inference_us": None,
    "feature_ms": 42.8,
    "flash_kb": None,
    "ram_kb": None,
    "esp32_agreement": None,
}

if __name__ == "__main__":
    evaluate("isolation_forest", make_model, kind="unsupervised", device=DEVICE, owner="Angelina Guasti",
             notes="TODO: describe your final settings")
