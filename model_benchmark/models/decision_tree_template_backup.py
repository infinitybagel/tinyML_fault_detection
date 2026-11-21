"""Decision tree (Devon Biscardi). STARTING TEMPLATE: runs as is with basic settings; tune it and make it yours.

    python models/decision_tree.py

TODO (Devon):
  - choose the search space (max_depth, min_samples_leaf, ...) in GRID
  - after converting to C (e.g. micromlgen or emlearn, or plain if/else) and testing on the ESP32, fill in DEVICE
  - write your section in the paper (Sections IV-C-4 and VI)
"""

import os
import sys

from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark import evaluate, tuned

GRID = {"max_depth": [2, 3, 4, 6, 8], "min_samples_leaf": [1, 5, 20]}      # TODO: your search space


def make_model():
    # Trees don't need standardized features.
    return tuned(DecisionTreeClassifier(class_weight="balanced", random_state=0), GRID, scoring="f1")


def make_multiclass_model():
    return tuned(DecisionTreeClassifier(class_weight="balanced", random_state=0), GRID, scoring="f1_macro")


DEVICE = {                       # TODO: fill in after the ESP32 tests (leave None until measured)
    "param_bytes": None,
    "inference_us": None,
    "feature_ms": 42.8,
    "flash_kb": None,
    "ram_kb": None,
    "esp32_agreement": None,
}

if __name__ == "__main__":
    evaluate("decision_tree", make_model, kind="supervised", make_multiclass_model=make_multiclass_model,
             device=DEVICE, owner="Devon Biscardi", notes="TODO: describe your final settings")
