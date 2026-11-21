"""Isolation forest (Angelina Guasti): her EXPORTED model, as it runs on the ESP32.

    python models/isolation_forest_exported.py

This does not retrain anything. It reads the 20 trees from
esp32/fault_detector_isolation_forest/isolation_forest_model.h and scores windows exactly the way
her C code does (she verified it against scikit-learn), so the benchmark metrics and the ESP32
verify step both describe the model that is actually deployed.

Because the model is already trained, only the shared split is reported (rounds=False): retraining-based
5-round scores would need her training script. If her training used any session-5 normal windows,
the shared-split false alarm rate is optimistic: check with Angelina.
"""

import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from benchmark import evaluate   # noqa: E402

HEADER = os.path.join(ROOT, "esp32", "fault_detector_isolation_forest", "isolation_forest_model.h")


class ExportedIsolationForest:
    """scikit-learn-style wrapper around the exported trees (outlier convention: -1 = anomaly)."""

    def __init__(self, header=HEADER):
        text = open(header).read()
        self.n_trees = int(re.search(r"IF_TREE_COUNT = (\d+)", text).group(1))
        self.max_samples = int(re.search(r"IF_MAX_SAMPLES = (\d+)", text).group(1))
        self.offset = float(re.search(r"IF_OFFSET = ([-\d.e+]+)", text).group(1))
        self.threshold = float(re.search(r"IF_THRESHOLD = ([-\d.e+]+)", text).group(1))
        self.tree_offsets = [int(v) for v in
                             re.search(r"IF_TREE_OFFSETS\[\] PROGMEM = \{([^}]*)\}", text).group(1).split(",")]
        self.nodes = [(int(a), int(b), int(c), int(d), float(e)) for a, b, c, d, e in
                      re.findall(r"\{(-?\d+), (-?\d+), (-?\d+), (\d+), ([-\d.e+]+)\}", text)]
        self.param_bytes = 16 * len(self.nodes) + 2 * len(self.tree_offsets)   # IFNode is 16 bytes on the ESP32

    @staticmethod
    def _c(n):
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        return 2.0 * (math.log(n - 1.0) + 0.5772156649015329) - 2.0 * (n - 1.0) / n

    def _score_one(self, x):
        total = 0.0
        for t in range(self.n_trees):
            i, depth = 0, 0
            while True:
                left, right, feat, n, thr = self.nodes[self.tree_offsets[t] + i]
                if left == -1:
                    total += depth + self._c(n)
                    break
                i = left if float(np.float32(x[feat])) <= thr else right   # board passes float32 features
                depth += 1
        return -2.0 ** (-total / (self.n_trees * self._c(self.max_samples))) - self.offset

    def fit(self, X, y=None):
        return self                                  # already trained; nothing to fit

    def score_samples(self, X):
        return np.array([self._score_one(x) for x in np.asarray(X)])

    def decision_function(self, X):
        """Negative = anomaly, using her threshold (her C code: anomaly if score < IF_THRESHOLD)."""
        return self.score_samples(X) - self.threshold

    def predict(self, X):
        return np.where(self.decision_function(X) < 0, -1, 1)


def make_model():
    return ExportedIsolationForest()


DEVICE = {                       # TODO (Angelina): fill in after esp32_test.py verify on the board
    "param_bytes": ExportedIsolationForest().param_bytes,
    "inference_us": None,
    "feature_ms": 42.8,
    "flash_kb": round((353271 - 326671) / 1000, 1),   # detector with this model vs. the logging sketch
    "ram_kb": round((47340 - 25284) / 1000, 1),
    "esp32_agreement": None,
}

if __name__ == "__main__":
    evaluate("isolation_forest_exported", make_model, kind="unsupervised", device=DEVICE, owner="Angelina Guasti",
             notes="Exported 20-tree isolation forest (max_samples 64), raw features, her threshold 0.0304",
             rounds=False)
