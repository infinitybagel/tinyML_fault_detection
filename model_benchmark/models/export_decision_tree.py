"""Export Devon's decision tree (models/decision_tree.py) to C for the ESP32.

    python models/export_decision_tree.py

Trains his model on the shared split exactly as the benchmark does (sessions 1-4 of each training
condition), then writes esp32/fault_detector_decision_tree/decision_tree_model.h as plain if/else:
  - every split compares the float feature against the tree's double threshold, the same way
    scikit-learn does (it casts features to float32 before comparing), so the results match exactly;
  - each leaf returns the tree's fault probability; FAULT when it is > 0.5, like predict().
It also checks the generated logic against predict() on every recorded window.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))
from benchmark import FEATURES, load_features          # noqa: E402
from benchmark.evaluate import _fit_round, _round_split  # noqa: E402
import decision_tree                                      # noqa: E402

OUT = os.path.join(ROOT, "esp32", "fault_detector_decision_tree", "decision_tree_model.h")

df = load_features()
train, _ = _round_split(df, 4)
search = _fit_round(decision_tree.make_model, df, train, "supervised")
clf = search.best_estimator_
t = clf.tree_
print("Chosen params:", search.best_params_)
print(f"Tree: depth {clf.get_depth()}, {clf.get_n_leaves()} leaves, {t.node_count} nodes")


def emit(node, depth):
    pad = "  " * depth
    if t.children_left[node] == -1:                       # leaf
        v = t.value[node][0]
        p_fault = float(v[1] / v.sum())
        return [f"{pad}return {p_fault:.9e}f;   // leaf {node}"]
    f, thr = int(t.feature[node]), float(t.threshold[node])
    lines = [f"{pad}if ((double)x[{f}] <= {thr:.17g}) {{   // {FEATURES[f]}"]
    lines += emit(t.children_left[node], depth + 1)
    lines += [f"{pad}}} else {{"]
    lines += emit(t.children_right[node], depth + 1)
    lines += [f"{pad}}}"]
    return lines


n_splits = int((t.children_left != -1).sum())
param_bytes = n_splits * (8 + 1) + clf.get_n_leaves() * 4   # double threshold + feature index per split, float per leaf
header = [
    "// Decision tree fan-fault detector (Devon Biscardi), exported by models/export_decision_tree.py.",
    f"// Params: {search.best_params_}",
    f"// depth {clf.get_depth()}, {clf.get_n_leaves()} leaves, {n_splits} splits. Input: the 9 shared features in order:",
    "//   " + ", ".join(FEATURES),
    "// Returns the fault probability of the leaf reached; FAULT when > 0.5 (scikit-learn predict()).",
    "#pragma once",
    "",
    f"#define DT_PARAM_BYTES {param_bytes}",
    "",
    "static inline float dt_fault_probability(const float *x) {",
    *emit(0, 1),
    "}",
    "",
]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(header))
print(f"Wrote {OUT} ({param_bytes} bytes of tree parameters)")

# check the generated logic against scikit-learn on every recorded window
X = df[FEATURES].to_numpy(dtype=np.float32)


def walk(x):
    node = 0
    while t.children_left[node] != -1:
        node = t.children_left[node] if float(x[t.feature[node]]) <= t.threshold[node] else t.children_right[node]
    v = t.value[node][0]
    return v[1] / v.sum()


ours = np.array([walk(x) for x in X]) > 0.5
print(f"Exported logic matches predict() on {np.mean(ours == (clf.predict(X) == 1)) * 100:.2f}% of {len(X)} windows")
