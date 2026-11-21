"""Score an exported isolation forest (isolation_forest_model.h) on every recorded window, in Python.

    python tools/isolation_forest_offline_check.py
    python tools/isolation_forest_offline_check.py path/to/isolation_forest_model.h

Reproduces the C scoring exactly, then reports detection per condition and how the threshold trades
false alarms against missed faults. Uses the shared benchmark features.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model_benchmark"))
sys.path.insert(0, os.path.join(ROOT, "model_benchmark", "models"))
from benchmark import FEATURES, load_features        # noqa: E402
from isolation_forest_exported import ExportedIsolationForest   # noqa: E402

HEADER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "model_benchmark", "esp32", "fault_detector_isolation_forest", "isolation_forest_model.h")


def main():
    model = ExportedIsolationForest(HEADER)
    print(f"{model.n_trees} trees, {len(model.nodes)} nodes, max_samples {model.max_samples}, threshold {model.threshold}")
    df = load_features()
    df["score"] = model.score_samples(df[FEATURES].to_numpy(dtype=np.float32))
    df["fault"] = (df.score < model.threshold).astype(int)
    pd.set_option("display.width", 200)
    print("\n% of windows flagged as fault, per session (0-4):")
    print((df.pivot_table(index="label", columns="session_idx", values="fault", aggfunc="mean") * 100).round(1).to_string())

    te = df[(df.session_idx == 4) & (df.label != "heldout")]
    y = (te.label != "normal").astype(int)
    print(f"\nShared split ROC-AUC: {roc_auc_score(y, -te.score):.3f}")
    nrm, flt = df[df.label == "normal"].score, df[df.label != "normal"].score
    for thr in [model.threshold, 0.0, -0.02, -0.04]:
        print(f"threshold {thr:+.4f}: normal flagged {(nrm < thr).mean() * 100:5.1f}%   "
              f"faults caught {(flt < thr).mean() * 100:5.1f}%")


if __name__ == "__main__":
    main()
