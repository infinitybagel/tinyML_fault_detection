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

# Search space
#GRID = {
#    "criterion" : ["gini", "entropy"], 
#    "max_depth": [2, 3, 4, 6, 8], 
#    "min_samples_split" : [2, 5, 10],
#    "min_samples_leaf": [1, 5, 10, 20]
#}

#Parameter	Role	Tuning Guidance
#max_depth	Caps the maximum number of levels in the tree.	Start small (3 to 10). Include None only to verify whether an unconstrained tree actually outperforms pruned ones.
#min_samples_leaf	Minimum samples required to be at a final leaf node.	One of the most effective smoothing parameters. Higher values (5–20) prevent the model from creating leaves for single outliers.
#min_samples_split	Minimum samples required to split an internal node.	Keep proportional to your dataset size (e.g., 2 to 20 for small/medium datasets, or fractions like 0.01, 0.05 for large ones).
#ccp_alpha	Complexity parameter used for Minimal Cost-Complexity Pruning.	0.0 means no pruning. Small positive values (0.001 to 0.05) trim weak branches after the tree is built.
#max_features	Number of features considered at each split.	Useful if you have high-dimensional data (many columns); 'sqrt' or 'log2' reduces variance.
#criterion	Function to measure split quality.	'gini' is faster; 'entropy' (or 'log_loss') sometimes produces slightly more balanced trees, though results are usually very similar.

GRID = {
    "max_depth": [None, 3, 4, 5, 6, 7, 8],
    "max_leaf_nodes": [8, 16, 32, 48, 64],
    "min_samples_leaf": [5, 10, 20],
    "min_samples_split": [2, 5, 10, 20],
    "criterion": ['gini', 'entropy'],  
    "ccp_alpha": [0.0, 0.001, 0.005, 0.01]
}

def make_model():
    # Trees don't need standardized features.
    return tuned(DecisionTreeClassifier(class_weight="balanced", random_state=0), GRID, scoring="f1")


def make_multiclass_model():
    return tuned(DecisionTreeClassifier(class_weight="balanced", random_state=0), GRID, scoring="f1_macro")

# TODO: fill in after the ESP32 tests (leave None until measured)
DEVICE = {
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
