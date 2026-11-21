// model.h: Devon's decision tree (exported to plain if/else in decision_tree_model.h by
// models/export_decision_tree.py), plugged into the shared fault detector.
#pragma once
#include "decision_tree_model.h"

#define MODEL_NAME        "decision tree (Devon, exported)"
#define MODEL_THRESHOLD   0.5000001f      // FAULT when probability > 0.5, exactly like scikit-learn's predict()
#define MODEL_PARAM_BYTES DT_PARAM_BYTES

static inline float model_fault_score(const float *x) {
  return dt_fault_probability(x);
}
