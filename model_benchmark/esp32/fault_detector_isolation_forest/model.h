// model.h: Angelina's isolation forest (20 trees, exported in isolation_forest_model.h), plugged into the
// shared fault detector. The scoring below is her verified isolationForestScore() from
// IsolationForestESP32_Corrected.ino, unchanged; only the wrapper at the bottom is new.
//
// Her model works on the raw 9 features (no scaling) and calls a window an anomaly when
// score < IF_THRESHOLD. The wrapper returns IF_THRESHOLD - score, so FAULT = value >= 0, which matches
// models/isolation_forest_exported.py in the benchmark.

#pragma once
#include <math.h>
#include <string.h>
#include "isolation_forest_model.h"

#define MODEL_NAME        "isolation forest (Angelina, exported)"
#define MODEL_THRESHOLD   0.0f
#define MODEL_PARAM_BYTES (sizeof(IF_NODES) + sizeof(IF_TREE_OFFSETS))

// Expected isolation path length for n samples
static inline double if_average_path_length(int n) {
  if (n <= 1) return 0.0;
  if (n == 2) return 1.0;
  const double eulerGamma = 0.5772156649015329;
  return 2.0 * (log((double)n - 1.0) + eulerGamma) - 2.0 * (n - 1.0) / n;
}

// scikit-learn decision_function: higher = more normal
static inline double if_score(const float *features) {
  double totalPathLength = 0.0;
  for (int t = 0; t < IF_TREE_COUNT; t++) {
    uint16_t treeOffset = pgm_read_word(&IF_TREE_OFFSETS[t]);
    int nodeIndex = 0;
    int depth = 0;
    while (true) {
      IFNode node;
      memcpy_P(&node, &IF_NODES[treeOffset + nodeIndex], sizeof(IFNode));
      if (node.left == -1) {
        totalPathLength += depth + if_average_path_length(node.n_samples);
        break;
      }
      nodeIndex = ((double)features[node.feature] <= node.threshold) ? node.left : node.right;
      depth++;
    }
  }
  double denominator = IF_TREE_COUNT * if_average_path_length(IF_MAX_SAMPLES);
  double rawScore = -pow(2.0, -totalPathLength / denominator);
  return rawScore - IF_OFFSET;
}

static inline float model_fault_score(const float *x) {
  return (float)(IF_THRESHOLD - if_score(x));      // >= 0 means FAULT (score below her threshold)
}
