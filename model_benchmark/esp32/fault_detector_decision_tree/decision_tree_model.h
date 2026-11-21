// Decision tree fan-fault detector (Devon Biscardi), exported by models/export_decision_tree.py.
// Params: {'ccp_alpha': 0.0, 'criterion': 'gini', 'max_depth': None, 'max_leaf_nodes': 8, 'min_samples_leaf': 5, 'min_samples_split': 2}
// depth 2, 4 leaves, 3 splits. Input: the 9 shared features in order:
//   rms, peak, std, kurtosis, crest_factor, band_5_25hz, band_25_75hz, band_75_200hz, band_200_500hz
// Returns the fault probability of the leaf reached; FAULT when > 0.5 (scikit-learn predict()).
#pragma once

#define DT_PARAM_BYTES 43

static inline float dt_fault_probability(const float *x) {
  if ((double)x[8] <= 7.9758369793125894e-06) {   // band_200_500hz
    if ((double)x[7] <= 0.00013173507613828406) {   // band_75_200hz
      return 0.000000000e+00f;   // leaf 5
    } else {
      return 7.722385142e-02f;   // leaf 6
    }
  } else {
    if ((double)x[3] <= 1.7098789811134338) {   // kurtosis
      return 1.000000000e+00f;   // leaf 3
    } else {
      return 5.724637681e-01f;   // leaf 4
    }
  }
}
