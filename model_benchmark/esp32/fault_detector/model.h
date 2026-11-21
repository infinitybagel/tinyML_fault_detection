// model.h: the ONLY file you change to run your model in fault_detector.ino.
//
// Provide three things:
//   MODEL_NAME         a short name printed at startup
//   MODEL_PARAM_BYTES  how many bytes your model's parameters take (for the comparison table)
//   float model_fault_score(const float *x)
//       x = the 9 shared features in this order:
//           rms, peak, std, kurtosis, crest_factor, band_5_25hz, band_25_75hz, band_75_200hz, band_200_500hz
//       return a score where >= MODEL_THRESHOLD means FAULT
//       (a probability with threshold 0.5, or 1.0 / 0.0 from a classifier, or your anomaly score)
//
// Example below: the logistic regression (Aaron). For your model, replace the include and the function,
// e.g. with the header micromlgen/emlearn writes for you, or your own if/else trees.

#pragma once
#include "logreg_model.h"

#define MODEL_NAME        "logistic regression"
#define MODEL_THRESHOLD   0.5f
#define MODEL_PARAM_BYTES (sizeof(LR_MEAN) + sizeof(LR_SCALE) + sizeof(LR_COEF) + sizeof(LR_INTERCEPT))

static inline float model_fault_score(const float *x) {
  return lr_predict_proba(x);          // probability of a fault
}

// --- Examples for the other models (sketches only; adapt to the code your converter writes) ---
//
// micromlgen (SVM / decision tree):   #include "model_svm.h"   Eloquent::ML::Port::SVM clf;
//   static inline float model_fault_score(const float *x) { return clf.predict((float *)x) == 1 ? 1.0f : 0.0f; }
//
// isolation forest (your own C): return the anomaly score with your chosen threshold as MODEL_THRESHOLD,
//   e.g. return -decision_function so that higher = more anomalous, matching the Python benchmark.
//   Remember: the Python pipeline standardizes features first; do the same in C if your model expects it.
