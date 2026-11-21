// model.h: Iritza's linear SVM (exported in linear_svm_model.h), plugged into the shared fault detector.
// Same shape as the logistic regression: standardize, weight, add; FAULT when the score >= 0.
#pragma once
#include "linear_svm_model.h"

#define MODEL_NAME        "linear SVM (Iritza, exported)"
#define MODEL_THRESHOLD   0.0f
#define MODEL_PARAM_BYTES (sizeof(SVM_MEAN) + sizeof(SVM_SCALE) + sizeof(SVM_COEF) + sizeof(SVM_INTERCEPT))

static inline float model_fault_score(const float *x) {
  return svm_predict_score(x);           // signed decision score, >= 0 means FAULT
}
