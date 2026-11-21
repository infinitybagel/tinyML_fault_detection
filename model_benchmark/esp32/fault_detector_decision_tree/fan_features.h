// Feature extraction for the fan-fault detector. Plain C++ (no Arduino code), so it can also
// be compiled and tested on a PC. It reproduces features.py exactly:
//
//   input  : WINDOW_SIZE samples of the acceleration magnitude sqrt(ax^2 + ay^2 + az^2) in m/s^2
//   output : rms, peak, std, kurtosis, crest_factor,
//            band_5_25hz, band_25_75hz, band_75_200hz, band_200_500hz
//
// features.py details reproduced here:
//   rms      = sqrt(mean(x^2))                 (includes gravity, like features.py)
//   peak     = max |x|
//   std      = population standard deviation
//   kurtosis = excess kurtosis m4 / m2^2 - 3   (scipy.stats.kurtosis, fisher=True, bias=True)
//   crest    = peak / rms
//   band     = mean over FFT bins k in [lo, hi) of (2 * |DFT(x - mean)[k]| / N)^2
//
// With N = 1000 samples at 1000 Hz, DFT bin k is exactly k Hz. N is not a power of two, so a
// plain DFT is computed for the bins that are needed (5-499) using a cosine/sine lookup table.

#pragma once
#include <math.h>

#define FF_WINDOW_SIZE 1000
#define FF_N_FEATURES  9

static const int FF_BANDS[4][2] = { {5, 25}, {25, 75}, {75, 200}, {200, 500} };

static float ff_cos[FF_WINDOW_SIZE];
static float ff_sin[FF_WINDOW_SIZE];
static float ff_dev[FF_WINDOW_SIZE];   // x - mean

// Call once at startup to fill the lookup tables.
static inline void ff_init() {
  for (int n = 0; n < FF_WINDOW_SIZE; n++) {
    double a = 2.0 * M_PI * (double)n / (double)FF_WINDOW_SIZE;
    ff_cos[n] = (float)cos(a);
    ff_sin[n] = (float)sin(a);
  }
}

// x: FF_WINDOW_SIZE magnitude samples in m/s^2. out: FF_N_FEATURES values.
static inline void ff_extract(const float *x, float *out) {
  const int N = FF_WINDOW_SIZE;

  float sum = 0.0f, peak = 0.0f;
  for (int n = 0; n < N; n++) {
    sum += x[n];
    float a = fabsf(x[n]);
    if (a > peak) peak = a;
  }
  float mean = sum / N;

  float m2 = 0.0f, m4 = 0.0f;
  for (int n = 0; n < N; n++) {
    float d = x[n] - mean;
    ff_dev[n] = d;
    float d2 = d * d;
    m2 += d2;
    m4 += d2 * d2;
  }
  m2 /= N;
  m4 /= N;

  float rms   = sqrtf(mean * mean + m2);          // = sqrt(mean(x^2)), computed more accurately
  float std   = sqrtf(m2);
  float kurt  = (m2 > 0.0f) ? m4 / (m2 * m2) - 3.0f : 0.0f;
  float crest = (rms > 0.0f) ? peak / rms : 0.0f;

  out[0] = rms;
  out[1] = peak;
  out[2] = std;
  out[3] = kurt;
  out[4] = crest;

  const float scale = 2.0f / N;
  for (int b = 0; b < 4; b++) {
    int lo = FF_BANDS[b][0], hi = FF_BANDS[b][1];
    float acc = 0.0f;
    for (int k = lo; k < hi; k++) {
      float re = 0.0f, im = 0.0f;
      int idx = 0;                                 // (k * n) mod N
      for (int n = 0; n < N; n++) {
        re += ff_dev[n] * ff_cos[idx];
        im -= ff_dev[n] * ff_sin[idx];
        idx += k;
        if (idx >= N) idx -= N;
      }
      float mag = scale * sqrtf(re * re + im * im);
      acc += mag * mag;
    }
    out[5 + b] = acc / (hi - lo);
  }
}
