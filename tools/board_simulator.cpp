#include <cstdio>
#include <cstdint>
#include <cmath>
#include <fcntl.h>
#include <io.h>
#include "fan_features.h"
#include "model.h"
int main() {
  _setmode(_fileno(stdin), _O_BINARY);
  ff_init();
  int16_t raw[3000]; float mag[1000], f[9];
  const float SCALE = 9.80665f / 4096.0f;
  while (fread(raw, sizeof(int16_t), 3000, stdin) == 3000) {
    for (int n = 0; n < 1000; n++) {
      float ax = raw[3*n], ay = raw[3*n+1], az = raw[3*n+2];
      mag[n] = sqrtf(ax*ax + ay*ay + az*az) * SCALE;
    }
    ff_extract(mag, f);
    float p = model_fault_score(f);
    for (int i = 0; i < 9; i++) printf("%.9g,", f[i]);
    printf("%.9g\n", p);
  }
}
