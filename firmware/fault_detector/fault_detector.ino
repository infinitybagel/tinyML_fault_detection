// ESP32 + MPU6050 on-device fan-fault detector (logistic regression)
//
// Every 1-second window (1000 samples at 1 kHz) it computes the 9 features from features.py,
// runs the logistic regression from logreg_model.h, and prints the result. Wiring and sensor
// settings are the SAME as vibration_logger.ino (+/-8 g, ~184 Hz filter, fan duty 204/255).
//
// Serial commands (921600 baud; esp32_test.py does this for you):
//   'l'  live: classify the fan continuously, one line per window
//   'L'  live, and also print each window's 9 features (F,... lines) for troubleshooting
//   'x'  stop live mode
//   'v'  verify: then send 6000 bytes = 1000 samples of (ax, ay, az) as little-endian int16.
//        The board replies with its features and probability for that window, so Python can
//        check the C version gives the same answer as the notebook.
//   'i'  info (sensor check, settings, memory)
//
// Output lines:
//   W,<window>,<pred 0/1>,<p_fault>,<rpm>,<feature_us>,<inference_us>
//   V,<9 features>,<p_fault>,<feature_us>,<inference_us>,<checksum of the received bytes>
//   # ALERT: fault detected        (after ALERT_AFTER fault windows in a row)
//   # clear: back to normal        (after ALERT_AFTER normal windows in a row)
//
// While the features are computed (~tens of ms) sampling pauses, so live windows are not
// back to back. That doesn't matter for classification: each window is judged on its own.

#include <Wire.h>
#include "fan_features.h"
#include "logreg_model.h"

const uint8_t  MPU_ADDR         = 0x68;
const int      SDA_PIN          = 21;
const int      SCL_PIN          = 22;
const uint32_t SAMPLE_PERIOD_US = 1000;   // 1 kHz

const int TACH_PIN    = 18;
const int FAN_PWM_PIN = 19;
const int FAN_DUTY    = 204;              // must match the recordings (80%)

const uint8_t ACCEL_CONFIG_VAL = 0x10;    // +/-8 g, must match the recordings
const float   COUNTS_PER_G     = 4096.0f;
const float   SCALE_MS2        = 9.80665f / COUNTS_PER_G;

const int ALERT_AFTER = 3;                // consecutive windows before raising / clearing an alert
const int LED_PIN     = 2;                // on-board LED on most ESP32 DevKits (lights on alert)

volatile uint32_t tachPulses  = 0;
volatile uint32_t lastPulseUs = 0;

float    mag[FF_WINDOW_SIZE];
float    feats[FF_N_FEATURES];
bool     live = false;
bool     showFeatures = false;   // 'L': live mode that also prints each window's features
uint32_t windowCount = 0;
int      faultStreak = 0, normalStreak = 0;
bool     alertOn = false;
uint32_t lastRpmUs = 0;
float    lastRpm = 0;

void IRAM_ATTR onTachPulse() {
  uint32_t now = micros();
  if (now - lastPulseUs > 5000) { tachPulses++; lastPulseUs = now; }
}

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR); Wire.write(reg); Wire.write(val); Wire.endTransmission();
}

uint8_t readReg(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR); Wire.write(reg); Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0xFF;
}

bool readAccel(int16_t &ax, int16_t &ay, int16_t &az) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)6) != 6) return false;
  uint8_t b[6];
  for (int i = 0; i < 6; i++) b[i] = Wire.read();
  ax = (int16_t)((b[0] << 8) | b[1]);
  ay = (int16_t)((b[2] << 8) | b[3]);
  az = (int16_t)((b[4] << 8) | b[5]);
  return true;
}

// Print a whole line and wait until it has left the board. Without the flush, the ESP32 serial
// driver can hold the end of a line in its buffer until more output comes along.
void sendLine(const char *s) {
  Serial.println(s);
  Serial.flush();
}

float magnitude(int16_t ax, int16_t ay, int16_t az) {
  float x = ax, y = ay, z = az;
  return sqrtf(x * x + y * y + z * z) * SCALE_MS2;
}

void setupFanPwm() {
  if (FAN_PWM_PIN < 0) return;
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(FAN_PWM_PIN, 25000, 8);
  ledcWrite(FAN_PWM_PIN, FAN_DUTY);
#else
  ledcSetup(0, 25000, 8);
  ledcAttachPin(FAN_PWM_PIN, 0);
  ledcWrite(0, FAN_DUTY);
#endif
}

void updateRpm() {
  uint32_t now = micros();
  if (TACH_PIN < 0 || now - lastRpmUs < 1000000) return;
  uint32_t pulses;
  noInterrupts(); pulses = tachPulses; tachPulses = 0; interrupts();
  lastRpm = pulses * 30.0f * 1e6f / (float)(now - lastRpmUs);   // 2 pulses per revolution
  lastRpmUs = now;
}

// Features + model on the current mag[] buffer. Returns p_fault; fills the timing outputs.
float classify(uint32_t &featUs, uint32_t &inferUs) {
  uint32_t t0 = micros();
  ff_extract(mag, feats);
  uint32_t t1 = micros();
  float p = lr_predict_proba(feats);
  uint32_t t2 = micros();
  featUs = t1 - t0;
  inferUs = t2 - t1;
  return p;
}

void printInfo() {
  uint8_t who = readReg(0x75);
  Serial.printf("# WHO_AM_I = 0x%02X%s\n", who, (who == 0xFF || who == 0x00) ? "  ERROR: MPU6050 not found" : "");
  Serial.printf("# Accel config = 0x%02X (expect 0x%02X, +/-8 g), fan duty %d/255\n",
                readReg(0x1C), ACCEL_CONFIG_VAL, FAN_DUTY);
  Serial.printf("# Model: logistic regression, %d features, %u bytes of parameters\n",
                N_FEATURES, (unsigned)(sizeof(LR_MEAN) + sizeof(LR_SCALE) + sizeof(LR_COEF) + sizeof(LR_INTERCEPT)));
  Serial.printf("# Feature buffers: %u bytes. Free heap: %u bytes\n",
                (unsigned)(sizeof(mag) + sizeof(ff_cos) + sizeof(ff_sin) + sizeof(ff_dev)), (unsigned)ESP.getFreeHeap());
  Serial.println("# Commands: 'l' live, 'x' stop, 'v' verify one window, 'i' info");
  Serial.flush();
}

// Read exactly n bytes, giving up after timeoutMs without new data. Returns bytes read.
size_t readExact(uint8_t *buf, size_t n, uint32_t timeoutMs) {
  size_t got = 0;
  uint32_t last = millis();
  while (got < n && millis() - last < timeoutMs) {
    int avail = Serial.available();
    if (avail > 0) {
      got += Serial.read(buf + got, min((size_t)avail, n - got));
      last = millis();
    }
  }
  return got;
}

// 'v': receive one window of raw counts from the PC, classify it, reply with the features.
// The reply ends with a checksum of the received bytes, so the PC can resend if a byte was lost.
void verifyWindow() {
  static int16_t raw[FF_WINDOW_SIZE * 3];
  size_t got = readExact((uint8_t *)raw, sizeof(raw), 500);
  if (got != sizeof(raw)) {
    char err[80];
    snprintf(err, sizeof(err), "# ERROR: verify got %u of %u bytes", (unsigned)got, (unsigned)sizeof(raw));
    sendLine(err);
    return;
  }
  uint32_t checksum = 0;
  for (size_t i = 0; i < sizeof(raw); i++) checksum += ((uint8_t *)raw)[i];
  for (int n = 0; n < FF_WINDOW_SIZE; n++) mag[n] = magnitude(raw[3 * n], raw[3 * n + 1], raw[3 * n + 2]);
  uint32_t featUs, inferUs;
  float p = classify(featUs, inferUs);
  char line[256];
  int len = snprintf(line, sizeof(line), "V");
  for (int i = 0; i < FF_N_FEATURES; i++) len += snprintf(line + len, sizeof(line) - len, ",%.9g", feats[i]);
  snprintf(line + len, sizeof(line) - len, ",%.9g,%lu,%lu,%lu", p, (unsigned long)featUs, (unsigned long)inferUs,
           (unsigned long)checksum);
  sendLine(line);
}

// Live mode: record one 1-second window at 1 kHz. Returns false if stopped with 'x'.
bool recordWindow() {
  uint32_t next = micros();
  int n = 0;
  while (n < FF_WINDOW_SIZE) {
    if (Serial.available() && Serial.peek() == 'x') return false;
    uint32_t now = micros();
    if ((int32_t)(now - next) < 0) continue;
    if (now - next > 10000) next = now;   // fell far behind: resync
    next += SAMPLE_PERIOD_US;
    int16_t ax, ay, az;
    if (readAccel(ax, ay, az)) mag[n++] = magnitude(ax, ay, az);
    updateRpm();
  }
  return true;
}

void setup() {
  Serial.setRxBufferSize(8192);   // room for a whole 6000-byte verify window
  Serial.begin(921600);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  Serial.setRxFIFOFull(32);       // move received bytes out of the 128-byte hardware FIFO sooner
#endif
  delay(500);
  Serial.println();

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  setupFanPwm();

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  writeReg(0x6B, 0x01);   // wake up, gyro clock
  delay(100);
  writeReg(0x1A, 0x01);   // ~184 Hz low-pass filter
  writeReg(0x19, 0x00);   // 1 kHz output rate
  writeReg(0x1C, ACCEL_CONFIG_VAL);

  if (TACH_PIN >= 0) {
    pinMode(TACH_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(TACH_PIN), onTachPulse, FALLING);
  }

  ff_init();
  printInfo();
  Serial.println("# Ready.");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'l' || c == 'L') {
      live = true; showFeatures = (c == 'L'); windowCount = 0; faultStreak = normalStreak = 0;
      alertOn = false; digitalWrite(LED_PIN, LOW);
      lastRpmUs = micros(); noInterrupts(); tachPulses = 0; interrupts();
      sendLine("# live (send 'x' to stop)");
    } else if (c == 'x') {
      if (live) sendLine("# stop");
      live = false;
    } else if (c == 'v') {
      verifyWindow();
    } else if (c == 'i') {
      printInfo();
    }
  }
  if (!live) return;

  if (!recordWindow()) return;   // 'x' arrived; handled at the top of the next loop

  uint32_t featUs, inferUs;
  float p = classify(featUs, inferUs);
  int pred = p >= 0.5f ? 1 : 0;
  char line[96];
  snprintf(line, sizeof(line), "W,%lu,%d,%.4f,%.0f,%lu,%lu", (unsigned long)windowCount++, pred, p, lastRpm,
           (unsigned long)featUs, (unsigned long)inferUs);
  sendLine(line);
  if (showFeatures) {
    char fl[256];
    int len = snprintf(fl, sizeof(fl), "F");
    for (int i = 0; i < FF_N_FEATURES; i++) len += snprintf(fl + len, sizeof(fl) - len, ",%.9g", feats[i]);
    sendLine(fl);
  }

  if (pred) { faultStreak++; normalStreak = 0; } else { normalStreak++; faultStreak = 0; }
  if (!alertOn && faultStreak >= ALERT_AFTER) {
    alertOn = true; digitalWrite(LED_PIN, HIGH);
    sendLine("# ALERT: fault detected");
  } else if (alertOn && normalStreak >= ALERT_AFTER) {
    alertOn = false; digitalWrite(LED_PIN, LOW);
    sendLine("# clear: back to normal");
  }
}
