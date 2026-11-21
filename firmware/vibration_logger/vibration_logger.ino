// ESP32 + MPU6050 vibration logger
// Reads the accelerometer at 1 kHz and streams raw readings over USB serial.
// Send 's' to start recording and 'x' to stop (record_session.py does this for you).
//
// Uses ONLY the board side with 3V3, GND, and GPIO 15-23.
//
// Wiring:
//   MPU6050 VCC  -> 3V3
//   MPU6050 GND  -> GND
//   MPU6050 SDA  -> GPIO 21
//   MPU6050 SCL  -> GPIO 22
//   Fan pin 1 (ground) and the 12V supply's ground -> GND
//   Fan pin 2 (+12V) -> 12V supply + (NOT to the ESP32)
//   Fan pin 3 (tach / RPM)  -> GPIO 18, plus a 10k resistor from GPIO 18 to 3V3 (never to 12V)
//   Fan pin 4 (PWM, 4-pin fans only) -> 1k resistor -> GPIO 19
//
// Tip: there is only one GND and one 3V3 pin on this side. Run each to a breadboard rail
// (or a free row) and connect everything that needs GND or 3V3 to that rail.
//
// No extra libraries needed: Wire is built into the ESP32 board package.

#include <Wire.h>

const uint8_t  MPU_ADDR         = 0x68;
const int      SDA_PIN          = 21;
const int      SCL_PIN          = 22;
const uint32_t SAMPLE_PERIOD_US = 1000;   // 1000 us = 1 kHz

// Fan RPM from the tach wire. Set to -1 if you haven't wired it.
const int TACH_PIN = 18;

// Fan speed control for 4-pin PWM fans. Set to -1 for a 3-pin fan (like the F12 TC).
const int FAN_PWM_PIN = 19;
const int FAN_DUTY    = 204;   // 0-255. 204 = 80% of full speed. Use the SAME value for every recording.

// Accelerometer range. +/-8 g so strong faults (e.g. imbalance at ~2000 RPM) don't clip.
// If you change it, also change COUNTS_PER_G in record_session.py and SCALE_MS2 in features.py.
const uint8_t ACCEL_CONFIG_VAL = 0x10;    // 0x00 = +/-2 g, 0x08 = +/-4 g, 0x10 = +/-8 g, 0x18 = +/-16 g
const int     ACCEL_RANGE_G    = 8;
const int     COUNTS_PER_G     = 4096;    // 16384 / 8192 / 4096 / 2048 for 2 / 4 / 8 / 16 g

volatile uint32_t tachPulses  = 0;
volatile uint32_t lastPulseUs = 0;

bool     recording    = false;
bool     rpmOnly      = false;   // 'r' test mode: print only RPM, no accelerometer lines
uint32_t nextSampleUs = 0;
uint32_t lastRpmUs    = 0;

void IRAM_ATTR onTachPulse() {
  uint32_t now = micros();
  if (now - lastPulseUs > 5000) {   // ignore electrical noise (real pulses are >20 ms apart)
    tachPulses++;
    lastPulseUs = now;
  }
}

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

uint8_t readReg(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0xFF;
}

bool readAccel(int16_t &ax, int16_t &ay, int16_t &az) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);                                  // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)6) != 6) return false;
  uint8_t b[6];
  for (int i = 0; i < 6; i++) b[i] = Wire.read();
  ax = (int16_t)((b[0] << 8) | b[1]);
  ay = (int16_t)((b[2] << 8) | b[3]);
  az = (int16_t)((b[4] << 8) | b[5]);
  return true;
}

void setupFanPwm() {
  if (FAN_PWM_PIN < 0) return;
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(FAN_PWM_PIN, 25000, 8);   // 25 kHz, 8-bit (0-255)
  ledcWrite(FAN_PWM_PIN, FAN_DUTY);
#else
  ledcSetup(0, 25000, 8);              // older ESP32 board package (2.x)
  ledcAttachPin(FAN_PWM_PIN, 0);
  ledcWrite(0, FAN_DUTY);
#endif
}

void printStatus() {
  uint8_t who = readReg(0x75);    // WHO_AM_I
  Serial.printf("# WHO_AM_I = 0x%02X\n", who);
  if (who == 0xFF || who == 0x00) {
    Serial.println("# ERROR: MPU6050 not found. Check wiring.");
  } else if (who != 0x68) {
    Serial.println("# Note: unexpected ID (common on clone boards). Continuing anyway.");
  }
  Serial.printf("# Pins: SDA=%d SCL=%d TACH=%d FAN_PWM=%d (duty %d/255)\n",
                SDA_PIN, SCL_PIN, TACH_PIN, FAN_PWM_PIN, FAN_DUTY);
  Serial.printf("# Accel range: +/-%d g (%d counts per g), ACCEL_CONFIG = 0x%02X\n",
                ACCEL_RANGE_G, COUNTS_PER_G, readReg(0x1C));
  Serial.println("# Commands: 's' record, 'r' RPM-only test, 'i' this info, 'x' stop.");
}

void setup() {
  Serial.setTxBufferSize(4096);   // must come before Serial.begin
  Serial.begin(921600);
  delay(500);
  Serial.println();               // end the garbled boot-message line so our first line shows

  setupFanPwm();

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);          // fast I2C so a 1 kHz read rate is easy

  writeReg(0x6B, 0x01);   // PWR_MGMT_1: wake up, use gyro clock
  delay(100);
  writeReg(0x1A, 0x01);   // CONFIG: low-pass filter, accel bandwidth ~184 Hz
  writeReg(0x19, 0x00);   // SMPLRT_DIV: 1 kHz output rate
  writeReg(0x1C, ACCEL_CONFIG_VAL);   // ACCEL_CONFIG: accelerometer range (see ACCEL_RANGE_G)

  if (TACH_PIN >= 0) {
    pinMode(TACH_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(TACH_PIN), onTachPulse, FALLING);
  }

  printStatus();
  Serial.println("# Ready.");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 's' || c == 'r') {
      recording    = true;
      rpmOnly      = (c == 'r');
      nextSampleUs = micros();
      lastRpmUs    = nextSampleUs;
      noInterrupts(); tachPulses = 0; interrupts();
      Serial.println(rpmOnly ? "# RPM-only test (send 'x' to stop)" : "# start");
    } else if (c == 'x') {
      recording = false;
      rpmOnly   = false;
      Serial.println("# stop");
    } else if (c == 'i') {
      printStatus();
    }
  }
  if (!recording) return;

  uint32_t now = micros();

  if (!rpmOnly && (int32_t)(now - nextSampleUs) >= 0) {
    if (now - nextSampleUs > 10000) nextSampleUs = now;   // fell far behind: resync
    nextSampleUs += SAMPLE_PERIOD_US;

    int16_t ax, ay, az;
    if (readAccel(ax, ay, az)) {
      Serial.printf("%lu,%d,%d,%d\n", (unsigned long)now, ax, ay, az);
    }
  }

  if (TACH_PIN >= 0 && now - lastRpmUs >= 1000000) {   // report RPM once per second
    uint32_t pulses;
    noInterrupts(); pulses = tachPulses; tachPulses = 0; interrupts();
    float rpm = pulses * 30.0f * 1e6f / (float)(now - lastRpmUs);   // 2 pulses per revolution
    lastRpmUs = now;
    Serial.printf("# rpm,%.0f\n", rpm);
  }
}
