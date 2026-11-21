// Temporary fan-speed tuner: same pins as the logger/detector (PWM GPIO 19 at 25 kHz, tach GPIO 18).
// Send "D<0-255>\n" to set the duty; prints "R,<duty>,<rpm>" once per second.
const int TACH_PIN = 18;
const int FAN_PWM_PIN = 19;
volatile uint32_t tachPulses = 0, lastPulseUs = 0;
int duty = 204;
uint32_t lastRpmUs = 0;

void IRAM_ATTR onTachPulse() {
  uint32_t now = micros();
  if (now - lastPulseUs > 5000) { tachPulses++; lastPulseUs = now; }
}

void setDuty(int d) {
  duty = constrain(d, 0, 255);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(FAN_PWM_PIN, duty);
#else
  ledcWrite(0, duty);
#endif
}

void setup() {
  Serial.begin(921600);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(FAN_PWM_PIN, 25000, 8);
#else
  ledcSetup(0, 25000, 8);
  ledcAttachPin(FAN_PWM_PIN, 0);
#endif
  setDuty(duty);
  pinMode(TACH_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(TACH_PIN), onTachPulse, FALLING);
  lastRpmUs = micros();
}

void loop() {
  if (Serial.available()) {
    String s = Serial.readStringUntil('\n');
    s.trim();
    if (s.length() > 1 && s[0] == 'D') setDuty(s.substring(1).toInt());
  }
  uint32_t now = micros();
  if (now - lastRpmUs >= 1000000) {
    uint32_t p;
    noInterrupts(); p = tachPulses; tachPulses = 0; interrupts();
    float rpm = p * 30.0f * 1e6f / (float)(now - lastRpmUs);
    lastRpmUs = now;
    Serial.printf("R,%d,%.0f\n", duty, rpm);
  }
}
