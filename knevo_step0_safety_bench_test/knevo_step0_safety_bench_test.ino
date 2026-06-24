/*
  KNEVO STEP 0 — SAFETY PATH BENCH TEST
  ----------------------------------------------------------------
  Purpose: verify the emergency freeze path (angle-error check +
  freezeSystem) actually halts the motor and turns the built-in RGB
  LED solid BLUE when a large commanded-vs-actual angle gap is forced
  — BEFORE anything new is built on top of the control firmware.

  Motor functions are copied as-is from the already tested
  moresmoother.ino (sendCmd, motorOn, sendPositionFast,
  readActualAngleQuick). The buzzer is replaced with the built-in RGB
  LED per your request: OFF while holding normally, solid BLUE once
  frozen. The only new logic here is holding a fixed target angle
  instead of running the gait FSM, so a failure means the existing
  safety code itself needs attention, not new integration code.

  NOTE: production's checkScenario2Emergency() only runs this check
  while supportMode == 1 (stance phase). There is no gait phase
  running here, so that gate is removed for this test — the threshold
  check runs unconditionally. Step 2+ restores the real gating.

  This sketch talks you through the test live over Serial — open the
  Serial Monitor at 115200 baud and just follow what it prints.
*/

#include <math.h>

#define MOTOR_RX_PIN 4
#define MOTOR_TX_PIN 5

HardwareSerial MotorSerial(1);

// Built-in RGB LED. Most ESP32-S3 dev boards use GPIO48.
// Change this if your board uses a different RGB LED pin.
#ifndef BUILTIN_RGB_LED_PIN
  #ifdef RGB_BUILTIN
    #define BUILTIN_RGB_LED_PIN RGB_BUILTIN
  #else
    #define BUILTIN_RGB_LED_PIN 48
  #endif
#endif
#define RGB_LED_BRIGHTNESS 35

// Same motor calibration constant as moresmoother.ino.
const float MOTOR_SCALE = 3722.0f;

// ----- TEST SETTINGS -----
const float HOLD_TARGET_DEG     = 20.0f;  // fixed angle to hold during the test
const float COLLAPSE_ERROR_DEG  = 20.0f;  // same threshold as production
const unsigned long CONTROL_MS  = 10;     // 100Hz hold streaming, same as production
const unsigned long FEEDBACK_MS = 250;    // 4Hz actual-angle read, same as production
const unsigned long PRINT_MS    = 1000;   // 1Hz status print, test-only
const int COUNTDOWN_SECONDS     = 5;

// ----- LOGIC-ONLY SANITY CHECK -----
// Set true to fake a bad reading and confirm the freeze branch + LED
// actually fire from a known bad number, independent of whether the
// real hardware can physically produce that number yet. Set back to
// false before doing the real mechanical-block test.
const bool USE_FAKE_ANGLE_TEST = true;

uint8_t cmdOn[5] = {0x3E, 0x88, 0x01, 0x00, 0xC7};

bool systemFrozen = false;
bool frozenMessagePrinted = false;
float actualAngle = 0.0f;
float commandedAngle = HOLD_TARGET_DEG;
unsigned long lastControlTime = 0;
unsigned long lastFeedbackTime = 0;
unsigned long lastPrintTime = 0;

// ============================================================
// BUILT-IN RGB LED
// ============================================================
void setBuiltinLedRGB(uint8_t r, uint8_t g, uint8_t b) {
  neopixelWrite(BUILTIN_RGB_LED_PIN, r, g, b);
}

void ledOff() {
  setBuiltinLedRGB(0, 0, 0);
}

void ledFrozenBlue() {
  setBuiltinLedRGB(0, 0, RGB_LED_BRIGHTNESS);
}

// ============================================================
// MOTOR FUNCTIONS — copied as-is from moresmoother.ino
// ============================================================
void sendCmd(uint8_t* cmd, int len) {
  while (MotorSerial.available()) MotorSerial.read();
  MotorSerial.write(cmd, len);
  MotorSerial.flush();
}

bool motorOn() {
  for (int attempt = 0; attempt < 20; attempt++) {
    while (MotorSerial.available()) MotorSerial.read();
    sendCmd(cmdOn, 5);
    delay(200);

    uint8_t buf[16];
    int n = 0;
    unsigned long t = millis();

    while (n < 16 && millis() - t < 200) {
      if (MotorSerial.available()) {
        buf[n++] = MotorSerial.read();
        t = millis();
      }
    }

    for (int i = 0; i < n; i++) {
      if (buf[i] == 0x3E) return true;
    }

    delay(100);
  }
  return false;
}

void sendPositionFast(float outputDeg) {
  int32_t angleRaw = (int32_t)(outputDeg * MOTOR_SCALE);

  uint8_t buf[14];
  buf[0]  = 0x3E;
  buf[1]  = 0xA3;
  buf[2]  = 0x01;
  buf[3]  = 0x08;
  buf[4]  = (0x3E + 0xA3 + 0x01 + 0x08) & 0xFF;
  buf[5]  = angleRaw & 0xFF;
  buf[6]  = (angleRaw >> 8)  & 0xFF;
  buf[7]  = (angleRaw >> 16) & 0xFF;
  buf[8]  = (angleRaw >> 24) & 0xFF;
  buf[9]  = 0x00;
  buf[10] = 0x00;
  buf[11] = 0x00;
  buf[12] = 0x00;

  uint8_t sum = 0;
  for (int i = 5; i < 13; i++) sum += buf[i];
  buf[13] = sum & 0xFF;

  // No delay here, same as production — avoids stop-go behavior.
  sendCmd(buf, 14);
}

float readActualAngleQuick() {
  uint8_t angleCmd[5] = {0x3E, 0x92, 0x01, 0x00, 0xD1};

  while (MotorSerial.available()) MotorSerial.read();
  MotorSerial.write(angleCmd, 5);
  MotorSerial.flush();

  uint8_t rx[32];
  int n = 0;
  unsigned long t = millis();

  // Short timeout so feedback never blocks the control loop.
  while (n < 32 && millis() - t < 30) {
    if (MotorSerial.available()) {
      rx[n++] = MotorSerial.read();
      t = millis();
    }
  }

  for (int i = 0; i < n - 8; i++) {
    if (rx[i] == 0x3E && rx[i + 1] == 0x92) {
      uint32_t rawUnsigned =
        ((uint32_t)rx[i + 5]) |
        ((uint32_t)rx[i + 6] << 8) |
        ((uint32_t)rx[i + 7] << 16) |
        ((uint32_t)rx[i + 8] << 24);

      int32_t rawAngle = (int32_t)rawUnsigned;
      return (float)rawAngle / MOTOR_SCALE;
    }
  }

  return actualAngle;  // stale-but-safe fallback, same as production
}

// ============================================================
// FREEZE
// ============================================================
void freezeSystem() {
  systemFrozen = true;
  ledFrozenBlue();
}

// ============================================================
// FUNCTION UNDER TEST
// Same threshold logic as production's checkScenario2Emergency(),
// minus the supportMode==1 gate (see note at top of file).
// ============================================================
void checkScenario2Emergency() {
  if (actualAngle > commandedAngle + COLLAPSE_ERROR_DEG) {
    freezeSystem();
  }
}

// ============================================================
// SETUP — talks you through the procedure step by step
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(3000);
  ledOff();

  MotorSerial.begin(115200, SERIAL_8N1, MOTOR_RX_PIN, MOTOR_TX_PIN);

  Serial.println();
  Serial.println("=========================================================");
  Serial.println(" KNEVO STEP 0: SAFETY PATH BENCH TEST");
  Serial.println("=========================================================");
  Serial.println("BEFORE YOU CONTINUE:");
  Serial.println("  - The joint must be OFF a person right now.");
  Serial.println("  - Either clamp it in a fixture, or have someone ready");
  Serial.println("    to firmly hand-resist it.");
  Serial.println();

  Serial.print("Connecting to motor... ");
  if (motorOn()) {
    Serial.println("OK (Motor ON confirmed)");
  } else {
    Serial.println("FAILED");
    Serial.println("Check wiring/power. Test cannot continue safely. Halting.");
    while (1) { delay(1000); }
  }

  sendPositionFast(commandedAngle);
  delay(500);
  actualAngle = readActualAngleQuick();

  Serial.println();
  Serial.print("The motor will start holding "); Serial.print(HOLD_TARGET_DEG);
  Serial.println(" deg in:");
  for (int s = COUNTDOWN_SECONDS; s > 0; s--) {
    Serial.print("  "); Serial.print(s); Serial.println("...");
    delay(1000);
  }

  Serial.println();
  Serial.println(">>> HOLDING NOW. <<<");
  Serial.print("Push/resist the joint until ActualDeg is more than ");
  Serial.print(COLLAPSE_ERROR_DEG, 1);
  Serial.println(" deg above CommandedDeg.");
  Serial.println("Expected result: LED turns solid BLUE, motor stops moving.");
  Serial.println();
  Serial.println("Live status below, once per second:");
  Serial.println("---------------------------------------------------------");
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  unsigned long now = millis();

  // 100Hz: hold position unless frozen.
  if (now - lastControlTime >= CONTROL_MS) {
    lastControlTime = now;
    if (!systemFrozen) {
      sendPositionFast(commandedAngle);
    }
  }

  // 4Hz: read actual angle + run the safety check, same cadence as production.
  if (now - lastFeedbackTime >= FEEDBACK_MS) {
    lastFeedbackTime = now;
    actualAngle = readActualAngleQuick();

    if (USE_FAKE_ANGLE_TEST) {
      // Override with a deliberately bad number to test the logic alone.
      actualAngle = commandedAngle + COLLAPSE_ERROR_DEG + 5.0f;
    }

    if (!systemFrozen) {
      checkScenario2Emergency();
    }
  }

  // 1Hz: human-readable status, plus a one-time freeze announcement.
  if (now - lastPrintTime >= PRINT_MS) {
    lastPrintTime = now;
    float error = actualAngle - commandedAngle;

    if (!systemFrozen) {
      Serial.print("[HOLDING] Commanded=");
      Serial.print(commandedAngle, 1);
      Serial.print(" deg | Actual=");
      Serial.print(actualAngle, 1);
      Serial.print(" deg | Error=");
      Serial.print(error, 1);
      Serial.print(" deg | need > ");
      Serial.print(COLLAPSE_ERROR_DEG, 1);
      Serial.println(" deg to trigger -> still safe");
    } else {
      if (!frozenMessagePrinted) {
        frozenMessagePrinted = true;
        Serial.println();
        Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        Serial.println("FREEZE TRIGGERED");
        Serial.print("  Error reached "); Serial.print(error, 1);
        Serial.print(" deg, past the "); Serial.print(COLLAPSE_ERROR_DEG, 1);
        Serial.println(" deg threshold.");
        Serial.println("  LED should now be solid BLUE.");
        Serial.println("  Motor has stopped accepting new position commands.");
        Serial.println("  Power-cycle the board to reset and run the test again.");
        Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        Serial.println();
      } else {
        Serial.println("[FROZEN] Waiting for power-cycle to reset...");
      }
    }
  }
}
