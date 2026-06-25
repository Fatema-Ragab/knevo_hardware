/*
  KNEVO STEP C — DUAL-CORE INTEGRATION
  ----------------------------------------------------------------
  Combines Step A (open-loop control, worn-ready) and Step B (real
  sensor + feature + DL pipeline, logging only) onto separate cores,
  so Step B's heavier workload and ~47ms inference call can NEVER
  stall Step A's motor timing.

  - CORE 1 (default Arduino loop()): Step A, UNCHANGED. Strict 10ms
    control loop, motor streaming, collapse-detection safety. This
    is what's actually worn tomorrow.
  - CORE 0 (new FreeRTOS task, coreBTask): Step B, UNCHANGED logic.
    Real IMU/FSR reads, the full causal 57-feature recipe, TFLite
    inference. Runs at its own pace. Logs to Serial only.

  NOT DATA-CONNECTED YET, deliberately: Core 0's tracker/CNN output
  does not feed Core 1's Gait_percent. Core 1 still runs on its own
  simulated input. This matches tonight's decision to keep the real
  sensor/DL pipeline observed-but-not-trusted for the live demo.

  NEW for this step (only new code beyond gluing A+B together):
    - A FreeRTOS mutex (serialMutex) guards Serial output, since both
      cores print independently and could otherwise interleave/garble
      each other's lines.
    - xTaskCreatePinnedToCore(..., 0) starts Core 0's task after all
      of Step B's setup (I2C, IMU init, FSR calibration, TFLite arena/
      model) has completed synchronously - the task only starts once
      everything it depends on already exists.
    - One small duplicate-definition cleanup: Step A and Step B both
      defined FSMState/fsmTable/LED helpers identically - kept ONE
      copy of each. No logic changed.

  WHAT TO ACTUALLY CHECK WHEN BENCH-TESTING THIS:
    1. Core 1's plotter/status output still looks exactly like Step A
       alone - same smoothness, same timing. If it doesn't, Core 0 is
       leaking into Core 1's timing somehow and that's the bug to chase.
    2. Core 0's [STATUS] line still updates and Hz/IMU/inference look
       the same as Step B alone.
    3. Serial output from both is readable, not garbled mid-line.
*/

#include <math.h>
#include <string.h>
#include <Wire.h>
#include <Arduino.h>
#include <esp_heap_caps.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "knevo_esp32_boosted_phase_cnn_6phase_model.h"
#include "knevo_esp32_boosted_phase_cnn_scaler.h"

SemaphoreHandle_t serialMutex;

// ============================================================
// SHARED: built-in RGB LED (was duplicated in A and B - one copy)
// ============================================================
#ifndef BUILTIN_RGB_LED_PIN
  #ifdef RGB_BUILTIN
    #define BUILTIN_RGB_LED_PIN RGB_BUILTIN
  #else
    #define BUILTIN_RGB_LED_PIN 48
  #endif
#endif
#define RGB_LED_BRIGHTNESS 35

void setBuiltinLedRGB(uint8_t r, uint8_t g, uint8_t b) { neopixelWrite(BUILTIN_RGB_LED_PIN, r, g, b); }
void ledOff()  { setBuiltinLedRGB(0, 0, 0); }
void ledFrozenBlue() { setBuiltinLedRGB(0, 0, RGB_LED_BRIGHTNESS); }  // Step A: freeze indicator
void ledGood() { setBuiltinLedRGB(0, RGB_LED_BRIGHTNESS, 0); }        // Step B: IMU init OK
void ledFail() { setBuiltinLedRGB(RGB_LED_BRIGHTNESS, 0, 0); }        // Step B: IMU init failed

// ============================================================
// SHARED: FSM phase table (was duplicated in A and B - one copy)
// ============================================================
struct FSMState { int id; float gpStart; float gpEnd; const char* name; };
const FSMState fsmTable[6] = {
  {1,  0.0f,   5.0f,  "HeelStrike"       },
  {2,  5.0f,  15.0f,  "LoadingResponse"  },
  {3, 15.0f,  50.0f,  "MidTerminalStance"},
  {4, 50.0f,  62.0f,  "PreSwingToeOff"   },
  {5, 62.0f,  87.0f,  "InitialMidSwing"  },
  {6, 87.0f, 100.0f,  "TerminalSwing"    }
};
float clampFloat(float x, float lo, float hi) { if (x < lo) return lo; if (x > hi) return hi; return x; }
int getFSMStatusCode(float gp) {
  gp = clampFloat(gp, 0.0f, 100.0f);
  for (int i = 0; i < 6; i++) {
    if (i < 5) { if (gp >= fsmTable[i].gpStart && gp < fsmTable[i].gpEnd) return fsmTable[i].id; }
    else       { if (gp >= fsmTable[i].gpStart && gp <= fsmTable[i].gpEnd) return fsmTable[i].id; }
  }
  return 6;
}
const char* fsmStatusName(int code) {
  for (int i = 0; i < 6; i++) if (fsmTable[i].id == code) return fsmTable[i].name;
  return "Unknown";
}
bool isStanceSupportState(int stateId) { return (stateId == 2 || stateId == 3); }


/* ============================================================
   ================   CORE 1 — STEP A (UNCHANGED)   ============
   ============================================================ */

#define MOTOR_RX_PIN 4
#define MOTOR_TX_PIN 5
HardwareSerial MotorSerial(1);

const float MOTOR_SCALE = 3722.0f;
float THERAPIST_MAX_ROM_DEG = 65.0f;
const float NORMAL_ROM_MIN_DEG = 5.0f;
const float NORMAL_ROM_MAX_DEG = 65.0f;

const unsigned long CONTROL_MS  = 10;
const unsigned long PLOT_MS     = 50;
const unsigned long FEEDBACK_MS = 250;

const float LOOKAHEAD_PERCENT = 4.0f;
const float COMMAND_SMOOTH_ALPHA = 0.12f;
// Combined bridge + dataset speed table, ordered slowest -> fastest by
// actual cycle length (not by mph label - 1.25mph and 1.50mph swap order
// here because of noise in the cycle-length estimate, flagged earlier).
// Runtime-adjustable: type the preset number + Enter in Serial Monitor.
struct SpeedPreset { float percentPerSec; const char* label; };
const int NUM_SPEED_PRESETS = 12;
const SpeedPreset SPEED_PRESETS[NUM_SPEED_PRESETS] = {
  { 8.0f,  "12.5s cycle - ORIGINAL (too slow, not based on real gait)" },
  {10.0f,  "10.0s cycle"                                               },
  {12.5f,  "8.0s cycle"                                                },
  {16.7f,  "6.0s cycle"                                                },
  {25.0f,  "4.0s cycle"                                                },
  {35.2f,  "2.84s cycle - 0.50 mph (slowest real dataset speed)"       },
  {48.1f,  "2.08s cycle - 0.75 mph"                                    },
  {58.0f,  "1.73s cycle - 1.00 mph"                                    },
  {67.3f,  "1.49s cycle - 1.50 mph"                                    },
  {70.1f,  "1.43s cycle - 1.25 mph"                                    },
  {71.7f,  "1.39s cycle - 1.75 mph"                                    },
  {74.5f,  "1.34s cycle - 2.00 mph (fastest real dataset speed)"       },
};
float simGaitSpeedPercentPerSec = SPEED_PRESETS[0].percentPerSec;  // start at slowest, per request
int currentSpeedPresetIndex = 1;

void printSpeedMenu() {
  Serial.println("=== SPEED PRESETS (type the number + Enter in Serial Monitor to switch) ===");
  for (int i = 0; i < NUM_SPEED_PRESETS; i++) {
    Serial.print(i + 1); Serial.print(": "); Serial.println(SPEED_PRESETS[i].label);
  }
  Serial.println("============================================================================");
}

String speedInputBuffer = "";

void checkSpeedCommand() {
  // Non-blocking by construction: only consumes bytes already sitting in
  // the input buffer, never waits for more. Serial.readStringUntil() (the
  // previous version) blocks for up to ~1s if a full line hasn't arrived
  // yet - while blocked, Core 1's control loop can't advance, so the next
  // getGaitPercentInput() call sees a huge dt and produces an oversized
  // jump, which gets rejected MAX_BAD_READINGS times in a row and freezes
  // the system. The jump size scales with speed, which is why this only
  // crossed the threshold at higher presets, not low ones.
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (speedInputBuffer.length() > 0) {
        int idx = speedInputBuffer.toInt();
        xSemaphoreTake(serialMutex, portMAX_DELAY);
        if (idx >= 1 && idx <= NUM_SPEED_PRESETS) {
          currentSpeedPresetIndex = idx;
          simGaitSpeedPercentPerSec = SPEED_PRESETS[idx - 1].percentPerSec;
          Serial.print(">>> Preset "); Serial.print(idx); Serial.print(": ");
          Serial.println(SPEED_PRESETS[idx - 1].label);
        } else {
          Serial.println(">>> Invalid. Type a number 1-12 and press Enter.");
        }
        xSemaphoreGive(serialMutex);
        speedInputBuffer = "";
      }
    } else {
      speedInputBuffer += c;
    }
  }
}

const float MAX_GAIT_JUMP_PERCENT = 10.0f;
const int   MAX_BAD_READINGS      = 3;
const float COLLAPSE_ERROR_DEG = 20.0f;

uint8_t cmdOn[5] = {0x3E, 0x88, 0x01, 0x00, 0xC7};

const int TRAJ_SIZE = 11;
const float gaitTable[TRAJ_SIZE] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
// DELIBERATE TRADE-OFF (confirmed tonight): original clinical curve had two
// humps - a small early-stance flexion wave (~10-20% gait) plus the big
// swing-phase peak (~70% gait) - matching standard gait biomechanics
// literature, not a data error. Each hump's reversal is a backlash-prone
// spot on this actuator. Smoothed to ONE peak (same start/end/peak values,
// monotonic rise into it instead of bump-dip-rise) to drop to a single
// direction reversal per cycle, trading clinical realism for hardware
// smoothness - flag this explicitly if asked about it tomorrow.
const float kneeTable[TRAJ_SIZE] = {4, 8, 14, 20, 28, 38, 50, 64.12, 53, 17, 2};

float Gait_percent = 0.0f;
float rawMLGaitPercent = 0.0f;
float usedGaitPercent = 0.0f;
float lookAheadGaitPercent = 0.0f;
float lastValidGaitPercent = 0.0f;
float predictedGaitPercent = 0.0f;
float gaitRatePerUpdate = 0.2f;
bool hasFirstValidGait = false;
bool badReading = false;
bool systemFrozen = false;
bool commandFilterReady = false;
int badReadingCounter = 0;
int statusCode = 1;
int supportMode = 0;
float desiredAngleNormal = 5.0f;
float desiredAngleROM = 5.0f;
float commandedAngle = 5.0f;
float actualAngle = 0.0f;
float lastSentAngle = -999.0f;
float frozenAngle = 5.0f;
unsigned long lastControlTime = 0;
unsigned long lastPlotTime = 0;
unsigned long lastFeedbackTime = 0;
unsigned long lastGaitUpdateTime = 0;

float wrapGaitPercent(float gp) {
  while (gp >= 100.0f) gp -= 100.0f;
  while (gp < 0.0f)    gp += 100.0f;
  return gp;
}
float gaitForwardDiff(float fromGp, float toGp) {
  float diff = toGp - fromGp;
  if (diff < -50.0f) diff += 100.0f;
  return diff;
}

float tableSlope(int i) {
  if (i <= 0) return (kneeTable[1] - kneeTable[0]) / (gaitTable[1] - gaitTable[0]);
  if (i >= TRAJ_SIZE - 1) return (kneeTable[TRAJ_SIZE - 1] - kneeTable[TRAJ_SIZE - 2]) / (gaitTable[TRAJ_SIZE - 1] - gaitTable[TRAJ_SIZE - 2]);
  return (kneeTable[i + 1] - kneeTable[i - 1]) / (gaitTable[i + 1] - gaitTable[i - 1]);
}

float smoothClinicalAngle(float gp) {
  gp = clampFloat(gp, 0.0f, 100.0f);
  if (gp <= gaitTable[0]) return kneeTable[0];
  if (gp >= gaitTable[TRAJ_SIZE - 1]) return kneeTable[TRAJ_SIZE - 1];
  for (int i = 0; i < TRAJ_SIZE - 1; i++) {
    if (gp >= gaitTable[i] && gp <= gaitTable[i + 1]) {
      float x0 = gaitTable[i], x1 = gaitTable[i + 1];
      float y0 = kneeTable[i], y1 = kneeTable[i + 1];
      float m0 = tableSlope(i), m1 = tableSlope(i + 1);
      float dx = x1 - x0;
      float t = (gp - x0) / dx, t2 = t * t, t3 = t2 * t;
      float h00 =  2.0f * t3 - 3.0f * t2 + 1.0f;
      float h10 =        t3 - 2.0f * t2 + t;
      float h01 = -2.0f * t3 + 3.0f * t2;
      float h11 =        t3 -       t2;
      float y = h00 * y0 + h10 * dx * m0 + h01 * y1 + h11 * dx * m1;
      return clampFloat(y, 0.0f, NORMAL_ROM_MAX_DEG);
    }
  }
  return kneeTable[0];
}

float scaleAngleToPatientROM(float normalAngle) {
  float maxROM = THERAPIST_MAX_ROM_DEG;
  if (maxROM < NORMAL_ROM_MIN_DEG) maxROM = NORMAL_ROM_MIN_DEG;
  float angleForScaling = normalAngle;
  if (angleForScaling < NORMAL_ROM_MIN_DEG) angleForScaling = NORMAL_ROM_MIN_DEG;
  float scaled = NORMAL_ROM_MIN_DEG + (angleForScaling - NORMAL_ROM_MIN_DEG) * (maxROM - NORMAL_ROM_MIN_DEG) / (NORMAL_ROM_MAX_DEG - NORMAL_ROM_MIN_DEG);
  return clampFloat(scaled, NORMAL_ROM_MIN_DEG, maxROM);
}

void freezeSystem(int emergencyMode) {
  systemFrozen = true;
  supportMode = emergencyMode;
  frozenAngle = commandedAngle;
  ledFrozenBlue();
}

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
    uint8_t buf[16]; int n = 0; unsigned long t = millis();
    while (n < 16 && millis() - t < 200) {
      if (MotorSerial.available()) { buf[n++] = MotorSerial.read(); t = millis(); }
    }
    for (int i = 0; i < n; i++) if (buf[i] == 0x3E) return true;
    delay(100);
  }
  return false;
}

void sendPositionFast(float outputDeg) {
  int32_t angleRaw = (int32_t)(outputDeg * MOTOR_SCALE);
  uint8_t buf[14];
  buf[0]  = 0x3E; buf[1]  = 0xA3; buf[2]  = 0x01; buf[3]  = 0x08;
  buf[4]  = (0x3E + 0xA3 + 0x01 + 0x08) & 0xFF;
  buf[5]  = angleRaw & 0xFF;
  buf[6]  = (angleRaw >> 8)  & 0xFF;
  buf[7]  = (angleRaw >> 16) & 0xFF;
  buf[8]  = (angleRaw >> 24) & 0xFF;
  buf[9] = 0x00; buf[10] = 0x00; buf[11] = 0x00; buf[12] = 0x00;
  uint8_t sum = 0;
  for (int i = 5; i < 13; i++) sum += buf[i];
  buf[13] = sum & 0xFF;
  sendCmd(buf, 14);
}

float readActualAngleQuick() {
  uint8_t angleCmd[5] = {0x3E, 0x92, 0x01, 0x00, 0xD1};
  while (MotorSerial.available()) MotorSerial.read();
  MotorSerial.write(angleCmd, 5);
  MotorSerial.flush();
  uint8_t rx[32]; int n = 0;
  unsigned long startTime = millis();  // fixed start - NOT reset per byte
  // Previous version reset the timeout clock every time a byte arrived,
  // so trickling bytes (e.g. under cross-core bus contention) could stall
  // this far past the intended 30ms, spiking dt on the next gait update
  // and triggering a freeze. This now bounds TOTAL elapsed time instead.
  while (n < 32 && millis() - startTime < 30) {
    if (MotorSerial.available()) { rx[n++] = MotorSerial.read(); }
  }
  for (int i = 0; i < n - 8; i++) {
    if (rx[i] == 0x3E && rx[i + 1] == 0x92) {
      uint32_t rawUnsigned = ((uint32_t)rx[i + 5]) | ((uint32_t)rx[i + 6] << 8) |
                             ((uint32_t)rx[i + 7] << 16) | ((uint32_t)rx[i + 8] << 24);
      return (float)((int32_t)rawUnsigned) / MOTOR_SCALE;
    }
  }
  return actualAngle;
}

float lastDt = 0.0f;

float getGaitPercentInput() {
  unsigned long now = millis();
  if (lastGaitUpdateTime == 0) { lastGaitUpdateTime = now; return Gait_percent; }
  float dt = (now - lastGaitUpdateTime) / 1000.0f;
  lastDt = dt;
  lastGaitUpdateTime = now;
  float step = simGaitSpeedPercentPerSec * dt;
  Gait_percent = wrapGaitPercent(Gait_percent + step);
  return Gait_percent;
}

float validateOrPredictGait(float newGaitPercent) {
  newGaitPercent = clampFloat(newGaitPercent, 0.0f, 100.0f);
  if (!hasFirstValidGait) {
    hasFirstValidGait = true; badReading = false; badReadingCounter = 0;
    lastValidGaitPercent = newGaitPercent; predictedGaitPercent = newGaitPercent;
    return newGaitPercent;
  }
  float jump = gaitForwardDiff(lastValidGaitPercent, newGaitPercent);
  bool validForwardMotion = (jump >= 0.0f && jump <= MAX_GAIT_JUMP_PERCENT);
  if (validForwardMotion) {
    badReading = false; badReadingCounter = 0;
    if (jump > 0.0f && jump < MAX_GAIT_JUMP_PERCENT) gaitRatePerUpdate = jump;
    lastValidGaitPercent = newGaitPercent; predictedGaitPercent = newGaitPercent;
    return newGaitPercent;
  }
  badReading = true; badReadingCounter++;
  xSemaphoreTake(serialMutex, portMAX_DELAY);
  Serial.print(">>> BAD READING #"); Serial.print(badReadingCounter);
  Serial.print(" dt="); Serial.print(lastDt * 1000.0f, 1); Serial.print("ms");
  Serial.print(" jump="); Serial.print(jump, 2);
  Serial.print(" from="); Serial.print(lastValidGaitPercent, 2);
  Serial.print(" to="); Serial.println(newGaitPercent, 2);
  xSemaphoreGive(serialMutex);
  predictedGaitPercent = wrapGaitPercent(predictedGaitPercent + gaitRatePerUpdate);
  if (badReadingCounter > MAX_BAD_READINGS) {
    xSemaphoreTake(serialMutex, portMAX_DELAY);
    Serial.println(">>> FREEZE: triggered from accumulated bad readings (validateOrPredictGait) <<<");
    xSemaphoreGive(serialMutex);
    freezeSystem(2);
  }
  return predictedGaitPercent;
}

void updateSmoothFSMAndTrajectory(float gp) {
  statusCode = getFSMStatusCode(gp);
  supportMode = isStanceSupportState(statusCode) ? 1 : 0;
  lookAheadGaitPercent = wrapGaitPercent(gp + LOOKAHEAD_PERCENT);
  desiredAngleNormal = smoothClinicalAngle(lookAheadGaitPercent);
  desiredAngleROM = scaleAngleToPatientROM(desiredAngleNormal);
  if (!commandFilterReady) { commandedAngle = desiredAngleROM; commandFilterReady = true; }
  else { commandedAngle = commandedAngle + COMMAND_SMOOTH_ALPHA * (desiredAngleROM - commandedAngle); }
  commandedAngle = clampFloat(commandedAngle, NORMAL_ROM_MIN_DEG, THERAPIST_MAX_ROM_DEG);
}

void sendMotorSmoothly() {
  if (systemFrozen) return;
  sendPositionFast(commandedAngle);
  lastSentAngle = commandedAngle;
}

void checkScenario2Emergency() {
  if (supportMode == 1) {
    if (actualAngle > commandedAngle + COLLAPSE_ERROR_DEG) {
      xSemaphoreTake(serialMutex, portMAX_DELAY);
      Serial.print(">>> FREEZE: triggered from checkScenario2Emergency, actualAngle=");
      Serial.print(actualAngle, 1); Serial.print(" commandedAngle="); Serial.println(commandedAngle, 1);
      xSemaphoreGive(serialMutex);
      freezeSystem(2);
    }
  }
}

// Set to false to silence Core A's plotter line and focus on Core B's
// [STATUS-B] output instead. Set back to true to watch the motor curve.
const bool SHOW_CORE_A_PLOTTER = false;

void printPlotterLine() {
  if (!SHOW_CORE_A_PLOTTER) return;
  xSemaphoreTake(serialMutex, portMAX_DELAY);
  Serial.print(rawMLGaitPercent, 1);      Serial.print(",");
  Serial.print(usedGaitPercent, 1);       Serial.print(",");
  Serial.print(lookAheadGaitPercent, 1);  Serial.print(",");
  Serial.print(statusCode * 10);          Serial.print(",");
  Serial.print(desiredAngleNormal, 1);    Serial.print(",");
  Serial.print(desiredAngleROM, 1);       Serial.print(",");
  Serial.print(commandedAngle, 1);        Serial.print(",");
  Serial.print(actualAngle, 1);           Serial.print(",");
  Serial.print(badReading ? 20 : 0);      Serial.print(",");
  Serial.print(badReadingCounter * 5);    Serial.print(",");
  Serial.print(supportMode * 15);         Serial.print(",");
  Serial.println(systemFrozen ? 70 : 0);
  xSemaphoreGive(serialMutex);
}


/* ============================================================
   ================   CORE 0 — STEP B (UNCHANGED)   =============
   ============================================================ */

#define BUS0_SDA 8
#define BUS0_SCL 9
#define BUS1_SDA 10
#define BUS1_SCL 11
#define FOOT_ADDR  0x68
#define SHANK_ADDR 0x69
#define THIGH_ADDR 0x68
#define REG_SMPLRT_DIV    0x19
#define REG_CONFIG        0x1A
#define REG_GYRO_CONFIG   0x1B
#define REG_ACCEL_CONFIG  0x1C
#define REG_PWR_MGMT_1    0x6B
#define REG_WHO_AM_I      0x75
#define REG_ACCEL_XOUT_H  0x3B
#define TARGET_ACCEL_CONFIG 0x18
#define TARGET_GYRO_CONFIG  0x18

TwoWire I2C_BUS0 = TwoWire(0);
TwoWire I2C_BUS1 = TwoWire(1);

struct ImuDef {
  TwoWire* bus; const char* busName; uint8_t addr; const char* name;
  bool ok; uint8_t accelCfg; uint8_t gyroCfg; float accelScale; float gyroScaleDps;
};

ImuDef imus[3] = {
  { &I2C_BUS0, "Bus0 GPIO8/9",   FOOT_ADDR,  "foot",  false, 0, 0, 16384.0f, 131.0f },
  { &I2C_BUS0, "Bus0 GPIO8/9",   SHANK_ADDR, "shank", false, 0, 0, 16384.0f, 131.0f },
  { &I2C_BUS1, "Bus1 GPIO10/11", THIGH_ADDR, "thigh", false, 0, 0, 16384.0f, 131.0f },
};

const float DEG_TO_RAD_F = 0.01745329252f;

uint8_t readReg(TwoWire &bus, uint8_t addr, uint8_t reg) {
  bus.beginTransmission(addr); bus.write(reg);
  if (bus.endTransmission(false) != 0) return 0xFF;
  bus.requestFrom(addr, (uint8_t)1);
  return bus.available() ? bus.read() : 0xFF;
}
bool writeReg(TwoWire &bus, uint8_t addr, uint8_t reg, uint8_t val) {
  bus.beginTransmission(addr); bus.write(reg); bus.write(val);
  return bus.endTransmission() == 0;
}
float accelScaleFromConfig(uint8_t cfg) {
  switch (cfg & 0x18) { case 0x00: return 16384.0f; case 0x08: return 8192.0f; case 0x10: return 4096.0f; case 0x18: return 2048.0f; }
  return 16384.0f;
}
float gyroScaleFromConfig(uint8_t cfg) {
  switch (cfg & 0x18) { case 0x00: return 131.0f; case 0x08: return 65.5f; case 0x10: return 32.8f; case 0x18: return 16.4f; }
  return 131.0f;
}
bool readMotion6(TwoWire &bus, uint8_t addr, int16_t &ax, int16_t &ay, int16_t &az, int16_t &gx, int16_t &gy, int16_t &gz) {
  bus.beginTransmission(addr); bus.write(REG_ACCEL_XOUT_H);
  if (bus.endTransmission(false) != 0) return false;
  if (bus.requestFrom(addr, (uint8_t)14) != 14) return false;
  ax = (int16_t)((bus.read() << 8) | bus.read());
  ay = (int16_t)((bus.read() << 8) | bus.read());
  az = (int16_t)((bus.read() << 8) | bus.read());
  bus.read(); bus.read();
  gx = (int16_t)((bus.read() << 8) | bus.read());
  gy = (int16_t)((bus.read() << 8) | bus.read());
  gz = (int16_t)((bus.read() << 8) | bus.read());
  return true;
}
bool initMPU(int idx) {
  ImuDef &imu = imus[idx];
  uint8_t who = readReg(*imu.bus, imu.addr, REG_WHO_AM_I);
  if (who != 0x68 && who != 0x70) {
    Serial.print("IMU "); Serial.print(imu.name); Serial.print(" FAIL WHO_AM_I=0x"); Serial.println(who, HEX);
    imu.ok = false; return false;
  }
  writeReg(*imu.bus, imu.addr, REG_PWR_MGMT_1, 0x00); delay(100);
  writeReg(*imu.bus, imu.addr, REG_SMPLRT_DIV, 0x04);
  writeReg(*imu.bus, imu.addr, REG_CONFIG, 0x03);
  writeReg(*imu.bus, imu.addr, REG_ACCEL_CONFIG, TARGET_ACCEL_CONFIG);
  writeReg(*imu.bus, imu.addr, REG_GYRO_CONFIG, TARGET_GYRO_CONFIG);
  delay(50);
  writeReg(*imu.bus, imu.addr, REG_ACCEL_CONFIG, TARGET_ACCEL_CONFIG);
  writeReg(*imu.bus, imu.addr, REG_GYRO_CONFIG, TARGET_GYRO_CONFIG);
  delay(50);
  imu.accelCfg = readReg(*imu.bus, imu.addr, REG_ACCEL_CONFIG);
  imu.gyroCfg  = readReg(*imu.bus, imu.addr, REG_GYRO_CONFIG);
  imu.accelScale = accelScaleFromConfig(imu.accelCfg);
  imu.gyroScaleDps = gyroScaleFromConfig(imu.gyroCfg);
  imu.ok = (imu.accelCfg != 0xFF && imu.gyroCfg != 0xFF);
  Serial.print("IMU "); Serial.print(imu.name);
  Serial.print(" addr=0x"); Serial.print(imu.addr, HEX);
  Serial.print(" WHO=0x"); Serial.print(who, HEX);
  Serial.print(" scale="); Serial.print(imu.accelScale, 1); Serial.print(" LSB/g, ");
  Serial.print(imu.gyroScaleDps, 1); Serial.println(" LSB/dps");
  return imu.ok;
}
bool allIMUsOK() { for (int i = 0; i < 3; i++) if (!imus[i].ok) return false; return true; }

#define FSR_HEEL 1
#define FSR_META 2

const unsigned long CAL_UNLOADED_MS = 3000;
const unsigned long CAL_LOADED_MS   = 3000;
float heelLow = 0, heelHigh = 4095;
float midLow = 0, midHigh = 4095;

float normalize01(int raw, float lo, float hi) {
  if (hi - lo < 1.0f) hi = lo + 1.0f;
  float v = (raw - lo) / (hi - lo);
  if (v < 0) v = 0; if (v > 1) v = 1;
  return v;
}

void calibrateFSR() {
  Serial.println(); Serial.println("=== FSR CALIBRATION ===");
  Serial.println("Step A: Keep both FSR sensors UNLOADED now.");
  for (int s = 3; s > 0; s--) { Serial.print(s); Serial.println("..."); delay(1000); }
  int heelMin = 4095, heelMax = 0, midMin = 4095, midMax = 0;
  unsigned long start = millis();
  while (millis() - start < CAL_UNLOADED_MS) {
    int h = analogRead(FSR_HEEL); int m = analogRead(FSR_META);
    heelMin = min(heelMin, h); heelMax = max(heelMax, h);
    midMin  = min(midMin, m);  midMax  = max(midMax, m);
    delay(10);
  }
  heelLow = heelMax; midLow = midMax;
  Serial.println(); Serial.println("Step B: Load BOTH FSR sensors fully now.");
  for (int s = 3; s > 0; s--) { Serial.print(s); Serial.println("..."); delay(1000); }
  heelMin = 4095; heelMax = 0; midMin = 4095; midMax = 0;
  start = millis();
  while (millis() - start < CAL_LOADED_MS) {
    int h = analogRead(FSR_HEEL); int m = analogRead(FSR_META);
    heelMin = min(heelMin, h); heelMax = max(heelMax, h);
    midMin  = min(midMin, m);  midMax  = max(midMax, m);
    delay(10);
  }
  heelHigh = heelMax; midHigh = midMax;
  Serial.print("Calibration -> heel ["); Serial.print(heelLow); Serial.print(", "); Serial.print(heelHigh);
  Serial.print("]  mid ["); Serial.print(midLow); Serial.print(", "); Serial.print(midHigh); Serial.println("]");
  const float MIN_DYNAMIC_RANGE = 200.0f;
  if ((heelHigh - heelLow) < MIN_DYNAMIC_RANGE) Serial.println("WARNING: heel dynamic range very small.");
  if ((midHigh - midLow) < MIN_DYNAMIC_RANGE) Serial.println("WARNING: midfoot dynamic range very small.");
  Serial.println("=== CALIBRATION DONE ==="); Serial.println();
}

const float CONTACT_HIGH = 0.40f;
const float CONTACT_LOW  = 0.20f;
const float MIN_CYCLE_FRACTION = 0.65f;
const unsigned long MAX_PLAUSIBLE_CYCLE_MS = 4000;
const int CYCLE_HISTORY_SIZE = 5;
const int SMOOTH_WINDOW = 5;

float contactBuf[SMOOTH_WINDOW];
int contactBufIdx = 0;
bool contactBufFilled = false;
bool inContact = false;
unsigned long lastHeelStrikeMs = 0;
bool hasLastHeelStrike = false;
float cycleHistoryMs[CYCLE_HISTORY_SIZE];
int cycleHistoryCount = 0;
int cycleHistoryIdx = 0;
float medianCycleMs = 1200.0f;
int completedCycles = 0;
bool recentCycleValid = false;
float trackerGaitFrac = 0.0f;

float medianOf(float* arr, int n) {
  float tmp[CYCLE_HISTORY_SIZE];
  for (int i = 0; i < n; i++) tmp[i] = arr[i];
  for (int i = 1; i < n; i++) {
    float key = tmp[i]; int j = i - 1;
    while (j >= 0 && tmp[j] > key) { tmp[j + 1] = tmp[j]; j--; }
    tmp[j + 1] = key;
  }
  return tmp[n / 2];
}

void updateHeelStrikeTracker(float heelNorm, float midNorm) {
  unsigned long now = millis();
  float contactScoreRaw = max(heelNorm, midNorm);
  contactBuf[contactBufIdx] = contactScoreRaw;
  contactBufIdx = (contactBufIdx + 1) % SMOOTH_WINDOW;
  if (contactBufIdx == 0) contactBufFilled = true;
  int count = contactBufFilled ? SMOOTH_WINDOW : contactBufIdx;
  float sum = 0;
  for (int i = 0; i < count; i++) sum += contactBuf[i];
  float contactScore = (count > 0) ? sum / count : contactScoreRaw;
  bool wasContact = inContact;
  if (!inContact && contactScore >= CONTACT_HIGH) inContact = true;
  else if (inContact && contactScore <= CONTACT_LOW) inContact = false;
  bool risingEdge = (!wasContact && inContact);
  if (risingEdge) {
    bool accept = true;
    if (hasLastHeelStrike) {
      unsigned long sinceLast = now - lastHeelStrikeMs;
      if (sinceLast < (unsigned long)(MIN_CYCLE_FRACTION * medianCycleMs)) accept = false;
    }
    if (accept) {
      if (hasLastHeelStrike) {
        unsigned long thisCycle = now - lastHeelStrikeMs;
        if (thisCycle <= MAX_PLAUSIBLE_CYCLE_MS) {
          cycleHistoryMs[cycleHistoryIdx] = (float)thisCycle;
          cycleHistoryIdx = (cycleHistoryIdx + 1) % CYCLE_HISTORY_SIZE;
          if (cycleHistoryCount < CYCLE_HISTORY_SIZE) cycleHistoryCount++;
          medianCycleMs = medianOf(cycleHistoryMs, cycleHistoryCount);
          completedCycles++;
          recentCycleValid = true;
        } else { recentCycleValid = false; }
      }
      lastHeelStrikeMs = now;
      hasLastHeelStrike = true;
    }
  }
  if (hasLastHeelStrike && recentCycleValid) {
    float elapsed = (float)(now - lastHeelStrikeMs);
    float frac = elapsed / medianCycleMs;
    if (frac < 0) frac = 0; if (frac > 1) frac = 1;
    trackerGaitFrac = frac;
  }
}

const char* trackerPhaseName(float gp01) {
  return fsmStatusName(getFSMStatusCode(gp01 * 100.0f));
}

const char* CNN_PHASE_NAMES[6] = {
  "HeelStrike", "LoadingResponse", "MidTerminalStance",
  "PreSwingToeOff", "InitialMidSwing", "TerminalSwing"
};

inline float safeSqrt(float x) { return sqrtf(x > 0 ? x : 0); }
float imuPitchDeg(float ax, float ay, float az) { return degrees(atan2f(ax, safeSqrt(ay * ay + az * az) + 1e-9f)); }
float imuRollDeg(float ax, float ay, float az) { return degrees(atan2f(ay, az + 1e-9f)); }

float prev_heel_fsr = 0, prev_mid_fsr = 0, prev_fsr_sum = 0, prev_fsr_diff = 0;
float prev_foot_gyro_mag = 0, prev_shank_gyro_mag = 0, prev_thigh_gyro_mag = 0;
float prev_knee_acc = 0, prev_knee_pitch = 0;
float prev_foot_pitch = 0, prev_shank_pitch = 0, prev_thigh_pitch = 0;
bool firstFeatureTick = true;
float sampleFeatures[KNEVO_TCN_FEATURE_COUNT];

void computeFeatures(float ax_g[3], float ay_g[3], float az_g[3],
                      float gx_rs[3], float gy_rs[3], float gz_rs[3],
                      float heelNorm, float midNorm) {
  sampleFeatures[0]=ax_g[0]; sampleFeatures[1]=ay_g[0]; sampleFeatures[2]=az_g[0];
  sampleFeatures[3]=gx_rs[0]; sampleFeatures[4]=gy_rs[0]; sampleFeatures[5]=gz_rs[0];
  sampleFeatures[6]=ax_g[1]; sampleFeatures[7]=ay_g[1]; sampleFeatures[8]=az_g[1];
  sampleFeatures[9]=gx_rs[1]; sampleFeatures[10]=gy_rs[1]; sampleFeatures[11]=gz_rs[1];
  sampleFeatures[12]=ax_g[2]; sampleFeatures[13]=ay_g[2]; sampleFeatures[14]=az_g[2];
  sampleFeatures[15]=gx_rs[2]; sampleFeatures[16]=gy_rs[2]; sampleFeatures[17]=gz_rs[2];
  sampleFeatures[18]=heelNorm; sampleFeatures[19]=midNorm;

  float foot_acc_mag  = sqrtf(ax_g[0]*ax_g[0] + ay_g[0]*ay_g[0] + az_g[0]*az_g[0]);
  float shank_acc_mag = sqrtf(ax_g[1]*ax_g[1] + ay_g[1]*ay_g[1] + az_g[1]*az_g[1]);
  float thigh_acc_mag = sqrtf(ax_g[2]*ax_g[2] + ay_g[2]*ay_g[2] + az_g[2]*az_g[2]);
  float foot_gyro_mag  = sqrtf(gx_rs[0]*gx_rs[0] + gy_rs[0]*gy_rs[0] + gz_rs[0]*gz_rs[0]);
  float shank_gyro_mag = sqrtf(gx_rs[1]*gx_rs[1] + gy_rs[1]*gy_rs[1] + gz_rs[1]*gz_rs[1]);
  float thigh_gyro_mag = sqrtf(gx_rs[2]*gx_rs[2] + gy_rs[2]*gy_rs[2] + gz_rs[2]*gz_rs[2]);
  sampleFeatures[20]=foot_acc_mag; sampleFeatures[21]=shank_acc_mag; sampleFeatures[22]=thigh_acc_mag;
  sampleFeatures[23]=foot_gyro_mag; sampleFeatures[24]=shank_gyro_mag; sampleFeatures[25]=thigh_gyro_mag;

  float foot_pitch  = imuPitchDeg(ax_g[0], ay_g[0], az_g[0]);
  float shank_pitch = imuPitchDeg(ax_g[1], ay_g[1], az_g[1]);
  float thigh_pitch = imuPitchDeg(ax_g[2], ay_g[2], az_g[2]);
  float foot_roll  = imuRollDeg(ax_g[0], ay_g[0], az_g[0]);
  float shank_roll = imuRollDeg(ax_g[1], ay_g[1], az_g[1]);
  float thigh_roll = imuRollDeg(ax_g[2], ay_g[2], az_g[2]);
  sampleFeatures[26]=foot_pitch; sampleFeatures[27]=shank_pitch; sampleFeatures[28]=thigh_pitch;
  sampleFeatures[29]=foot_roll; sampleFeatures[30]=shank_roll; sampleFeatures[31]=thigh_roll;

  float dot = ax_g[2]*ax_g[1] + ay_g[2]*ay_g[1] + az_g[2]*az_g[1];
  float normProd = (sqrtf(ax_g[2]*ax_g[2]+ay_g[2]*ay_g[2]+az_g[2]*az_g[2]) + 1e-9f) *
                    (sqrtf(ax_g[1]*ax_g[1]+ay_g[1]*ay_g[1]+az_g[1]*az_g[1]) + 1e-9f);
  float cosAngle = dot / normProd;
  if (cosAngle > 1) cosAngle = 1; if (cosAngle < -1) cosAngle = -1;
  float knee_angle_acc = degrees(acosf(cosAngle));
  float knee_angle_pitch = thigh_pitch - shank_pitch;
  sampleFeatures[32]=knee_angle_acc; sampleFeatures[33]=knee_angle_pitch;

  float fsr_sum = heelNorm + midNorm;
  float fsr_diff = heelNorm - midNorm;
  float fsr_max = max(heelNorm, midNorm);
  sampleFeatures[34]=fsr_sum; sampleFeatures[35]=fsr_diff; sampleFeatures[36]=fsr_max;

  sampleFeatures[37] = shank_pitch - thigh_pitch;
  sampleFeatures[38] = foot_pitch - shank_pitch;
  sampleFeatures[39] = shank_gyro_mag - thigh_gyro_mag;
  sampleFeatures[40] = foot_gyro_mag - shank_gyro_mag;

  if (firstFeatureTick) {
    prev_heel_fsr = heelNorm; prev_mid_fsr = midNorm;
    prev_fsr_sum = fsr_sum; prev_fsr_diff = fsr_diff;
    prev_foot_gyro_mag = foot_gyro_mag; prev_shank_gyro_mag = shank_gyro_mag; prev_thigh_gyro_mag = thigh_gyro_mag;
    prev_knee_acc = knee_angle_acc; prev_knee_pitch = knee_angle_pitch;
    prev_foot_pitch = foot_pitch; prev_shank_pitch = shank_pitch; prev_thigh_pitch = thigh_pitch;
    firstFeatureTick = false;
  }
  sampleFeatures[41] = heelNorm - prev_heel_fsr;
  sampleFeatures[42] = midNorm - prev_mid_fsr;
  sampleFeatures[43] = fsr_sum - prev_fsr_sum;
  sampleFeatures[44] = fsr_diff - prev_fsr_diff;
  sampleFeatures[45] = foot_gyro_mag - prev_foot_gyro_mag;
  sampleFeatures[46] = shank_gyro_mag - prev_shank_gyro_mag;
  sampleFeatures[47] = thigh_gyro_mag - prev_thigh_gyro_mag;
  sampleFeatures[48] = knee_angle_acc - prev_knee_acc;
  sampleFeatures[49] = knee_angle_pitch - prev_knee_pitch;
  sampleFeatures[50] = foot_pitch - prev_foot_pitch;
  sampleFeatures[51] = shank_pitch - prev_shank_pitch;
  sampleFeatures[52] = thigh_pitch - prev_thigh_pitch;

  prev_heel_fsr = heelNorm; prev_mid_fsr = midNorm;
  prev_fsr_sum = fsr_sum; prev_fsr_diff = fsr_diff;
  prev_foot_gyro_mag = foot_gyro_mag; prev_shank_gyro_mag = shank_gyro_mag; prev_thigh_gyro_mag = thigh_gyro_mag;
  prev_knee_acc = knee_angle_acc; prev_knee_pitch = knee_angle_pitch;
  prev_foot_pitch = foot_pitch; prev_shank_pitch = shank_pitch; prev_thigh_pitch = thigh_pitch;

  float gp = trackerGaitFrac;
  bool valid = hasLastHeelStrike && recentCycleValid;
  sampleFeatures[53] = gp;
  sampleFeatures[54] = valid ? sinf(2.0f * PI * gp) : 0.0f;
  sampleFeatures[55] = valid ? cosf(2.0f * PI * gp) : 0.0f;
  sampleFeatures[56] = valid ? 1.0f : 0.0f;
}

const uint32_t SAMPLE_PERIOD_US = 10000;
const int INFERENCE_EVERY_N_SAMPLES = 5;
const size_t kTensorArenaSize = 130 * 1024;
const int NUM_PHASES = 6;

uint8_t* tensor_arena = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int8_t* quantized_window_buffer = nullptr;
int current_sample_row = 0;
bool window_ready = false;
float q_mult[KNEVO_TCN_FEATURE_COUNT];
float q_bias[KNEVO_TCN_FEATURE_COUNT];
uint32_t next_sample_time_us = 0;
int samples_since_inference = 0;
int last_predicted_phase = -1;
float last_probability = 0.0f;
uint32_t last_prep_us = 0, last_invoke_us = 0;

inline int8_t clampInt8(int32_t x) { if (x > 127) return 127; if (x < -128) return -128; return (int8_t)x; }
inline int8_t fastQuantizeFeature(float raw_val, int f) {
  int32_t q = (int32_t)lrintf(raw_val * q_mult[f] + q_bias[f]);
  return clampInt8(q);
}
void pushQuantizedSampleToWindow(const float* row) {
  int8_t* dest = quantized_window_buffer + (current_sample_row * KNEVO_TCN_FEATURE_COUNT);
  for (int f = 0; f < KNEVO_TCN_FEATURE_COUNT; f++) dest[f] = fastQuantizeFeature(row[f], f);
  current_sample_row++;
  if (current_sample_row >= KNEVO_TCN_WINDOW_SIZE) { current_sample_row = 0; window_ready = true; }
}
void copyCircularWindowToInputTensor() {
  const int feature_count = KNEVO_TCN_FEATURE_COUNT;
  const int start_row = current_sample_row;
  int first_rows = KNEVO_TCN_WINDOW_SIZE - start_row;
  int first_bytes = first_rows * feature_count * sizeof(int8_t);
  memcpy(input->data.int8, quantized_window_buffer + (start_row * feature_count), first_bytes);
  if (start_row > 0) {
    int second_bytes = start_row * feature_count * sizeof(int8_t);
    memcpy(input->data.int8 + (first_rows * feature_count), quantized_window_buffer, second_bytes);
  }
}
void runInference() {
  uint32_t prep_start = micros();
  copyCircularWindowToInputTensor();
  last_prep_us = micros() - prep_start;
  uint32_t invoke_start = micros();
  TfLiteStatus status = interpreter->Invoke();
  last_invoke_us = micros() - invoke_start;
  if (status != kTfLiteOk) { Serial.println("Inference failed."); return; }
  int predicted_phase = 0; float top_prob = -999.0f;
  for (int i = 0; i < NUM_PHASES; i++) {
    int8_t raw = output->data.int8[i];
    float prob = (raw - KNEVO_TCN_OUTPUT_ZERO_POINT) * KNEVO_TCN_OUTPUT_SCALE;
    if (prob > top_prob) { top_prob = prob; predicted_phase = i; }
  }
  last_predicted_phase = predicted_phase;
  last_probability = top_prob;
}

unsigned long lastHumanPrintTime_B = 0;
unsigned long rateWindowStart = 0;
int samplesThisWindow = 0;
float achievedHz = 0.0f;

void setupStepB() {
  analogReadResolution(12);
  pinMode(FSR_HEEL, INPUT);
  pinMode(FSR_META, INPUT);

  Serial.println("=== STEP B SETUP: SENSOR + FEATURE + DL PIPELINE ===");

  I2C_BUS0.begin(BUS0_SDA, BUS0_SCL, 400000);
  I2C_BUS1.begin(BUS1_SDA, BUS1_SCL, 400000);

  for (int i = 0; i < 3; i++) imus[i].ok = initMPU(i);
  if (allIMUsOK()) { Serial.println("All IMUs OK."); }
  else { Serial.println("One or more IMUs FAILED - features will be wrong until fixed."); }

  calibrateFSR();

  for (int f = 0; f < KNEVO_TCN_FEATURE_COUNT; f++) {
    if (KNEVO_TCN_STD[f] == 0.0f) { q_mult[f] = 0.0f; q_bias[f] = KNEVO_TCN_INPUT_ZERO_POINT; }
    else {
      q_mult[f] = 1.0f / (KNEVO_TCN_STD[f] * KNEVO_TCN_INPUT_SCALE);
      q_bias[f] = KNEVO_TCN_INPUT_ZERO_POINT - (KNEVO_TCN_MEAN[f] * q_mult[f]);
    }
  }

  size_t window_buffer_size = KNEVO_TCN_WINDOW_SIZE * KNEVO_TCN_FEATURE_COUNT * sizeof(int8_t);
  quantized_window_buffer = (int8_t*)heap_caps_malloc(window_buffer_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!quantized_window_buffer) { Serial.println("CRITICAL: window buffer alloc failed."); while (1) delay(1000); }
  memset(quantized_window_buffer, 0, window_buffer_size);

  tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!tensor_arena) { Serial.println("CRITICAL: tensor arena alloc failed."); while (1) delay(1000); }

  model = tflite::GetModel(g_knevo_boosted_phase_cnn_6phase_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) { Serial.println("ERROR: model schema mismatch."); while (1) delay(1000); }

  static tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddReshape(); resolver.AddConv2D(); resolver.AddAdd(); resolver.AddAveragePool2D();
  resolver.AddShape(); resolver.AddStridedSlice(); resolver.AddPack();
  resolver.AddFullyConnected(); resolver.AddMul(); resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) { Serial.println("ERROR: tensor allocation failed."); while (1) delay(1000); }
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TFLite Micro ready.");

  next_sample_time_us = micros() + SAMPLE_PERIOD_US;
  rateWindowStart = millis();
}

uint32_t last_i2c_us = 0, last_feature_us = 0;

void coreBTask(void* pvParameters) {
  for (;;) {
    uint32_t now_us = micros();

    if ((int32_t)(now_us - next_sample_time_us) >= 0) {
      next_sample_time_us += SAMPLE_PERIOD_US;

      uint32_t t_i2c_start = micros();
      int16_t ax_raw[3], ay_raw[3], az_raw[3], gx_raw[3], gy_raw[3], gz_raw[3];
      for (int i = 0; i < 3; i++) {
        if (imus[i].ok) {
          readMotion6(*imus[i].bus, imus[i].addr, ax_raw[i], ay_raw[i], az_raw[i], gx_raw[i], gy_raw[i], gz_raw[i]);
        } else {
          ax_raw[i]=0; ay_raw[i]=0; az_raw[i]=0; gx_raw[i]=0; gy_raw[i]=0; gz_raw[i]=0;
        }
      }
      last_i2c_us = micros() - t_i2c_start;

      float ax_g[3], ay_g[3], az_g[3], gx_rs[3], gy_rs[3], gz_rs[3];
      for (int i = 0; i < 3; i++) {
        ax_g[i] = ax_raw[i] / imus[i].accelScale;
        ay_g[i] = ay_raw[i] / imus[i].accelScale;
        az_g[i] = az_raw[i] / imus[i].accelScale;
        gx_rs[i] = (gx_raw[i] / imus[i].gyroScaleDps) * DEG_TO_RAD_F;
        gy_rs[i] = (gy_raw[i] / imus[i].gyroScaleDps) * DEG_TO_RAD_F;
        gz_rs[i] = (gz_raw[i] / imus[i].gyroScaleDps) * DEG_TO_RAD_F;
      }

      int heelRaw = analogRead(FSR_HEEL);
      int midRaw  = analogRead(FSR_META);
      float heelNorm = normalize01(heelRaw, heelLow, heelHigh);
      float midNorm  = normalize01(midRaw, midLow, midHigh);

      uint32_t t_feat_start = micros();
      updateHeelStrikeTracker(heelNorm, midNorm);
      computeFeatures(ax_g, ay_g, az_g, gx_rs, gy_rs, gz_rs, heelNorm, midNorm);
      last_feature_us = micros() - t_feat_start;

      pushQuantizedSampleToWindow(sampleFeatures);
      if (window_ready) {
        samples_since_inference++;
        if (samples_since_inference >= INFERENCE_EVERY_N_SAMPLES) {
          samples_since_inference = 0;
          runInference();
        }
      }
      samplesThisWindow++;
    }

    if (millis() - rateWindowStart >= 1000) {
      achievedHz = samplesThisWindow * 1000.0f / (millis() - rateWindowStart);
      samplesThisWindow = 0;
      rateWindowStart = millis();
    }

    if (millis() - lastHumanPrintTime_B >= 1000) {
      lastHumanPrintTime_B = millis();
      xSemaphoreTake(serialMutex, portMAX_DELAY);
      Serial.print("[STATUS-B] Hz="); Serial.print(achievedHz, 1);
      Serial.print(" TrackerProgress="); Serial.print(trackerGaitFrac * 100.0f, 1);
      Serial.print("% TrackerPhase="); Serial.print(trackerPhaseName(trackerGaitFrac));
      Serial.print(" TrackerValid="); Serial.print(recentCycleValid ? 1 : 0);
      Serial.print(" | CNN_Phase=");
      Serial.print(last_predicted_phase >= 0 ? CNN_PHASE_NAMES[last_predicted_phase] : "none-yet");
      Serial.print(" Prob="); Serial.print(last_probability, 3);
      Serial.print(" Prep="); Serial.print(last_prep_us); Serial.print("us");
      Serial.print(" Invoke="); Serial.print(last_invoke_us); Serial.print("us");
      Serial.print(" | I2C="); Serial.print(last_i2c_us); Serial.print("us");
      Serial.print(" Feature="); Serial.print(last_feature_us); Serial.println("us");
      xSemaphoreGive(serialMutex);
    }

    vTaskDelay(1);  // yield - prevents watchdog trip, negligible vs. the workload above
  }
}


/* ============================================================
   ===================   SETUP / LOOP   ========================
   ============================================================ */

void setup() {
  Serial.begin(115200);
  delay(3000);

  ledOff();
  MotorSerial.begin(115200, SERIAL_8N1, MOTOR_RX_PIN, MOTOR_TX_PIN);

  Serial.println("=== KNEVO STEP C: DUAL-CORE INTEGRATION ===");
  Serial.println("Core 1 = Step A (open-loop control, worn). Core 0 = Step B (sensor+DL, logging only).");
  Serial.print("Simulated gait cycle length: "); Serial.print(100.0f / simGaitSpeedPercentPerSec, 2); Serial.println(" s");
  printSpeedMenu();

  if (motorOn()) { Serial.println("Motor ON confirmed"); }
  else { Serial.println("Motor ON failed"); }

  // Step B's full setup must complete before its task starts.
  setupStepB();

  serialMutex = xSemaphoreCreateMutex();
  xTaskCreatePinnedToCore(coreBTask, "CoreB_SensorDL", 16384, NULL, 1, NULL, 0);

  // Step A's tail: first gait target + initial motor position.
  rawMLGaitPercent = 0.0f;
  usedGaitPercent = 0.0f;
  updateSmoothFSMAndTrajectory(usedGaitPercent);
  sendPositionFast(commandedAngle);
  lastSentAngle = commandedAngle;
  delay(500);
  actualAngle = readActualAngleQuick();

  Serial.println("RawGait,UsedGait,LookAheadGait,State_x10,DesiredNormal_deg,DesiredROM_deg,Commanded_deg,Actual_deg,BadReading_x20,ErrorCount_x5,SupportMode_x15,Frozen_x70");
  Serial.println("=== BOTH CORES RUNNING ===");
}

// Runs on Core 1 (default Arduino loop core). This is Step A, unchanged.
void loop() {
  unsigned long now = millis();

  checkSpeedCommand();

  if (now - lastControlTime >= CONTROL_MS) {
    lastControlTime = now;
    if (!systemFrozen) {
      rawMLGaitPercent = getGaitPercentInput();
      usedGaitPercent = validateOrPredictGait(rawMLGaitPercent);
      if (!systemFrozen) {
        updateSmoothFSMAndTrajectory(usedGaitPercent);
        sendMotorSmoothly();
      } else {
        commandedAngle = frozenAngle;
        supportMode = 2;
      }
    } else {
      ledFrozenBlue();
    }
  }

  if (now - lastFeedbackTime >= FEEDBACK_MS) {
    lastFeedbackTime = now;
    actualAngle = readActualAngleQuick();
    if (!systemFrozen) checkScenario2Emergency();
  }

  if (now - lastPlotTime >= PLOT_MS) {
    lastPlotTime = now;
    printPlotterLine();
  }
}
