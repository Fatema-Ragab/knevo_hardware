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
#include <esp_timer.h>   // esp_timer_get_time() for monotonic per-sample timestamps

// Step D — mobile-contract integration.
// NimBLE-Arduino (install via Library Manager; tested against the 1.4.x API).
// WiFi.h/WiFiClient are part of the ESP32 Arduino core. BLE and WiFi share the
// radio, so the contract keeps them time-separated (BLE pre/post-set, WiFi only
// for the post-set upload) — they are never active simultaneously.
#include <WiFi.h>
#include <NimBLEDevice.h>

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

// ---- Step D device state (declared early: freezeSystem() below reports FAULT) ----
enum DeviceState : uint8_t { DEV_IDLE = 0, DEV_RUNNING = 1, DEV_DONE = 2, DEV_FAULT = 3 };
volatile DeviceState deviceState = DEV_IDLE;
volatile uint8_t lastFaultCode = 0;   // 0 none; safety modes set this; 0x10 = config_rejected (D-ext1.6)
const uint8_t BATTERY_PCT_UNKNOWN = 0xFF;   // R5: no battery sense line on this hardware
volatile bool setActive = false;      // true while RUNNING and Core 0 should buffer samples

// WiFi/TCP net-machine state (declared early so cmdStart can refuse a new set while
// a provisioning test or upload is still in flight).
enum NetState : uint8_t { NET_IDLE = 0, NET_WIFI_CONNECTING = 1 };
volatile NetState netState = NET_IDLE;

// Step D forward declarations — these are called from coreBTask (Core 0) but defined
// further down; declare them explicitly so we don't depend on Arduino's auto-prototype
// generation (which hoists above custom types and is fragile — see handoff §7.7).
void serviceCalibration(int heelRaw, int midRaw);
void appendSample(uint64_t ts_us, float ax_g[3], float ay_g[3], float az_g[3],
                  float gx_rs[3], float gy_rs[3], float gz_rs[3], uint16_t heelRaw, uint16_t midRaw);
void serviceNet();
void deviceStatusNotify();
// Command API (called from checkSpeedCommand on Core 1; defined in the Step D section).
void cmdSetSpeedIndex(int idx);
bool cmdSetExtension(float deg);
bool cmdSetFlexion(float deg);
void cmdStart(const uint8_t setId[16], uint16_t durationS);
void cmdStop();
void cmdCalibrateUnloaded();
void cmdCalibrateStatic();
// Globals referenced by checkSpeedCommand()'s "B" diagnostic branch but
// declared later in the file (next to the buffer/BLE/WiFi code they belong
// with) - forward-declared here for the same reason as the functions above.
extern volatile uint32_t bufferCount;
extern volatile int sessionSpeedIndex;
extern uint16_t appPort;

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
// Per-session prescribed limits (mutable; set by Serial E/F or the app's BLE
// SetConfig). Defaults match the mobile contract (D-ext1.6): extension 5, flexion 60.
float THERAPIST_MIN_ROM_DEG = 5.0f;   // extension floor (was implicit NORMAL_ROM_MIN_DEG)
float THERAPIST_MAX_ROM_DEG = 60.0f;  // flexion ceiling
const float NORMAL_ROM_MIN_DEG = 5.0f;
const float NORMAL_ROM_MAX_DEG = 65.0f;

// Mobile-contract safety-limit ranges (D-ext1.6) — enforced for BOTH the Serial
// E/F commands and the BLE SetConfig. Speed range is the existing 1..12 preset index.
const float CONTRACT_EXT_MIN_DEG  = 1.0f;
const float CONTRACT_EXT_MAX_DEG  = 5.0f;
const float CONTRACT_FLEX_MIN_DEG = 30.0f;
const float CONTRACT_FLEX_MAX_DEG = 65.0f;

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
// Stepping: speed changes now walk through the existing 12 dataset-
// correlated preset speeds one at a time, advancing exactly one preset
// per second toward the target - no continuous interpolation, no fixed
// 5s ramp. 0 is treated as a 13th step, below preset 1, for a full stop.
const unsigned long SPEED_STEP_MS = 1000;  // one preset step per second
int currentSpeedIndex = 0;       // 0 = stop, 1-12 = SPEED_PRESETS index; starts stopped (Core 1 only)
// Step D: boot AT REST (target 0, resting at extension) so DeviceState IDLE truly
// means "motor at rest". The motor stays still until a speed command (1-12) or a
// START (G / BLE) is given. (stepC booted aiming at preset 1 and walked on its own.)
volatile int targetSpeedIndexUser = 0;    // boots stopped (also written from BLE)
unsigned long lastSpeedStepTime = 0;
float simGaitSpeedPercentPerSec = 0.0f;  // current actual speed, derived from currentSpeedIndex
volatile bool pendingStopAtExtension = false;  // true while decelerating toward a full stop (cross-core)
volatile bool fullyStoppedAtExtension = true;  // boot resting at extension; cleared by any speed/START cmd

float speedForIndex(int idx) {
  if (idx <= 0) return 0.0f;
  if (idx > NUM_SPEED_PRESETS) idx = NUM_SPEED_PRESETS;
  return SPEED_PRESETS[idx - 1].percentPerSec;
}

void updateSpeedStepping() {
  unsigned long now = millis();
  if (now - lastSpeedStepTime < SPEED_STEP_MS) return;
  lastSpeedStepTime = now;
  if (currentSpeedIndex < targetSpeedIndexUser) currentSpeedIndex++;
  else if (currentSpeedIndex > targetSpeedIndexUser) currentSpeedIndex--;
  simGaitSpeedPercentPerSec = speedForIndex(currentSpeedIndex);
}

void printSpeedMenu() {
  Serial.println("=== SPEED PRESETS (type the number + Enter in Serial Monitor to switch) ===");
  Serial.println("0: STOP (steps down one preset per second to a full stop)");
  for (int i = 0; i < NUM_SPEED_PRESETS; i++) {
    Serial.print(i + 1); Serial.print(": "); Serial.println(SPEED_PRESETS[i].label);
  }
  Serial.println("A speed (1-12) only CONFIGURES the preset; the motor moves on START, not before.");
  Serial.println("--- Step D session/config commands (same actions the app drives over BLE) ---");
  Serial.println("E<deg> : set max extension (1-5)      F<deg> : set max flexion (30-65)");
  Serial.println("CU     : calibrate UNLOADED (idle)    CS     : calibrate STATIC (idle)");
  Serial.println("G<sec> : START a set (0 = no auto-stop)   0 : STOP the set / motor");
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
        // Serial dispatcher: every action routes through the same command API the
        // BLE handlers use, so Serial-bench and app-driven runs exercise one path.
        String s = speedInputBuffer; s.trim();
        String up = s; up.toUpperCase();
        xSemaphoreTake(serialMutex, portMAX_DELAY);
        if (up.startsWith("E")) {                       // E<deg> -> max extension (floor)
          float v = s.substring(1).toFloat();
          if (cmdSetExtension(v)) { Serial.print(">>> Extension set to "); Serial.println(v, 1); }
          else { Serial.print(">>> Invalid extension. Allowed "); Serial.print(CONTRACT_EXT_MIN_DEG, 0); Serial.print("-"); Serial.print(CONTRACT_EXT_MAX_DEG, 0); Serial.println(" and below flexion."); }
        } else if (up.startsWith("F")) {                // F<deg> -> max flexion (ceiling)
          float v = s.substring(1).toFloat();
          if (cmdSetFlexion(v)) { Serial.print(">>> Flexion set to "); Serial.println(v, 1); }
          else { Serial.print(">>> Invalid flexion. Allowed "); Serial.print(CONTRACT_FLEX_MIN_DEG, 0); Serial.print("-"); Serial.print(CONTRACT_FLEX_MAX_DEG, 0); Serial.println(" and above extension."); }
        } else if (up == "CU") {                        // calibrate UNLOADED (step 1)
          cmdCalibrateUnloaded();
          Serial.println(deviceState == DEV_IDLE ? ">>> Calibrate UNLOADED requested" : ">>> Ignored: device must be IDLE to calibrate");
        } else if (up == "CS") {                        // calibrate STATIC (step 2)
          cmdCalibrateStatic();
          Serial.println(deviceState == DEV_IDLE ? ">>> Calibrate STATIC requested" : ">>> Ignored: device must be IDLE to calibrate");
        } else if (up == "B") {                         // B -> dump device/session state (bench diagnostic)
          Serial.print(">>> [STATE] dev="); Serial.print((int)deviceState);
          Serial.print(" fault=0x"); Serial.print(lastFaultCode, HEX);
          Serial.print(" buffer="); Serial.print(bufferCount);
          Serial.print(" ext="); Serial.print(THERAPIST_MIN_ROM_DEG, 1);
          Serial.print(" flex="); Serial.print(THERAPIST_MAX_ROM_DEG, 1);
          Serial.print(" spdIdx="); Serial.print(sessionSpeedIndex);
          Serial.print(" wifiPort="); Serial.println(appPort);
        } else if (up.startsWith("G")) {                // G<sec> -> start a bench set (0 = no auto-stop)
          uint16_t dur = (uint16_t)s.substring(1).toInt();
          uint8_t benchId[16]; for (int i = 0; i < 16; i++) benchId[i] = (uint8_t)(i + 1);
          cmdStart(benchId, dur);
          Serial.print(">>> START set (bench id) duration="); Serial.print(dur); Serial.println("s");
        } else if (s == "0") {                          // stop
          if (currentSpeedIndex == 0 && !pendingStopAtExtension && deviceState != DEV_RUNNING) {
            Serial.println(">>> Already stopped at full extension. <<<");
          } else {
            cmdStop();
            Serial.println(">>> STOP - finishing stride, will rest at full extension <<<");
          }
        } else {                                        // 1-12 -> speed preset
          int idx = s.toInt();
          if (idx >= 1 && idx <= NUM_SPEED_PRESETS) {
            cmdSetSpeedIndex(idx);
            Serial.print(">>> Speed configured: preset "); Serial.print(idx); Serial.print(" (");
            Serial.print(SPEED_PRESETS[idx - 1].label); Serial.println(") - applies on START (G), motor stays at rest <<<");
          } else {
            Serial.println(">>> Invalid. 0=stop, 1-12=speed, E<deg>, F<deg>, CU/CS=calibrate, G<sec>=start set");
          }
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
// RETIMED (this turn): breakpoints redistributed so each segment demands
// roughly EQUAL angular speed, instead of evenly-spaced 10% steps that let
// the 80-90% segment spike to ~3x the speed of every other segment - that
// spike is what was causing the real actuator to lag and trip the collapse
// check at higher presets. Same knee angle waypoints (kneeTable unchanged),
// same total cycle time - only WHEN each transition happens shifted.
const float gaitTable[TRAJ_SIZE] = {0, 3.3, 8.2, 13.1, 19.6, 27.8, 37.6, 49.2, 58.3, 87.7, 100};
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
  // The fixed reference scale [NORMAL_ROM_MIN_DEG, NORMAL_ROM_MAX_DEG] is the
  // DENOMINATOR (the curve's shape) and is never patient-configurable. The output
  // is mapped onto the per-session prescribed window [THERAPIST_MIN, THERAPIST_MAX].
  // CHANGED for Step D: the output floor is now THERAPIST_MIN_ROM_DEG (was the
  // const NORMAL_ROM_MIN_DEG), so a prescribed extension below 5 deg is honored.
  float minROM = THERAPIST_MIN_ROM_DEG;
  float maxROM = THERAPIST_MAX_ROM_DEG;
  if (maxROM < minROM) maxROM = minROM;
  float angleForScaling = normalAngle;
  if (angleForScaling < NORMAL_ROM_MIN_DEG) angleForScaling = NORMAL_ROM_MIN_DEG;
  float scaled = minROM + (angleForScaling - NORMAL_ROM_MIN_DEG) * (maxROM - minROM) / (NORMAL_ROM_MAX_DEG - NORMAL_ROM_MIN_DEG);
  return clampFloat(scaled, minROM, maxROM);
}

void freezeSystem(int emergencyMode) {
  systemFrozen = true;
  supportMode = emergencyMode;
  frozenAngle = commandedAngle;
  ledFrozenBlue();
  // Step D: surface the (one-way, hardware-truth) safety latch as DeviceState FAULT.
  // This is separate from the transient config_rejected code (0x10) and, like the
  // latch itself, is only cleared by a power cycle.
  deviceState = DEV_FAULT;
  lastFaultCode = (uint8_t)emergencyMode;
  setActive = false;
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
  float prevGaitPercent = Gait_percent;
  Gait_percent = wrapGaitPercent(Gait_percent + step);

  // Stride just completed (percent wrapped from ~100 back to ~0) while
  // already down to the slowest preset and a stop was requested - this
  // is the extension point (gait%=0, the curve's minimum). Finalize the
  // stop here instead of continuing, so it always rests at full
  // extension rather than wherever in the cycle it happened to be.
  if (pendingStopAtExtension && currentSpeedIndex == 1 && Gait_percent < prevGaitPercent) {
    pendingStopAtExtension = false;
    fullyStoppedAtExtension = true;
    currentSpeedIndex = 0;
    targetSpeedIndexUser = 0;
    simGaitSpeedPercentPerSec = 0.0f;
    Gait_percent = 0.0f;
  }

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

  if (fullyStoppedAtExtension) {
    // Bypass the look-ahead entirely - it would otherwise keep aiming
    // ~4% into the curve ahead of true extension (landing ~8-14 deg
    // instead of the actual floor). Target the floor directly.
    desiredAngleNormal = kneeTable[0];
    desiredAngleROM = THERAPIST_MIN_ROM_DEG;
  } else {
    lookAheadGaitPercent = wrapGaitPercent(gp + LOOKAHEAD_PERCENT);
    desiredAngleNormal = smoothClinicalAngle(lookAheadGaitPercent);
    desiredAngleROM = scaleAngleToPatientROM(desiredAngleNormal);
  }

  if (!commandFilterReady) { commandedAngle = desiredAngleROM; commandFilterReady = true; }
  else { commandedAngle = commandedAngle + COMMAND_SMOOTH_ALPHA * (desiredAngleROM - commandedAngle); }
  commandedAngle = clampFloat(commandedAngle, THERAPIST_MIN_ROM_DEG, THERAPIST_MAX_ROM_DEG);
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

  // Step D: FSR calibration is now ON-DEMAND and non-blocking (Serial CU/CS or the
  // app's BLE calibration opcodes), not a blocking boot step. Until calibrated, the
  // FSR bounds keep their full-range defaults (heel/mid Low=0, High=4095). The old
  // blocking calibrateFSR() is retained below for reference but no longer called.

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

      // Step D: drive on-demand calibration, and buffer raw samples while a set runs.
      serviceCalibration(heelRaw, midRaw);
      if (setActive) {
        appendSample((uint64_t)esp_timer_get_time(),
                     ax_g, ay_g, az_g, gx_rs, gy_rs, gz_rs,
                     (uint16_t)heelRaw, (uint16_t)midRaw);
      }

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
      deviceStatusNotify();   // Step D: periodic DeviceStatus push (also covers Serial-driven state changes)
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

    // Step D: WiFi provisioning-test / post-set upload state machine. Only does real
    // work when a request is pending (device IDLE or DONE) - never during a RUNNING set,
    // so it cannot disturb buffering or Core 1's motor loop.
    serviceNet();

    vTaskDelay(1);  // yield - prevents watchdog trip, negligible vs. the workload above
  }
}


/* ============================================================
   ============   STEP D — MOBILE-CONTRACT INTEGRATION   ========
   ============================================================

   New for Step D: a BLE GATT control plane + post-set WiFi/TCP batch
   upload that mirror the mobile/backend contract, driven through ONE
   internal command API that BOTH the Serial console and the BLE
   characteristic handlers call. Serial and the app work simultaneously.

   Core 1 (motor, 10ms) is NOT touched by any of this beyond reading the
   shared speed/ROM/stop flags it already read. All BLE/WiFi work happens
   in BLE callbacks and on Core 0; handlers only set flags, never block.
   ============================================================ */

// (DeviceState enum + deviceState/lastFaultCode/BATTERY_PCT_UNKNOWN declared near the
// top of the file so freezeSystem() can report FAULT.)

// ---- Shared session state (written by the command API, read by both cores) ----
// (setActive declared near the top so freezeSystem() can clear it.)
uint8_t currentSetId[16] = {0};            // from SetConfig, echoed in the TCP frame header
volatile int sessionSpeedIndex = 5;        // resolved from SetConfig.max_speed (1..12)
volatile bool durationActive = false;
volatile unsigned long sessionStartMs = 0;
volatile unsigned long sessionDurationMs = 0;  // 0 = no auto-stop

// ---- Non-blocking calibration state machine (Core 0) ----
// Replaces the old blocking calibrateFSR(). Two independently-triggerable,
// idle-gated steps: UNLOADED (foot lifted -> low bound) and STATIC (standing
// still -> high bound). Driven by the command API (Serial keys or BLE opcodes).
enum CalStep : uint8_t { CAL_NONE = 0, CAL_UNLOADED = 1, CAL_STATIC = 2 };
volatile CalStep calRequest = CAL_NONE;    // set by command API, consumed by Core 0
volatile bool calRunning = false;
CalStep calActive = CAL_NONE;
unsigned long calStartMs = 0;
int calHeelMin, calHeelMax, calMidMin, calMidMax;
const unsigned long CAL_CAPTURE_MS = 3000; // matches the old per-step window

// ---- Command API: the single source of truth (Serial + BLE both call these) ----

int speedIndexFromMaxSpeed(float maxSpeed) {     // R2: contract max_speed -> preset index
  int idx = (int)lroundf(maxSpeed);
  if (idx < 1) idx = 1;
  if (idx > NUM_SPEED_PRESETS) idx = NUM_SPEED_PRESETS;
  return idx;
}

// Speed is CONFIGURATION ONLY: it records the preset to use, but does NOT move the
// motor or change the device state. The motor only moves on START (which applies
// sessionSpeedIndex). Each config command is independent — speed does not touch
// duration/ROM, and configuring speed never starts a set. (STOP, by contrast, is an
// immediate action handled by cmdStop, not by this function.)
void cmdSetSpeedIndex(int idx) {
  if (idx < 1) idx = 1;
  if (idx > NUM_SPEED_PRESETS) idx = NUM_SPEED_PRESETS;
  sessionSpeedIndex = idx;
}

bool cmdSetExtension(float deg) {                // returns false if out of contract range
  if (deg < CONTRACT_EXT_MIN_DEG || deg > CONTRACT_EXT_MAX_DEG) return false;
  if (deg >= THERAPIST_MAX_ROM_DEG) return false; // extension must stay below flexion
  THERAPIST_MIN_ROM_DEG = deg;
  return true;
}

bool cmdSetFlexion(float deg) {
  if (deg < CONTRACT_FLEX_MIN_DEG || deg > CONTRACT_FLEX_MAX_DEG) return false;
  if (deg <= THERAPIST_MIN_ROM_DEG) return false;
  THERAPIST_MAX_ROM_DEG = deg;
  return true;
}

// Forward decls for buffer/upload (defined in the BLE/WiFi section below).
void bufferReset();
void requestUpload();

void cmdStart(const uint8_t setId[16], uint16_t durationS) {
  if (deviceState == DEV_FAULT) return;          // never start while frozen/faulted
  if (deviceState != DEV_IDLE) return;           // a set is already running/finishing — ignore
  if (netState != NET_IDLE) return;              // previous upload/test still in flight (review Issue 2)
  memcpy(currentSetId, setId, 16);
  pendingStopAtExtension = false;
  fullyStoppedAtExtension = false;
  targetSpeedIndexUser = sessionSpeedIndex;       // resolved earlier from SetConfig
  sessionDurationMs = (unsigned long)durationS * 1000UL;
  sessionStartMs = millis();
  durationActive = (durationS > 0);
  bufferReset();
  setActive = true;                               // Core 0 begins appending samples
  deviceState = DEV_RUNNING;
}

void cmdStop() {
  // R4: data is "done" the instant STOP is processed -> freeze the buffer now and
  // let the motor's graceful stop-at-extension run in parallel. Reuses the EXISTING
  // stop path (same as Serial '0'); no second stop implementation.
  bool wasActive = setActive;
  setActive = false;
  durationActive = false;
  if (!(currentSpeedIndex == 0 && !pendingStopAtExtension)) {
    targetSpeedIndexUser = 1;          // step down to slowest; existing logic rests at extension
    pendingStopAtExtension = true;
  }
  if (deviceState == DEV_RUNNING) {
    deviceState = DEV_DONE;            // batch ready; upload may begin while motor settles
    if (wasActive) requestUpload();    // no-op when WiFi not provisioned (Serial-only bench)
  }
}

void cmdCalibrateUnloaded() { if (deviceState == DEV_IDLE && !calRunning) calRequest = CAL_UNLOADED; } // §5.5 idle-gated
void cmdCalibrateStatic()   { if (deviceState == DEV_IDLE && !calRunning) calRequest = CAL_STATIC; }

// Non-blocking calibration, serviced once per Core 0 sensor tick with the raw FSR reads.
void serviceCalibration(int heelRaw, int midRaw) {
  if (!calRunning) {
    if (calRequest == CAL_NONE) return;
    calActive = calRequest;
    calRequest = CAL_NONE;
    calRunning = true;
    calStartMs = millis();
    calHeelMin = 4095; calHeelMax = 0; calMidMin = 4095; calMidMax = 0;
    xSemaphoreTake(serialMutex, portMAX_DELAY);
    Serial.print(">>> CAL start: "); Serial.println(calActive == CAL_UNLOADED ? "UNLOADED (lift foot)" : "STATIC (stand still)");
    xSemaphoreGive(serialMutex);
    return;
  }
  calHeelMin = min(calHeelMin, heelRaw); calHeelMax = max(calHeelMax, heelRaw);
  calMidMin  = min(calMidMin, midRaw);   calMidMax  = max(calMidMax, midRaw);
  if (millis() - calStartMs < CAL_CAPTURE_MS) return;
  if (calActive == CAL_UNLOADED) { heelLow = calHeelMax; midLow = calMidMax; }
  else                           { heelHigh = calHeelMax; midHigh = calMidMax; }
  calRunning = false;
  calActive = CAL_NONE;
  xSemaphoreTake(serialMutex, portMAX_DELAY);
  Serial.print(">>> CAL done -> heel ["); Serial.print(heelLow); Serial.print(","); Serial.print(heelHigh);
  Serial.print("] mid ["); Serial.print(midLow); Serial.print(","); Serial.print(midHigh); Serial.println("]");
  xSemaphoreGive(serialMutex);
}

/* ============================================================
   =========   STEP D — BLE GATT + WiFi/TCP DATA PLANE   ========
   ============================================================
   Byte layouts here are byte-for-byte mirrors of the iOS reference
   (KnevoPatient/.../Bluetooth/KnevoCodec.swift). All multi-byte fields
   are little-endian. Serialization is explicit (no packed-struct casts)
   to avoid alignment-padding drift (gap-analysis §5.1).
   ============================================================ */

// GATT UUIDs (verbatim from the agreed contract / KnevoGATT.swift)
#define KNEVO_SVC_UUID        "94A4B5CC-14F9-40BF-9631-62954DC8D647"
#define KNEVO_WIFICONFIG_UUID "A4F7B9FF-AC9E-4171-A9FF-461F1180F124"
#define KNEVO_WIFISTATUS_UUID "23F1E73A-68CD-4D6A-85DC-B33BA4ED6153"
#define KNEVO_SETCONFIG_UUID  "4FCD372C-910F-46B1-94AB-F3C06597EB60"
#define KNEVO_CONTROL_UUID    "93413C6D-977F-490B-AEC3-AF800C18FFA9"
#define KNEVO_DEVSTATUS_UUID  "0439E3D3-AA83-4A13-8AE4-CE19AB85B832"
#define KNEVO_BLE_NAME        "knevo_exo"   // must keep the knevo_ prefix the app scans for

const char* FIRMWARE_VERSION = "knevo-stepD-1.0";

#define OP_START        0x01
#define OP_STOP         0x02
#define OP_CAL_UNLOADED 0x10
#define OP_CAL_STATIC   0x11

// ---- little-endian writers / readers ----
static inline void wrU16(uint8_t* p, uint16_t v) { p[0] = v & 0xFF; p[1] = (v >> 8) & 0xFF; }
static inline void wrU32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (v >> (8 * i)) & 0xFF; }
static inline void wrU64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (v >> (8 * i)) & 0xFF; }
static inline void wrF32(uint8_t* p, float f) { uint32_t b; memcpy(&b, &f, 4); wrU32(p, b); }
static inline uint16_t rdU16(const uint8_t* p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
static inline float rdF32(const uint8_t* p) {
  uint32_t b = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
  float f; memcpy(&f, &b, 4); return f;
}

// ---- Sample buffer (PSRAM); records pre-serialized in the 88-byte contract layout ----
const int SAMPLE_SIZE = 88;
const uint32_t MAX_BUFFER_SAMPLES = 20000;   // ~1.76 MB PSRAM; minutes at the real ~16-91 Hz rate
uint8_t* sampleBuffer = nullptr;
volatile uint32_t bufferCount = 0;
volatile bool bufferTruncated = false;
uint32_t sampleSeq = 0;

void bufferReset() { bufferCount = 0; bufferTruncated = false; sampleSeq = 0; }  // flags/memory only, no mutex/print

// Called from Core 0 each sensor tick while a set is active. Record layout matches
// KnevoCodec.encodeSensorBatch: u64 ts, u32 sample_id, 18x f32 (foot/shank/thigh
// each ax,ay,az,gx,gy,gz), u16 heel_raw, u16 mid_raw.
void appendSample(uint64_t ts_us,
                  float ax_g[3], float ay_g[3], float az_g[3],
                  float gx_rs[3], float gy_rs[3], float gz_rs[3],
                  uint16_t heelRaw, uint16_t midRaw) {
  if (sampleBuffer == nullptr) return;
  if (bufferCount >= MAX_BUFFER_SAMPLES) { bufferTruncated = true; return; }
  uint8_t* p = sampleBuffer + (size_t)bufferCount * SAMPLE_SIZE;
  wrU64(p, ts_us); p += 8;
  wrU32(p, sampleSeq++); p += 4;
  for (int i = 0; i < 3; i++) {   // foot(0), shank(1), thigh(2)
    wrF32(p, ax_g[i]);  p += 4; wrF32(p, ay_g[i]);  p += 4; wrF32(p, az_g[i]);  p += 4;
    wrF32(p, gx_rs[i]); p += 4; wrF32(p, gy_rs[i]); p += 4; wrF32(p, gz_rs[i]); p += 4;
  }
  wrU16(p, heelRaw); p += 2; wrU16(p, midRaw); p += 2;
  bufferCount++;
}

// ---- WiFi provisioning state (from WiFiConfig writes) ----
char     wifiSsid[33] = {0};
char     wifiPass[64] = {0};
uint8_t  appIp[4]     = {0, 0, 0, 0};
uint16_t appPort      = 0;            // 0 => not provisioned (Serial-only mode skips upload)
char     lastTestedSsid[33] = {0};
volatile bool wifiTestRequested = false;   // provisioning connection test
volatile bool uploadRequested   = false;   // post-set batch upload

// ---- BLE characteristic handles + notify helpers ----
NimBLECharacteristic* chWifiStatus = nullptr;
NimBLECharacteristic* chDevStatus  = nullptr;

void deviceStatusNotify() {
  if (!chDevStatus) return;
  uint8_t b[3] = { (uint8_t)deviceState, BATTERY_PCT_UNKNOWN, lastFaultCode };  // §3.5: state, battery, fault
  chDevStatus->setValue(b, 3);
  chDevStatus->notify();
}
void wifiStatusNotify(uint8_t status, uint8_t err) {   // §3.3: 0=OK, else err code
  if (!chWifiStatus) return;
  uint8_t b[2] = { status, err };
  chWifiStatus->setValue(b, 2);
  chWifiStatus->notify();
}

volatile unsigned long uploadReqMs = 0;
const unsigned long UPLOAD_SETTLE_TIMEOUT_MS = 8000;   // proceed even if "fully stopped" never asserts

void requestUpload() {
  if (appPort == 0) { deviceState = DEV_IDLE; return; }  // Serial-only bench: nothing to upload
  uploadReqMs = millis();
  uploadRequested = true;
}

// ---- WiFi/TCP non-blocking-ish state machine (runs on Core 0, device IDLE/DONE) ----
// (NetState enum + netState are declared near the top so cmdStart can check them.)
bool netForUpload = false;
unsigned long netWifiStartMs = 0;
const unsigned long WIFI_CONNECT_TIMEOUT_MS = 15000;

// Returns true on a successful send + ACK. Yields between chunks so a multi-second
// upload doesn't starve Core 0's IDLE task and trip the task watchdog (review Issue 2).
bool doUpload() {
  WiFiClient client;
  IPAddress ip(appIp[0], appIp[1], appIp[2], appIp[3]);
  if (!client.connect(ip, appPort, 5000)) {
    xSemaphoreTake(serialMutex, portMAX_DELAY);
    Serial.println(">>> Upload: TCP connect to app failed");
    xSemaphoreGive(serialMutex);
    return false;
  }
  uint32_t n = bufferCount;
  uint32_t frameLen = 28 + n * SAMPLE_SIZE;   // header(28) + records
  uint8_t hdr[32]; uint8_t* p = hdr;
  wrU32(p, frameLen); p += 4;                 // 4-byte length prefix
  memcpy(p, "KNVO", 4); p += 4;
  *p++ = 1;                                    // version
  *p++ = 0;                                    // flags
  memcpy(p, currentSetId, 16); p += 16;
  wrU32(p, n); p += 4;                         // sample_count
  wrU16(p, (uint16_t)SAMPLE_SIZE); p += 2;     // sample_size = 88  (total hdr = 32)
  client.write(hdr, 32);
  size_t total = (size_t)n * SAMPLE_SIZE, off = 0;
  while (off < total) {
    size_t chunk = min((size_t)1024, total - off);
    client.write(sampleBuffer + off, chunk);
    off += chunk;
    vTaskDelay(1);                             // yield between chunks (watchdog-safe)
  }
  client.flush();
  unsigned long t = millis();                  // await 1-byte ACK (0x06)
  while (!client.available() && millis() - t < 5000) vTaskDelay(2);
  bool acked = client.available() && (client.read() == 0x06);
  client.stop();
  xSemaphoreTake(serialMutex, portMAX_DELAY);
  Serial.print(">>> Upload sent "); Serial.print(n); Serial.print(" samples, ACK=");
  Serial.println(acked ? "yes" : "no"); if (bufferTruncated) Serial.println(">>> WARNING: buffer was truncated (set exceeded cap)");
  xSemaphoreGive(serialMutex);
  return acked;
}

void startWifiConnect(bool forUpload) {
  netForUpload = forUpload;
  WiFi.mode(WIFI_STA);
  WiFi.begin(wifiSsid, wifiPass);
  netWifiStartMs = millis();
  netState = NET_WIFI_CONNECTING;
}

void serviceNet() {
  if (netState == NET_IDLE) {
    if (wifiTestRequested) {
      if (deviceState != DEV_IDLE) return;                           // never run a WiFi test mid-set (Issue 1) — defer
      wifiTestRequested = false;
      if (wifiSsid[0] == 0) { wifiStatusNotify(1, 2); return; }       // 2 = ssid_not_found / not set
      if (strcmp(wifiSsid, lastTestedSsid) == 0) { wifiStatusNotify(0, 0); return; } // already validated
      startWifiConnect(false);
    } else if (uploadRequested) {
      if (wifiSsid[0] == 0) { uploadRequested = false; deviceState = DEV_IDLE; return; }
      // Keep the radio OFF until the motor has physically settled — the graceful
      // stop stride is still motion (review smaller-note). Data was already frozen
      // at STOP; only the radio waits. A safety timeout prevents a missed settle
      // from stranding the upload.
      if (!fullyStoppedAtExtension && (millis() - uploadReqMs < UPLOAD_SETTLE_TIMEOUT_MS)) return;
      uploadRequested = false;
      startWifiConnect(true);
    }
    return;
  }
  // NET_WIFI_CONNECTING
  if (WiFi.status() == WL_CONNECTED) {
    if (netForUpload) {
      bool ok = doUpload();
      if (!ok) ok = doUpload();                                      // one retry over the live WiFi (review Issue 3)
      WiFi.disconnect(true); WiFi.mode(WIFI_OFF);
      lastFaultCode = ok ? 0 : 0x12;                                 // 0x12 = upload_failed (recoverable; buffer kept)
      deviceState = DEV_IDLE;
      deviceStatusNotify();
    } else {
      strncpy(lastTestedSsid, wifiSsid, sizeof(lastTestedSsid) - 1);
      wifiStatusNotify(0, 0);                                        // provisioning OK
      WiFi.disconnect(true); WiFi.mode(WIFI_OFF);                    // device disconnects WiFi after the test
    }
    netState = NET_IDLE;
  } else if (millis() - netWifiStartMs > WIFI_CONNECT_TIMEOUT_MS) {
    WiFi.disconnect(true); WiFi.mode(WIFI_OFF);
    if (netForUpload) { lastFaultCode = 0x13; deviceState = DEV_IDLE; deviceStatusNotify(); } // 0x13 = wifi_join_failed
    else wifiStatusNotify(1, 3);                                    // 3 = timeout
    netState = NET_IDLE;
  }
}

// ---- BLE write-handler state shared with Control START ----
uint8_t  pendingSetId[16] = {0};
uint16_t pendingDurationS = 0;

// ---- NimBLE characteristic callbacks (tiny, non-blocking: parse + set flags only) ----
class WiFiConfigCB : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* c, NimBLEConnInfo& connInfo) override {
    std::string v = c->getValue();
    const uint8_t* d = (const uint8_t*)v.data();
    if (v.size() < 7) return;                       // u16 port + 4B ip + u8 ssid_len + u8 pass_len minimum
    size_t o = 0;
    appPort = rdU16(d + o); o += 2;                 // ALWAYS refresh IP + port
    appIp[0] = d[o]; appIp[1] = d[o + 1]; appIp[2] = d[o + 2]; appIp[3] = d[o + 3]; o += 4;
    uint8_t ssidLen = d[o++];
    if (ssidLen == 0) {
      // Per-set "refresh IP/port only" write (the coordinator sends empty creds
      // before every set): KEEP the provisioned SSID/password and do NOT re-test.
      // Without this, the empty write would wipe the credentials provisioned
      // earlier and the post-set WiFi upload could never connect.
      return;
    }
    if (o + ssidLen + 1 > v.size()) return;
    uint8_t n1 = ssidLen < 32 ? ssidLen : 32; memcpy(wifiSsid, d + o, n1); wifiSsid[n1] = 0; o += ssidLen;
    uint8_t passLen = d[o++]; if (o + passLen > v.size()) return;
    uint8_t n2 = passLen < 63 ? passLen : 63; memcpy(wifiPass, d + o, n2); wifiPass[n2] = 0;
    // Only kick off a provisioning test when IDLE — never bring WiFi up mid-set
    // (review Issue 1). Creds are still stored above; the test just waits for idle.
    if (deviceState == DEV_IDLE) {
      lastTestedSsid[0] = 0;                          // new creds -> force a fresh provisioning test
      wifiTestRequested = true;                       // serviceNet() connects, tests, reports WiFiStatus
    }
  }
};

class SetConfigCB : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* c, NimBLEConnInfo& connInfo) override {
    std::string v = c->getValue();
    if (v.size() < 30) return;                      // 16B id + u16 dur + 3x f32
    if (deviceState != DEV_IDLE) return;            // ignore config changes mid-set — no mid-stride ROM jump (Issue 1)
    const uint8_t* d = (const uint8_t*)v.data();
    float ms = rdF32(d + 18), me = rdF32(d + 22), mf = rdF32(d + 26);
    bool ok = (ms >= 1.0f && ms <= 12.0f)
           && (me >= CONTRACT_EXT_MIN_DEG && me <= CONTRACT_EXT_MAX_DEG)
           && (mf >= CONTRACT_FLEX_MIN_DEG && mf <= CONTRACT_FLEX_MAX_DEG)
           && (me < mf);
    if (!ok) {                                      // D-ext1.6: reject out-of-range, do NOT start
      lastFaultCode = 0x10;                         // config_rejected (NOT the safety latch; state stays IDLE)
      deviceStatusNotify();
      return;
    }
    memcpy(pendingSetId, d, 16);
    pendingDurationS = rdU16(d + 16);
    sessionSpeedIndex = speedIndexFromMaxSpeed(ms);
    cmdSetExtension(me);
    cmdSetFlexion(mf);
    lastFaultCode = 0;
    deviceStatusNotify();
  }
};

class ControlCB : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* c, NimBLEConnInfo& connInfo) override {
    std::string v = c->getValue();
    if (v.size() < 1) return;
    switch ((uint8_t)v[0]) {
      case OP_START:        cmdStart(pendingSetId, pendingDurationS); break;
      case OP_STOP:         cmdStop(); break;
      case OP_CAL_UNLOADED: cmdCalibrateUnloaded(); break;
      case OP_CAL_STATIC:   cmdCalibrateStatic(); break;
      default: return;
    }
    deviceStatusNotify();
  }
};

void bleSetup() {
  NimBLEDevice::init(KNEVO_BLE_NAME);
  NimBLEServer* server = NimBLEDevice::createServer();
  NimBLEService* svc = server->createService(KNEVO_SVC_UUID);

  NimBLECharacteristic* chWifiCfg = svc->createCharacteristic(KNEVO_WIFICONFIG_UUID, NIMBLE_PROPERTY::WRITE);
  chWifiCfg->setCallbacks(new WiFiConfigCB());

  chWifiStatus = svc->createCharacteristic(KNEVO_WIFISTATUS_UUID, NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);

  NimBLECharacteristic* chSetCfg = svc->createCharacteristic(KNEVO_SETCONFIG_UUID, NIMBLE_PROPERTY::WRITE);
  chSetCfg->setCallbacks(new SetConfigCB());

  NimBLECharacteristic* chControl = svc->createCharacteristic(KNEVO_CONTROL_UUID, NIMBLE_PROPERTY::WRITE);
  chControl->setCallbacks(new ControlCB());

  chDevStatus = svc->createCharacteristic(KNEVO_DEVSTATUS_UUID, NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);
  uint8_t initStatus[3] = { (uint8_t)deviceState, BATTERY_PCT_UNKNOWN, 0 };
  chDevStatus->setValue(initStatus, 3);

  svc->start();
  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->addServiceUUID(KNEVO_SVC_UUID);     // the app scans/filters by this service UUID
  NimBLEDevice::startAdvertising();
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
  Serial.println("Boot state: IDLE, motor at rest at extension. Type a speed (1-12) or G<sec> to start.");
  printSpeedMenu();

  if (motorOn()) { Serial.println("Motor ON confirmed"); }
  else { Serial.println("Motor ON failed"); }

  // Step B's full setup must complete before its task starts.
  setupStepB();

  // Step D: PSRAM sample buffer + BLE peripheral, before Core 0's task starts using them.
  size_t bufBytes = (size_t)MAX_BUFFER_SAMPLES * SAMPLE_SIZE;
  sampleBuffer = (uint8_t*)heap_caps_malloc(bufBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!sampleBuffer) Serial.println("WARNING: PSRAM sample buffer alloc failed - session upload disabled until fixed.");
  else { Serial.print("Sample buffer: "); Serial.print(bufBytes / 1024); Serial.println(" KB in PSRAM."); }
  bleSetup();
  Serial.println("BLE advertising as '" KNEVO_BLE_NAME "'.");

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
unsigned long lastSpeedPrintTime = 0;
const unsigned long SPEED_PRINT_MS = 250;  // 4x/sec - enough resolution to watch the 1-per-second steps clearly

void printSpeedStatus() {
  xSemaphoreTake(serialMutex, portMAX_DELAY);
  Serial.print("[SPEED] currentIdx="); Serial.print(currentSpeedIndex);
  Serial.print(" targetIdx="); Serial.print(targetSpeedIndexUser);
  Serial.print(" speed="); Serial.print(simGaitSpeedPercentPerSec, 1);
  Serial.print(" pendingStop="); Serial.print(pendingStopAtExtension ? 1 : 0);
  Serial.print(" stoppedAtExt="); Serial.print(fullyStoppedAtExtension ? 1 : 0);
  Serial.print(" commandedAngle="); Serial.print(commandedAngle, 1);
  Serial.print(" cycle=");
  if (simGaitSpeedPercentPerSec > 0.01f) { Serial.print(100.0f / simGaitSpeedPercentPerSec, 2); Serial.println("s"); }
  else { Serial.println("stopped"); }
  xSemaphoreGive(serialMutex);
}

void loop() {
  unsigned long now = millis();

  checkSpeedCommand();
  updateSpeedStepping();

  // Step D (D6): session duration elapsed -> the SAME stop path as a manual STOP.
  if (durationActive && (millis() - sessionStartMs >= sessionDurationMs)) {
    cmdStop();
  }

  if (now - lastSpeedPrintTime >= SPEED_PRINT_MS) {
    lastSpeedPrintTime = now;
    printSpeedStatus();
  }

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
