/*
  KNEVO STEP A — OPEN-LOOP CONTROL FIRMWARE
  ----------------------------------------------------------------
  This is the original, tested moresmoother.ino control logic
  (FSM status, cubic Hermite spline trajectory, look-ahead, command
  smoothing, jump-validation, motor streaming, collapse-detection
  safety path verified in Step 0) - with the real-time FSR/heel-
  strike tracker DELIBERATELY REMOVED.

  Why: the tracker is new, untested under real worn conditions, and
  has already surfaced three real bugs in bench testing today. The
  control path itself has a long, clean track record. For tomorrow's
  live demo, this firmware drives the motor exactly like the
  original tested version - the sensor + DL work (Step B) runs
  separately, logs only, and is not in this firmware's path at all.

  WHAT CHANGED vs. the original tested moresmoother.ino:
    1. SIM_GAIT_SPEED_PERCENT_PER_SEC: 8.0 -> 55.6, i.e. cycle length
       12.5s -> ~1.8s. The old value was only ever chosen to make a
       curve easy to watch on a bench plotter; 1.8s sits in the
       middle of the real measured cycle range (1.3-2.9s) from the
       actual dataset. Retune this single constant if the demo needs
       a faster/slower pace - SIM_GAIT_SPEED_PERCENT_PER_SEC =
       100.0 / desired_cycle_seconds.
    2. Buzzer -> built-in RGB LED (solid BLUE on freeze), carried
       over from Step 0/2 per your earlier request.
  Everything else - FSM table, spline math, ROM scaling, motor
  protocol, jump-validation, collapse-detection - is unchanged from
  the version already bench-tested today.
*/

#include <math.h>

#define MOTOR_RX_PIN 4
#define MOTOR_TX_PIN 5

HardwareSerial MotorSerial(1);

// Built-in RGB LED.
#ifndef BUILTIN_RGB_LED_PIN
  #ifdef RGB_BUILTIN
    #define BUILTIN_RGB_LED_PIN RGB_BUILTIN
  #else
    #define BUILTIN_RGB_LED_PIN 48
  #endif
#endif
#define RGB_LED_BRIGHTNESS 35

// ============================================================
// MOTOR CALIBRATION / SESSION SETTINGS
// ============================================================
const float MOTOR_SCALE = 3722.0f;

// Doctor/session ROM. Change this per patient/session.
float THERAPIST_MAX_ROM_DEG = 65.0f;

const float NORMAL_ROM_MIN_DEG = 5.0f;
const float NORMAL_ROM_MAX_DEG = 65.0f;

// ============================================================
// SMOOTH MOTION SETTINGS
// ============================================================
const unsigned long CONTROL_MS  = 10;   // stream motor target every 10 ms = 100 Hz
const unsigned long PLOT_MS     = 50;   // serial plotter update
const unsigned long FEEDBACK_MS = 250;  // actual angle read slower to avoid blocking motion

const float LOOKAHEAD_PERCENT = 4.0f;
// Softened from 0.18 -> 0.12: the early-stance region (0-30% gait) is the
// only spot with a quick direction reversal, which likely shows up as
// gearbox backlash (a fast snap at that one spot, smooth elsewhere even at
// higher speed - confirmed from video). A gentler filter eases into the
// reversal more gradually instead of commanding it abruptly.
const float COMMAND_SMOOTH_ALPHA = 0.12f;

// Simulated gait input - retuned to a realistic walking cycle.
// (100.0 / SIM_GAIT_SPEED_PERCENT_PER_SEC = cycle length in seconds)
// NOTE: COMMAND_SMOOTH_ALPHA and LOOKAHEAD_PERCENT below were tuned and
// proven smooth at the ORIGINAL ~12.5s pace. Speeding the cycle up makes
// the actuator track the same curve shape proportionally faster, which
// shows up first in the most curvature-dense part of the curve (the
// 0-30% early-stance "knee flexion wave"). If it's still rough at this
// pace, slow it further before retuning ALPHA/LOOKAHEAD blind.
const float SIM_GAIT_SPEED_PERCENT_PER_SEC = 45.5f;

// ML / gait_percent plausibility (kept as-is; not exercised by the
// simulated input, which never jumps, but left in place since this
// is otherwise the same tested validation path).
const float MAX_GAIT_JUMP_PERCENT = 10.0f;
const int   MAX_BAD_READINGS      = 3;

// Scenario 2B emergency collapse threshold.
const float COLLAPSE_ERROR_DEG = 20.0f;

// ============================================================
// MOTOR COMMANDS
// ============================================================
uint8_t cmdOn[5] = {0x3E, 0x88, 0x01, 0x00, 0xC7};

// ============================================================
// FSM PHASE TABLE: STATUS ONLY
// ============================================================
struct FSMState {
  int id;
  float gpStart;
  float gpEnd;
  const char* name;
};

const FSMState fsmTable[6] = {
  {1,  0.0f,   5.0f,  "HeelStrike"       },
  {2,  5.0f,  15.0f,  "LoadingResponse"  },
  {3, 15.0f,  50.0f,  "MidTerminalStance"},
  {4, 50.0f,  62.0f,  "PreSwingToeOff"   },
  {5, 62.0f,  87.0f,  "InitialMidSwing"  },
  {6, 87.0f, 100.0f,  "TerminalSwing"    }
};

// ============================================================
// CLINICAL KNEE ANGLE TABLE
// ============================================================
const int TRAJ_SIZE = 11;
const float gaitTable[TRAJ_SIZE] = {
  0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
};

const float kneeTable[TRAJ_SIZE] = {
  3.97, 19.84, 18.86, 11.09, 7.72, 13.86, 38.74, 64.12, 53.27, 17.27, 2.21
};

// ============================================================
// RUNTIME VARIABLES
// ============================================================
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
int supportMode = 0;  // 0 normal, 1 stance support, 2 emergency freeze

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

// ============================================================
// BASIC UTILITIES
// ============================================================
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

float clampFloat(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

// ============================================================
// SMOOTH CUBIC HERMITE INTERPOLATION
// ============================================================
float tableSlope(int i) {
  if (i <= 0) {
    return (kneeTable[1] - kneeTable[0]) / (gaitTable[1] - gaitTable[0]);
  }
  if (i >= TRAJ_SIZE - 1) {
    return (kneeTable[TRAJ_SIZE - 1] - kneeTable[TRAJ_SIZE - 2]) /
           (gaitTable[TRAJ_SIZE - 1] - gaitTable[TRAJ_SIZE - 2]);
  }
  return (kneeTable[i + 1] - kneeTable[i - 1]) /
         (gaitTable[i + 1] - gaitTable[i - 1]);
}

float smoothClinicalAngle(float gp) {
  gp = clampFloat(gp, 0.0f, 100.0f);
  if (gp <= gaitTable[0]) return kneeTable[0];
  if (gp >= gaitTable[TRAJ_SIZE - 1]) return kneeTable[TRAJ_SIZE - 1];

  for (int i = 0; i < TRAJ_SIZE - 1; i++) {
    if (gp >= gaitTable[i] && gp <= gaitTable[i + 1]) {
      float x0 = gaitTable[i];
      float x1 = gaitTable[i + 1];
      float y0 = kneeTable[i];
      float y1 = kneeTable[i + 1];
      float m0 = tableSlope(i);
      float m1 = tableSlope(i + 1);

      float dx = x1 - x0;
      float t = (gp - x0) / dx;
      float t2 = t * t;
      float t3 = t2 * t;

      float h00 =  2.0f * t3 - 3.0f * t2 + 1.0f;
      float h10 =        t3 - 2.0f * t2 + t;
      float h01 = -2.0f * t3 + 3.0f * t2;
      float h11 =        t3 -       t2;

      float y = h00 * y0 + h10 * dx * m0 + h01 * y1 + h11 * dx * m1;
      y = clampFloat(y, 0.0f, NORMAL_ROM_MAX_DEG);
      return y;
    }
  }
  return kneeTable[0];
}

// ============================================================
// ROM SCALING THAT PRESERVES GAIT SHAPE
// ============================================================
float scaleAngleToPatientROM(float normalAngle) {
  float maxROM = THERAPIST_MAX_ROM_DEG;
  if (maxROM < NORMAL_ROM_MIN_DEG) {
    maxROM = NORMAL_ROM_MIN_DEG;
  }

  float angleForScaling = normalAngle;
  if (angleForScaling < NORMAL_ROM_MIN_DEG) {
    angleForScaling = NORMAL_ROM_MIN_DEG;
  }

  float scaled = NORMAL_ROM_MIN_DEG +
                 (angleForScaling - NORMAL_ROM_MIN_DEG) *
                 (maxROM - NORMAL_ROM_MIN_DEG) /
                 (NORMAL_ROM_MAX_DEG - NORMAL_ROM_MIN_DEG);

  return clampFloat(scaled, NORMAL_ROM_MIN_DEG, maxROM);
}

// ============================================================
// FSM STATUS
// ============================================================
int getFSMStatusCode(float gp) {
  gp = clampFloat(gp, 0.0f, 100.0f);
  for (int i = 0; i < 6; i++) {
    if (i < 5) {
      if (gp >= fsmTable[i].gpStart && gp < fsmTable[i].gpEnd) return fsmTable[i].id;
    } else {
      if (gp >= fsmTable[i].gpStart && gp <= fsmTable[i].gpEnd) return fsmTable[i].id;
    }
  }
  return 6;
}

const char* fsmStatusName(int code) {
  for (int i = 0; i < 6; i++) {
    if (fsmTable[i].id == code) return fsmTable[i].name;
  }
  return "Unknown";
}

bool isStanceSupportState(int stateId) {
  return (stateId == 2 || stateId == 3);
}

// ============================================================
// LED / FREEZE
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

void freezeSystem(int emergencyMode) {
  systemFrozen = true;
  supportMode = emergencyMode;
  frozenAngle = commandedAngle;
  ledFrozenBlue();
}

// ============================================================
// MOTOR FUNCTIONS
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

  // No delay here. This avoids stop-go behavior.
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
  return actualAngle;
}

// ============================================================
// GAIT INPUT + VALIDATION
// ============================================================
float getGaitPercentInput() {
  // Simulated gait input. This is the open-loop reference signal
  // for tomorrow's demo - the wearer moves along with this fixed
  // rhythm rather than the device sensing/adapting to them.
  unsigned long now = millis();

  if (lastGaitUpdateTime == 0) {
    lastGaitUpdateTime = now;
    return Gait_percent;
  }

  float dt = (now - lastGaitUpdateTime) / 1000.0f;
  lastGaitUpdateTime = now;

  float step = SIM_GAIT_SPEED_PERCENT_PER_SEC * dt;
  Gait_percent = wrapGaitPercent(Gait_percent + step);

  return Gait_percent;
}

float validateOrPredictGait(float newGaitPercent) {
  newGaitPercent = clampFloat(newGaitPercent, 0.0f, 100.0f);

  if (!hasFirstValidGait) {
    hasFirstValidGait = true;
    badReading = false;
    badReadingCounter = 0;
    lastValidGaitPercent = newGaitPercent;
    predictedGaitPercent = newGaitPercent;
    return newGaitPercent;
  }

  float jump = gaitForwardDiff(lastValidGaitPercent, newGaitPercent);
  bool validForwardMotion = (jump >= 0.0f && jump <= MAX_GAIT_JUMP_PERCENT);

  if (validForwardMotion) {
    badReading = false;
    badReadingCounter = 0;
    if (jump > 0.0f && jump < MAX_GAIT_JUMP_PERCENT) {
      gaitRatePerUpdate = jump;
    }
    lastValidGaitPercent = newGaitPercent;
    predictedGaitPercent = newGaitPercent;
    return newGaitPercent;
  }

  badReading = true;
  badReadingCounter++;
  predictedGaitPercent = wrapGaitPercent(predictedGaitPercent + gaitRatePerUpdate);

  if (badReadingCounter > MAX_BAD_READINGS) {
    freezeSystem(2);
  }
  return predictedGaitPercent;
}

// ============================================================
// TRAJECTORY UPDATE
// ============================================================
void updateSmoothFSMAndTrajectory(float gp) {
  statusCode = getFSMStatusCode(gp);
  supportMode = isStanceSupportState(statusCode) ? 1 : 0;

  lookAheadGaitPercent = wrapGaitPercent(gp + LOOKAHEAD_PERCENT);
  desiredAngleNormal = smoothClinicalAngle(lookAheadGaitPercent);
  desiredAngleROM = scaleAngleToPatientROM(desiredAngleNormal);

  if (!commandFilterReady) {
    commandedAngle = desiredAngleROM;
    commandFilterReady = true;
  } else {
    commandedAngle = commandedAngle + COMMAND_SMOOTH_ALPHA * (desiredAngleROM - commandedAngle);
  }

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
      freezeSystem(2);
    }
  }
}

// ============================================================
// SERIAL PLOTTER OUTPUT
// ============================================================
void printPlotterLine() {
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
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(3000);

  ledOff();

  MotorSerial.begin(115200, SERIAL_8N1, MOTOR_RX_PIN, MOTOR_TX_PIN);

  Serial.println("=== KNEVO STEP A: OPEN-LOOP CONTROL FIRMWARE ===");
  Serial.print("Simulated gait cycle length: ");
  Serial.print(100.0f / SIM_GAIT_SPEED_PERCENT_PER_SEC, 2);
  Serial.println(" s");

  if (motorOn()) {
    Serial.println("Motor ON confirmed");
  } else {
    Serial.println("Motor ON failed");
  }

  rawMLGaitPercent = 0.0f;
  usedGaitPercent = 0.0f;
  updateSmoothFSMAndTrajectory(usedGaitPercent);

  sendPositionFast(commandedAngle);
  lastSentAngle = commandedAngle;

  delay(500);
  actualAngle = readActualAngleQuick();

  Serial.println("RawGait,UsedGait,LookAheadGait,State_x10,DesiredNormal_deg,DesiredROM_deg,Commanded_deg,Actual_deg,BadReading_x20,ErrorCount_x5,SupportMode_x15,Frozen_x70");
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  unsigned long now = millis();

  // ----------------------------------------------------------
  // 1) CONTROL UPDATE: ultra smooth 100 Hz streaming
  // ----------------------------------------------------------
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

  // ----------------------------------------------------------
  // 2) FEEDBACK UPDATE: slower so it does not break smoothness
  // ----------------------------------------------------------
  if (now - lastFeedbackTime >= FEEDBACK_MS) {
    lastFeedbackTime = now;
    actualAngle = readActualAngleQuick();

    if (!systemFrozen) {
      checkScenario2Emergency();
    }
  }

  // ----------------------------------------------------------
  // 3) SERIAL PLOTTER UPDATE
  // ----------------------------------------------------------
  if (now - lastPlotTime >= PLOT_MS) {
    lastPlotTime = now;
    printPlotterLine();
  }
}
