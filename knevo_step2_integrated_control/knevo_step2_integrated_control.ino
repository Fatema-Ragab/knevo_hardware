/*
  KNEVO STEP 2 — INTEGRATED CONTROL FIRMWARE
  ----------------------------------------------------------------
  This merges:
    - The real, causal heel-strike tracker validated in Step 1
      (FSR calibration, hysteresis contact detection, debounce,
      gap-timeout, rolling-median cycle length -> Gait_percent).
    - The tested control firmware from moresmoother.ino (FSM status,
      cubic Hermite spline trajectory, look-ahead, command smoothing,
      jump-validation, motor streaming, collapse-detection safety
      path verified in Step 0).

  WHAT CHANGED vs. the previously tested code, and why:
    1. getGaitPercentInput() (the SIM_GAIT_SPEED_PERCENT_PER_SEC sine
       simulation) is REMOVED. Gait_percent now comes from
       updateHeelStrikeTracker(), driven by live FSR sensors, per
       Decision #2 (tracker drives the trajectory directly; no CNN
       in this build yet - that's Step 3+).
    2. Buzzer -> built-in RGB LED (solid BLUE on freeze), carried
       over from the Step 0 bench test per your request. If you
       actually want the buzzer back for the real device, say so.
    3. When the tracker has not yet confirmed a real cycle
       (recentCycleValid == false, e.g. right after a long pause),
       Gait_percent now HOLDS its last value instead of snapping to
       0 - snapping would cause a real motor jerk; holding lets the
       existing jump-rejection logic absorb it smoothly.
    4. rawMLGaitPercent renamed to rawTrackerGaitPercent (it was
       never ML output; keeping that name would be confusing once
       Step 3 adds the actual CNN logging alongside it).

  This is still SINGLE-CORE (Step 3 adds the dual-core split so a
  slow CNN can never block this loop). Test this BENCH-SIDE first
  (not worn): confirm smoothing/look-ahead/jump-rejection behave
  correctly against real, noisy tracker input instead of the clean
  sine wave this code was tested against before.

  HARDWARE:
    Heel FSR -> GPIO1, Midfoot FSR -> GPIO2 (3.3V -> FSR -> GPIO -> 10k -> GND)
    Motor: MotorSerial on GPIO4 (RX) / GPIO5 (TX)
    Built-in RGB LED: GPIO48 default (change BUILTIN_RGB_LED_PIN if different)
*/

#include <math.h>

#define MOTOR_RX_PIN 4
#define MOTOR_TX_PIN 5
#define FSR_HEEL 1
#define FSR_META 2

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
// MOTOR CALIBRATION / SESSION SETTINGS (unchanged from moresmoother.ino)
// ============================================================
const float MOTOR_SCALE = 3722.0f;
float THERAPIST_MAX_ROM_DEG = 65.0f;
const float NORMAL_ROM_MIN_DEG = 5.0f;
const float NORMAL_ROM_MAX_DEG = 65.0f;

// ============================================================
// SMOOTH MOTION SETTINGS (unchanged)
// ============================================================
const unsigned long CONTROL_MS  = 10;   // 100Hz: tracker update + motor streaming
const unsigned long PLOT_MS     = 50;   // 20Hz: Serial Plotter CSV line
const unsigned long FEEDBACK_MS = 250;  // 4Hz: actual angle read + collapse check
const unsigned long PRINT_HUMAN_MS = 1000;  // 1Hz: human-readable status line

const float LOOKAHEAD_PERCENT = 4.0f;
const float COMMAND_SMOOTH_ALPHA = 0.18f;

const float MAX_GAIT_JUMP_PERCENT = 10.0f;
const int   MAX_BAD_READINGS      = 3;

const float COLLAPSE_ERROR_DEG = 20.0f;

// ============================================================
// MOTOR COMMANDS (unchanged)
// ============================================================
uint8_t cmdOn[5] = {0x3E, 0x88, 0x01, 0x00, 0xC7};

// ============================================================
// FSM PHASE TABLE: STATUS ONLY (unchanged)
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
// CLINICAL KNEE ANGLE TABLE (unchanged)
// ============================================================
const int TRAJ_SIZE = 11;
const float gaitTable[TRAJ_SIZE] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
const float kneeTable[TRAJ_SIZE] = {3.97, 19.84, 18.86, 11.09, 7.72, 13.86, 38.74, 64.12, 53.27, 17.27, 2.21};

// ============================================================
// FSR / HEEL-STRIKE TRACKER SETTINGS (from Step 1)
// ============================================================
const unsigned long CAL_UNLOADED_MS = 3000;
const unsigned long CAL_LOADED_MS   = 3000;
const float CONTACT_HIGH = 0.40f;
const float CONTACT_LOW  = 0.20f;
const float MIN_CYCLE_FRACTION = 0.65f;
const unsigned long MAX_PLAUSIBLE_CYCLE_MS = 4000;
const int CYCLE_HISTORY_SIZE = 5;
const int SMOOTH_WINDOW = 5;

float heelLow = 0, heelHigh = 4095;
float midLow = 0, midHigh = 4095;

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
int completedCycles = 0;       // monotonic running total, for telemetry only
bool recentCycleValid = false; // drives whether Gait_percent is trustworthy

float trackerGaitPercent = 0.0f;  // holds last value when not yet valid

// ============================================================
// RUNTIME VARIABLES (unchanged, aside from the rename noted above)
// ============================================================
float Gait_percent = 0.0f;
float rawTrackerGaitPercent = 0.0f;
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
unsigned long lastHumanPrintTime = 0;

// ============================================================
// BASIC UTILITIES (unchanged)
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
// SMOOTH CUBIC HERMITE INTERPOLATION (unchanged)
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
// ROM SCALING THAT PRESERVES GAIT SHAPE (unchanged)
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
// FSM STATUS (unchanged)
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
// MOTOR FUNCTIONS (unchanged)
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
// FSR CALIBRATION (from Step 1, unchanged)
// ============================================================
float normalize(int raw, float lo, float hi) {
  if (hi - lo < 1.0f) hi = lo + 1.0f;
  float v = (raw - lo) / (hi - lo);
  if (v < 0) v = 0;
  if (v > 1) v = 1;
  return v;
}

float medianOf(float* arr, int n) {
  float tmp[CYCLE_HISTORY_SIZE];
  for (int i = 0; i < n; i++) tmp[i] = arr[i];
  for (int i = 1; i < n; i++) {
    float key = tmp[i];
    int j = i - 1;
    while (j >= 0 && tmp[j] > key) { tmp[j + 1] = tmp[j]; j--; }
    tmp[j + 1] = key;
  }
  return tmp[n / 2];
}

void calibrateFSR() {
  Serial.println();
  Serial.println("=== FSR CALIBRATION ===");
  Serial.println("Step A: Keep the sensor UNLOADED (no contact) now.");
  for (int s = 3; s > 0; s--) { Serial.print(s); Serial.println("..."); delay(1000); }

  int heelMin = 4095, heelMax = 0, midMin = 4095, midMax = 0;
  unsigned long start = millis();
  while (millis() - start < CAL_UNLOADED_MS) {
    int h = analogRead(FSR_HEEL);
    int m = analogRead(FSR_META);
    heelMin = min(heelMin, h); heelMax = max(heelMax, h);
    midMin  = min(midMin, m);  midMax  = max(midMax, m);
    delay(10);
  }
  heelLow = heelMax;
  midLow  = midMax;

  Serial.println();
  Serial.println("Step B: Press/load the sensor FULLY now.");
  for (int s = 3; s > 0; s--) { Serial.print(s); Serial.println("..."); delay(1000); }

  heelMin = 4095; heelMax = 0; midMin = 4095; midMax = 0;
  start = millis();
  while (millis() - start < CAL_LOADED_MS) {
    int h = analogRead(FSR_HEEL);
    int m = analogRead(FSR_META);
    heelMin = min(heelMin, h); heelMax = max(heelMax, h);
    midMin  = min(midMin, m);  midMax  = max(midMax, m);
    delay(10);
  }
  heelHigh = heelMax;
  midHigh  = midMax;

  Serial.print("Calibration bounds -> heel [");
  Serial.print(heelLow); Serial.print(", "); Serial.print(heelHigh);
  Serial.print("]  mid [");
  Serial.print(midLow); Serial.print(", "); Serial.print(midHigh);
  Serial.println("]");

  const float MIN_DYNAMIC_RANGE = 200.0f;
  if ((heelHigh - heelLow) < MIN_DYNAMIC_RANGE) {
    Serial.println("WARNING: heel dynamic range is very small - sensor may not have been pressed during Step B.");
  }
  if ((midHigh - midLow) < MIN_DYNAMIC_RANGE) {
    Serial.println("WARNING: midfoot dynamic range is very small - sensor may not have been pressed during Step B.");
  }
  Serial.println("=== CALIBRATION DONE ===");
  Serial.println();
}

// ============================================================
// HEEL-STRIKE TRACKER (from Step 1, runs once per CONTROL_MS tick)
// ============================================================
void updateHeelStrikeTracker() {
  unsigned long now = millis();

  int heelRaw = analogRead(FSR_HEEL);
  int midRaw  = analogRead(FSR_META);

  float heelN = normalize(heelRaw, heelLow, heelHigh);
  float midN  = normalize(midRaw, midLow, midHigh);
  float contactScoreRaw = max(heelN, midN);

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
  bool fallingEdge = (wasContact && !inContact);

  if (risingEdge) {
    bool accept = true;
    if (hasLastHeelStrike) {
      unsigned long sinceLast = now - lastHeelStrikeMs;
      if (sinceLast < (unsigned long)(MIN_CYCLE_FRACTION * medianCycleMs)) {
        accept = false;
      }
    }

    Serial.print(now); Serial.print(" ms | CONTACT START | heelRaw=");
    Serial.print(heelRaw); Serial.print(" midRaw="); Serial.print(midRaw);
    Serial.print(" score="); Serial.print(contactScore, 2);
    Serial.print(" | accepted="); Serial.println(accept ? "YES" : "NO (debounce)");

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
        } else {
          Serial.print("  -> gap "); Serial.print(thisCycle);
          Serial.println(" ms exceeds MAX_PLAUSIBLE_CYCLE_MS, treating as restart (holding Gait_percent until next confirmed cycle)");
          recentCycleValid = false;
        }
      }
      lastHeelStrikeMs = now;
      hasLastHeelStrike = true;
    }
  } else if (fallingEdge) {
    Serial.print(now); Serial.print(" ms | CONTACT END   | heelRaw=");
    Serial.print(heelRaw); Serial.print(" midRaw="); Serial.print(midRaw);
    Serial.print(" score="); Serial.println(contactScore, 2);
  }

  if (hasLastHeelStrike && recentCycleValid) {
    float elapsed = (float)(now - lastHeelStrikeMs);
    float frac = elapsed / medianCycleMs;
    if (frac < 0) frac = 0;
    if (frac > 1) frac = 1;
    trackerGaitPercent = frac * 100.0f;
  }
  // else: hold trackerGaitPercent at its last value (see header note).
}

// ============================================================
// GAIT VALIDATION (unchanged - same jump-rejection/badReading logic
// that was tested against the simulated sine input, now applied to
// the real tracker's output)
// ============================================================
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
// TRAJECTORY UPDATE (unchanged)
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
// SERIAL PLOTTER OUTPUT (unchanged CSV line, for Tools > Serial Plotter)
// ============================================================
void printPlotterLine() {
  Serial.print(rawTrackerGaitPercent, 1); Serial.print(",");
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
// HUMAN-READABLE STATUS LINE (new, for Serial Monitor bench testing)
// ============================================================
void printHumanStatus() {
  Serial.print("[STATUS] Used=");      Serial.print(usedGaitPercent, 1);
  Serial.print("% Phase=");            Serial.print(fsmStatusName(statusCode));
  Serial.print(" Commanded=");         Serial.print(commandedAngle, 1);
  Serial.print("deg Actual=");         Serial.print(actualAngle, 1);
  Serial.print("deg Support=");        Serial.print(supportMode);
  Serial.print(" BadReading=");        Serial.print(badReading ? 1 : 0);
  Serial.print(" TrackerValid=");      Serial.print(recentCycleValid ? 1 : 0);
  Serial.print(" MedianCycle=");       Serial.print(medianCycleMs, 0);
  Serial.print("ms Frozen=");          Serial.println(systemFrozen ? 1 : 0);
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(3000);

  analogReadResolution(12);
  pinMode(FSR_HEEL, INPUT);
  pinMode(FSR_META, INPUT);
  ledOff();

  MotorSerial.begin(115200, SERIAL_8N1, MOTOR_RX_PIN, MOTOR_TX_PIN);

  Serial.println("=== KNEVO STEP 2: INTEGRATED CONTROL FIRMWARE ===");
  Serial.println("Bench-test only - not worn. Real FSR tracker now drives Gait_percent.");

  if (motorOn()) {
    Serial.println("Motor ON confirmed");
  } else {
    Serial.println("Motor ON failed - check wiring/power before testing");
  }

  calibrateFSR();

  rawTrackerGaitPercent = 0.0f;
  usedGaitPercent = 0.0f;
  updateSmoothFSMAndTrajectory(usedGaitPercent);
  sendPositionFast(commandedAngle);
  lastSentAngle = commandedAngle;

  delay(500);
  actualAngle = readActualAngleQuick();

  Serial.println();
  Serial.println("Running. Open Serial Plotter (Tools > Serial Plotter) to see the trajectory curve,");
  Serial.println("or watch the [STATUS] lines below in Serial Monitor.");
  Serial.println("PlotterHeader: RawTracker,UsedGait,LookAheadGait,State_x10,DesiredNormal_deg,DesiredROM_deg,Commanded_deg,Actual_deg,BadReading_x20,ErrorCount_x5,SupportMode_x15,Frozen_x70");
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  unsigned long now = millis();

  // ----------------------------------------------------------
  // 1) CONTROL UPDATE: 100Hz tracker + ultra-smooth motor streaming
  // ----------------------------------------------------------
  if (now - lastControlTime >= CONTROL_MS) {
    lastControlTime = now;

    if (!systemFrozen) {
      updateHeelStrikeTracker();
      rawTrackerGaitPercent = trackerGaitPercent;
      usedGaitPercent = validateOrPredictGait(rawTrackerGaitPercent);

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
  // 2) FEEDBACK UPDATE: 4Hz actual-angle read + collapse check
  // ----------------------------------------------------------
  if (now - lastFeedbackTime >= FEEDBACK_MS) {
    lastFeedbackTime = now;
    actualAngle = readActualAngleQuick();
    if (!systemFrozen) {
      checkScenario2Emergency();
    }
  }

  // ----------------------------------------------------------
  // 3) SERIAL PLOTTER CSV LINE (20Hz)
  // ----------------------------------------------------------
  if (now - lastPlotTime >= PLOT_MS) {
    lastPlotTime = now;
    printPlotterLine();
  }

  // ----------------------------------------------------------
  // 4) HUMAN-READABLE STATUS (1Hz, Serial Monitor)
  // ----------------------------------------------------------
  if (now - lastHumanPrintTime >= PRINT_HUMAN_MS) {
    lastHumanPrintTime = now;
    printHumanStatus();
  }
}
