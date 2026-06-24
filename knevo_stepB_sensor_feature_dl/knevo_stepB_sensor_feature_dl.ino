/*
  KNEVO STEP B — SENSOR + FEATURE EXTRACTION + DL INFERENCE (LOGGING ONLY)
  ----------------------------------------------------------------
  Real I2C/FSR sensor reads -> the real causal 57-feature recipe from
  Decision #1 -> TFLite Micro inference. Output goes to Serial only.
  ZERO connection to the motor - this cannot move anything.

  Reused VERBATIM from already-tested code:
    - I2C/MPU6050 init+read (readReg/writeReg/initMPU/readMotion6/
      accelScaleFromConfig/gyroScaleFromConfig) from knevo_final_status.ino.
    - TFLite Micro setup (130KB internal SRAM arena, op resolver,
      quantization, circular window buffer) from Gait_DL.ino - this
      arena size was already confirmed working on real hardware.

  NEW for this build (Decision #1's causal approximations):
    - Gap 1 (FSR normalization): per-session calibration bounds
      (same approach validated in Step 1/2), used for BOTH the
      heel_fsr_raw/midfoot_fsr_raw feature values AND the heel-strike
      contact score - matching training, where the same normalized
      FSR value feeds both roles.
    - Gap 2 (derivative features): backward difference (current -
      previous), NOT divided by elapsed time - matches training's
      np.gradient(), which was also called without a dt argument.
    - Gap 3 (phase-tracker features): the same causal heel-strike
      tracker validated in Step 1 (hysteresis + debounce + gap-
      timeout + rolling median), now producing gait_progress_est in
      0..1 (not 0..100 - this is a separate instance from Step 2's
      motor-facing tracker; the two are not connected yet).

  All 57 features are assembled in the EXACT order from metadata.json
  (sample_feature_cols), which is what the model was actually trained
  on - this is the part most likely to need debugging once real
  sensors are attached, same as every other step tonight.

  EXPECTED LIMITATION (intentional, not a bug): this is single-core.
  Real IMU+feature math is heavier than the test_stride.h playback
  Gait_DL.ino used, so the achieved sample rate may dip below 100Hz,
  especially during the ~47ms inference call. That's fine here since
  this build drives nothing safety-critical - Step C's dual-core
  split is what protects timing once this is combined with Step A.

  HARDWARE (same as knevo_final_status.ino):
    Bus0 SDA GPIO8, SCL GPIO9: FOOT 0x68, SHANK 0x69
    Bus1 SDA GPIO10, SCL GPIO11: THIGH 0x68
    Heel FSR -> GPIO1, Midfoot FSR -> GPIO2
*/

#include <Wire.h>
#include <math.h>
#include <string.h>
#include <Arduino.h>
#include <esp_heap_caps.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "knevo_esp32_boosted_phase_cnn_6phase_model.h"
#include "knevo_esp32_boosted_phase_cnn_scaler.h"

// ============================================================
// I2C / MPU6050 (verbatim from knevo_final_status.ino)
// ============================================================
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

#define TARGET_ACCEL_CONFIG 0x18  // +/-16g
#define TARGET_GYRO_CONFIG  0x18  // +/-2000 deg/s

TwoWire I2C_BUS0 = TwoWire(0);
TwoWire I2C_BUS1 = TwoWire(1);

struct ImuDef {
  TwoWire* bus;
  const char* busName;
  uint8_t addr;
  const char* name;
  bool ok;
  uint8_t accelCfg;
  uint8_t gyroCfg;
  float accelScale;
  float gyroScaleDps;
};

ImuDef imus[3] = {
  { &I2C_BUS0, "Bus0 GPIO8/9",   FOOT_ADDR,  "foot",  false, 0, 0, 16384.0f, 131.0f },
  { &I2C_BUS0, "Bus0 GPIO8/9",   SHANK_ADDR, "shank", false, 0, 0, 16384.0f, 131.0f },
  { &I2C_BUS1, "Bus1 GPIO10/11", THIGH_ADDR, "thigh", false, 0, 0, 16384.0f, 131.0f },
};

const float DEG_TO_RAD_F = 0.01745329252f;

uint8_t readReg(TwoWire &bus, uint8_t addr, uint8_t reg) {
  bus.beginTransmission(addr);
  bus.write(reg);
  if (bus.endTransmission(false) != 0) return 0xFF;
  bus.requestFrom(addr, (uint8_t)1);
  return bus.available() ? bus.read() : 0xFF;
}

bool writeReg(TwoWire &bus, uint8_t addr, uint8_t reg, uint8_t val) {
  bus.beginTransmission(addr);
  bus.write(reg);
  bus.write(val);
  return bus.endTransmission() == 0;
}

float accelScaleFromConfig(uint8_t cfg) {
  switch (cfg & 0x18) {
    case 0x00: return 16384.0f;
    case 0x08: return 8192.0f;
    case 0x10: return 4096.0f;
    case 0x18: return 2048.0f;
  }
  return 16384.0f;
}

float gyroScaleFromConfig(uint8_t cfg) {
  switch (cfg & 0x18) {
    case 0x00: return 131.0f;
    case 0x08: return 65.5f;
    case 0x10: return 32.8f;
    case 0x18: return 16.4f;
  }
  return 131.0f;
}

bool readMotion6(TwoWire &bus, uint8_t addr,
                 int16_t &ax, int16_t &ay, int16_t &az,
                 int16_t &gx, int16_t &gy, int16_t &gz) {
  bus.beginTransmission(addr);
  bus.write(REG_ACCEL_XOUT_H);
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

bool initMPU(ImuDef &imu) {
  uint8_t who = readReg(*imu.bus, imu.addr, REG_WHO_AM_I);
  if (who != 0x68 && who != 0x70) {
    Serial.print("IMU "); Serial.print(imu.name);
    Serial.print(" FAIL WHO_AM_I=0x"); Serial.println(who, HEX);
    imu.ok = false;
    return false;
  }

  writeReg(*imu.bus, imu.addr, REG_PWR_MGMT_1, 0x00);
  delay(100);
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

bool allIMUsOK() {
  for (int i = 0; i < 3; i++) if (!imus[i].ok) return false;
  return true;
}

// ============================================================
// FSR + BUILT-IN LED
// ============================================================
#define FSR_HEEL 1
#define FSR_META 2

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
void ledGood() { setBuiltinLedRGB(0, RGB_LED_BRIGHTNESS, 0); }
void ledFail() { setBuiltinLedRGB(RGB_LED_BRIGHTNESS, 0, 0); }

// ============================================================
// FSR CALIBRATION (Decision #1, Gap 1 - same approach as Step 1/2)
// ============================================================
const unsigned long CAL_UNLOADED_MS = 3000;
const unsigned long CAL_LOADED_MS   = 3000;
float heelLow = 0, heelHigh = 4095;
float midLow = 0, midHigh = 4095;

float normalize01(int raw, float lo, float hi) {
  if (hi - lo < 1.0f) hi = lo + 1.0f;
  float v = (raw - lo) / (hi - lo);
  if (v < 0) v = 0;
  if (v > 1) v = 1;
  return v;
}

void calibrateFSR() {
  Serial.println();
  Serial.println("=== FSR CALIBRATION ===");
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

  Serial.println();
  Serial.println("Step B: Load BOTH FSR sensors fully now (e.g. step on them).");
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
  Serial.println("=== CALIBRATION DONE ===");
  Serial.println();
}

// ============================================================
// HEEL-STRIKE TRACKER (Decision #1, Gap 3 - same logic as Step 1,
// output here is gait_progress_est in 0..1, a SEPARATE instance
// from Step 2's motor-facing tracker; not connected to it yet)
// ============================================================
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
float trackerGaitFrac = 0.0f;  // 0..1, holds last value when not valid

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

// Returns the normalized heel/mid values used both for the tracker's
// contact score AND directly as feature values 18/19 (matching how
// training uses the same normalized FSR value for both roles).
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
        } else {
          recentCycleValid = false;
        }
      }
      lastHeelStrikeMs = now;
      hasLastHeelStrike = true;
    }
  }

  if (hasLastHeelStrike && recentCycleValid) {
    float elapsed = (float)(now - lastHeelStrikeMs);
    float frac = elapsed / medianCycleMs;
    if (frac < 0) frac = 0;
    if (frac > 1) frac = 1;
    trackerGaitFrac = frac;
  }
  // else: hold last value
}

// fsmTable reused purely for printing a comparable "TrackerPhase" name
// alongside the CNN's predicted phase - not connected to any motor.
struct FSMState { int id; float gpStart; float gpEnd; const char* name; };
const FSMState fsmTable[6] = {
  {1,  0.0f,   5.0f,  "HeelStrike"       },
  {2,  5.0f,  15.0f,  "LoadingResponse"  },
  {3, 15.0f,  50.0f,  "MidTerminalStance"},
  {4, 50.0f,  62.0f,  "PreSwingToeOff"   },
  {5, 62.0f,  87.0f,  "InitialMidSwing"  },
  {6, 87.0f, 100.0f,  "TerminalSwing"    }
};
const char* trackerPhaseName(float gp01) {
  float gp = gp01 * 100.0f;
  for (int i = 0; i < 6; i++) {
    if (i < 5) { if (gp >= fsmTable[i].gpStart && gp < fsmTable[i].gpEnd) return fsmTable[i].name; }
    else       { if (gp >= fsmTable[i].gpStart && gp <= fsmTable[i].gpEnd) return fsmTable[i].name; }
  }
  return "Unknown";
}

const char* CNN_PHASE_NAMES[6] = {
  "HeelStrike", "LoadingResponse", "MidTerminalStance",
  "PreSwingToeOff", "InitialMidSwing", "TerminalSwing"
};

// ============================================================
// FEATURE ENGINEERING (Decision #1 - exact recipe from cnn_99.ipynb)
// ============================================================
inline float safeSqrt(float x) { return sqrtf(x > 0 ? x : 0); }

float imuPitchDeg(float ax, float ay, float az) {
  return degrees(atan2f(ax, safeSqrt(ay * ay + az * az) + 1e-9f));
}
float imuRollDeg(float ax, float ay, float az) {
  return degrees(atan2f(ay, az + 1e-9f));
}

// Previous-tick values for the 12 backward-difference derivative features.
float prev_heel_fsr = 0, prev_mid_fsr = 0, prev_fsr_sum = 0, prev_fsr_diff = 0;
float prev_foot_gyro_mag = 0, prev_shank_gyro_mag = 0, prev_thigh_gyro_mag = 0;
float prev_knee_acc = 0, prev_knee_pitch = 0;
float prev_foot_pitch = 0, prev_shank_pitch = 0, prev_thigh_pitch = 0;
bool firstFeatureTick = true;

float sampleFeatures[KNEVO_TCN_FEATURE_COUNT];

void computeFeatures(float ax_g[3], float ay_g[3], float az_g[3],
                      float gx_rs[3], float gy_rs[3], float gz_rs[3],
                      float heelNorm, float midNorm) {
  // indices: 0=foot,1=shank,2=thigh

  // --- raw 20 (0-19) ---
  sampleFeatures[0]=ax_g[0]; sampleFeatures[1]=ay_g[0]; sampleFeatures[2]=az_g[0];
  sampleFeatures[3]=gx_rs[0]; sampleFeatures[4]=gy_rs[0]; sampleFeatures[5]=gz_rs[0];
  sampleFeatures[6]=ax_g[1]; sampleFeatures[7]=ay_g[1]; sampleFeatures[8]=az_g[1];
  sampleFeatures[9]=gx_rs[1]; sampleFeatures[10]=gy_rs[1]; sampleFeatures[11]=gz_rs[1];
  sampleFeatures[12]=ax_g[2]; sampleFeatures[13]=ay_g[2]; sampleFeatures[14]=az_g[2];
  sampleFeatures[15]=gx_rs[2]; sampleFeatures[16]=gy_rs[2]; sampleFeatures[17]=gz_rs[2];
  sampleFeatures[18]=heelNorm; sampleFeatures[19]=midNorm;

  // --- magnitudes (20-25) ---
  float foot_acc_mag  = sqrtf(ax_g[0]*ax_g[0] + ay_g[0]*ay_g[0] + az_g[0]*az_g[0]);
  float shank_acc_mag = sqrtf(ax_g[1]*ax_g[1] + ay_g[1]*ay_g[1] + az_g[1]*az_g[1]);
  float thigh_acc_mag = sqrtf(ax_g[2]*ax_g[2] + ay_g[2]*ay_g[2] + az_g[2]*az_g[2]);
  float foot_gyro_mag  = sqrtf(gx_rs[0]*gx_rs[0] + gy_rs[0]*gy_rs[0] + gz_rs[0]*gz_rs[0]);
  float shank_gyro_mag = sqrtf(gx_rs[1]*gx_rs[1] + gy_rs[1]*gy_rs[1] + gz_rs[1]*gz_rs[1]);
  float thigh_gyro_mag = sqrtf(gx_rs[2]*gx_rs[2] + gy_rs[2]*gy_rs[2] + gz_rs[2]*gz_rs[2]);
  sampleFeatures[20]=foot_acc_mag; sampleFeatures[21]=shank_acc_mag; sampleFeatures[22]=thigh_acc_mag;
  sampleFeatures[23]=foot_gyro_mag; sampleFeatures[24]=shank_gyro_mag; sampleFeatures[25]=thigh_gyro_mag;

  // --- pitch/roll (26-31) ---
  float foot_pitch  = imuPitchDeg(ax_g[0], ay_g[0], az_g[0]);
  float shank_pitch = imuPitchDeg(ax_g[1], ay_g[1], az_g[1]);
  float thigh_pitch = imuPitchDeg(ax_g[2], ay_g[2], az_g[2]);
  float foot_roll  = imuRollDeg(ax_g[0], ay_g[0], az_g[0]);
  float shank_roll = imuRollDeg(ax_g[1], ay_g[1], az_g[1]);
  float thigh_roll = imuRollDeg(ax_g[2], ay_g[2], az_g[2]);
  sampleFeatures[26]=foot_pitch; sampleFeatures[27]=shank_pitch; sampleFeatures[28]=thigh_pitch;
  sampleFeatures[29]=foot_roll; sampleFeatures[30]=shank_roll; sampleFeatures[31]=thigh_roll;

  // --- knee angles (32-33) ---
  float dot = ax_g[2]*ax_g[1] + ay_g[2]*ay_g[1] + az_g[2]*az_g[1];
  float normProd = (sqrtf(ax_g[2]*ax_g[2]+ay_g[2]*ay_g[2]+az_g[2]*az_g[2]) + 1e-9f) *
                    (sqrtf(ax_g[1]*ax_g[1]+ay_g[1]*ay_g[1]+az_g[1]*az_g[1]) + 1e-9f);
  float cosAngle = dot / normProd;
  if (cosAngle > 1) cosAngle = 1; if (cosAngle < -1) cosAngle = -1;
  float knee_angle_acc = degrees(acosf(cosAngle));
  float knee_angle_pitch = thigh_pitch - shank_pitch;
  sampleFeatures[32]=knee_angle_acc; sampleFeatures[33]=knee_angle_pitch;

  // --- fsr combos (34-36) ---
  float fsr_sum = heelNorm + midNorm;
  float fsr_diff = heelNorm - midNorm;
  float fsr_max = max(heelNorm, midNorm);
  sampleFeatures[34]=fsr_sum; sampleFeatures[35]=fsr_diff; sampleFeatures[36]=fsr_max;

  // --- cross-limb diffs (37-40) ---
  sampleFeatures[37] = shank_pitch - thigh_pitch;       // shank_thigh_pitch_diff
  sampleFeatures[38] = foot_pitch - shank_pitch;        // foot_shank_pitch_diff
  sampleFeatures[39] = shank_gyro_mag - thigh_gyro_mag;  // shank_thigh_gyro_diff
  sampleFeatures[40] = foot_gyro_mag - shank_gyro_mag;   // foot_shank_gyro_diff

  // --- derivatives (41-52), backward difference, NOT divided by dt ---
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

  // --- phase tracker features (53-56) ---
  float gp = trackerGaitFrac;
  bool valid = hasLastHeelStrike && recentCycleValid;
  sampleFeatures[53] = gp;
  sampleFeatures[54] = valid ? sinf(2.0f * PI * gp) : 0.0f;
  sampleFeatures[55] = valid ? cosf(2.0f * PI * gp) : 0.0f;
  sampleFeatures[56] = valid ? 1.0f : 0.0f;
}

// ============================================================
// TFLITE MICRO (reused from Gait_DL.ino)
// ============================================================
const uint32_t SAMPLE_PERIOD_US = 10000;  // target 10ms = 100Hz
const int INFERENCE_EVERY_N_SAMPLES = 5;  // target 20Hz inference
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

// ============================================================
// MAIN TIMING / RATE TRACKING
// ============================================================
unsigned long lastHumanPrintTime = 0;
unsigned long rateWindowStart = 0;
int samplesThisWindow = 0;
float achievedHz = 0.0f;

void setup() {
  Serial.begin(115200);
  delay(3000);

  analogReadResolution(12);
  pinMode(FSR_HEEL, INPUT);
  pinMode(FSR_META, INPUT);
  ledOff();

  Serial.println("=== KNEVO STEP B: SENSOR + FEATURE + DL PIPELINE (LOGGING ONLY) ===");
  Serial.println("No motor connection. Output goes to Serial only.");

  I2C_BUS0.begin(BUS0_SDA, BUS0_SCL, 400000);
  I2C_BUS1.begin(BUS1_SDA, BUS1_SCL, 400000);

  for (int i = 0; i < 3; i++) imus[i].ok = initMPU(imus[i]);
  if (allIMUsOK()) { Serial.println("All IMUs OK."); ledGood(); }
  else { Serial.println("One or more IMUs FAILED - features will be wrong until fixed."); ledFail(); }

  calibrateFSR();

  // Precompute quantization constants once (same as Gait_DL.ino).
  for (int f = 0; f < KNEVO_TCN_FEATURE_COUNT; f++) {
    if (KNEVO_TCN_STD[f] == 0.0f) {
      q_mult[f] = 0.0f;
      q_bias[f] = KNEVO_TCN_INPUT_ZERO_POINT;
    } else {
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
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: model schema version mismatch."); while (1) delay(1000);
  }

  static tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddReshape(); resolver.AddConv2D(); resolver.AddAdd(); resolver.AddAveragePool2D();
  resolver.AddShape(); resolver.AddStridedSlice(); resolver.AddPack();
  resolver.AddFullyConnected(); resolver.AddMul(); resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: tensor allocation failed."); while (1) delay(1000);
  }
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TFLite Micro ready.");
  Serial.println();
  Serial.println("Running. [STATUS] lines print once per second.");

  next_sample_time_us = micros() + SAMPLE_PERIOD_US;
  rateWindowStart = millis();
}

void loop() {
  uint32_t now_us = micros();

  if ((int32_t)(now_us - next_sample_time_us) >= 0) {
    next_sample_time_us += SAMPLE_PERIOD_US;

    // --- read real sensors ---
    int16_t ax_raw[3], ay_raw[3], az_raw[3], gx_raw[3], gy_raw[3], gz_raw[3];
    for (int i = 0; i < 3; i++) {
      if (imus[i].ok) {
        readMotion6(*imus[i].bus, imus[i].addr,
          ax_raw[i], ay_raw[i], az_raw[i], gx_raw[i], gy_raw[i], gz_raw[i]);
      } else {
        ax_raw[i]=0; ay_raw[i]=0; az_raw[i]=0; gx_raw[i]=0; gy_raw[i]=0; gz_raw[i]=0;
      }
    }

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

    // --- heel-strike tracker (drives features 53-56) ---
    updateHeelStrikeTracker(heelNorm, midNorm);

    // --- assemble all 57 features in the exact trained order ---
    computeFeatures(ax_g, ay_g, az_g, gx_rs, gy_rs, gz_rs, heelNorm, midNorm);

    // --- window + inference ---
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

  // --- achieved sample-rate tracking ---
  if (millis() - rateWindowStart >= 1000) {
    achievedHz = samplesThisWindow * 1000.0f / (millis() - rateWindowStart);
    samplesThisWindow = 0;
    rateWindowStart = millis();
  }

  // --- 1Hz human-readable status ---
  if (millis() - lastHumanPrintTime >= 1000) {
    lastHumanPrintTime = millis();
    Serial.print("[STATUS] Hz="); Serial.print(achievedHz, 1);
    Serial.print(" TrackerProgress="); Serial.print(trackerGaitFrac * 100.0f, 1);
    Serial.print("% TrackerPhase="); Serial.print(trackerPhaseName(trackerGaitFrac));
    Serial.print(" TrackerValid="); Serial.print(recentCycleValid ? 1 : 0);
    Serial.print(" | CNN_Phase=");
    Serial.print(last_predicted_phase >= 0 ? CNN_PHASE_NAMES[last_predicted_phase] : "none-yet");
    Serial.print(" Prob="); Serial.print(last_probability, 3);
    Serial.print(" Prep="); Serial.print(last_prep_us); Serial.print("us");
    Serial.print(" Invoke="); Serial.print(last_invoke_us); Serial.println("us");
  }
}
