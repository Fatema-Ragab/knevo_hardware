/*
  KNEVO STEP 1 — CAUSAL HEEL-STRIKE TRACKER (GUIDED BENCH TEST)
  ----------------------------------------------------------------
  Purpose: real-time Gait_percent (0-100%) from live FSR sensors only,
  using a causal (real-time-safe) approximation of the heel-strike-
  cycle-progress idea the training notebook used for pseudo-labels
  and the four "boosted phase" model features.

  Per Decision #1 (Gap 3): FSR-only hysteresis detection, no gyro-peak
  fallback yet. No motor, no CNN.

  GUIDED TEST MODE (this version):
  After calibration, the sketch runs through fixed phases. Each phase
  prints ONE instruction, gives you a countdown to act, then prints a
  compact SUMMARY block when the phase ends and moves to the next one
  automatically. You don't need to judge the raw stream yourself —
  copy the SUMMARY blocks (or the whole output) back for analysis.

  During each phase it also prints one line per detected contact
  transition (not a continuous flood), so the log stays short and
  readable: CONTACT START (with accepted=YES/NO) and CONTACT END.

  HARDWARE: Heel FSR -> GPIO1, Midfoot FSR -> GPIO2
  (3.3V -> FSR -> GPIO -> 10k resistor -> GND)

  No need to wear/walk: pressing the FSR pads with your fingers is
  electrically identical input for everything this test checks.
*/

#include <math.h>

#define FSR_HEEL 1
#define FSR_META 2

// ----- CALIBRATION SETTINGS -----
const unsigned long CAL_UNLOADED_MS = 3000;
const unsigned long CAL_LOADED_MS   = 3000;

// ----- HYSTERESIS THRESHOLDS (0..1 normalized contact score) -----
const float CONTACT_HIGH = 0.40f;
const float CONTACT_LOW  = 0.20f;

// ----- DEBOUNCE -----
const float MIN_CYCLE_FRACTION = 0.65f;

// ----- ROLLING MEDIAN CYCLE WINDOW -----
const int CYCLE_HISTORY_SIZE = 5;

// ----- CAUSAL SMOOTHING -----
const int SMOOTH_WINDOW = 5;

// ----- TIMING -----
const unsigned long SAMPLE_MS = 10;  // 100Hz sampling

// ----- CALIBRATION RESULT -----
float heelLow = 0, heelHigh = 4095;
float midLow = 0, midHigh = 4095;

// ----- SMOOTHING BUFFER -----
float contactBuf[SMOOTH_WINDOW];
int contactBufIdx = 0;
bool contactBufFilled = false;

// ----- HEEL-STRIKE / CYCLE STATE -----
bool inContact = false;
unsigned long lastHeelStrikeMs = 0;
bool hasLastHeelStrike = false;
float cycleHistoryMs[CYCLE_HISTORY_SIZE];
int cycleHistoryCount = 0;
int cycleHistoryIdx = 0;
float medianCycleMs = 1200.0f;   // placeholder guess until the first real cycle completes
int completedCycles = 0;

// ----- EVENT COUNTERS (for phase summaries) -----
int risingEdgeCount = 0;      // every detected rising edge, before debounce
int acceptedStrikeCount = 0;  // rising edges accepted as real strikes, after debounce

unsigned long lastSampleTime = 0;

float GaitPercent = 0.0f;
bool gaitValid = false;

// ----- GUIDED TEST PHASES -----
struct TestPhase {
  const char* instruction;
  unsigned long durationMs;
};

TestPhase phases[] = {
  {"Do NOT touch the sensors. Stay idle.", 5000},
  {"Press and fully release the FSR ONCE.", 5000},
  {"Press and release TWICE at a normal pace (about 1 press every 1-2 seconds).", 8000},
  {"Press and release TWICE very fast, less than half a second apart.", 5000},
  {"Press and release at a steady rhythm, about once per second, for the full duration.", 10000},
  {"Press and release SLOWER now, about once every 2 seconds, for the full duration.", 10000},
};
const int NUM_PHASES = 6;
int currentPhase = 0;
bool phaseAnnounced = false;
unsigned long phaseStartTime = 0;
int risingEdgeAtPhaseStart = 0;
int acceptedAtPhaseStart = 0;
int completedAtPhaseStart = 0;
bool allPhasesDone = false;

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

void calibrate() {
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
  Serial.print("Unloaded heel range: "); Serial.print(heelMin); Serial.print(" - "); Serial.println(heelMax);
  Serial.print("Unloaded mid range: ");  Serial.print(midMin);  Serial.print(" - "); Serial.println(midMax);

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
  Serial.print("Loaded heel range: "); Serial.print(heelMin); Serial.print(" - "); Serial.println(heelMax);
  Serial.print("Loaded mid range: ");  Serial.print(midMin);  Serial.print(" - "); Serial.println(midMax);

  Serial.println();
  Serial.print("Calibration bounds -> heel [");
  Serial.print(heelLow); Serial.print(", "); Serial.print(heelHigh);
  Serial.print("]  mid [");
  Serial.print(midLow); Serial.print(", "); Serial.print(midHigh);
  Serial.println("]");
  Serial.println("=== CALIBRATION DONE ===");
}

void announcePhase() {
  Serial.println();
  Serial.print("=== PHASE "); Serial.print(currentPhase + 1); Serial.print("/"); Serial.print(NUM_PHASES);
  Serial.print(" ("); Serial.print(phases[currentPhase].durationMs / 1000); Serial.println("s) ===");
  Serial.println(phases[currentPhase].instruction);
  Serial.println(">>> GO NOW <<<");
}

void printPhaseSummary() {
  int rising = risingEdgeCount - risingEdgeAtPhaseStart;
  int accepted = acceptedStrikeCount - acceptedAtPhaseStart;
  int rejected = rising - accepted;
  int cyclesThisPhase = completedCycles - completedAtPhaseStart;

  Serial.println("--- PHASE SUMMARY ---");
  Serial.print("Rising edges detected: ");      Serial.println(rising);
  Serial.print("Accepted as real strikes: ");   Serial.println(accepted);
  Serial.print("Rejected by debounce: ");        Serial.println(rejected);
  Serial.print("Completed cycles this phase: "); Serial.println(cyclesThisPhase);
  Serial.print("Current median cycle estimate: "); Serial.print(medianCycleMs, 0); Serial.println(" ms");
  Serial.println("---------------------");
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  analogReadResolution(12);
  pinMode(FSR_HEEL, INPUT);
  pinMode(FSR_META, INPUT);

  Serial.println("=== KNEVO STEP 1: CAUSAL HEEL-STRIKE TRACKER (GUIDED TEST) ===");
  calibrate();
  Serial.println();
  Serial.println("Starting guided test phases now.");
}

void loop() {
  unsigned long now = millis();

  if (now - lastSampleTime >= SAMPLE_MS) {
    lastSampleTime = now;

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

    bool risingEdge  = (!wasContact && inContact);
    bool fallingEdge = (wasContact && !inContact);

    if (risingEdge) {
      risingEdgeCount++;
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
        acceptedStrikeCount++;
        if (hasLastHeelStrike) {
          float thisCycle = (float)(now - lastHeelStrikeMs);
          cycleHistoryMs[cycleHistoryIdx] = thisCycle;
          cycleHistoryIdx = (cycleHistoryIdx + 1) % CYCLE_HISTORY_SIZE;
          if (cycleHistoryCount < CYCLE_HISTORY_SIZE) cycleHistoryCount++;
          medianCycleMs = medianOf(cycleHistoryMs, cycleHistoryCount);
          completedCycles++;
        }
        lastHeelStrikeMs = now;
        hasLastHeelStrike = true;
      }
    } else if (fallingEdge) {
      Serial.print(now); Serial.print(" ms | CONTACT END   | heelRaw=");
      Serial.print(heelRaw); Serial.print(" midRaw="); Serial.print(midRaw);
      Serial.print(" score="); Serial.println(contactScore, 2);
    }

    if (hasLastHeelStrike && completedCycles >= 1) {
      float elapsed = (float)(now - lastHeelStrikeMs);
      float frac = elapsed / medianCycleMs;
      if (frac < 0) frac = 0;
      if (frac > 1) frac = 1;
      GaitPercent = frac * 100.0f;
      gaitValid = true;
    } else {
      GaitPercent = 0.0f;
      gaitValid = false;
    }
  }

  // ----- GUIDED PHASE SEQUENCING -----
  if (!allPhasesDone) {
    if (!phaseAnnounced) {
      phaseAnnounced = true;
      phaseStartTime = now;
      risingEdgeAtPhaseStart = risingEdgeCount;
      acceptedAtPhaseStart = acceptedStrikeCount;
      completedAtPhaseStart = completedCycles;
      announcePhase();
    }

    if (now - phaseStartTime >= phases[currentPhase].durationMs) {
      printPhaseSummary();
      currentPhase++;
      phaseAnnounced = false;

      if (currentPhase >= NUM_PHASES) {
        allPhasesDone = true;
        Serial.println();
        Serial.println("=== ALL TEST PHASES COMPLETE ===");
        Serial.println("Copy everything above (or at least each PHASE SUMMARY block) and send it back.");
        Serial.println("Free-run continues below if you want to keep experimenting.");
      }
    }
  } else {
    // Free-run after the guided sequence: still prints GaitPercent on change, low rate.
    static unsigned long lastFreeRunPrint = 0;
    if (now - lastFreeRunPrint >= 500) {
      lastFreeRunPrint = now;
      Serial.print("[FREE RUN] GaitPercent="); Serial.print(GaitPercent, 1);
      Serial.print(" Valid="); Serial.print(gaitValid ? 1 : 0);
      Serial.print(" CompletedCycles="); Serial.println(completedCycles);
    }
  }
}
