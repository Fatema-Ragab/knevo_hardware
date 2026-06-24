# Knevo — AI + Control Integration Plan & Decision Log

**Project:** Smart Active Bionic Knee Brace for Rehabilitation and Assistance (Knevo)
**Goal of this doc:** capture everything decided/found while planning the merge of the gait-classifier CNN (`Gait_DL.ino`) into the motor control firmware (`moresmoother.ino`), so this thread can be picked back up or handed off without re-deriving anything.
**Status at time of writing:** 3 architecture decisions locked. Integration not yet built. Deadline: live demo (person wearing the brace) + code, due tomorrow.

---

## 1. File Inventory (what exists, what's confirmed working)

| File | Role | Status |
|---|---|---|
| `moresmoother.ino` | Motor control firmware. FSM = status-only, cubic Hermite spline drives the actual knee angle, 100Hz/20Hz/4Hz multi-frequency loop (control/plot/feedback). | Tested & working — but `getGaitPercentInput()` is currently a **simulated sine-style input** (`SIM_GAIT_SPEED_PERCENT_PER_SEC`), not live sensor/AI data. |
| `knevo_final_status.ino` | Data-collection firmware. Dual I2C buses, 3× MPU6050 (foot/shank/thigh), 2× FSR, Wi-Fi TCP streaming to laptop, anti-burst guard against Wi-Fi stalls. | Tested & working. CSV always writes `gait_phase_id = -1` (hardcoded) — every recorded sample is **unlabeled**. Minor schema bug: header lists 5 trailing columns but only 4 values are ever written (`knee_angle_est_deg` is always blank; harmless, unused by training). |
| `Gait_DL.ino` | TFLite Micro inference test sketch. 130KB tensor arena in **internal SRAM** (confirmed working on real hardware). Single-core, single-thread. Infers every 5th sample (20Hz cadence target). | Tested & working in isolation, but only against `test_stride.h` **placeholder/zero-padded** features — never run with real causal feature math. |
| `knevo_esp32_boosted_phase_cnn_6phase_model.h` | INT8 TFLite model, 27,248 bytes. "Boosted Phase Lite Temporal CNN" — Conv2D residual TCN blocks → dense → softmax. | — |
| `knevo_esp32_boosted_phase_cnn_scaler.h` | Window=71 samples, 57 features, per-feature mean/std, INT8 quant scale/zero-point. | — |
| `test_stride.h` | 120×57 placeholder playback data, built from a quick notebook with zero-padded engineered features. | **Not representative** of the real feature recipe — only useful for confirming the TFLite pipeline executes, not that it classifies meaningfully. |
| `cnn_99.ipynb` | The actual training notebook. Ground truth for the real 57-feature recipe, labeling logic, model architecture. | Read in full — see §3. |
| `metadata.json` | Real export metadata from the actual trained model. | Confirms 99.47% int8 / 99.64% float accuracy, `split_mode: random`. |
| `raw.zip` | The actual training dataset. 4 subjects (S01–S04) × 7 speeds (0.50–2.00 mph) × 3 trials = 84 files, ~60s each. | Inspected directly — see §3. |

---

## 2. Key Technical Findings (from direct inspection, not assumptions)

- `PHASE_BOUNDARIES` in the training notebook `[0, .05, .15, .50, .62, .87, 1.01]` **exactly matches** `fsmTable` in `moresmoother.ino`. The two were always meant to line up.
- The real 57-feature vector = 20 raw IMU/FSR + 3 accel-magnitudes + 3 gyro-magnitudes + 3 pitch + 3 roll + 2 knee-angle + 3 FSR combos (sum/diff/max) + 2 pitch-diffs + 2 gyro-diffs + 12 derivative (`_d`) features + 4 "boosted phase tracker" features (`gait_progress_est`, `gait_sin_est`, `gait_cos_est`, `gait_valid_est`).
- **FSR behavior (measured across all 84 files):** heel FSR's 97th percentile is **4095 (full saturation) in every single file** — very reliable. Midfoot FSR's 97th percentile ranges **730–4095** (5.6× spread) depending on trial — genuinely variable, confirms per-file/per-session normalization is necessary, a fixed constant won't work.
- **Derivative features:** the training code calls `np.gradient()` with **no `dt` argument** — meaning the "derivative" features are per-sample-step, not per-second. Measured real sampling rate across files ranges **94.76–100.02 Hz** (not a clean 100Hz), so the training-time derivatives already silently assumed uniform spacing that wasn't always true. **Implication:** the ESP version should replicate this simplified (slightly inexact) convention, not "fix" it to use real elapsed time — doing so would create a new train/serve mismatch in the opposite direction.
- **Real gait cycle durations** (estimated from FSR heel-strike timing across the dataset): median **2.84s at 0.5 mph** down to **1.34s at 2.0 mph**. A 10ms processing delay is ~0.4–0.75% of a cycle — negligible.
- **Labels are 100% pseudo-labels** — confirmed by direct inspection: `gait_phase_id == -1` in every row of all 84 raw files. No manual labeling was ever done.
- **Accuracy context:** 99.47% (int8) / 99.64% (float) is real, but measured on a **random window-level split**, not a subject-held-out split, with only **4 subjects** total. Likely optimistic; true generalization to new people (especially impaired/asymmetric gait) is unverified.
- **Measured real inference time on hardware: ~47ms.** (`metadata.json`'s `colab_tflite_inference_ms_per_window` was 0.18ms — the ESP is ~260× slower than a laptop/Colab run, which is expected for a first unoptimized embedded deployment.)
- Dataset is treadmill-only, fixed prescribed speeds, presumably able-bodied volunteers — **not representative of an actual impaired wearer.**

---

## 3. Decision Log

### Decision #1 — Feature Engineering Parity ✅ LOCKED

**Core problem:** the real training recipe uses several computations that aren't causally reproducible on the ESP in real time (they use whole-file knowledge the chip doesn't have yet).

| Gap | Problem | Solution (locked) | Drawback / criticality |
|---|---|---|---|
| **1. FSR normalization** | `robust_norm()` uses the 3rd/97th percentile of the **entire file** — not knowable in real time. | Per-session calibration using the existing `CAL_STATIC` / `CAL_UNLOADED` / `CAL_STANDING` / `CAL_KNEE` commands already in `knevo_final_status.ino`; recalibrate at every speed/intensity change, not just once per session. | **Not critical** — a workflow discipline issue, not a code risk. Skipping a recalibration degrades accuracy for that bout only; doesn't crash anything. |
| **2. Derivative features** | `np.gradient()` is centered (needs future sample `i+1`) and not divided by real Δt. | Simple backward difference `x[i] - x[i-1]`, **no dt division** — matches the (slightly inexact) training convention exactly. | **Not critical** — basically free to implement correctly, no real downside. |
| **3. Phase-tracker features** | `gait_progress_est`/`sin`/`cos`/`valid` are built from a whole-file-aware heel-strike anchor detector (hysteresis + gyro-peak fallback + whole-file cycle-length cleanup). | Causal rolling-median heel-strike tracker (FSR threshold + hysteresis), **with a required debounce/lockout** — ignore a new candidate heel-strike if it arrives sooner than ~60–70% of the current rolling median cycle length. | **This is the one that actually matters.** Without the debounce, FSR noise can fake extra heel-strikes, snapping the progress estimate to near-zero at the worst real-time moments, quietly feeding the model bad input. Still a small, well-defined addition — not a redesign. |
| **4. Label mode** | Needed to know if this model was trained on real or pseudo labels. | **Confirmed via direct data inspection:** 100% pseudo-labels. No manual labeling occurred. | Informational — shapes how much to trust the 99.4% figure, especially on out-of-distribution (impaired) gait. |

### Decision #2 — What drives `Gait_percent`, and what is the CNN actually for? ✅ LOCKED

**Core problem:** `fsmTable` bins are wildly uneven (5/10/35/12/25/13%), so no simple "class index → percent" formula works for most bins. More fundamentally: the CNN's 6-class output is, by construction, a coarser, ~47ms-delayed, thresholded version of the *same* continuous heel-strike-timing signal that Decision #1 Gap 3 already requires building. Two sources of "where am I in the cycle" would otherwise exist on the chip for no good reason.

**Decision:**
- The **causal heel-strike tracker** (Decision #1, Gap 3) drives `Gait_percent` directly and continuously, every 10ms, zero added latency.
- The **CNN's predicted phase is compared** against the tracker's current FSM bin (`getFSMStatusCode(Gait_percent)`) and logged/counted as a disagreement — it does **not** edit the trajectory.
- **Shadow mode:** a single boolean flag (e.g. `aiCorrectionEnabled = false`) gates whether a CNN/tracker disagreement is ever allowed to nudge `Gait_percent`. Default is **off**. The correction logic gets built either way; the flag just controls whether it's live.
- **Safety stays 100% hardware-truth-based, no exceptions.** `checkScenario2Emergency()` continues to use only measured-vs-commanded motor angle and the tracker-driven FSM. The CNN never touches the freeze/support-mode path. Rule of thumb: *the CNN can raise suspicion; only hardware truth can freeze the system.*

**Why shadow mode stays off for tomorrow's deadline (confirmed as the project owner's own call, not a committee requirement):** the dataset is 4 able-bodied subjects on a treadmill at fixed prescribed speeds — not representative of an actual impaired wearer, and there's no time before the deadline to validate correction behavior on representative gait. Live correction decided same-day as a live human demo is the exact scenario shadow mode exists to prevent.

**What ships tomorrow:** a live, real-time gait classifier running on hardware, logging and visibly comparing its output against the tracker — a genuine, demoable result — **without** being allowed to alter the motor trajectory.

### Decision #3 — 47ms inference vs. 10ms control loop ✅ LOCKED

**Core problem:** `Gait_DL.ino` is single-core/single-thread. A 47ms `Invoke()` call blocks *everything* — sensors, tracker, motor streaming — for its full duration. At the current 50ms inference cadence, that's ~47 of every 50ms as dead time on a single core; smoothness would visibly break.

**Decision:** Build a true dual-core split with FreeRTOS (`xTaskCreatePinnedToCore`):
- **Core A** (pinned): sensors → heel-strike tracker → spline → motor streaming. Strict 10ms cadence, never blocked by inference, no exceptions.
- **Core B** (pinned): feature window + CNN inference. Runs whenever free; writes its result to a simple **double-buffer** (no semaphore/queue needed — since the CNN is shadow/logging-only per Decision #2, a few cycles of staleness doesn't matter).

This is considered essential infrastructure regardless of shadow-mode status — it's pure architecture, not contingent on whether correction is ever turned on. Locked as the top build priority.

*(Optional, non-blocking: profile which op inside `Invoke()` actually costs the most — `Conv2D` layers are the likely culprit per the `tflite_ops` list — and consider shrinking the model later. Not required for tomorrow.)*

---

## 4. Today's Build & Test Plan

Approach: **step-by-step, gated** — confirm each step actually works before moving to the next, so any failure is easy to isolate.

**Step 0 — Verify the safety path alone, before anything new touches it.**
Bench-test `checkScenario2Emergency()` / `freezeSystem()` with the motor under load but *not worn* (fixture or hand-resistance). Confirm a forced angle error actually halts the motor and fires the buzzer. Must hold true no matter what gets built later today.

**Step 1 — Causal heel-strike tracker, standalone.**
Real-time `Gait_percent` from live IMU/FSR sensors, implementing Decision #1's Gap 1 (calibration) and Gap 3 (debounced rolling-median heel-strike detection). No motor, no CNN yet. Sanity-check: numbers look like real strides while walking/stepping, not noise.

**Step 2 — Wire the tracker into the existing control loop.**
Replace `moresmoother.ino`'s simulated `getGaitPercentInput()` with Step 1's real tracker output. Bench-test the motor (not worn): confirm smoothing, look-ahead, and jump-rejection still behave correctly against real noisy input instead of the clean sine wave it was tested against before.

**Step 3 — Dual-core bring-up, CNN in shadow mode.**
Core A (sensors/tracker/spline/motor) pinned, strict 10ms loop. Core B (CNN) using the real causal feature recipe from Decision #1, running window+inference whenever free, logging its result and comparing it to the tracker's FSM bin via the double-buffer handoff. `aiCorrectionEnabled = false`. Test: confirm Core A's 10ms timing holds steady regardless of Core B's load.

**Step 4 — First worn test, conservative settings.**
Small ROM, cautious `THERAPIST_MAX_ROM_DEG`, spotter present. Confirm: freeze path still works while worn, motion is smooth, CNN's logged output looks reasonable next to the tracker.

**Step 5 — Practice the demo.**

---

## 5. Open Items (not yet decided / not yet verified)

- Exact debounce lockout fraction for the heel-strike tracker — starting point ~60–70% of rolling median cycle length, needs live tuning on the day.
- Whether/when to revisit live AI correction — after collecting more representative (impaired/slower/asymmetric) gait data, post-deadline.
- Whether to profile/shrink the 47ms inference — optional, not blocking.
- Exact conservative `THERAPIST_MAX_ROM_DEG` value for the first worn test — pick cautiously at Step 4, on the day.
- Re-confirm `AllocateTensors()` / the 130KB internal-SRAM arena still succeeds once real causal feature code (heavier than `test_stride.h` playback) runs alongside the dual-core split. Should hold — arena size is about the model graph, not the feature math — but worth re-checking after Step 3.

---

## 6. Reference Numbers Worth Remembering

- **FSM / phase boundaries:** Heel Strike 0–5%, Loading Response 5–15%, Mid-Terminal Stance 15–50%, Pre-Swing Toe-off 50–62%, Initial-Mid Swing 62–87%, Terminal Swing 87–100%.
- **Model:** 57 features × 71-sample window, INT8 quantized, ~27KB `.tflite`, 130KB internal-SRAM tensor arena (confirmed working).
- **Inference timing:** ~47ms measured on real ESP32-S3 hardware vs. 0.18ms on Colab/laptop (~260× slower on-device — expected for first deployment).
- **Real gait cycle durations:** ~2.84s at 0.5 mph down to ~1.34s at 2.0 mph.
- **Dataset:** S01–S04, 7 speeds × 3 trials = 84 files, ~60s each, ~94.7–100.02 Hz actual sampling.
- **Reported accuracy:** 99.47% (int8) / 99.64% (float), random split, pseudo-labels (no manual `gait_phase_id` ever recorded).
