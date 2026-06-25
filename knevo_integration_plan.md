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

**UPDATE (later same night):** Steps 0–2 below were executed and surfaced real bugs in the heel-strike tracker (median contamination from sparse test presses, a jump-validation/reset mismatch that caused false freezes). Given the live-demo deadline, the plan was deliberately revised: the tracker was pulled out of the motor's control path entirely. See §7 for the current plan (Steps A/B/C), which supersedes Steps 3–5 below. Steps 0–2's bug fixes and lessons (debounce, gap-timeout, reset-bypass) remain valid and are reused inside Step B.

Approach: **step-by-step, gated** — confirm each step actually works before moving to the next, so any failure is easy to isolate.

**Step 0 — Verify the safety path alone, before anything new touches it.**
Bench-test `checkScenario2Emergency()` / `freezeSystem()` with the motor under load but *not worn* (fixture or hand-resistance). Confirm a forced angle error actually halts the motor and fires the buzzer. Must hold true no matter what gets built later today.

**Step 1 — Causal heel-strike tracker, standalone.**
Real-time `Gait_percent` from live IMU/FSR sensors, implementing Decision #1's Gap 1 (calibration) and Gap 3 (debounced rolling-median heel-strike detection). No motor, no CNN yet. Sanity-check: numbers look like real strides while walking/stepping, not noise.

**Step 2 — Wire the tracker into the existing control loop.**
Replace `moresmoother.ino`'s simulated `getGaitPercentInput()` with Step 1's real tracker output. Bench-test the motor (not worn): confirm smoothing, look-ahead, and jump-rejection still behave correctly against real noisy input instead of the clean sine wave it was tested against before.

**Step 3 — ~~Dual-core bring-up, CNN in shadow mode~~ (superseded, see §7 Step C).**

**Step 4 — ~~First worn test~~ (superseded — see §7; worn testing now happens on the open-loop Step A firmware, not the tracker-driven loop).**

**Step 5 — Practice the demo.** (Still applies, against whatever firmware is actually used for the demo.)

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

---

## 7. Revised Plan (current) — Steps A / B / C

Triggered by Step 2 bench testing: the heel-strike tracker surfaced three real bugs in a row (median contamination from sparse/irregular test presses, a jump-validation-vs-reset mismatch causing false freezes) right at the point of driving the motor. Given the live-demo deadline, the tracker was deliberately pulled out of the motor's control path — same shadow-mode principle already used for the CNN (Decision #2), just applied one layer earlier.

**Step A — Open-loop control firmware (no tracker, no FSR dependency).**
Reverted to the original tested simulated-curve control logic (FSM status, spline, look-ahead, smoothing, jump-validation, collapse-detection safety — all unchanged from the proven version). Two deliberate tweaks: buzzer → built-in RGB LED (solid blue on freeze), and the simulated cycle retuned from a 12.5s bench-test pace to a realistic walking pace. Settled at **2.2s/cycle, `COMMAND_SMOOTH_ALPHA = 0.12`** after live testing showed a brief snap specifically in the 0–30% gait range (the only direction-reversal in the early-stance "knee flexion wave") — most likely gearbox backlash, not a timing/math bug, since a bigger but single-direction sweep later in the cycle stayed smooth even though faster. This is what gets worn for the demo. **Open-loop**, by deliberate choice: the wearer moves along with the device's fixed rhythm rather than the device sensing/adapting to them — true and defensible to say outright if asked, not something to gloss over.

**Step B — Sensor + feature extraction + DL inference pipeline (logging only).**
Real I2C/FSR reads (verbatim from `knevo_final_status.ino`'s tested MPU6050 code) → the real, full 57-feature recipe from Decision #1 (causal FSR calibration, backward-difference derivatives matching training's non-dt-scaled `np.gradient`, the debounced heel-strike tracker for the 4 boosted-phase features) → TFLite Micro inference (reusing the confirmed-working 130KB internal-SRAM arena setup from `Gait_DL.ino`). Output (predicted phase, probability, tracker's own progress/phase) goes to Serial only — **zero connection to the motor.**

**Step C — Dual-core integration.**
Step A pinned to one core (strict 10ms, never blocked), Step B pinned to the other (runs at its own pace — its ~47ms inference and heavier per-tick feature math can never stall Step A's timing). The two are not data-connected yet: Step B logs independently; Step A still runs on its own simulated input. A Serial mutex guards console output since both cores print independently. *(Status: in progress.)*

**What did NOT get rebuilt tonight, and why that's fine:** the heel-strike tracker actually driving `Gait_percent` for the motor (the original Step 1→2 goal) is shelved, not abandoned — it's real, working logic (after the bug fixes) but unproven on a worn device under time pressure. Revisit post-deadline with more runway to bench-test the tracker→motor connection properly before trusting it on a person.

---

## 8. Confirmed Future Architecture — Mobile App Integration

Not built tonight; a confirmed design decision for the next phase, logged so it isn't lost:

- **BLE** for short, intermittent config/control: the app sends **max ROM** and **speed/gait time** before a session starts, and sends **start**/**stop** commands. Session ends on timer completion or a stop press.
- **WiFi** for the bulk data dump: only turned on **after** a session ends, to upload that session's data to the backend. WiFi is never active during motion — consistent with the principle already established tonight (network stalls introduce latency the control loop can't afford during real motion).
- **Backend** runs anomaly/statistics analysis on uploaded sessions — appropriately, since this is heavy analysis that has no business running on the ESP32 alongside the control loop + model. This is **after-the-fact** review (e.g. for a clinician), not real-time safety — same principle as the CNN: backend/AI can flag concerns after the fact, only on-device hardware-truth can freeze the motor in the moment. Don't let "the backend will catch it" erode that boundary later.
- **Technical note for later:** BLE and WiFi share the ESP32's radio hardware. This design's natural separation in time (BLE only pre-session, WiFi only post-session) avoids the throughput/latency cost of running both simultaneously — worth deliberately preserving if the architecture evolves.
- **Direct line to existing firmware:** the two BLE-configured parameters (**max ROM**, **speed/gait time**) are exactly `THERAPIST_MAX_ROM_DEG` and the cycle-pacing constant already being hand-tuned in Step A tonight. Making those runtime-configurable instead of hardcoded is a small, contained future task, not a redesign.
- **Separately, already resolved tonight:** the 4 original data-collection commands (`CAL_STATIC`/`CAL_UNLOADED`/`CAL_STANDING`/`CAL_KNEE`) in `knevo_final_status.ino` are unrelated to this — they were never used in training this model (confirmed: not present in `raw.zip`, and `robust_norm()` reads percentiles straight from each walking file, not from a separate calibration file). They stay exactly as-is in `knevo_final_status.ino` for future data collection; no need to merge them into the live-inference firmware.
