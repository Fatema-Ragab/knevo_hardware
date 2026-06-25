# Knevo — Implementation Document

Merged master reference combining the integration decision log and the technical handoff/constraints notes from this build session. Step/Decision naming is kept exactly as originally used throughout (Decisions #1-#3, Steps 0-5, Steps A/B/C) to avoid confusion across the two source documents.

---

## 1. Project & File Inventory

**Project:** Smart Active Bionic Knee Brace (Knevo) — integrating the gait-classifier CNN with the motor control firmware.

| File | Role | Status |
|---|---|---|
| `moresmoother.ino` | Original control firmware prototype. FSM = status-only, cubic Hermite spline drives angle, 100Hz/20Hz/4Hz multi-frequency loop. | Superseded by Step A / Step C below. |
| `knevo_final_status.ino` | Data-collection firmware. Dual I2C, 3x MPU6050, 2x FSR, Wi-Fi TCP streaming, 4 calibration commands (`CAL_STATIC`/`CAL_UNLOADED`/`CAL_STANDING`/`CAL_KNEE`). | Tested & working. Confirmed unrelated to the current model's training (not in `raw.zip`, not used by `robust_norm()`). Stays as-is for future data collection; not merged into the live-inference firmware. |
| `Gait_DL.ino` | TFLite Micro inference test sketch. 130KB internal-SRAM tensor arena (confirmed working on real hardware). | Superseded by Step B / Step C below, which reuse its TFLite setup verbatim. |
| `knevo_esp32_boosted_phase_cnn_6phase_model.h` / `..._scaler.h` | Real trained INT8 TFLite model (~27KB) + 71-step/57-feature scaler constants. | Ground truth — `#include`d directly into Step B/C, never redefined. |
| `cnn_99.ipynb` | The actual training notebook — ground truth for the real 57-feature recipe, labeling logic, model architecture. | Reference only. |
| `metadata.json` / `raw.zip` | Real export metadata + real training dataset (4 subjects, 7 speeds, 3 trials = 84 files). | Reference only — see §2. |
| `knevo_stepC_dual_core.ino` | **Current working firmware.** Dual-core: Step A (control) on Core 1, Step B (sensor+DL) on Core 0. | Active development target — see §5-§13. |

---

## 2. Key Technical Findings (from dataset/notebook inspection)

- `PHASE_BOUNDARIES` in the training notebook exactly matches the `fsmTable` boundaries used in firmware (HeelStrike 0-5%, LoadingResponse 5-15%, MidTerminalStance 15-50%, PreSwingToeOff 50-62%, InitialMidSwing 62-87%, TerminalSwing 87-100%).
- Real 57-feature vector = 20 raw IMU/FSR + 3 accel-mag + 3 gyro-mag + 3 pitch + 3 roll + 2 knee-angle + 3 FSR combos + 2 pitch-diffs + 2 gyro-diffs + 12 derivative (`_d`) features + 4 "boosted phase tracker" features.
- Heel FSR's 97th percentile is **4095 (saturates) in every one of the 84 dataset files**; midfoot FSR's 97th percentile ranges **730-4095** (5.6x spread) — confirms per-session FSR normalization (not a fixed constant) is necessary.
- Training's derivative features use `np.gradient()` with **no `dt` argument** — per-sample-step, not per-second. Real measured sampling rate across files ranges 94.76-100.02 Hz (not perfectly clean 100Hz).
- Real measured gait cycle durations (from FSR heel-strike timing): median **2.84s at 0.5 mph** down to **1.34s at 2.0 mph**.
- **Labels are 100% pseudo-labels** — confirmed directly: `gait_phase_id == -1` in every row of all 84 raw files. No manual labeling occurred.
- Reported accuracy (99.47% int8 / 99.64% float) is on a **random** window-level split with only **4 subjects** — likely optimistic; generalization to new/impaired gait is unverified.
- Measured real inference time on ESP32-S3 hardware: **~47-50ms** (vs. metadata.json's Colab figure of 0.18ms — expected for first on-device deployment, ~260x slower).
- Dataset is treadmill-only, fixed prescribed speeds, presumably able-bodied volunteers — **not representative of an actual impaired wearer.**

---

## 3. Decision Log

### Decision #1 — Feature Engineering Parity ✅ LOCKED
Real training recipe requires causal (real-time-safe) approximations for things the offline pipeline computed with whole-file knowledge:

| Gap | Problem | Solution | Criticality |
|---|---|---|---|
| 1. FSR normalization | `robust_norm()` uses whole-file 3rd/97th percentiles | Per-session calibration (unloaded/loaded bounds) | Not critical — workflow discipline, not a code risk |
| 2. Derivative features | `np.gradient()` centered, no dt | Backward difference, **no dt division** (matches training's actual convention) | Not critical — free to implement correctly |
| 3. Phase-tracker features | Whole-file-aware heel-strike anchor detection | Causal rolling-median heel-strike tracker with debounce/lockout | **The one that matters** — without debounce, FSR noise fakes extra strikes |
| 4. Label mode | Needed to know real vs. pseudo labels | Confirmed: 100% pseudo-labels | Informational |

### Decision #2 — What drives `Gait_percent`, and what is the CNN for? ✅ LOCKED
- The causal heel-strike tracker (Decision #1 Gap 3) drives `Gait_percent` directly and continuously — not the CNN.
- CNN's predicted phase is compared against the tracker's FSM bin and logged as a disagreement — never edits the trajectory.
- **Shadow mode:** a boolean flag gates whether a CNN/tracker disagreement can ever nudge `Gait_percent`. Default off.
- **Safety stays 100% hardware-truth-based.** The CNN never touches the freeze/support-mode path. Rule: *the CNN can raise suspicion, only hardware truth can freeze the system.*
- Rationale: dataset is 4 able-bodied subjects on a treadmill — not representative of an actual impaired wearer, and there was no time to validate correction behavior before a live demo.

### Decision #3 — 47ms inference vs. 10ms control loop ✅ LOCKED
- `Gait_DL.ino` is single-core; a 47ms `Invoke()` call would block sensors/tracker/motor for its full duration if combined naively.
- **Decision:** true dual-core split via FreeRTOS (`xTaskCreatePinnedToCore`). One core: sensors → tracker → spline → motor, strict 10ms, never blocked. Other core: feature window + CNN inference, runs whenever free, results logged via simple handoff (no queue needed — shadow/logging only, staleness doesn't matter).
- This is essential infrastructure regardless of shadow-mode status — pure architecture.

**Note on Decision #2 in light of later events (see §5):** the heel-strike tracker actually driving `Gait_percent` for the motor was the original Step 1→2 goal, but was pulled from the motor control path after Step 2 testing surfaced real bugs under time pressure (see §4 update and §5). The tracker's logic (after its bugs were fixed) lives on inside Step B's feature pipeline, still producing the CNN's input features — it just no longer drives Core 1's motor control. This is consistent with, not a reversal of, Decision #2's shadow-mode principle: it was already established that nothing not yet proven trustworthy gets to touch the motor; the tracker itself turned out to need more runway than was available before the deadline.

---

## 4. Original Build Plan (Steps 0-5) — status update

**UPDATE:** Steps 0-2 were executed and surfaced real bugs in the heel-strike tracker (median contamination from sparse test presses, a jump-validation/reset mismatch causing false freezes). Given the live-demo deadline, the plan was deliberately revised: the tracker was pulled out of the motor's control path entirely. Steps 0-2's bug fixes and lessons (debounce, gap-timeout, reset-bypass) remain valid and are reused inside Step B. Steps 3-5 below were superseded by Steps A/B/C (§5).

- **Step 0 — Verify the safety path alone.** `checkScenario2Emergency()`/`freezeSystem()` bench-tested with motor under load, not worn. Confirmed: forced angle error → motor halt + (originally buzzer, later changed to LED — see §5). ✅ Done.
- **Step 1 — Causal heel-strike tracker, standalone.** Real-time `Gait_percent` from live FSR, implementing Decision #1 Gaps 1 & 3. Bugs found and fixed here: median contamination from sparse presses (gap-timeout fix), negative phase-summary counts (running-total vs. validity-flag split). ✅ Done, bugs fixed.
- **Step 2 — Wire the tracker into the control loop.** Replaced simulated gait input with Step 1's real tracker. Surfaced the jump-validation-vs-reset mismatch (false freezes on irregular press timing) — fixed via a heel-strike-reset bypass of the jump-suspicion check. ⚠️ Superseded — see §4 update above.
- **Step 3 — ~~Dual-core bring-up, CNN in shadow mode~~** → superseded, see Step C (§5).
- **Step 4 — ~~First worn test~~** → superseded, see Step A (§5); worn testing now happens on the open-loop firmware, not the tracker-driven loop.
- **Step 5 — Practice the demo.** Still applies, against whatever firmware is actually used.

---

## 5. Revised Plan — Steps A / B / C

### Step A — Open-loop control firmware (no tracker, no FSR dependency)
Reverted to the original tested simulated-curve control logic (FSM status, spline, look-ahead, smoothing, jump-validation, collapse-detection safety — all unchanged from the proven version). **This is what gets worn.**

Changes made within Step A over the course of testing:
- Buzzer → built-in RGB LED (solid blue on freeze).
- Simulated cycle retuned from a 12.5s bench-test pace through a full bridge table up to real dataset-correlated speeds (see §10 for the full speed table).
- `COMMAND_SMOOTH_ALPHA` lowered 0.18 → 0.12 after live testing showed a snap specifically at a direction-reversal point — diagnosed as actuator gearbox backlash (see §9).
- `kneeTable` smoothed from a two-hump clinical curve to a one-peak curve, and `gaitTable` breakpoints retimed — see §7.5 for full details; this was the actual fix for motor freezes at higher speeds, not a filter/timing tweak.
- Runtime-configurable speed (12 dataset-correlated presets + stop) and ROM (extension/flexion) added — see §10.
- Stop-at-extension behavior added — see §11.

**Open-loop, by deliberate choice:** the wearer moves along with the device's fixed rhythm rather than the device sensing/adapting to them. True and defensible to say outright if asked, not something to gloss over.

### Step B — Sensor + feature extraction + DL inference pipeline (logging only)
Real I2C/FSR reads (verbatim from `knevo_final_status.ino`'s tested MPU6050 code) → the real, full 57-feature recipe from Decision #1 → TFLite Micro inference (reusing the confirmed-working 130KB internal-SRAM arena setup from `Gait_DL.ino`). Output (predicted phase, probability, tracker's own progress/phase) goes to Serial only — **zero connection to the motor.** See §12-§13 for full status.

### Step C — Dual-core integration ✅ ACTIVE, WORKING (with known open items)
Step A pinned to one core (strict 10ms, never blocked), Step B pinned to the other (runs at its own pace). Not data-connected: Step B logs independently; Step A runs on its own simulated input. A Serial mutex guards console output. See §6-§13 for full current state, all bugs found/fixed, and what's still open.

**What did NOT get rebuilt, and why that's fine:** the heel-strike tracker actually driving `Gait_percent` for the motor (the original Step 1→2 goal) is shelved, not abandoned — it's real, working logic but unproven on a worn device under time pressure. Revisit post-deadline with more runway to bench-test the tracker→motor connection properly before trusting it on a person.

---

## 6. Current Architecture Snapshot (Step C)

Single sketch (`knevo_stepC_dual_core.ino`), two FreeRTOS tasks pinned to separate cores:

- **Core 1 (default Arduino `loop()`)** = Step A. Open-loop simulated gait input, cubic Hermite spline trajectory, command smoothing, jump-validation, motor streaming, collapse-detection safety. **This is what's worn.**
- **Core 0 (`coreBTask`, a separate FreeRTOS task)** = Step B. Real IMU/FSR sensor reads, the full causal 57-feature recipe, TFLite Micro inference. Runs independently, logs to Serial only. **Not connected to Core 1.**

A Serial mutex (`serialMutex`) guards all console output since both cores print independently.

---

## 7. DO NOT EDIT WITHOUT EXTREME CARE

### 7.1 The safety check must never depend on anything except real hardware feedback
`checkScenario2Emergency()` compares `actualAngle` (read back from the motor over UART) against `commandedAngle`. **This must never be gated by, blended with, or overridden by the CNN/DL pipeline, network state, or anything else.** Rule from Decision #2, never relaxed: *the CNN can raise suspicion, only hardware truth can freeze the system.*

### 7.2 The freeze is a one-way latch, by design
Once `systemFrozen` is set (via `freezeSystem()`), there is **no auto-recovery** anywhere in this firmware — requires a power cycle. Deliberately verified in Step 0 and relied on ever since. Don't add auto-clear without redoing that safety review.

### 7.3 Motor protocol bytes — tested, exact, do not improvise
```
Enable:        {0x3E, 0x88, 0x01, 0x00, 0xC7}
Set position:  0x3E, 0xA3, 0x01, 0x08, checksum(header)=0xEA,
               then int32 (angle_deg * MOTOR_SCALE) little-endian (4 bytes),
               then 4 zero bytes, then checksum = sum(bytes 5-12) & 0xFF
Read position: query {0x3E, 0x92, 0x01, 0x00, 0xD1} -> response parsed for
               0x3E,0x92 header, then int32 little-endian / MOTOR_SCALE
MOTOR_SCALE = 3722.0f
```
`MotorSerial` on UART1, GPIO4 (RX) / GPIO5 (TX), 115200 baud. A future actuator/protocol swap needs full re-verification of `sendCmd`/`motorOn`/`sendPositionFast`/`readActualAngleQuick`, not just new byte values.

### 7.4 Two *different* ROM constant pairs exist — do not confuse them
- `NORMAL_ROM_MIN_DEG` / `NORMAL_ROM_MAX_DEG` (both `const`, 5/65) — the **fixed clinical reference scale**, the denominator basis for `scaleAngleToPatientROM()`'s proportional mapping. Represents the *shape* of the reference curve. **Never make these patient-configurable.**
- `THERAPIST_MIN_ROM_DEG` / `THERAPIST_MAX_ROM_DEG` (both mutable, default 5/65) — the **actual per-session prescribed limits**, runtime-configurable via `E<deg>`/`F<deg>` (§10). This is what the mobile app's BLE config should map onto.
- The motor clamp and the stop-at-extension rest angle (§11) both correctly use the *configurable* pair — re-verified during this session specifically because the naming invites confusion.

### 7.5 The trajectory curve has been deliberately altered twice
- `kneeTable` originally `{3.97, 19.84, 18.86, 11.09, 7.72, 13.86, 38.74, 64.12, 53.27, 17.27, 2.21}` — a real, clinically-standard **two-hump** curve (small early-stance flexion wave + big swing-phase peak; documented gait biomechanics, not a data error). Now `{4, 8, 14, 20, 28, 38, 50, 64.12, 53, 17, 2}` — smoothed to **one** peak, deliberately trading clinical realism for fewer actuator-backlash-prone direction reversals (two per cycle → one). **Revert here if clinical accuracy must win over hardware smoothness** — expect the backlash snap to return at the early-stance bump.
- `gaitTable` originally evenly spaced every 10%. Now retimed to `{0, 3.3, 8.2, 13.1, 19.6, 27.8, 37.6, 49.2, 58.3, 87.7, 100}` so every segment demands roughly equal angular speed instead of one segment (the old 80-90% span) spiking to ~3x the others. Derived from `Δangle / segment_duration` per segment, reallocated proportional to `|Δangle|`. **If `kneeTable` values ever change again, this retiming must be redone.**
- `fsmTable` (status boundaries: HeelStrike 0-5%, LoadingResponse 5-15%, MidTerminalStance 15-50%, PreSwingToeOff 50-62%, InitialMidSwing 62-87%, TerminalSwing 87-100%) is a **separate, independent** labeling scheme, not tied to `gaitTable`'s spline knots, and was *not* touched by the retiming.

### 7.6 The look-ahead bypass at rest
`updateSmoothFSMAndTrajectory()` has a branch for `fullyStoppedAtExtension` that **skips the look-ahead entirely** and targets `THERAPIST_MIN_ROM_DEG` directly. Without this, the look-ahead (`LOOKAHEAD_PERCENT = 4.0`) would keep aiming ~4% into the curve ahead of true rest, landing 8-14° instead of the actual floor. Preserve this bypass (or an equivalent) in any future rework of "stopped" state.

### 7.7 Arduino IDE auto-prototype gotcha
The Arduino preprocessor auto-generates function prototypes immediately after `#include` lines — **before any custom struct/type defined later in the same `.ino` file.** A function whose *signature* uses a custom struct (e.g. `bool initMPU(ImuDef &imu)`) fails with a confusing "type not declared" error, even though the struct is defined earlier in the visible code, because the auto-prototype hoists above it. Header/built-in types (e.g. `TwoWire`) don't have this problem. **Fix used:** functions take a plain `int` index into a global array instead of a struct reference in the signature (the struct can still be used freely *inside* the function body). Keep this pattern if ever splitting into real `.h`/`.cpp` files, which sidesteps the issue entirely.

### 7.8 `motorOn()` has the same kind of timeout bug as one we fixed elsewhere — known, unfixed, low-risk
`readActualAngleQuick()` originally had a *sliding* timeout (reset on every received byte instead of bounding total elapsed time) — fixed (§8). `motorOn()` has the identical pattern, **deliberately left alone** because it only runs once during `setup()`, not on every control tick. Apply the same fixed-total-elapsed-time fix if `motorOn()`'s reliability ever becomes suspect.

---

## 8. Bugs Found and Fixed (chronological, with root cause)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | (Step 1) Heel-strike tracker median ballooned to 8-9s | A long real-world pause between sparse test presses got recorded as one real (very slow) stride, then the inflated median made the debounce reject legitimate faster presses too, with no way to self-correct | Max-plausible-cycle timeout: a gap longer than ~4s is treated as "stopped/restarted," not a slow stride, and isn't added to the rolling median |
| 2 | (Step 1) Phase-summary diagnostic printed negative numbers | A "running total" counter was being reset to 0 by the same timeout logic that should only affect a separate "is this currently trustworthy" flag | Split into a monotonic running total and a separate boolean that can drop without corrupting the total |
| 3 | (Step 2) False freezes on irregular tracker press timing | `validateOrPredictGait()`'s jump-suspicion check (designed for noisy classifier output) treated a legitimate heel-strike reset as a suspicious large jump if the previous cycle hadn't yet reached ~100% | Heel-strike-reset bypass: a known-good reset (already vetted by the tracker's own debounce/gap-timeout) skips the jump-suspicion check for that one tick |
| 4 | (Step C) Compile error: `'ImuDef' was not declared in this scope` | Arduino auto-prototype-hoisting issue (§7.7) | Changed `initMPU`'s signature to take an `int` index instead of a struct reference |
| 5 | (Step C) Motor froze during speed testing, worse at higher presets | `checkSpeedCommand()` used a *blocking* `Serial.readStringUntil('\n')`. While blocked, Core 1 couldn't advance; the next `dt` was huge, producing an oversized gait-percent jump rejected repeatedly, freezing the system. Jump size scales with speed, explaining why only higher presets crossed the threshold | Rewrote the Serial command reader to be non-blocking by construction |
| 6 | (Step C) Suspected, then disproven: `readActualAngleQuick()`'s sliding timeout | Real latent bug, but disproven as *this* freeze's cause — an older file with the identical pattern ran fine at comparable speeds | Fixed anyway (bound total elapsed time, not per-byte) as a real risk under cross-core contention, just not the active cause that time |
| 7 | (Step C) Motor froze again at presets 9-10, reproducibly after ~2-3 cycles | **Not a software bug.** Instrumented the freeze moment: `actualAngle=32.3, commandedAngle=8.3` — the real actuator physically couldn't keep up with the commanded trajectory (one segment demanded ~252°/s) | Not "fixed" by loosening the check — fixed by retiming `gaitTable` (§7.5) so no segment demands more speed than the actuator can deliver |
| 8 | (Step C) "Two range-of-motion loops" suspected | Wasn't two loops — the single curve genuinely has two humps of very different height, which can look like two alternating things at slow speed | No code bug; led to the deliberate `kneeTable` smoothing (§7.5) |
| 9 | (Step C) Speed ramp "didn't feel linear" | Continuous fixed-acceleration ramp wasn't tied to the actual dataset-correlated preset speeds | Replaced with discrete stepping through the 12 presets, one step per second, with 0 as a 13th step below preset 1 |

---

## 9. Empirically Discovered Hardware Limits

- **The actuator has a real, finite maximum angular speed** — not in any spec sheet available; discovered by instrumenting the exact freeze moment (bug #7) and computing the literal `Δangle / time` the curve demanded. Original evenly-spaced curve demanded up to **~252°/s** in its worst segment; after retiming, every segment demands a roughly uniform **~85°/s** (at the 1.43s-cycle preset tested). **Before increasing demo speed beyond what's tested, re-run the diagnostic sweep** rather than assuming the retiming covers all future speeds — only verified up to preset 10 so far.
- **Cross-core contention was suspected multiple times and was wrong every time.** Every actual freeze traced back to either a software bug or a genuine actuator speed limit — not Core 0 disrupting Core 1. Don't reach for cross-core interference as an explanation without instrumented evidence (the `BAD READING #`/freeze-source print pattern used throughout this session) — instrument before guessing.
- **Core 0 (sensor+DL) was measured running well below its 100Hz target** (~16.5-91Hz depending on conditions; I2C read ~1.4-1.5ms/tick, feature computation ~50-90µs/tick, separate from the ~47-50ms inference cost every 5th sample). Never fully root-caused — see §14.

---

## 10. Runtime-Configurable Parameters (current Serial command interface)

Directly relevant to mobile app wiring — these are exactly the parameters the BLE config screen needs to set, and the current Serial interface is a working bench stand-in for that future feature.

| Command | Effect | Maps to |
|---|---|---|
| `0` | Step down 1 preset/sec to the slowest, continue at that speed until the current stride completes, then rest at full extension (§11) | "Stop" button |
| `1`-`12` | Set target speed preset; steps there 1 preset/sec from wherever it currently is | "Speed / gait time" config |
| `E<deg>` | Set max extension (floor). Must be `>= 0` and `< current flexion limit` | "Max ROM" (extension side) |
| `F<deg>` | Set max flexion (ceiling). Must be `> current extension limit` and `<= 90` | "Max ROM" (flexion side) |

The 12 preset speeds (index : %/sec : real-world equivalent):

| Idx | %/sec | Cycle (s) | Real-world equivalent |
|---|---|---|---|
| 1 | 8.0 | 12.5 | Original bench value — not based on real gait, avoid exposing to end users |
| 2 | 10.0 | 10.0 | bridge value |
| 3 | 12.5 | 8.0 | bridge value |
| 4 | 16.7 | 6.0 | bridge value |
| 5 | 25.0 | 4.0 | bridge value |
| 6 | 35.2 | 2.84 | 0.50 mph (slowest real dataset speed) |
| 7 | 48.1 | 2.08 | 0.75 mph |
| 8 | 58.0 | 1.73 | 1.00 mph |
| 9 | 67.3 | 1.49 | 1.50 mph |
| 10 | 70.1 | 1.43 | 1.25 mph |
| 11 | 71.7 | 1.39 | 1.75 mph |
| 12 | 74.5 | 1.34 | 2.00 mph (fastest real dataset speed) |

Rows 9-10 swap relative to their mph labels — known noise in the original cycle-length estimate, not a new error.

**For the mobile app:** presets 1-5 are bench/bridge values with no real-gait basis and probably shouldn't be exposed as user-facing speed options — likely only 6-12 (the actual dataset-correlated speeds) make sense for a patient-facing UI.

---

## 11. Stop-at-Extension Behavior Contract

A real behavioral contract the mobile app's "Stop" button needs to expect, not an instant motor stop:

1. On a stop request, speed steps down one preset per second (same as normal speed changes) until it reaches the slowest preset (index 1).
2. It then **keeps moving** at that speed — does not freeze — until the current gait stride actually completes (`Gait_percent` wrapping from ~100% back to ~0%).
3. Only then does it snap to a true stop and ease smoothly to rest at exactly `THERAPIST_MIN_ROM_DEG` (max extension).

A "stop" can take anywhere from just-under-one-stride-length up to (deceleration time + one full stride). **The mobile app should not assume the brace is immediately still after sending a stop command** — design around a graceful finish, not an immediate halt; ideally surface both states (`pendingStopAtExtension` = finishing stride, `fullyStoppedAtExtension` = actually at rest) distinctly in any future status protocol, not collapsed into one "stopped" boolean.

---

## 12. Dual-Core Architecture & Shared Resources

- **Core 1** = Step A (control). **Core 0** = Step B (sensor + DL, `coreBTask`).
- **Shared: Serial.** Both cores print independently; `serialMutex` guards every multi-line print block on both sides. Any new print statement added to either core must be wrapped the same way, or output will garble.
- **Shared: built-in RGB LED** (`neopixelWrite`) — used by both cores, **not** mutex-protected (deliberate: worst case is a momentary color flicker, not corrupted multi-byte output). Revisit only if LED behavior looks genuinely wrong, not preemptively.
- **Not shared:** I2C buses (Core 0 only), motor UART (Core 1 only), FSR analog pins (Core 0 only). No GPIO conflicts currently exist.
- **Core 0's task stack** sized at 16384 bytes — generous given TFLite Micro's call depth and floating-point math, never verified against actual stack high-water-mark. Profile before shrinking if memory ever gets tight.

---

## 13. Sensor + DL Pipeline (Step B) Status

- Real I2C reads (3x MPU6050, dual bus) and 2x FSR reads, feeding the **full real 57-feature recipe** (matching the actual training notebook): causal per-session FSR calibration standing in for offline `robust_norm()`, backward-difference derivatives matching training's non-dt-scaled `np.gradient`, and a debounced heel-strike tracker producing the four "boosted phase" features.
- TFLite Micro inference reuses the **confirmed-working** 130KB internal-SRAM tensor arena (not PSRAM — model is only ~27KB, internal SRAM already proven faster and working on real hardware).
- **Deliberately not connected to Core 1's motor control** — not a placeholder waiting to be wired up casually. A considered decision (Decision #2) that sensor/DL output stays observed-only until tested on a population more representative of an actual wearer than the 4 able-bodied bench subjects the model was trained on.
- Performance measured, not optimized: achieved sample rate seen anywhere from ~16.5Hz to ~91Hz, well under the 100Hz target. I2C and feature-computation time individually instrumented (`I2C=`, `Feature=` in `[STATUS-B]`), but overall throughput gap never fully root-caused.

---

## 14. Mobile App Integration — What Aligns, What's Missing, What to Watch For

**Already aligned (confirmed architecture):**
- BLE sends max ROM + speed/gait time before a session, sends start/stop. WiFi only turns on after a session ends, to upload data — never active during motion (consistent with: network stalls shouldn't introduce latency during real motion).
- The two BLE-configured parameters map **directly** onto `THERAPIST_MIN_ROM_DEG`/`THERAPIST_MAX_ROM_DEG` and the speed preset system (§10) — the existing Serial interface is a legitimate bench stand-in for that BLE config, not a separate thing to reconcile later.
- Backend anomaly/statistics analysis is after-the-fact review, not real-time safety — same boundary as the CNN: it can flag, only on-device hardware-truth can freeze.
- BLE and WiFi share the ESP32's radio hardware; the design's clean time separation (BLE pre-session, WiFi post-session) avoids any throughput/latency cost of running both simultaneously — worth deliberately preserving if the architecture evolves.

**What's missing (not built yet):**
- No actual BLE server code exists yet. The Serial command interface is a deliberate stand-in, not a partial BLE implementation.
- No WiFi-based post-session upload exists in this firmware. (`knevo_final_status.ino` already has WiFi + the 4 calibration commands, but for a different purpose — recording new training data, not session telemetry — and doesn't need to be merged in.)
- No connection between Core 0 (sensor/DL) and Core 1 (motor) exists. If a future requirement is "show live gait phase from the sensor in the app," that's Core 0's `[STATUS-B]` data — already computed, never transmitted anywhere beyond Serial.

**What to watch for when wiring BLE config in:**
- Validate ROM bounds the same way the Serial parser does (extension `< current flexion`, flexion `> current extension` and `<= 90`) — these bounds exist because of the proportional-scaling math in §7.4, not arbitrarily.
- Presets 1-5 (§10) aren't real gait speeds — don't expose them in a patient-facing speed selector without relabeling/hiding them.
- Honor the stop behavior contract (§11) — a BLE "stop" command should trigger the same graceful finish-then-rest, not an instant cut; surface "finishing stride" vs. "fully stopped" as distinct states if showing real-time status.
- If BLE/WiFi are ever both active closer together in time than the current pre-/post-session split, double check radio-sharing implications — don't casually narrow that gap.

---

## 15. Open Items / Not Yet Validated

- Core 0's actual throughput bottleneck (well under 100Hz) was instrumented but never fully root-caused.
- The actuator speed ceiling (§9) was only verified up to preset 10 with the retimed curve — presets 11-12 unconfirmed.
- `motorOn()`'s sliding-timeout pattern (§7.8) is known-but-unfixed, low risk only because it's setup-only.
- No worn human testing logged — everything here is bench-validated (motor under no load / hand-resistance / fixture), not validated on an actual wearer.
- Feature-engineering parity (Decision #1) is implemented but the causal approximations (FSR calibration cadence, debounce thresholds, etc.) have only been bench-tested with finger presses, not real walking — worth a real-walking validation pass before trusting Step B's output even as a logged/observed signal.
- The heel-strike tracker that was originally meant to drive `Gait_percent` directly (Steps 1→2) is real, working code (after its bugs were fixed) but deliberately shelved from the motor control path. It still exists inside Step B's feature pipeline, driving the 4 boosted-phase features — it just no longer drives anything on Core 1. Revisit the tracker→motor connection with more runway before trusting it on a worn device.

---

## 16. Quick Reference

**Pins:** Motor UART1 GPIO4(RX)/GPIO5(TX) @ 115200. Heel FSR GPIO1, Midfoot FSR GPIO2. I2C Bus0 SDA8/SCL9 (FOOT 0x68, SHANK 0x69), Bus1 SDA10/SCL11 (THIGH 0x68). Built-in RGB LED GPIO48 default.

**Key constants:** `MOTOR_SCALE=3722.0`, `CONTROL_MS=10`, `FEEDBACK_MS=250`, `PLOT_MS=50`, `LOOKAHEAD_PERCENT=4.0`, `COMMAND_SMOOTH_ALPHA=0.12`, `MAX_GAIT_JUMP_PERCENT=10.0`, `MAX_BAD_READINGS=3`, `COLLAPSE_ERROR_DEG=20.0`, `SPEED_STEP_MS=1000`.

**TFLite:** 130KB internal-SRAM arena, 57 features, 71-sample window, INT8 quantized, ~27KB model, inference every 5th sample, real measured invoke time ~47-50ms.
