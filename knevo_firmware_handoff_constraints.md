# Knevo Firmware — Constraints, Lessons Learned & Mobile App Integration Notes

**Purpose of this document:** this is the accumulated, hard-won knowledge from one long debugging/integration session on `knevo_stepC_dual_core.ino` — every real bug found, every empirical hardware limit discovered, and every design decision made. Read this *before* restructuring this firmware or wiring it to the mobile app, so the same mistakes aren't repeated and the same fragile spots aren't broken again.

---

## 1. Current Architecture Snapshot

Single sketch, two FreeRTOS tasks pinned to separate cores:

- **Core 1 (default Arduino `loop()`)** — the control firmware. Open-loop simulated gait input (no live sensor feedback driving it yet), cubic Hermite spline trajectory, command smoothing, jump-validation, motor streaming, collapse-detection safety. **This is what's worn.**
- **Core 0 (`coreBTask`, a separate FreeRTOS task)** — real IMU/FSR sensor reads, the full causal 57-feature recipe, TFLite Micro inference. Runs independently, logs to Serial only. **Not connected to Core 1 at all.** Its output goes nowhere — this is deliberate (see §8 and §9).

A Serial mutex (`serialMutex`) guards all console output since both cores print independently.

---

## 2. DO NOT EDIT WITHOUT EXTREME CARE

These are either safety-critical, empirically validated against real hardware, or fragile in a non-obvious way.

### 2.1 The safety check must never depend on anything except real hardware feedback
`checkScenario2Emergency()` compares `actualAngle` (read back from the motor over UART) against `commandedAngle`. **This must never be gated by, blended with, or overridden by the CNN/DL pipeline, network state, or anything else.** Rule established early and never relaxed: *the CNN can raise suspicion, only hardware truth can freeze the system.* If the mobile app or backend ever wants to "help" with safety, it can only ever add a second, independent check — never replace or weaken this one.

### 2.2 The freeze is a one-way latch, by design
Once `systemFrozen` is set (via `freezeSystem()`), there is **no auto-recovery** anywhere in this firmware. It requires a power cycle. This was deliberately verified back in early bench testing (Step 0) and has been relied on ever since. Do not add an auto-clear/auto-resume path without re-doing that safety review from scratch.

### 2.3 Motor protocol bytes — tested, exact, do not improvise
```
Enable:        {0x3E, 0x88, 0x01, 0x00, 0xC7}
Set position:  0x3E, 0xA3, 0x01, 0x08, checksum(header)=0xEA,
               then int32 (angle_deg * MOTOR_SCALE) little-endian (4 bytes),
               then 4 zero bytes, then checksum = sum(bytes 5-12) & 0xFF
Read position: query {0x3E, 0x92, 0x01, 0x00, 0xD1} -> response parsed for
               0x3E,0x92 header, then int32 little-endian / MOTOR_SCALE
MOTOR_SCALE = 3722.0f
```
`MotorSerial` on UART1, GPIO4 (RX) / GPIO5 (TX), 115200 baud. If a future actuator/protocol swap happens, all of `sendCmd`/`motorOn`/`sendPositionFast`/`readActualAngleQuick` need re-verification, not just the byte values.

### 2.4 Two *different* ROM constant pairs exist — do not confuse them
- `NORMAL_ROM_MIN_DEG` / `NORMAL_ROM_MAX_DEG` (both `const`, 5/65) — the **fixed clinical reference scale**, the denominator basis for `scaleAngleToPatientROM()`'s proportional mapping. These represent the *shape* of the reference curve. **Never make these patient-configurable** — changing them silently distorts the proportional scaling math for everyone.
- `THERAPIST_MIN_ROM_DEG` / `THERAPIST_MAX_ROM_DEG` (both mutable floats, default 5/65) — the **actual per-session prescribed limits**, runtime-configurable via `E<deg>`/`F<deg>` Serial commands (see §5). This is what the mobile app's BLE config should map onto.
- The final motor clamp (`clampFloat(commandedAngle, THERAPIST_MIN_ROM_DEG, THERAPIST_MAX_ROM_DEG)`) correctly uses the *configurable* pair. The stop-at-extension rest angle (§6) was checked and also correctly uses `THERAPIST_MIN_ROM_DEG`, not the fixed constant — this was specifically re-verified during this session because the naming is easy to mix up.

### 2.5 The trajectory curve has been deliberately altered twice — know why before touching it again
- `kneeTable` (knee angle waypoints) was originally `{3.97, 19.84, 18.86, 11.09, 7.72, 13.86, 38.74, 64.12, 53.27, 17.27, 2.21}` — a real, clinically-standard two-hump curve (small early-stance flexion wave + big swing-phase peak; this shape is documented gait biomechanics, not a data error). It is now `{4, 8, 14, 20, 28, 38, 50, 64.12, 53, 17, 2}` — smoothed to **one** peak, deliberately trading clinical realism for fewer actuator-backlash-prone direction reversals (the original had two reversals per cycle; this has one). **If clinical accuracy ever needs to win out over hardware smoothness, this is the line to revert** — but expect the backlash snap to come back at the early-stance bump.
- `gaitTable` (the percentage breakpoints those waypoints occur at) was originally evenly spaced every 10%. It is now `{0, 3.3, 8.2, 13.1, 19.6, 27.8, 37.6, 49.2, 58.3, 87.7, 100}` — **retimed** so every segment demands roughly equal angular speed instead of one segment (the old 80-90% span) spiking to ~3x the speed of every other segment. This was derived by computing `Δangle / segment_duration` per segment and reallocating breakpoint spacing proportional to `|Δangle|`. **If `kneeTable` values are ever changed again, this retiming must be redone** — the current breakpoints are only correct for the current waypoint values.
- `fsmTable` (phase status boundaries: HeelStrike 0-5%, LoadingResponse 5-15%, MidTerminalStance 15-50%, PreSwingToeOff 50-62%, InitialMidSwing 62-87%, TerminalSwing 87-100%) is a **separate, independent** labeling scheme — it is not tied to `gaitTable`'s spline knot positions and was *not* touched by the retiming. It's just a status/telemetry label.

### 2.6 The look-ahead bypass at rest
`updateSmoothFSMAndTrajectory()` has a branch for `fullyStoppedAtExtension` that **skips the normal look-ahead entirely** and targets `THERAPIST_MIN_ROM_DEG` directly. This exists because the look-ahead (`LOOKAHEAD_PERCENT = 4.0`) would otherwise keep aiming ~4% into the curve ahead of true rest, landing the brace 8-14° instead of the actual floor. Any future change to how "stopped" is represented must preserve this bypass or re-derive an equivalent.

### 2.7 A genuine Arduino IDE gotcha, worth remembering for any future structural reorganization
The Arduino preprocessor auto-generates function prototypes and inserts them immediately after `#include` lines — **before any custom struct/type defined later in the same `.ino` file.** A function whose *signature* uses a custom struct (e.g. `bool initMPU(ImuDef &imu)`) will fail to compile with a confusing "type not declared" error, even though the struct is defined earlier in the visible code, because the auto-prototype hoists above it. Built-in/header types (like `TwoWire`) don't have this problem since they're already known from their `#include`. **Fix used here:** change such functions to take a plain `int` index into a global array instead of a struct reference in the signature (the struct can still be used freely *inside* the function body — only the signature is the problem). Keep this pattern if restructuring into multiple files/headers, since splitting into a real `.h`/`.cpp` pair sidesteps this Arduino-specific issue entirely.

### 2.8 `motorOn()` has the same kind of timeout bug as a fixed one we found — never triggered yet, but known and unfixed
`readActualAngleQuick()` originally had a *sliding* timeout (reset on every received byte instead of a fixed total elapsed time) — fixed during this session (see §3). `motorOn()` has the identical pattern and was **deliberately left alone** because it only runs once during `setup()`, not on every control tick, so it can't cause a live freeze. If `motorOn()`'s reliability ever becomes suspect (e.g. intermittent boot failures), apply the same fixed-total-elapsed-time fix used in `readActualAngleQuick()`.

---

## 3. Bugs Found and Fixed This Session (chronological, with root cause)

Useful as a "don't reintroduce these" checklist, especially if this firmware gets refactored for the mobile app.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Heel-strike tracker (early prototype, since shelved from the motor path) median ballooned to 8-9s | A long real-world pause between sparse test presses got recorded as if it were one real (very slow) stride, then the inflated median made the debounce reject legitimate faster presses too, with no way to self-correct | Added a max-plausible-cycle timeout: a gap longer than ~4s is treated as "stopped/restarted," not a slow stride, and isn't added to the rolling median |
| 2 | A phase-summary diagnostic printed negative numbers | A "running total" counter was being reset to 0 by the same timeout logic that should only affect a separate "is this currently trustworthy" flag | Split into two variables: a monotonic running total, and a separate boolean that can drop without corrupting the total |
| 3 | Dual-core build failed to compile: `'ImuDef' was not declared in this scope` | The Arduino auto-prototype-hoisting issue (§2.7) | Changed `initMPU`'s signature to take an `int` index instead of a struct reference |
| 4 | Motor froze partway through speed testing, worse at higher presets | `checkSpeedCommand()` used a *blocking* `Serial.readStringUntil('\n')`. While blocked, Core 1's control loop couldn't advance; the next `dt` was huge, producing an oversized gait-percent jump that got rejected repeatedly and froze the system. Jump size scales with speed, which is why it only crossed the rejection threshold at higher presets | Rewrote the Serial command reader to be non-blocking by construction — only consumes bytes already buffered, never waits |
| 5 | (Initially suspected, then disproven) `readActualAngleQuick()`'s sliding timeout (resets on every byte instead of bounding total elapsed time) | Real bug, but **not** the cause of bug #6 below — disproven when an older file with the identical pattern was shown to run fine at comparable speeds | Fixed anyway (bound total elapsed time, not per-byte) since it's a real latent risk under cross-core contention, just not the active cause that time |
| 6 | Motor froze again at presets 9-10 after fixes #4/#5, reproducibly after ~2-3 cycles | **Not a software bug at all.** Instrumented the exact freeze moment: `actualAngle=32.3, commandedAngle=8.3` — the real actuator physically couldn't keep up with the commanded trajectory's required angular velocity at that speed (one segment demanded ~252°/s, see §2.5/§4), so it legitimately lagged far enough to trip the real collapse-detection check | Not "fixed" in the sense of removing the check (that would mean trusting a real angle error as if it were noise) — fixed by retiming `gaitTable` so no segment demands more speed than the actuator can deliver (§2.5) |
| 7 | "Two range-of-motion loops" suspected | Wasn't two loops — the single existing curve genuinely has two humps of very different height (small early bump + big late peak), which can look like two alternating things when watched at slow speed | No code bug; led to the deliberate kneeTable smoothing in §2.5 |
| 8 | Speed ramp "didn't feel linear," disliked | Original continuous fixed-acceleration ramp wasn't tied to the actual dataset-correlated preset speeds at all | Replaced with discrete stepping through the 12 existing presets, one step per second, including 0 as a 13th step below preset 1 for a full stop |

---

## 4. Empirically Discovered Hardware Limits

- **The actuator has a real, finite maximum angular speed.** This was not in any spec sheet we had — it was discovered by instrumenting the exact freeze moment (bug #6 above) and computing the literal `Δangle / time` demanded by the curve at the speed that failed. The original evenly-spaced curve demanded as much as **~252°/s** in its worst segment; after retiming, every segment demands a roughly uniform **~85°/s** (at the 1.43s-cycle preset that was tested). **Before increasing demo speed beyond what's already been tested, re-run the same diagnostic sweep** (set a preset, let it run untouched, watch for `actualAngle`/`commandedAngle` divergence) rather than assuming the retiming fixed it for all future speeds — it was only verified up to preset 10 at last check.
- **Cross-core contention was suspected multiple times and was wrong every time it was blamed for a freeze.** Every actual freeze in this session traced back to either a software bug (blocking Serial read) or a genuine actuator speed limit — not Core 0 disrupting Core 1. That doesn't mean cross-core interference is impossible, just that it wasn't the cause of anything observed so far. Don't reach for it as an explanation without instrumented evidence (see the `BAD READING` / freeze-source print pattern used throughout — instrument before guessing).
- **Core 0 (sensor+DL) was measured running well below its 100Hz target** (~16.5-91Hz depending on conditions, with per-tick I2C read ~1.4-1.5ms and feature computation ~50-90µs, separate from the ~47-50ms inference cost every 5th sample). This was never fully root-caused or optimized — see §10.

---

## 5. Runtime-Configurable Parameters (current Serial command interface)

This is the part most directly relevant to mobile app wiring — these are exactly the parameters the BLE config screen needs to set, and the current Serial commands are a working bench stand-in for that exact future feature.

| Command | Effect | Maps to |
|---|---|---|
| `0` | Step down 1 preset/sec to the slowest, then continue at that speed until the current stride completes, then rest at full extension (see §6) | "Stop" button |
| `1`-`12` | Set target speed preset; steps there 1 preset/sec from wherever it currently is | "Speed / gait time" config |
| `E<deg>` | Set max extension (floor). Must be `>= 0` and `< current flexion limit` | "Max ROM" (extension side) |
| `F<deg>` | Set max flexion (ceiling). Must be `> current extension limit` and `<= 90` | "Max ROM" (flexion side) |

The 12 preset speeds (index : %/sec : real-world equivalent), in case the mobile app needs to present a friendlier speed selector than raw index numbers:

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

Note rows 9-10 swap relative to their mph labels — known noise in the original cycle-length estimate, not a new error.

**For the mobile app:** presets 1-5 are bench/bridge values with no real-gait basis and probably shouldn't be exposed as user-facing speed options at all — likely only 6-12 (the actual dataset-correlated speeds) make sense as real "walking speed" choices for a patient-facing UI.

---

## 6. Stop-at-Extension Behavior Contract

This is a real behavioral contract the mobile app's "Stop" button needs to expect, not an instant motor stop:

1. On a stop request, speed steps down one preset per second (same as normal speed changes) until it reaches the slowest preset (index 1).
2. It then **keeps moving** at that speed — does not freeze — until the current gait stride actually completes (detected as `Gait_percent` wrapping from ~100% back to ~0%).
3. Only then does it snap to a true stop and ease (smoothly, via the normal command filter) to rest at exactly `THERAPIST_MIN_ROM_DEG` (max extension).

So a "stop" can take anywhere from just-under-one-stride-length up to (deceleration time + one full stride) depending on what speed it was at when requested. **The mobile app should not assume the brace is immediately still after sending a stop command** — it should either wait for a status signal indicating `stoppedAtExt`, or be designed around the expectation of a graceful finish, not an immediate halt.

State flags involved: `pendingStopAtExtension` (decelerating/finishing stride) and `fullyStoppedAtExtension` (actually at rest). Any future status protocol to the mobile app should expose both states distinctly, not collapse them into one "stopped" boolean — a user watching mid-stop should see "finishing up" differently from "fully stopped."

---

## 7. Dual-Core Architecture & Shared Resources

- **Core 1** = control (Step A logic). **Core 0** = sensor + DL (Step B logic, `coreBTask`).
- **Shared resource: Serial.** Both cores print independently; `serialMutex` (a FreeRTOS mutex) guards every multi-line print block on both sides. Any new print statement added to either core's code must be wrapped the same way, or output will garble.
- **Shared resource: the built-in RGB LED** (`neopixelWrite`) — used by both cores (Core 1 for freeze-blue, Core 0 for init green/red) but **not** mutex-protected. This was a deliberate choice: worst case is a momentary color flicker, not corrupted multi-byte output, so it wasn't judged worth the overhead. Revisit only if LED behavior ever looks genuinely wrong, not preemptively.
- **Not shared at all:** I2C buses (Core 0 only), motor UART (Core 1 only), FSR analog pins (Core 0 only). No GPIO conflicts currently exist between the two cores.
- **Core 0's task stack** is sized at 16384 bytes — chosen generously given TFLite Micro's call depth and floating-point feature math; never verified against actual stack high-water-mark. If memory ever gets tight, profile this specifically before shrinking it blind.

---

## 8. Sensor + DL Pipeline (Step B) Status

- Real I2C reads (3x MPU6050, dual bus) and 2x FSR reads, feeding the **full real 57-feature recipe** (matching the actual training notebook, not a placeholder): causal per-session FSR calibration standing in for the offline `robust_norm()`, backward-difference derivatives matching training's non-dt-scaled `np.gradient`, and a debounced heel-strike tracker producing the four "boosted phase" features.
- TFLite Micro inference reuses the **confirmed-working** 130KB internal-SRAM tensor arena setup (not PSRAM — the model is only ~27KB, internal SRAM is faster and was already proven to work on real hardware before this session).
- **Deliberately not connected to Core 1's motor control.** This is not a placeholder waiting to be wired up casually — it's a considered decision (see §9) that the sensor/DL output stays observed-only until there's been real testing on a population more representative of an actual wearer than the 4 able-bodied bench-test subjects the original model was trained on.
- Performance was measured but not optimized: achieved sample rate has been seen anywhere from ~16.5Hz to ~91Hz depending on conditions, well under the 100Hz target. I2C read time and feature computation time are individually instrumented (`I2C=`, `Feature=` fields in `[STATUS-B]`) but the overall throughput gap was never fully root-caused (see §10).

---

## 9. Mobile App Integration — What Aligns, What's Missing, What to Watch For

**Already aligned (confirmed architecture, logged earlier in the broader planning doc too):**
- BLE sends max ROM + speed/gait time before a session, sends start/stop. WiFi only turns on after a session ends, to upload data — never active during motion (consistent with the principle that network stalls shouldn't be able to introduce latency during real motion).
- The two BLE-configured parameters map **directly** onto `THERAPIST_MIN_ROM_DEG`/`THERAPIST_MAX_ROM_DEG` and the speed preset system (§5) — this firmware's existing Serial interface is a legitimate bench stand-in for exactly that BLE config, not a separate thing that needs reconciling later.
- Backend anomaly/statistics analysis is after-the-fact review, not real-time safety — same boundary as the CNN: it can flag, only on-device hardware-truth can freeze.

**What's missing (not built yet, don't assume it exists):**
- No actual BLE server code exists in this firmware yet. The Serial command interface is a deliberate stand-in, not a partial implementation of BLE.
- No WiFi-based post-session upload exists in this firmware. (Separately: the *data collection* firmware, `knevo_final_status.ino`, already has WiFi + 4 calibration commands (`CAL_STATIC`/`CAL_UNLOADED`/`CAL_STANDING`/`CAL_KNEE`) for a completely different purpose — recording new training data, not session telemetry. Confirmed during this project that those 4 commands were never used in training the current model and don't need to be merged into this firmware.)
- No connection between Core 0 (sensor/DL) and Core 1 (motor) exists — if a future requirement is "the app should show live gait phase from the sensor," that's Core 0's `[STATUS-B]` data, already computed, just never transmitted anywhere beyond Serial.

**What to watch for when wiring BLE config in:**
- Validate ROM bounds the same way the Serial parser does (extension `< current flexion`, flexion `> current extension` and `<= 90`) — these bounds exist because of the proportional-scaling math in §2.4, not arbitrarily.
- Remember presets 1-5 (§5) aren't real gait speeds — don't expose them in a patient-facing speed selector without relabeling/hiding them.
- Remember the stop behavior contract (§6) — a BLE "stop" command should trigger the same graceful finish-then-rest behavior, not an instant cut, and the app's UI should reflect "finishing stride" vs. "fully stopped" as distinct states if it shows real-time status.
- If BLE/WiFi are ever both active closer together in time than the current pre-session/post-session split, double check radio-sharing implications (ESP32 BLE and WiFi share the same radio hardware) — the current design's clean time separation avoids this, don't casually narrow that gap.

---

## 10. Known Open Items / Not Yet Validated

- Core 0's actual throughput bottleneck (well under 100Hz) was instrumented but never fully root-caused — `I2C=`/`Feature=` timings were captured but a definitive explanation for the lower-than-expected Hz numbers wasn't reached.
- The actuator speed ceiling (§4) was only verified up to preset 10 with the retimed curve — presets 11-12 haven't been confirmed clean.
- `motorOn()`'s sliding-timeout pattern (§2.8) is a known-but-unfixed latent issue, low risk only because it's setup-only.
- No worn human testing has been logged in this document's scope — everything described here is bench-validated (motor under no load / hand-resistance / fixture), not validated on an actual wearer.
- The heel-strike tracker that was originally meant to drive `Gait_percent` directly (before the pivot to open-loop control for the demo) is real, working code (after its bugs were fixed) but was deliberately shelved from the motor control path — see the broader integration plan doc for that decision's full reasoning. It still exists inside Core 0's feature pipeline (driving the 4 boosted-phase features) but no longer drives anything on Core 1.
