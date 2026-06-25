# Phase 2 — Embedded Gap Analysis (Mobile/Backend Contract vs. Current Firmware)

> Companion to `phase_2_decisions.md`, `phase_2_implementation.md`,
> `phase_2_decisions_ext1.md`, `phase_2_implementation_ext1.md`, and the embedded
> side's `implementation.md`. This document exists because the mobile/backend
> contract was written and partially built (with `MockBLETransport`) **without
> visibility into the actual current embedded firmware**, and the embedded
> firmware was built **without this contract in front of it**. This is the
> reconciliation pass before any embedded BLE/WiFi code gets written.

---

## 1. What the mobile side has actually built, and what that means

The mobile/backend side is materially further along than "a contract on paper" —
`MockBLETransport` simulates a full `knevo_` device end-to-end (accepts
WiFiConfig/SetConfig/Control, emits WiFiStatus/DeviceStatus, pushes a fake TCP
batch), and the iOS app, backend ingest endpoint, and doctor-portal graphs have
all been built and tested against that mock. **This means the mock is a more
precise, already-validated specification of expected device behavior than the
markdown contracts alone — recommend getting a copy of `MockBLETransport.swift`
itself as a reference when implementing the real firmware side, not just these
docs.**

The embedded side, by contrast, currently has **zero BLE code, zero WiFi code,
and zero session/state-machine concept** — see `implementation.md` for the full
current state. Everything in this document is gap analysis against that real
baseline, not against an idealized "what we'd build from scratch."

---

## 2. Alignment already present (good news — low-risk matches)

- **Endianness.** Contract specifies little-endian for all multi-byte fields;
  ESP32 (Xtensa LX7) is natively little-endian. No byte-swapping needed — but
  see §5.6 for a real packing gotcha to watch for anyway.
- **No WiFi during motion.** The contract's core architectural override (BLE
  control plane during the set, WiFi data plane only after STOP) matches a
  principle the embedded side independently arrived at early in this project:
  network stalls shouldn't be able to introduce latency during real-time motor
  control. Same conclusion, reached separately — good cross-validation.
- **Phase 3/M14 boundary matches Decision #2's shadow-mode status.** The
  contract explicitly defers `heel_contact`, `midfoot_contact`, `gait_phase_id`,
  `gait_phase_label`, `knee_angle_est_deg` to Phase 3 (M14) — i.e., the CNN/DL
  pipeline's output. This lines up exactly with the embedded side's own Decision
  #2 (CNN stays shadow-mode/logged-only, never feeds anything downstream yet).
  Nothing needs to change here; just confirms the embedded roadmap and the
  app/backend roadmap were already pointed at the same eventual integration
  point without coordinating on it directly.
- **Raw sensor channels match what's already being read.** The 88-byte TCP
  record's 18 IMU floats + 2 FSR values are exactly the raw channels Step B's
  I2C/FSR reading code already produces (reused verbatim from
  `knevo_final_status.ino`). **No new sensor-acquisition code is needed** —
  only new buffering + serialization + transport code around data that's
  already being read correctly.
- **D6 (cleartext WiFi password over BLE, accepted for v1)** is a mobile-side
  risk acceptance, not something embedded should "fix" unilaterally by adding
  ad hoc encryption that breaks the documented wire format.

---

## 3. Gaps — not yet implemented in embedded firmware at all

| Area | Contract requirement | Current firmware state |
|---|---|---|
| BLE stack | Full GATT peripheral: 1 service + 5 characteristics (§6 below), advertised as `knevo_*` | No BLE code exists at all |
| WiFi data plane | STA mode, connect using app-provided SSID/password, TCP **client** to app's IP:port, send framed batch, read 1-byte ACK, disconnect | No WiFi code exists at all (the *old* `knevo_final_status.ino` has WiFi STA + TCP code, architecturally different — live streaming, not post-set upload — but is a reasonable starting reference for the STA-connect portion) |
| Session timer | `SetConfig.duration_s` — set auto-stops when time elapses, same as a manual stop | No timer/duration concept exists. Current firmware only stops on a manual Serial `0` command |
| Batch buffering | All samples during a set must be buffered on-device (PSRAM/flash) for upload after STOP | No buffering exists — Step B currently only computes features in-place per tick, never stores raw samples for later transport |
| `set_record_id` passthrough | 16-byte UUID arrives via `SetConfig`, must be echoed in the TCP frame header | No concept of "current set ID" exists anywhere in firmware |
| Device state machine | `DeviceStatus.state`: IDLE / RUNNING / DONE / FAULT | No formal state machine exists. Closest analogues: `systemFrozen` (boolean), `currentSpeedIndex` (int) — neither maps cleanly; see §5.2 |
| Battery reporting | `DeviceStatus.battery_pct` (0-100) | No battery sensing exists anywhere in this project's hardware or code, as far as this session's history shows. **Worth confirming whether the physical hardware even has a battery voltage sense line** — this may be a hardware gap, not just firmware |
| Calibration as discrete BLE-triggered steps | Two independent opcodes (`0x10`/`0x11`), each triggerable any time the device is idle, non-blocking | Current `calibrateFSR()` is a single blocking function (two sequential 3s `delay()`-based phases) called once during `setup()` — fundamentally the wrong shape for this contract (see §5.4) |
| `firmware_version` read | Plain BLE read of a version string | Doesn't exist |
| Config-rejection fault reporting | Device must validate `SetConfig` ranges and reject (D-ext1.6) — recommended via a fault code | No BLE exists yet to report through; Serial-side validation exists but with different ranges (see §5.1) |

---

## 4. Discrepancies that need explicit resolution before coding (not just "missing" — actively different or ambiguous)

### 4.1 ROM validation ranges don't match, and the contract's are stricter
Current firmware (Serial `E<deg>`/`F<deg>`, see `implementation.md` §10):
- Extension: `>= 0` and `< current flexion` — **no upper bound**.
- Flexion: `> current extension` and `<= 90`.

Mobile contract (D-ext1.6), which the device **must** enforce per its own text
("the device MUST also validate these... and reject the session if any value is
out of range"):
- Extension: **1–5**, default 5.
- Flexion: **30–65**, default 60.

Firmware's current flexion ceiling (90°) is meaningfully more permissive than the
contract's (65°) — this needs tightening before any BLE wiring, not after,
since it's a stated safety requirement. Defaults also differ slightly (flexion
65 vs. contract's 60) — low risk, but worth a deliberate decision rather than
leaving the mismatch unexamined.

### 4.2 `max_speed` semantics — strong likely match, but never explicitly confirmed
Contract: `SetConfig.max_speed` is a `float32`, range **1–12**, default 5.
Firmware's existing speed system (`implementation.md` §10) is **also** exactly a
1–12 index into a 12-entry dataset-correlated preset table. The identical range
is almost certainly not a coincidence — but nothing in either document set
explicitly states "the app's speed value *is* the embedded preset index." Before
building the BLE handler, get explicit confirmation of this rather than assuming
it silently:
- If confirmed: the handler is trivial — round the incoming float, use it
  directly as `targetSpeedIndexUser`.
- If *not* confirmed (e.g. the app side means something else, like a literal
  mph or %-per-second value): a translation layer is needed, and the default
  value mismatch (mobile default 5, firmware boots aiming at preset 1) becomes
  actually meaningful rather than incidental.

Related, smaller naming question: the field is called `max_speed`, not `speed`.
Firmware's existing stepping behavior (steps up to and holds at the commanded
preset) already behaves like a target that's ramped up to, which fits "max"
naturally — flagged only so it's a deliberate read, not an assumed one.

### 4.2.1 Recommendation
Resolve §4.2 explicitly (a one-line confirmation from whoever owns the mobile
contract) before writing the `SetConfig` BLE handler — it changes whether that
handler is one line or a small translation function.

### 4.3 STOP grace period is undocumented on the mobile side, but firmware already has opinionated behavior here
Mobile contract: `Control` byte `0x02 STOP` — no detail on timing/grace period.
Firmware: STOP already triggers a deliberate graceful sequence — steps down to
the slowest preset, finishes the current stride, *then* rests at full extension
(`implementation.md` §11). This is arguably better behavior than an instant cut,
but the mobile UI needs to know about it: a "Stop" button shouldn't assume the
brace is immediately still, and ideally the app's UX (spinner, "finishing up..."
state) should reflect the two distinct states firmware already tracks
(`pendingStopAtExtension` vs. `fullyStoppedAtExtension`) rather than collapsing
both into a single "stopped" assumption the instant the byte is sent.

### 4.4 Calibration step duration: app counts 5s, device currently calibrates for 3s — not a conflict, but worth a deliberate choice
The contract is explicit that the app's 5s countdown is independent of any
device timing and there's no device ack to wait for — so the current firmware's
3s-per-step capture window finishing *before* the app's 5s UI countdown ends is
not a race condition, just an asymmetry. Low risk, but worth either matching the
durations for cleanliness or explicitly documenting why they differ.

### 4.5 "DONE (batch ready)" timing — recommend decoupling data-readiness from motor-rest
The contract's `DeviceStatus.state = DONE` is described as "batch ready" with no
detail on exactly when that transition happens relative to the motor's physical
stop. Recommend an explicit design choice now rather than letting it default to
whatever's easiest to wire up: **stop appending new samples to the upload buffer
the instant STOP/duration-elapsed is processed** (data is "done" immediately),
while the motor's graceful deceleration-to-extension continues as a separate,
slower process in parallel. Coupling the two would add several unnecessary
seconds to getting data ready for upload, for no benefit.

---

## 5. Other integration notes worth flagging

### 5.1 Struct packing — a real embedded gotcha, not a style nitpick
When implementing the GATT codec and the TCP frame header/records, do **not**
build them via naive C struct casts/memcpy. The compiler may insert padding
bytes for alignment that don't exist in the documented wire format (e.g. the
TCP frame header mixes `char[4]`, `uint8`, `uint8[16]`, `uint32`, `uint16` —
exactly the kind of layout where default struct alignment silently adds padding
that breaks the byte-for-byte contract). Use explicit byte-level
serialization/deserialization (or `__attribute__((packed))` structs verified
against the exact byte offsets in the contract), not assumed in-memory layout.

### 5.2 Device state machine needs to be purpose-built, not inferred from existing variables
None of `systemFrozen`, `currentSpeedIndex`, `pendingStopAtExtension`, or
`fullyStoppedAtExtension` map cleanly onto IDLE/RUNNING/DONE/FAULT on their own.
Recommend a single new explicit state variable that these existing signals feed
into, rather than scattering `DeviceStatus`-reporting logic across multiple
existing flags inferred ad hoc. Rough mapping to start from:
- **FAULT** = `systemFrozen == true` (the one unambiguous case)
- **IDLE** = no active set, motor at rest (`fullyStoppedAtExtension == true` or
  never started)
- **RUNNING** = an active set, motor moving or `pendingStopAtExtension == true`
- **DONE** = set's data collection has ended and a batch is buffered, awaiting
  WiFi upload (per §4.5, independent of whether the motor has physically settled
  yet)

### 5.3 Reuse the existing stop-at-extension path for the duration timer — don't write a second stop implementation
`SetConfig.duration_s` elapsing should trigger the **exact same** code path as
the manual Serial `0` / future `Control STOP` — set `pendingStopAtExtension =
true; targetSpeedIndexUser = 1;` and let the existing logic finish the stride
and rest at extension. Two independent stop implementations would be a real
risk of behavioral drift between them.

### 5.4 Calibration needs restructuring, not just new opcodes
The current `calibrateFSR()` is blocking (`delay()`-based) and runs once at
boot, sequentially. The contract needs two independently-triggerable,
non-blocking steps available any time the device is idle. This is a real
restructuring of existing code, not just adding two new `case` branches — a
blocking 3-second loop on whichever core handles BLE would stall all other BLE
characteristic handling and `DeviceStatus` notifications for that whole window.
Recommend converting calibration into the same kind of non-blocking,
state-machine-driven pattern already used for the speed-stepping system
(`updateSpeedStepping()` in `implementation.md` §10) rather than a blocking
function call.

### 5.5 Mid-set calibration guard
Contract: calibration opcodes are "valid only while the device is IDLE...
recommend: ignore" if received mid-set. Once the device state machine (§5.2)
exists, this is a one-line guard — flagging now so it isn't forgotten once
calibration gets wired to real opcodes.

### 5.6 The embedded side currently has no automated test harness equivalent to `MockBLETransport`/Pepper
The mobile side can develop and verify against a fully simulated device with no
real hardware or radio involved. The embedded side has no equivalent — testing
so far in this project has been live Serial commands against real hardware,
every time. Given real BLE/WiFi-to-phone testing requires the actual ESP32 board
present either way, this gap is lower-priority than the functional ones above,
but worth naming: a small PC-side script (Python, acting as a BLE central +
TCP server matching the contract) would let embedded BLE/WiFi logic be exercised
without needing the real iOS app every iteration — mirroring what
`MockBLETransport` already does for the iOS side, just from the embedded
team's perspective instead.

### 5.7 GATT UUIDs and byte layouts are already fixed — adopt them as firmware constants verbatim, now
These are already agreed and the iOS side is already built against them — there
is no reason to defer copying them into firmware as literal constants, even
before the rest of the BLE stack exists:

```
Knevo Service:              94A4B5CC-14F9-40BF-9631-62954DC8D647
WiFiConfig   (Write):       A4F7B9FF-AC9E-4171-A9FF-461F1180F124
WiFiStatus   (Read/Notify): 23F1E73A-68CD-4D6A-85DC-B33BA4ED6153
SetConfig    (Write):       4FCD372C-910F-46B1-94AB-F3C06597EB60
Control      (Write):       93413C6D-977F-490B-AEC3-AF800C18FFA9
DeviceStatus (Read/Notify): 0439E3D3-AA83-4A13-8AE4-CE19AB85B832

Control opcodes: 0x01 START, 0x02 STOP, 0x10 CALIBRATE_UNLOADED, 0x11 CALIBRATE_STATIC
```

---

## 6. Recommended embedded-side task breakdown

Mirroring the mobile side's `T4-T9`/`E1-E6` style, in dependency order:

1. **Prep:** tighten ROM validation ranges to match D-ext1.6 (§4.1) — independent of BLE, can land first.
2. **Prep:** restructure `calibrateFSR()` into two independent, non-blocking, idle-gated steps (§5.4).
3. **Device state machine** (§5.2) — needed before `DeviceStatus` can report anything meaningful.
4. **BLE peripheral skeleton:** advertise `knevo_*` + the service UUID; implement `DeviceStatus` (Read/Notify) and `Control` (Write, opcodes 0x01/0x02/0x10/0x11) first, since they touch the fewest new concepts.
5. **`SetConfig` handler:** parse + validate (§4.1), resolve `max_speed` semantics (§4.2) before writing this, store `set_record_id`.
6. **Duration timer:** reuse the existing stop path (§5.3).
7. **Sample buffering:** start storing raw 88-byte-format-compatible records during a RUNNING set.
8. **WiFiConfig handler + WiFi STA connect + WiFiStatus reporting.**
9. **TCP client + frame construction + ACK handling** — the actual post-set upload.
10. **End-to-end bench test** against either a real iOS test build or a PC-side mock (§5.6) before attempting a real phone.

---

## 7. Open questions to resolve before writing code (not to assume silently)

1. **Does `max_speed` mean the embedded preset index directly, or something else?** (§4.2)
2. **Does the physical hardware have a battery-voltage sense line at all?** If not, `battery_pct` needs either a hardware change or an explicit placeholder/0% decision communicated back to the mobile side.
3. **Should flexion default be reconciled to 60 (contract) or kept at 65 (current firmware), and should the embedded validation logic just adopt the contract's ranges outright?** (§4.1)
4. **Is there an actual `MockBLETransport.swift` (or equivalent test fixture) available to the embedded team as a reference, not just these markdown docs?** (§1, §5.6)
5. **Confirm the DONE/data-ready vs. motor-rest decoupling (§4.5)** is the desired behavior, not just this document's recommendation.
