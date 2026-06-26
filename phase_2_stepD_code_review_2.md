# Phase 2 — Step D Code Review (`knevo_stepD_integration.ino`)

> Review of the first integrated implementation combining Step A (control),
> Step B (sensor+DL), and the new BLE/WiFi mobile-contract layer (Step D).
> Companion to `phase_2_embedded_gap_analysis.md` — this document records what
> that analysis's recommendations actually became in code, what's correct,
> and what still needs fixing. Written before any further edits, per request,
> so the next session has a clear record independent of what gets changed next.

---

## Overall verdict

Substantial, careful work — this directly answers most of what the gap
analysis flagged, not a surface-level pass. Three real issues found, all
fixable without a structural rewrite. Documented here before any further
changes are made.

---

## What's done well

- **Calibration restructuring (gap analysis §5.4) — done correctly.**
  `serviceCalibration()` is a proper non-blocking state machine
  (`calRequest`/`calRunning`), idle-gated
  (`if (deviceState == DEV_IDLE && !calRunning)`), serviced once per Core 0
  tick instead of `delay()`-blocking. The old blocking `calibrateFSR()` is
  removed from the call path (left as dead code with an explanatory comment —
  minor cleanliness nit, harmless).
- **ROM validation tightened to match the contract exactly (gap analysis §4.1).**
  `CONTRACT_EXT_MIN/MAX_DEG` (1/5) and `CONTRACT_FLEX_MIN/MAX_DEG` (30/65),
  defaults now 5/60. Both the absolute range *and* the extension-below-flexion
  ordering are enforced in `cmdSetExtension`/`cmdSetFlexion`. This was the
  most safety-relevant gap flagged — done correctly.
- **Device state machine (IDLE/RUNNING/DONE/FAULT, gap analysis §5.2) is real
  and sensibly wired.** FAULT comes from the one-way `freezeSystem()` latch
  (consistent with that being non-negotiable — see `implementation.md` §7.2).
  DONE fires the instant `cmdStop()` is processed rather than waiting for the
  motor to physically settle — exactly the data-readiness-vs-motor-rest
  decoupling recommended in gap analysis §4.5.
- **One command API, two callers.** Serial and BLE both call the same
  `cmdSetSpeedIndex`/`cmdSetExtension`/`cmdStart`/`cmdStop`/etc. functions —
  Serial bench-testing exercises the same code path the app will use.
- **Duration timer and manual stop reuse the identical stop-at-extension
  path** (gap analysis §5.3) — no second stop implementation.
- **Explicit byte-level little-endian serialization** (`wrU16`/`wrF32`/etc.)
  instead of struct casts — correctly avoids the alignment-padding trap
  flagged in gap analysis §5.1. Verified by hand: `frameLen = 28 + n*88` and
  the 32-byte header buffer (4-byte length prefix + 28-byte contract header)
  are both correct against the documented wire format.
- **`max_speed` → preset index** resolved via `speedIndexFromMaxSpeed()`
  (round + clamp 1-12). Comment tags (`R2`, `R4`, `R5`) suggest these were
  explicitly confirmed rather than silently assumed — the right call, given
  gap analysis §4.2 flagged this as needing confirmation, not assumption.

---

## Issue 1 (most important): BLE write handlers inconsistently check device state

**STATUS: FIXED, verified.** `SetConfigCB::onWrite` now opens with
`if (deviceState != DEV_IDLE) return;`. `WiFiConfigCB` takes a more precise
approach: credential storage (SSID/password/IP/port) always happens (harmless
— just memory), but the thing that actually brings the radio up
(`wifiTestRequested = true`) is gated on `deviceState == DEV_IDLE`, and
`serviceNet()` independently re-checks the same condition before acting on
the flag — double-gated. Residual, minor: genuinely *new* credentials (not
the normal empty "refresh IP/port" write sent before every set) arriving
mid-set would still silently overwrite stored SSID/password without the
radio turning on — narrow edge case, not the original architectural risk.

---

## Issue 2: no mutual exclusion between "start a new set" and "previous upload still in flight"

**STATUS: FIXED, verified — and the fix addresses a more serious risk than
originally described.** `cmdStart()` now refuses unless `deviceState ==
DEV_IDLE` **and** `netState == NET_IDLE`, closing the data-loss path
described below. Separately, `doUpload()`'s inter-chunk `delay()` calls
became `vTaskDelay()` — a multi-second `delay()` inside a FreeRTOS task can
actually trip the task watchdog and reboot the device, not just stall a few
seconds; this fix closes that more severe failure mode too, which the
original review under-described.

Original description, for the record:

1. `doUpload()` is fully blocking — `client.connect(..., 5000)` plus the
   write loop plus the ACK wait — called synchronously inside `serviceNet()`,
   which runs inside `coreBTask`'s own loop body. While blocked, Core 0
   cannot read sensors, compute features, or run inference. This residual
   characteristic still exists (Core 0 pauses sensor/DL work for the upload's
   duration) but is no longer watchdog-unsafe, and since uploads only happen
   between sets, this is now a contained trade-off, not a bug.
2. ~~If a new set starts during that window, the first several seconds of its
   samples were never appended.~~ Closed by the `netState` guard above.

---

## Issue 3: upload/connection failures currently lose data silently

**STATUS: MOSTLY FIXED, one loose end.** `doUpload()` now returns
success/failure and gets one automatic retry over the same live connection.
Failure produces a distinct fault code surfaced via `DeviceStatus`
(`0x12 upload_failed`, `0x13 wifi_join_failed`) instead of failing quietly.

Loose end: the `0x12` comment says "recoverable; buffer kept" — true,
`bufferReset()` is never called on a failure path — but there is currently
**no command that re-triggers an upload of that preserved buffer**;
`requestUpload()` is only ever called from `cmdStop()`. The data survives in
memory but nothing yet acts on it; the next real recovery path is a fresh
set, which wipes it. Recommend either adding a small "retry upload" trigger
or adjusting the comment to match what's currently possible.

---

## Smaller notes — worth knowing, not urgent

- `bufferTruncated` (buffer cap of 20,000 samples ≈ 4-20 min depending on
  achieved Hz) is **still** logged to Serial only, not surfaced to the app —
  unchanged from the original review.
- The WiFi-after-full-stop refinement suggested in the original review was
  implemented: `serviceNet()` now waits for `fullyStoppedAtExtension` (with
  an `UPLOAD_SETTLE_TIMEOUT_MS` safety fallback) before bringing the radio up
  for an upload, instead of starting as soon as `setActive == false`.
- `bad_auth` (1) and `app_unreachable` (4) WiFiStatus error codes are still
  never produced — only `ssid_not_found`-as-proxy and `timeout` are
  distinguished. Likely an ESP32 `WiFi.status()` API limitation rather than
  an oversight.
- **Behavior change worth knowing about, not a bug:** `cmdSetSpeedIndex()` no
  longer moves the motor directly — it only sets `sessionSpeedIndex`; the
  motor stays at rest until `START` (Serial `G<sec>` or BLE `Control`)
  applies it. This correctly mirrors the real contract's two-step
  SetConfig-then-Control(START) flow, and is self-documented in the Serial
  reply text, but changes the bench-testing flow from earlier in this
  project (typing a number alone used to move the motor immediately).

---

## Prioritized fix order (for when editing resumes)

1. ~~Add the device-state guard to `SetConfigCB`/`WiFiConfigCB`~~ — **done.**
2. ~~Guard `cmdStart()` against `netState != NET_IDLE`~~ — **done.**
3. ~~Implement something for upload-failure visibility~~ — **done** (retry +
   distinct fault codes); remaining loose end is the missing "retry upload"
   trigger noted in Issue 3 above.
4. Still open, lower urgency: surface `bufferTruncated` to the app (fault
   code or the header's reserved `flags` byte); add a way to actually retry
   sending a preserved-but-unsent buffer.

**Status: compiles cleanly after the 5 mechanical fixes from the first
compile pass (declaration-order externs, `onWrite` signature, removed
`setScanResponse` call). Logic review of this revision is complete; no new
compile-blocking issues found by static read, though real compilation
remains the authoritative check.**
