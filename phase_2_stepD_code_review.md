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

Calibration correctly refuses to run unless `deviceState == DEV_IDLE`.
**`SetConfigCB` and `WiFiConfigCB` don't have the equivalent guard.**

1. `SetConfigCB::onWrite` applies new ROM limits
   (`cmdSetExtension`/`cmdSetFlexion`) with no check on current state. If the
   app sends a `SetConfig` while a set is **already running**, the ROM clamp
   (`clampFloat(commandedAngle, THERAPIST_MIN_ROM_DEG, THERAPIST_MAX_ROM_DEG)`)
   changes on the very next 10ms control tick — a real possibility of a
   sudden angle clamp/jump while the brace is mid-motion on a leg.
2. `WiFiConfigCB::onWrite` will happily start a WiFi connection test mid-set
   if new credentials arrive while `deviceState == DEV_RUNNING` — directly
   violating the contract's central architectural rule (WiFi never active
   during motion). Currently only true because the app is *expected* not to
   do this, not because the device enforces it.

This is the same missing pattern in two places, not two unrelated bugs.
Recommend auditing every BLE write handler for "should this be acted on right
now," not just patching these two individually.

---

## Issue 2: no mutual exclusion between "start a new set" and "previous upload still in flight"

`cmdStart()` only refuses when `deviceState == DEV_FAULT` — it doesn't check
whether a previous set's upload is still running. Two concrete consequences:

1. `doUpload()` is **fully blocking** — `client.connect(..., 5000)` plus the
   write loop plus `while (!client.available()...) delay(2)` up to another
   5s — called synchronously inside `serviceNet()`, which runs inside
   `coreBTask`'s own loop body. While blocking, **Core 0 cannot read sensors,
   compute features, or run inference at all**, for up to ~10 seconds.
2. If a new set starts during that window, `cmdStart()` calls `bufferReset()`
   and sets `setActive = true` — but Core 0 is stuck inside the blocking
   upload call, so the first several seconds of the *new* set's samples are
   never appended. Silent data loss, not a crash — harder to notice.

Since `serviceNet()` already implements a clean non-blocking state machine for
the WiFi *connect* phase, the natural fix is extending that same pattern to
the upload itself (additional states for connecting/sending/awaiting-ACK)
rather than introducing a new concept. A smaller near-term fix: refuse
`cmdStart()` while `netState != NET_IDLE`.

---

## Issue 3: upload/connection failures currently lose data silently

Both failure paths — WiFi STA connect timing out (`WIFI_CONNECT_TIMEOUT_MS`)
and the TCP connect-to-phone failing inside `doUpload()` — drop straight back
to `DEV_IDLE` with no retry and no app-visible failure signal. Buffered
samples aren't preserved for a later retry; they're just gone. A real clinic
WiFi network failing to connect within 15s is plausible, not an edge case —
worth a deliberate decision (retry once? keep the buffer until a successful
upload? surface a distinct fault code?) rather than the current default of
failing quietly.

---

## Smaller notes — worth knowing, not urgent

- `bufferTruncated` (buffer cap of 20,000 samples ≈ 4-20 min depending on
  achieved Hz) is logged to Serial but never surfaced to the app — the
  frame header's `flags` byte is reserved/always-0 and would be a natural
  place to signal this if both sides agree to extend its meaning.
- Starting the WiFi upload as soon as `setActive == false` (rather than
  waiting for `fullyStoppedAtExtension == true`) means the radio can be
  active while the motor still has up to one full stride of graceful
  deceleration left. This followed the gap analysis's own §4.5 recommendation,
  but seeing it concretely implemented, recommend gating *specifically the
  WiFi connect* on full motor settlement (keep the data-freeze immediate,
  just delay turning the radio on) — given the entire reason for "no WiFi
  during motion" was avoiding exactly this kind of overlap.
- `bad_auth` (1) and `app_unreachable` (4) WiFiStatus error codes are never
  produced — only `ssid_not_found`-as-proxy and `timeout` are distinguished.
  May be an ESP32 `WiFi.status()` API limitation rather than an oversight,
  but worth knowing it's not yet differentiated.

---

## Prioritized fix order (for when editing resumes)

1. Add the device-state guard to `SetConfigCB`/`WiFiConfigCB` (small,
   mechanical, directly safety-adjacent).
2. Guard `cmdStart()` against `netState != NET_IDLE` (small, prevents the
   data-loss scenario in Issue 2).
3. Decide and implement *something* for upload-failure visibility (Issue 3;
   size depends on how much retry logic is wanted).
4. The non-blocking-upload refactor and the WiFi-after-full-stop refinement
   are real improvements but lower urgency than 1-2.

**Status: compilation has not yet been verified. Per current direction, get
the file compiling cleanly before acting on any of the above.**
