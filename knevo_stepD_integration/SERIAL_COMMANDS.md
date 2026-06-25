# Step D — Serial Monitor Commands (Manual Testing)

Bench-test reference for `knevo_stepD_integration.ino`. Serial monitor at **115200
baud**, line ending **Newline** (or Carriage return). Every command below routes
through the same internal command API the mobile app drives over BLE, so a
Serial-driven test exercises the exact same code path as an app-driven one.

Type the command, press **Enter**.

## Commands

| Command | Argument | Action | App / BLE equivalent |
|---------|----------|--------|----------------------|
| `1`–`12` | — | Set target **speed preset** (steps 1 preset/sec toward it). | `SetConfig.max_speed` (preset index) |
| `0` | — | **STOP**: step down to the slowest preset, finish the current stride, rest at full extension. Also ends an active set (freezes the buffer; uploads if WiFi is provisioned). | `Control` STOP (`0x02`) |
| `E<deg>` | `1`–`5` | Set **max extension** (floor). Rejected if outside 1–5 or not below the flexion limit. | `SetConfig.max_extension_angle_deg` |
| `F<deg>` | `30`–`65` | Set **max flexion** (ceiling). Rejected if outside 30–65 or not above the extension limit. | `SetConfig.max_flexion_angle_deg` |
| `CU` | — | **Calibrate UNLOADED** (step 1): capture the unloaded FSR bound. Device must be **IDLE**. Non-blocking, ~3 s. | `Control` CALIBRATE_UNLOADED (`0x10`) |
| `CS` | — | **Calibrate STATIC** (step 2): capture the loaded FSR bound. Device must be **IDLE**. Non-blocking, ~3 s. | `Control` CALIBRATE_STATIC (`0x11`) |
| `G<sec>` | seconds | **START a set** with a bench `set_record_id`. `0` = run until a manual `0`. While running, raw samples buffer in PSRAM. | `SetConfig` + `Control` START (`0x01`) |
| `B` | — | **Dump state**: device state, fault code, buffered sample count, current ext/flex, speed index, WiFi port. | (diagnostic only) |

Notes:
- Examples: `5` (speed preset 5), `E5`, `F60`, `G30` (30 s set), `G0` (open-ended set).
- Presets **1–5** are bench/bridge values (not real gait speeds); **6–12** map to the real dataset speeds. The mobile UI should expose only 6–12.
- `E`/`F` accept decimals (e.g. `E2.5`); they are validated against the mobile contract ranges (D-ext1.6).
- Calibration is **idle-gated**: `CU`/`CS` are ignored while a set is running.

## Device states (shown by `B` and over BLE DeviceStatus)

| State | Meaning |
|-------|---------|
| `0` IDLE | No active set; motor at rest or never started. |
| `1` RUNNING | A set is active; samples are being buffered. |
| `2` DONE | Set ended; batch frozen and ready (upload may be in progress). |
| `3` FAULT | Safety freeze (`freezeSystem`) — **one-way latch, requires a power cycle**. |

Fault codes: `0` none; `0x10` config_rejected (a `SetConfig`/`E`/`F` outside the
allowed range — state stays IDLE, not a safety freeze).

## Example bench sequences

**Calibration (two steps):**
```
CU        # lift the foot off the device, wait ~3 s for ">>> CAL done"
CS        # stand still / load the FSRs, wait ~3 s for ">>> CAL done"
B         # confirm state and bounds
```

**A device-assisted set (Serial-only, no WiFi upload):**
```
E5        # extension
F60       # flexion
6         # pick a real-gait speed preset
G20       # start a 20 s set (auto-stops; or use G0 then 0 to stop manually)
B         # while running: dev=1 (RUNNING), buffer climbing
...        # after 20 s it stops gracefully; dev=2 (DONE) then 0 (IDLE)
B         # buffer shows the captured sample count
```
Without WiFi provisioned (`wifiPort=0` in `B`), the set buffers but skips the
upload and returns to IDLE — expected for a Serial-only bench run. The full
upload path (WiFi + TCP batch) is exercised by the real mobile app, which
provisions WiFi and provides its IP/port over BLE.

## What still needs the app (not testable over Serial)

- WiFi provisioning + the post-set TCP batch upload (needs the phone's IP/port and the app's TCP listener).
- BLE pairing/discovery and the `DeviceStatus`/`WiFiStatus` notifications themselves.
