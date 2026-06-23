# Knevo ESP32-S3 DL Deployment Handoff Notes

**Project:** Knevo / Smart Active Bionic Knee Brace  
**Target board:** ESP32-S3-N16R8  
**Main goal:** Deploy a gait-phase deep learning model on ESP32-S3 for a 100 Hz rehabilitation knee-brace project.  
**Current status:** Model can run on ESP32-S3, but current fast model is still too slow for true 100 Hz inference and classification quality became worse than the older model.

---

## 1. High-level objective

The project needs a lightweight gait-phase classifier for an ESP32-S3-based knee brace. The sensor/control loop target is **100 Hz**, meaning a new sample arrives every:

```text
100 Hz = 10 ms per sample
```

Therefore, if the neural network is expected to run every sample, the full inference pipeline must be below:

```text
10 ms = 10000 microseconds
```

In practice, the safer architecture is:

```text
100 Hz:
- read sensors
- compute safety limits
- update knee-angle / FSM / motor control

20–50 Hz:
- run CNN gait phase inference
- hold last predicted phase between inference calls
```

---

## 2. Model files and data files used

The main files used across the trials included:

```text
knevo_esp32_boosted_phase_cnn_6phase_model.h
knevo_esp32_boosted_phase_cnn_scaler.h
test_stride.h
cnn_99.ipynb / cnn_99 variants
knevo_cnn_100hz.ipynb
knevo_cnn_100hz_ARDUINO_IDE_TFLM_CONVERTER.ipynb
```

Important generated output examples from Colab:

```text
accuracy.png
boosted_phase_lite_cnn_best_float.keras
boosted_phase_lite_cnn_int8.tflite
candidate_results.csv
classification_report.txt
confusion_matrix.png
dataset_summary.csv
ESP32_BOOSTED_PHASE_CNN_100Hz_README.txt
knevo_esp32_boosted_phase_cnn_100hz_6phase_int8.tflite
knevo_esp32_boosted_phase_cnn_100hz_6phase_model.h
knevo_esp32_boosted_phase_cnn_100hz_scaler.h
knevo_esp32_tcn_100hz_scaler.npz
label_map.json
metadata.json
sample_feature_cols.json
```

Later, the Arduino IDE converter workflow produced / was expected to produce:

```text
knevo_esp32s3_100hz_6phase_int8_model.h
knevo_ESP32_S3_BOOSTED_PHASE_CNN_100hz_6phase_package.zip
```

---

## 3. Old model: key properties and behavior

The older working model used:

```text
Window size: 71
Feature count: 57
Input type: INT8
Output type: INT8
Output classes: 6 gait phases
```

From the scaler header:

```cpp
const int KNEVO_TCN_WINDOW_SIZE = 71;
const int KNEVO_TCN_FEATURE_COUNT = 57;
const float KNEVO_TCN_INPUT_SCALE = 0.06274510175f;
const int KNEVO_TCN_INPUT_ZERO_POINT = -1;
const float KNEVO_TCN_OUTPUT_SCALE = 0.00390625f;
const int KNEVO_TCN_OUTPUT_ZERO_POINT = -128;
```

The old Arduino-style inference code manually registered TFLite Micro ops and used the model header as a plain C array:

```cpp
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "knevo_esp32_boosted_phase_cnn_6phase_model.h"
#include "knevo_esp32_boosted_phase_cnn_scaler.h"
#include "test_stride.h"
```

Old model inference time on Arduino IDE was reported around:

```text
~50 ms
```

This is too slow for 100 Hz true inference.

---

## 4. First Arduino code optimization attempt

The first optimization focused on Arduino-side preprocessing, not model architecture.

### Original problem

The original loop re-standardized and quantized the entire window every inference:

```cpp
float standardized = (raw_val - KNEVO_TCN_MEAN[f]) / KNEVO_TCN_STD[f];
int32_t q_val = (int32_t)round(standardized / KNEVO_TCN_INPUT_SCALE)
                + KNEVO_TCN_INPUT_ZERO_POINT;
input->data.int8[tensor_idx++] = (int8_t)constrain(q_val, -128, 127);
```

This was repeated for:

```text
WINDOW_SIZE × FEATURE_COUNT
```

every inference.

### Optimization applied

A quantized circular buffer was proposed:

```cpp
int8_t* quantized_window_buffer = nullptr;

float q_mult[KNEVO_TCN_FEATURE_COUNT];
float q_bias[KNEVO_TCN_FEATURE_COUNT];

q_mult[f] = 1.0f / (KNEVO_TCN_STD[f] * KNEVO_TCN_INPUT_SCALE);
q_bias[f] = KNEVO_TCN_INPUT_ZERO_POINT - (KNEVO_TCN_MEAN[f] * q_mult[f]);
```

Then each new sample is quantized once:

```cpp
inline int8_t fastQuantizeFeature(float raw_val, int feature_index) {
    int32_t q = (int32_t)lrintf(raw_val * q_mult[feature_index] + q_bias[feature_index]);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}
```

The circular buffer is then copied to the input tensor in chronological order using `memcpy`.

### Insight

This reduced preprocessing cost dramatically. On later ESP output, preprocessing was about:

```text
Prep: 7–12 us
```

So preprocessing was no longer the bottleneck.

The bottleneck remained the model `Invoke()`.

---

## 5. PlatformIO migration attempt

The user switched from Arduino IDE to PlatformIO hoping that `-O3`, better board configuration, and ESP-NN flags would reduce inference time.

Initial `platformio.ini` included:

```ini
[env:esp32s3]
platform = espressif32
board = esp32-s3-devkitc-1
framework = arduino

board_build.f_cpu = 240000000L
board_build.arduino.memory_type = opi_opi
board_build.flash_size = 16MB
board_build.partitions = default_16MB.csv

build_unflags = -std=gnu++11
build_flags =
    -D BOARD_HAS_PSRAM
    -D ESP_NN
    -O3
    -std=gnu++17

lib_compat_mode = off
lib_extra_dirs =
    .pio/libdeps/esp32s3/TFLiteMicro_ArduinoESP32S3/src
    .pio/libdeps/esp32s3/TFLiteMicro_ArduinoESP32S3/src/third_party

lib_deps =
    https://github.com/j-siderius/TFLiteMicro_ArduinoESP32S3.git
```

### Insight

`lib_extra_dirs` was the wrong tool for fixing compiler include paths. It is for extra libraries and is deprecated/undesirable in this workflow.

Also, `.pio/libdeps/...` does not exist until PlatformIO successfully downloads/builds the library, so pointing include paths there can fail early.

---

## 6. PlatformIO issue 1: GitHub connection failed

PlatformIO first failed to install the library from GitHub:

```text
fatal: unable to access 'https://github.com/j-siderius/TFLiteMicro_ArduinoESP32S3.git/':
Failed to connect to github.com port 443 after 75002 ms
```

### Diagnosis

This was a network/install problem, not a code/model problem.

### Fix attempted

Manual library install:

```text
project/
├── lib/
│   └── TFLiteMicro_ArduinoESP32S3-main/
│       └── src/
├── src/
│   └── main.cpp
└── platformio.ini
```

Then remove `lib_deps` and use local include paths.

---

## 7. PlatformIO issue 2: FlatBuffers include error

After manual install, PlatformIO saw the local library, but failed with:

```text
fatal error: flatbuffers/array.h: No such file or directory
#include "flatbuffers/array.h"
```

The file `array.h` was present inside:

```text
lib/TFLiteMicro_ArduinoESP32S3-main/src/third_party/flatbuffers/array.h
```

### Fix attempted

Add include paths:

```ini
-I$PROJECT_DIR/lib/TFLiteMicro_ArduinoESP32S3-main/src
-I$PROJECT_DIR/lib/TFLiteMicro_ArduinoESP32S3-main/src/third_party
-I$PROJECT_DIR/lib/TFLiteMicro_ArduinoESP32S3-main/src/third_party/flatbuffers
```

A symlink fix was also suggested:

```bash
cd lib/TFLiteMicro_ArduinoESP32S3-main/src/third_party/flatbuffers
ln -s . flatbuffers
```

### Result

The FlatBuffers missing include error was fixed.

---

## 8. PlatformIO issue 3: KissFFT include error

Next error:

```text
fatal error: kiss_fft.h: No such file or directory
#include "kiss_fft.h"
```

### Fix attempted

Add include paths:

```ini
-I$PROJECT_DIR/lib/TFLiteMicro_ArduinoESP32S3-main/src/third_party/kissfft
-I$PROJECT_DIR/lib/TFLiteMicro_ArduinoESP32S3-main/src/third_party/kissfft/tools
```

### Result

The missing `kiss_fft.h` issue was fixed.

---

## 9. PlatformIO issue 4: memcpy / memset / strlen errors

Next errors came from library files:

```text
error: 'memcpy' was not declared in this scope
error: 'memset' was not declared in this scope
error: 'strlen' was not declared in this scope
```

Logs pointed to files such as:

```text
third_party/flatbuffers/flexbuffers.h
third_party/flatbuffers/vector_downward.h
third_party/flatbuffers/flatbuffer_builder.h
signal/micro/kernels/rfft.cpp
```

A build flag was proposed:

```ini
-include string.h
```

or patching specific source files to include:

```cpp
#include <string.h>
```

### Important insight

At this point the user correctly questioned whether a released library would really be missing these includes. That was the turning point.

---

## 10. Library README insight

After checking the intended workflow of `TFLiteMicro_ArduinoESP32S3`, the conclusion was:

### We were not fully on the correct track with PlatformIO

The library is designed primarily for Arduino IDE and its own converter workflow, not as a normal raw source library compiled entirely by PlatformIO.

The library expects:

```cpp
#include "TFLiteMicro_ArduinoESP32S3.h"
#include "generated_model.h"
```

and setup using something like:

```cpp
TFLMinterpreter = TFLMsetupModel<TFLMnumberOperators, ARENA_SIZE>(
    TFLM_model,
    TFLMgetResolver,
    true
);
```

The important part is that the model header should be generated using the library’s converter:

```text
tools/tflm_converter.py
```

not a generic bytes-to-C-array converter.

### Why this matters

The converter-generated header includes:

```text
1. Model byte array
2. TFLMgetResolver()
3. TFLMnumberOperators
4. Correct model variable name
5. Operator list extracted from the .tflite model
```

The old plain C-array header only included:

```cpp
const unsigned char g_model[] = { ... };
```

and did not include the resolver/setup pieces.

### Important conclusion

The PlatformIO errors were mostly caused by forcing PlatformIO to compile unnecessary library source folders, including audio/signal kernels that the gait model does not need.

For this project, the recommended path became:

```text
Option A:
Arduino IDE + official release ZIP + tflm_converter.py-generated header
```

---

## 11. Option A: Arduino IDE + tflm_converter.py workflow

### Concept

Instead of:

```text
.tflite → plain C array .h
```

use:

```text
.tflite → TFLiteMicro_ArduinoESP32S3 tools/tflm_converter.py → Arduino-ready .h
```

### Required final files

Arduino sketch folder should contain:

```text
Knevo_TFLM_ArduinoIDE_100Hz/
├── Knevo_TFLM_ArduinoIDE_100Hz.ino
├── knevo_esp32s3_100hz_6phase_int8_model.h
├── knevo_esp32_boosted_phase_cnn_100hz_scaler.h
└── test_stride.h
```

### Important check

The generated model header should contain:

```text
TFLMgetResolver
TFLMnumberOperators
TFLM_knevo_esp32s3_100hz_6phase_int8_model
```

If it does not, the header is not the correct converter-generated file.

---

## 12. New Colab notebook generated

A new Colab notebook was generated:

```text
knevo_cnn_100hz_ARDUINO_IDE_TFLM_CONVERTER.ipynb
```

It was intended to:

```text
1. Keep the original 100 Hz training/export pipeline.
2. Find the exported .tflite.
3. Find the scaler .h.
4. Clone TFLiteMicro_ArduinoESP32S3.
5. Run tools/tflm_converter.py.
6. Generate a converter-style Arduino model header.
7. Generate an Arduino IDE .ino package.
```

### Actual issue

The user later showed screenshots of the package and said:

```text
No .ino was generated from Colab.
```

This means the notebook/package generation was incomplete or the `.ino` was not included in the zip despite being expected.

A separate `.ino` was then provided manually.

---

## 13. Arduino IDE `.ino` created manually

The manually provided Arduino `.ino` used:

```cpp
#include "TFLiteMicro_ArduinoESP32S3.h"
#include "knevo_esp32s3_100hz_6phase_int8_model.h"
#include "knevo_esp32_boosted_phase_cnn_100hz_scaler.h"
#include "test_stride.h"
```

It used:

```cpp
TFLMinterpreter = TFLMsetupModel<TFLMnumberOperators, TENSOR_ARENA_SIZE>(
    TFLM_knevo_esp32s3_100hz_6phase_int8_model,
    TFLMgetResolver,
    true
);
```

It also kept the optimized quantized circular buffer approach.

---

## 14. New 100 Hz model results on ESP32-S3

After deploying the new converter-generated workflow, ESP32-S3 output showed:

```text
DEBUG:Prediction took 24208 microseconds.

Phase: 0 | Prob: 0.9961 | Prep: 11 us | Invoke: 24496 us | Total: 24510 us

DEBUG:Prediction took 24195 microseconds.

Phase: 0 | Prob: 0.9961 | Prep: 11 us | Invoke: 24501 us | Total: 24514 us

DEBUG:Prediction took 24220 microseconds.

Phase: 1 | Prob: 0.9414 | Prep: 10 us | Invoke: 24524 us | Total: 24537 us
```

Typical values:

```text
Prep: ~7–12 us
Invoke: ~24,500 us
Total: ~24,510 us
```

### Insight

The preprocessing optimization worked very well:

```text
Prep ≈ 10 us
```

But model inference is still:

```text
Invoke ≈ 24.5 ms
```

This is better than the old 50 ms result, but still too slow for true 100 Hz per-sample inference:

```text
Target for 100 Hz: < 10 ms
Current: ~24.5 ms
```

So the current new model can run at approximately:

```text
1000 / 24.5 ≈ 40 Hz max inference rate
```

but it should be run at 20 Hz or maybe 25 Hz in a real control loop.

---

## 15. Classification got worse

The user reported:

```text
old vs new model and codes. the classification went bad
```

The new model outputs were mostly phases 0 and 1 with very high confidence:

```text
Phase: 0 | Prob: 0.9961
Phase: 1 | Prob: 0.9961
```

### Possible reasons

#### A) New model is weaker

The new 100 Hz model likely uses a shorter window. It was discussed that:

```text
Old window: 71 samples
New window: around 36 samples if WINDOW_SECONDS = 0.35 at 100 Hz
```

A shorter window reduces gait context and may hurt phase classification.

#### B) `test_stride.h` mismatch

There is a high risk that the Arduino test file was from the older pipeline.

All of these must match exactly:

```text
model header
scaler header
feature order
window size
test_stride.h
```

If the model and scaler are new but `test_stride.h` is old, the model may output wrong predictions with high confidence.

#### C) Feature order mismatch

If `sample_feature_cols.json` or feature order changed between old and new notebook, Arduino playback must use the new feature order.

#### D) Label mapping mismatch

If `label_map.json` changed, phase IDs may not correspond to the same gait phases as before.

---

## 16. Recommended debug test for classification

A new `test_stride.h` should be generated from the **current 100 Hz notebook pipeline**, not reused from the old model.

Suggested Colab cell:

```python
# ============================================================
# Generate test_stride.h from the CURRENT 100Hz model pipeline
# Must match the current model + scaler + feature order
# ============================================================

from pathlib import Path
import numpy as np

TEST_STRIDE_ROWS_TO_EXPORT = min(120, X_test_raw.shape[1])
TEST_WINDOW_INDEX = 0   # Change this to test another sample/window

test_window = X_test_raw[TEST_WINDOW_INDEX]
expected_label = int(y_test[TEST_WINDOW_INDEX])

print("Selected test window:", TEST_WINDOW_INDEX)
print("Expected label:", expected_label)
print("Window shape:", test_window.shape)
print("Model window size:", WINDOW_SIZE)
print("Feature count:", len(sample_cols_final))

assert test_window.shape[0] == WINDOW_SIZE
assert test_window.shape[1] == len(sample_cols_final)

test_stride_h_path = OUTPUT_DIR / "test_stride.h"

with open(test_stride_h_path, "w") as f:
    f.write("#pragma once\n")
    f.write("// Generated from CURRENT 100Hz notebook pipeline\n")
    f.write("// This must be used with the matching model + scaler header\n\n")

    f.write(f"const int TEST_STRIDE_ROWS = {test_window.shape[0]};\n")
    f.write(f"const int TEST_STRIDE_COLS = {test_window.shape[1]};\n")
    f.write(f"const int TEST_STRIDE_EXPECTED_LABEL = {expected_label};\n\n")

    f.write(f"const float TEST_STRIDE_DATA[{test_window.shape[0]}][{test_window.shape[1]}] = {{\n")

    for r in range(test_window.shape[0]):
        row = test_window[r]
        row_str = ", ".join([f"{float(v):.6f}f" for v in row])
        comma = "," if r < test_window.shape[0] - 1 else ""
        f.write(f"  {{ {row_str} }}{comma}\n")

    f.write("};\n")

print("Generated:", test_stride_h_path)

try:
    from google.colab import files
    files.download(str(test_stride_h_path))
except Exception:
    pass
```

Then Arduino should compare prediction to:

```cpp
TEST_STRIDE_EXPECTED_LABEL
```

If the Arduino prediction does not match the expected label from the same notebook, then the deployment/preprocessing is wrong.

If it does match but classification is still bad on other samples, then the model itself is weak.

---

## 17. Old vs new results table

| Item | Old model/code | New 100 Hz converter workflow |
|---|---:|---:|
| Deployment route | Raw TFLite Micro style / manual resolver | Arduino IDE + TFLiteMicro_ArduinoESP32S3 converter |
| Model header | Plain C-array `.h` | Converter-generated `.h` with resolver |
| Window | 71 samples | likely shorter, around 36 samples |
| Features | 57 | 57 if unchanged |
| Inference time | ~50 ms | ~24.5 ms |
| Preprocessing time | included in old timing / heavier | ~7–12 us |
| True 100 Hz inference capable? | No | Still no |
| Classification | Better | Worse from observed test |
| Current recommendation | Accuracy baseline | Speed experiment, not final |

---

## 18. Important architecture recommendation

Do not force CNN inference every 10 ms unless `Invoke < 10000 us`.

Recommended final firmware architecture:

```text
Every 10 ms / 100 Hz:
- read 3 IMUs + 2 FSRs
- compute features
- update circular buffer
- safety checks
- ROM limiter
- motor control / FSM fallback

Every 40–50 ms / 20–25 Hz:
- run CNN inference
- update last predicted gait phase

Between CNN calls:
- hold last predicted phase
- optionally smooth with FSM / transition constraints
```

Suggested Arduino setting:

```cpp
const int INFERENCE_EVERY_N_SAMPLES = 5; // CNN at 20 Hz, sample loop at 100 Hz
```

For debugging only:

```cpp
const int INFERENCE_EVERY_N_SAMPLES = 1;
```

but current model is too slow for this in real-time.

---

## 19. Board configuration insights

The board is ESP32-S3-N16R8. PlatformIO logs sometimes showed:

```text
ESP32-S3-DevKitC-1-N8
8 MB QD, No PSRAM
```

while hardware was expected to be:

```text
16 MB Flash
8 MB PSRAM
```

For many ESP32-S3-WROOM-1-N16R8 boards, likely config:

```ini
board_build.arduino.memory_type = qio_opi
board_build.flash_mode = qio
board_build.psram_type = opi
board_upload.flash_size = 16MB
board_upload.maximum_size = 16777216
board_build.partitions = default_16MB.csv
```

For WROOM-2 variants, `opi_opi` may be correct, but this must be verified from the actual module.

For Arduino IDE, board settings should be:

```text
CPU Frequency: 240 MHz
Flash Size: 16 MB
PSRAM: OPI PSRAM if available
USB CDC on Boot: Enabled if needed for Serial
Serial Monitor: 115200
```

---

## 20. What not to do next

Do not continue patching random PlatformIO library headers unless PlatformIO is absolutely required.

This path led to repeated errors:

```text
flatbuffers/array.h missing
kiss_fft.h missing
memcpy/memset/strlen missing
```

The better conclusion was:

```text
Use Arduino IDE with the library as intended.
Use the library converter-generated header.
```

---

## 21. Recommended next steps for the next LLM / engineer

### Step 1 — Verify new header

Open:

```text
knevo_esp32s3_100hz_6phase_int8_model.h
```

Confirm it contains:

```text
TFLMgetResolver
TFLMnumberOperators
TFLM_knevo_esp32s3_100hz_6phase_int8_model
```

If the model variable name is different, update the `.ino`.

### Step 2 — Regenerate matching `test_stride.h`

Do not reuse old `test_stride.h`.

Generate it from the current model pipeline and include:

```cpp
const int TEST_STRIDE_EXPECTED_LABEL = ...
```

### Step 3 — Add expected-label print in Arduino

Arduino should print:

```cpp
Serial.print("Expected: ");
Serial.print(TEST_STRIDE_EXPECTED_LABEL);
Serial.print(" | Predicted: ");
Serial.println(predicted_phase);
```

### Step 4 — Compare Colab vs Arduino on the same sample

For the exact same test window:

```text
Colab quantized model prediction
Arduino TFLM prediction
```

They must match. If not:

```text
- feature order mismatch
- scaler mismatch
- window order mismatch
- quantization mismatch
- model/header mismatch
```

### Step 5 — If matching but bad, retrain better small model

Target:

```text
Accuracy close to old model
Invoke < 10 ms if possible
Or accept 20–25 Hz inference if accuracy is good
```

Potential training changes:

```text
- Increase window from 0.35 s to 0.45–0.55 s
- Keep model small but not too weak
- Use depthwise separable convolutions
- Avoid dynamic graph ops: Shape, Pack, StridedSlice if possible
- Prefer ops that ESP-NN accelerates: Conv2D, DepthwiseConv2D, FullyConnected
- Reduce dense layers
- Use INT8 full quantization
- Export with representative dataset
```

### Step 6 — Consider non-DL alternative

If hard real-time 100 Hz classification is required:

```text
- Try TinyML classical ML
- XGBoost/decision tree converted to C
- Smaller feature-based model
- FSM + threshold model for control, CNN only for monitoring
```

---

## 22. Current honest conclusion

Current work achieved:

```text
- Arduino-side preprocessing optimized successfully
- TFLiteMicro_ArduinoESP32S3 converter workflow identified as correct
- New model inference reduced from ~50 ms to ~24.5 ms
```

But current work did not yet achieve:

```text
- true 100 Hz CNN inference
- acceptable classification quality from the new small model
```

The next effort should not be more build-system debugging. It should be:

```text
1. Verify deployment correctness with matching test_stride.h.
2. If deployment is correct, retrain a better small model.
3. Decide whether CNN inference at 20–25 Hz is acceptable for the final project.
```

---

## 23. Most important takeaway

The bottleneck is no longer preprocessing.

```text
Prep ≈ 10 us
Invoke ≈ 24500 us
```

So the next optimization must focus on:

```text
- model architecture
- window size tradeoff
- operator selection
- quantization
- inference frequency strategy
```

not on circular buffer code or Arduino-side preprocessing.
