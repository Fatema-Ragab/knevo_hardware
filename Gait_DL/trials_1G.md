# Technical Report & Handoff: Knevo ESP32-S3-N16R8 Gait Phase Classification System

## 1. Project Overview & Context
* **Objective:** Implement a real-time, 6-phase human gait classification system on an edge microcontroller.
* **Target Hardware:** ESP32-S3-N16R8 (Xtensa LX7 dual-core, 16MB Flash, 8MB Octal PSRAM).
* **Input Data:** 57 feature channels derived from lower-limb IMUs (foot, shank, thigh) and Force-Sensing Resistors (FSRs).
* **Classification Targets (6 Phases):** * 0: Heel Strike
    * 1: Loading Response
    * 2: Mid-Terminal Stance
    * 3: Pre-Swing Toe-off
    * 4: Initial-Mid Swing
    * 5: Terminal Swing

---

## 2. Core Model Architecture
* **Type:** Quantized INT8 Lite Temporal Convolutional Network (TCN).
* **Sliding Window Dimensions:** 71 timesteps $\times$ 57 features (`float32` compressed to `int8_t` via quantization factors).
* **Layers Configuration:**
    1.  **Input Shape:** `(None, 71, 57)`.
    2.  **Reshape Layer:** Converts sequence input into image-like shapes `(71, 57, 1)`.
    3.  **Feature Mixer (`Conv2D`):** Kernel size `(1, 57)`, valid padding, 16 filters. Contracts spatial feature correlation per timestep down to 16 channels.
    4.  **Temporal Blocks:** 2 consecutive TCN blocks (Conv2D layer, kernel 5x1 and 3x1, batch normalization, and residual additions).
    5.  **Global Pooling:** Fixed `AveragePooling2D` over the temporal block size `(71, 1)`.
    6.  **Dense Network:** 48-unit fully connected layer, Dropout (0.10), and a 6-unit Softmax activation.
* **Quantized Size:** 26.61 KB.

---

## 3. The Realism and Data Leak Analysis
A foundational caveat was established regarding the initial **99.40% INT8 accuracy**:
* **Tautology Artifact:** The dataset used pseudo-labels built programmatically from heel-strike anchors (FSR). In training, the model used `USE_PHASE_TRACKER_FEATURES = True`, injecting `gait_progress_est` (and its sine/cosine mappings) into the input space.
* **The Leak:** The model was given the exact progress variables used to define the boundaries of the target classes, turning the neural network into a rule estimator rather than a biological sensor processor.
* **Mitigation Flag:** Future iterations require retraining with `USE_PHASE_TRACKER_FEATURES = False` to assess unassisted IMU-to-gait classification accuracy, which is expected to normalize between 88% and 93%.

---

## 4. Hardware Deployment Iterations & Trials

The implementation underwent several compilation and hardware runtime iterations to achieve operational stability on the ESP32-S3:

### Trial 1: The Initial Architecture Verification
* **Action:** Deployed via standard C++ structures on a generic Arduino library.
* **Result:** Compilation error: `class tflite::MicroMutableOpResolver` has no member named `AddFlatten`.
* **Resolution:** Removed `AddFlatten()` from the setup graph. `Flatten` is a zero-cost dimension reshaping operation handled automatically by the TFLite Micro kernel without requiring manual registration memory slots. The resolver size was reduced from `<8>` to `<7>`.

### Trial 2: Moving to Accelerated Kernels
* **Action:** Shifted to the **`Chirale_TensorFlowLite`** library to leverage native ESP-NN micro-operator optimizations.
* **Result:** Repeated registration runtime failures for custom optimization nodes:
    1.  `Didn't find op for builtin opcode 'SHAPE' version '1'`
    2.  `Didn't find op for builtin opcode 'STRIDED_SLICE' version '1'`
* **Resolution:** Sequentially scale-registered the custom math operations. Upgraded the `MicroMutableOpResolver` count to `<9>` and explicitly mapped `AddShape()` and `AddStridedSlice()`.

### Trial 3: Memory Segment Overflow (`dram0_0_seg` crash)
* **Action:** Attempted compilation with full operation mappings.
* **Result:** Linker panic: `section .dram0.bss' will not fit in region 'dram0_0_seg' overflowed by 111024 bytes`.
* **Root Cause:** Declaring a large timeline window matrix (`float window_buffer[71][57]`) and allocating a 350KB static Tensor Arena forced variables into the ESP32-S3's internal high-speed Data RAM (DRAM), instantly exhausting its space.
* **Resolution:** Swapped static allocations for dynamic pointers (`float** window_buffer` and `uint8_t* tensor_arena`) initialized at runtime inside the **8MB external OPI PSRAM** space using `heap_caps_malloc` with the `MALLOC_CAP_SPIRAM` flag.

### Trial 4: Missing Library Op Node (`PACK`)
* **Action:** Executed the model within the updated memory layout.
* **Result:** Runtime panic: `Didn't find op for builtin opcode 'PACK'`.
* **Resolution:** Modified the explicit mutable resolver to an operator count of `<10>` and added the missing `res.AddPack()` mapping to complete the deep graph connection.

---

## 5. Final Verified C++ Firmware Code
The final firmware operates a **Hardware-in-the-Loop simulation**. It completely clears static internal RAM bounds, configures the 8MB Octal PSRAM safely, loads the 10 core operational nodes, and streams real-time step predictions.

```cpp
#include <Arduino.h>

// Specialized hardware-accelerated ESP32-S3 TFLM libraries
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

// Local sketch configuration files
#include "knevo_esp32_boosted_phase_cnn_6phase_model.h"
#include "knevo_esp32_boosted_phase_cnn_scaler.h"
#include "test_stride.h" // Dataset playback sequence file

namespace {
    const size_t kTensorArenaSize = 350 * 1024; 
    uint8_t* tensor_arena = nullptr;

    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    
    float** window_buffer = nullptr;
    int buffer_index = 0;
    bool window_ready = false;
    int playback_row_counter = 0; 
}

// Explicit definition of the 10 structural math nodes
constexpr int TFLMnumberOperators = 10;
tflite::MicroMutableOpResolver<TFLMnumberOperators> TFLMgetResolver() {
    tflite::MicroMutableOpResolver<TFLMnumberOperators> res;
    res.AddReshape();
    res.AddConv2D();
    res.AddAdd();
    res.AddAveragePool2D();
    res.AddShape();
    res.AddStridedSlice();
    res.AddPack();           
    res.AddFullyConnected();
    res.AddMul();
    res.AddSoftmax();
    return res;
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("Initializing Knevo Neural Network (Explicit PSRAM Mode)...");

    // Allocate the sliding matrix buffer inside external PSRAM
    window_buffer = (float**)heap_caps_malloc(KNEVO_TCN_WINDOW_SIZE * sizeof(float*), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    for (int i = 0; i < KNEVO_TCN_WINDOW_SIZE; i++) {
        window_buffer[i] = (float*)heap_caps_malloc(KNEVO_TCN_FEATURE_COUNT * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        memset(window_buffer[i], 0, KNEVO_TCN_FEATURE_COUNT * sizeof(float));
    }

    // Allocate the 350KB working scratchpad arena inside external PSRAM
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (tensor_arena == nullptr) {
        Serial.println("CRITICAL ERROR: Failed to allocate Tensor Arena in PSRAM!");
        while (1);
    }

    model = tflite::GetModel(g_knevo_boosted_phase_cnn_6phase_model);
    static tflite::MicroInterpreter static_interpreter(model, TFLMgetResolver(), tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Error: Tensor allocation failed inside arena!");
        while (1);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    Serial.println("System Ready! Generating streaming playback classifications...");
}

void loop() {
    float simulated_sensors[KNEVO_TCN_FEATURE_COUNT];

    // Read sequentially from the dataset array header file
    for(int i = 0; i < KNEVO_TCN_FEATURE_COUNT; i++) {
        simulated_sensors[i] = TEST_STRIDE_DATA[playback_row_counter][i]; 
    }

    playback_row_counter++;
    if (playback_row_counter >= TEST_STRIDE_ROWS) {
        playback_row_counter = 0; 
    }

    // Push into rolling window history
    for (int f = 0; f < KNEVO_TCN_FEATURE_COUNT; f++) {
        window_buffer[buffer_index][f] = simulated_sensors[f];
    }
    buffer_index++;
    if (buffer_index >= KNEVO_TCN_WINDOW_SIZE) {
        buffer_index = 0;
        window_ready = true;
    }

    if (window_ready) {
        uint32_t start_time = micros();

        // Standardize & Quantize sliding window into the INT8 Input Tensor
        int tensor_idx = 0;
        for (int t = 0; t < KNEVO_TCN_WINDOW_SIZE; t++) {
            int chronological_idx = (buffer_index + t) % KNEVO_TCN_WINDOW_SIZE;
            for (int f = 0; f < KNEVO_TCN_FEATURE_COUNT; f++) {
                float raw_val = window_buffer[chronological_idx][f];
                float standardized = (raw_val - KNEVO_TCN_MEAN[f]) / KNEVO_TCN_STD[f];
                int32_t q_val = (int32_t)round(standardized / KNEVO_TCN_INPUT_SCALE) + KNEVO_TCN_INPUT_ZERO_POINT;
                input->data.int8[tensor_idx++] = (int8_t)constrain(q_val, -128, 127);
            }
        }

        if (interpreter->Invoke() == kTfLiteOk) {
            uint32_t execution_duration = micros() - start_time;
            int predicted_phase = 0;
            float top_probability = -1.0f;

            for (int i = 0; i < 6; i++) {
                int8_t raw_output = output->data.int8[i];
                float probability = (raw_output - KNEVO_TCN_OUTPUT_ZERO_POINT) * KNEVO_TCN_OUTPUT_SCALE;
                if (probability > top_probability) {
                    top_probability = probability;
                    predicted_phase = i;
                }
            }

            Serial.print("Predicted Gait Phase: ");
            Serial.print(predicted_phase);
            Serial.print(" | Hardware Accelerated Speed: ");
            Serial.print(execution_duration);
            Serial.println(" microseconds");
        }
    }
    delay(10); // 100Hz loop execution
}
