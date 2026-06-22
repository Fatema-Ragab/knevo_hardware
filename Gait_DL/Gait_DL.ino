#include <Arduino.h>

// Use the core optimized engine files from your new library
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

// Your model arrays from your local sketch folder
#include "knevo_esp32_boosted_phase_cnn_6phase_model.h"
#include "knevo_esp32_boosted_phase_cnn_scaler.h"

namespace {
    // 350KB working arena buffer
    const size_t kTensorArenaSize = 350 * 1024; 
    uint8_t* tensor_arena = nullptr;

    // Core TensorFlow pointers
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    
    // Sliding history buffer matrix pointer
    float** window_buffer = nullptr;
    int buffer_index = 0;
    bool window_ready = false;
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("Initializing Knevo Neural Network (Explicit PSRAM Mode)...");

    // 1. Allocate the sliding timeline matrix buffer in external PSRAM
    window_buffer = (float**)heap_caps_malloc(KNEVO_TCN_WINDOW_SIZE * sizeof(float*), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    for (int i = 0; i < KNEVO_TCN_WINDOW_SIZE; i++) {
        window_buffer[i] = (float*)heap_caps_malloc(KNEVO_TCN_FEATURE_COUNT * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        memset(window_buffer[i], 0, KNEVO_TCN_FEATURE_COUNT * sizeof(float));
    }
    Serial.println("-> Timeline Matrix allocated in external PSRAM.");

    // 2. Allocate the 350KB Tensor Arena in external PSRAM
    // This removes the 94KB overflow error completely!
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (tensor_arena == nullptr) {
        Serial.println("CRITICAL ERROR: Failed to allocate Tensor Arena in PSRAM!");
        while (1);
    }
    Serial.println("-> 350KB Tensor Arena allocated in external PSRAM.");

    // 3. Map the compiled model array
    model = tflite::GetModel(g_knevo_boosted_phase_cnn_6phase_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Error: Model schema version mismatch.");
        while (1);
    }

    // 4. Register the 10 math operators your model needs
    static tflite::MicroMutableOpResolver<10> micro_op_resolver;
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddShape();
    micro_op_resolver.AddStridedSlice();
    micro_op_resolver.AddPack();           
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddSoftmax();

    // 5. Build the interpreter object manually
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // 6. Allocate memory tensors inside our safe PSRAM allocation
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Error: Tensor allocation failed inside arena!");
        while (1);
    }

    // 7. Establish pointers to input/output channels
    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("System Ready! Generating hardware-accelerated predictions...");
}

void loop() {
    // Generate simulated sensor inputs (57 channels)
    float simulated_sensors[KNEVO_TCN_FEATURE_COUNT];
    for(int i = 0; i < KNEVO_TCN_FEATURE_COUNT; i++) {
        simulated_sensors[i] = random(-200, 200) / 100.0f; 
    }

    // Push sample into sliding buffer
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

        // Quantize and map floating features directly into the INT8 input tensor
        int tensor_idx = 0;
        for (int t = 0; t < KNEVO_TCN_WINDOW_SIZE; t++) {
            int chronological_idx = (buffer_index + t) % KNEVO_TCN_WINDOW_SIZE;
            for (int f = 0; f < KNEVO_TCN_FEATURE_COUNT; f++) {
                float raw_val = window_buffer[chronological_idx][f];
                
                // Standardize
                float standardized = (raw_val - KNEVO_TCN_MEAN[f]) / KNEVO_TCN_STD[f];
                
                // Quantize to INT8
                int32_t q_val = (int32_t)round(standardized / KNEVO_TCN_INPUT_SCALE) + KNEVO_TCN_INPUT_ZERO_POINT;
                input->data.int8[tensor_idx++] = (int8_t)constrain(q_val, -128, 127);
            }
        }

        // Invoke execution
        if (interpreter->Invoke() == kTfLiteOk) {
            uint32_t execution_duration = micros() - start_time;

            // Sort classification results
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
        } else {
            Serial.println("Prediction execution failed.");
        }
    }
    delay(10); // Run loop at ~100Hz
}