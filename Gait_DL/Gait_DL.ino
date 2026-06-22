#include <Arduino.h>

// Explicit library headers
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "esp_heap_caps.h" 

#include "knevo_esp32_boosted_phase_cnn_6phase_model.h"
#include "knevo_esp32_boosted_phase_cnn_scaler.h"
#include "test_stride.h"  

namespace {
    // FIXED: Downscaled to 240KB to safely fit inside your 279.99KB largest continuous block
    const size_t kTensorArenaSize = 240 * 1024; 
    uint8_t* tensor_arena = nullptr;

    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    
    // Flat 1D array allocated internally for high-speed block copies
    float* flat_window_buffer = nullptr;
    int current_sample_row = 0;
    bool window_ready = false;
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("Initializing Knevo Neural Network (Optimized Internal SRAM)...");

    // 1. Allocate flat window feature buffer in fast internal RAM
    size_t buffer_size = KNEVO_TCN_WINDOW_SIZE * KNEVO_TCN_FEATURE_COUNT * sizeof(float);
    flat_window_buffer = (float*)heap_caps_malloc(buffer_size, MALLOC_CAP_INTERNAL);

    // 2. Allocate the 240KB Arena in internal SRAM to force ESP-NN vector speed
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (tensor_arena == nullptr) {
        Serial.println("CRITICAL ERROR: Internal SRAM allocation failed even at 240KB!");
        while (1);
    }
    Serial.println("-> Success! 240KB Tensor Arena allocated in fast internal SRAM.");

    model = tflite::GetModel(g_knevo_boosted_phase_cnn_6phase_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Error: Model schema version mismatch.");
        while (1);
    }

    // Registering operators
    static tflite::MicroMutableOpResolver<12> micro_op_resolver;
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

    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Test allocation space boundaries
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Error: Tensor allocation failed! 240KB is too small for layer scratchpads.");
        Serial.println("Increment kTensorArenaSize slightly (e.g., to 260 * 1024).");
        while (1);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    Serial.println("System Ready! Hardware-accelerated predictions unlocked.");
}

int playback_row_counter = 0; 

void loop() {
    float simulated_sensors[KNEVO_TCN_FEATURE_COUNT];

    // Dataset playback loop
    for(int i = 0; i < KNEVO_TCN_FEATURE_COUNT; i++) {
        simulated_sensors[i] = TEST_STRIDE_DATA[playback_row_counter][i]; 
    }
    playback_row_counter++;
    if (playback_row_counter >= TEST_STRIDE_ROWS) {
        playback_row_counter = 0; 
    }

    // High-speed sequential copy into flat continuous buffer
    float* dest = flat_window_buffer + (current_sample_row * KNEVO_TCN_FEATURE_COUNT);
    memcpy(dest, simulated_sensors, KNEVO_TCN_FEATURE_COUNT * sizeof(float));
    
    current_sample_row++;
    if (current_sample_row >= KNEVO_TCN_WINDOW_SIZE) {
        current_sample_row = 0;
        window_ready = true;
    }

    if (window_ready) {
        uint32_t start_time = micros();

        // Standardize, Quantize, and Map directly into the flat tensor layout
        int tensor_idx = 0;
        for (int t = 0; t < KNEVO_TCN_WINDOW_SIZE; t++) {
            int chronological_idx = (current_sample_row + t) % KNEVO_TCN_WINDOW_SIZE;
            float* row_ptr = flat_window_buffer + (chronological_idx * KNEVO_TCN_FEATURE_COUNT);
            
            for (int f = 0; f < KNEVO_TCN_FEATURE_COUNT; f++) {
                float raw_val = row_ptr[f];
                
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
        } else {
            Serial.println("Prediction execution failed.");
        }
    }
    delay(10); // Perfect 10ms delay match for your 100Hz physical stride stream
}