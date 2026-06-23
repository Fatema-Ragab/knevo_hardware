#include <Arduino.h>
#include <math.h>
#include <string.h>

// TensorFlow Lite Micro headers
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

// ESP32 memory allocation
#include <esp_heap_caps.h>

// Your exported model / scaler / test stride files.
// Put these .h files in the SAME folder as this .ino file.
#include "knevo_esp32_boosted_phase_cnn_6phase_model.h"
#include "knevo_esp32_boosted_phase_cnn_scaler.h"
#include "test_stride.h"

namespace {

// ========================= USER SETTINGS =========================

// 100 Hz sample timing.
const uint32_t SAMPLE_PERIOD_US = 10000;  // 10 ms = 100 Hz

// Run ML every N samples.
// 1 = infer at 100 Hz, every 10 ms
// 2 = infer at 50 Hz, every 20 ms
// 5 = infer at 20 Hz, every 50 ms
//
// Recommended for your project:
// keep sensors/control/safety at 100 Hz,
// run CNN at 20 Hz and hold the last phase between predictions.
const int INFERENCE_EVERY_N_SAMPLES = 5;

// Tensor arena size.
// Your previous code used 130 KB internal SRAM.
const size_t kTensorArenaSize = 130 * 1024;

// Number of gait phases in your model.
const int NUM_PHASES = 6;

// ========================= TFLITE GLOBALS =========================

uint8_t* tensor_arena = nullptr;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// ========================= BUFFERS =========================

// Quantized circular window buffer.
// Shape logically = [KNEVO_TCN_WINDOW_SIZE][KNEVO_TCN_FEATURE_COUNT]
// Stored as flat int8 array.
int8_t* quantized_window_buffer = nullptr;

int current_sample_row = 0;
bool window_ready = false;

// Precomputed quantization constants.
// q = raw * q_mult[f] + q_bias[f]
float q_mult[KNEVO_TCN_FEATURE_COUNT];
float q_bias[KNEVO_TCN_FEATURE_COUNT];

// ========================= TIMING =========================

uint32_t next_sample_time_us = 0;
int samples_since_inference = 0;

// Playback index for test_stride.h
int playback_row_counter = 0;

// Last prediction, held between inference calls.
int last_predicted_phase = -1;
float last_probability = 0.0f;

}  // namespace

// ========================= HELPERS =========================

inline int8_t clampInt8(int32_t x) {
  if (x > 127) return 127;
  if (x < -128) return -128;
  return (int8_t)x;
}

inline int8_t fastQuantizeFeature(float raw_val, int feature_index) {
  int32_t q = (int32_t)lrintf(raw_val * q_mult[feature_index] + q_bias[feature_index]);
  return clampInt8(q);
}

// This is only for testing using test_stride.h.
// Later, replace this function with your real IMU + FSR feature extraction.
void loadNextPlaybackSample(float* sensor_row) {
  for (int i = 0; i < KNEVO_TCN_FEATURE_COUNT; i++) {
    sensor_row[i] = TEST_STRIDE_DATA[playback_row_counter][i];
  }

  playback_row_counter++;

  if (playback_row_counter >= TEST_STRIDE_ROWS) {
    playback_row_counter = 0;
  }
}

// Push one new sample into the quantized circular window.
void pushQuantizedSampleToWindow(const float* sensor_row) {
  int8_t* dest = quantized_window_buffer + (current_sample_row * KNEVO_TCN_FEATURE_COUNT);

  for (int f = 0; f < KNEVO_TCN_FEATURE_COUNT; f++) {
    dest[f] = fastQuantizeFeature(sensor_row[f], f);
  }

  current_sample_row++;

  if (current_sample_row >= KNEVO_TCN_WINDOW_SIZE) {
    current_sample_row = 0;
    window_ready = true;
  }
}

// Copy circular window to input tensor in chronological order.
void copyCircularWindowToInputTensor() {
  const int feature_count = KNEVO_TCN_FEATURE_COUNT;
  const int start_row = current_sample_row;

  // Copy from start_row to last row.
  int first_rows = KNEVO_TCN_WINDOW_SIZE - start_row;
  int first_bytes = first_rows * feature_count * sizeof(int8_t);

  memcpy(
    input->data.int8,
    quantized_window_buffer + (start_row * feature_count),
    first_bytes
  );

  // Copy from row 0 to start_row - 1.
  if (start_row > 0) {
    int second_bytes = start_row * feature_count * sizeof(int8_t);

    memcpy(
      input->data.int8 + (first_rows * feature_count),
      quantized_window_buffer,
      second_bytes
    );
  }
}

void runInference() {
  uint32_t total_start_us = micros();

  uint32_t prep_start_us = micros();
  copyCircularWindowToInputTensor();
  uint32_t prep_time_us = micros() - prep_start_us;

  uint32_t invoke_start_us = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  uint32_t invoke_time_us = micros() - invoke_start_us;

  uint32_t total_time_us = micros() - total_start_us;

  if (invoke_status != kTfLiteOk) {
    Serial.println("Prediction execution failed.");
    return;
  }

  int predicted_phase = 0;
  float top_probability = -999.0f;

  for (int i = 0; i < NUM_PHASES; i++) {
    int8_t raw_output = output->data.int8[i];
    float probability =
      (raw_output - KNEVO_TCN_OUTPUT_ZERO_POINT) * KNEVO_TCN_OUTPUT_SCALE;

    if (probability > top_probability) {
      top_probability = probability;
      predicted_phase = i;
    }
  }

  last_predicted_phase = predicted_phase;
  last_probability = top_probability;

  Serial.print("Phase: ");
  Serial.print(predicted_phase);

  Serial.print(" | Prob: ");
  Serial.print(top_probability, 4);

  Serial.print(" | Prep: ");
  Serial.print(prep_time_us);
  Serial.print(" us");

  Serial.print(" | Invoke: ");
  Serial.print(invoke_time_us);
  Serial.print(" us");

  Serial.print(" | Total: ");
  Serial.print(total_time_us);
  Serial.println(" us");
}

// ========================= SETUP =========================

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println();
  Serial.println("Knevo ESP32-S3 CNN inference test");
  Serial.println("Optimized Arduino sketch");
  Serial.println("Quantized circular buffer + non-blocking 100 Hz timing");
  Serial.println();

  Serial.print("Window size: ");
  Serial.println(KNEVO_TCN_WINDOW_SIZE);

  Serial.print("Feature count: ");
  Serial.println(KNEVO_TCN_FEATURE_COUNT);

  Serial.print("Input scale: ");
  Serial.println(KNEVO_TCN_INPUT_SCALE, 10);

  Serial.print("Input zero point: ");
  Serial.println(KNEVO_TCN_INPUT_ZERO_POINT);

  Serial.print("Output scale: ");
  Serial.println(KNEVO_TCN_OUTPUT_SCALE, 10);

  Serial.print("Output zero point: ");
  Serial.println(KNEVO_TCN_OUTPUT_ZERO_POINT);

  Serial.print("Internal SRAM free before allocation: ");
  Serial.println(heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

  // Precompute quantization constants once.
  for (int f = 0; f < KNEVO_TCN_FEATURE_COUNT; f++) {
    if (KNEVO_TCN_STD[f] == 0.0f) {
      q_mult[f] = 0.0f;
      q_bias[f] = KNEVO_TCN_INPUT_ZERO_POINT;
    } else {
      q_mult[f] = 1.0f / (KNEVO_TCN_STD[f] * KNEVO_TCN_INPUT_SCALE);
      q_bias[f] = KNEVO_TCN_INPUT_ZERO_POINT - (KNEVO_TCN_MEAN[f] * q_mult[f]);
    }
  }

  Serial.println("Quantization constants prepared.");

  // Allocate quantized circular window buffer.
  size_t window_buffer_size =
    KNEVO_TCN_WINDOW_SIZE * KNEVO_TCN_FEATURE_COUNT * sizeof(int8_t);

  quantized_window_buffer = (int8_t*)heap_caps_malloc(
    window_buffer_size,
    MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
  );

  if (quantized_window_buffer == nullptr) {
    Serial.println("CRITICAL ERROR: Quantized window buffer allocation failed.");
    while (1) {
      delay(1000);
    }
  }

  memset(quantized_window_buffer, 0, window_buffer_size);

  Serial.print("Quantized window buffer allocated: ");
  Serial.print(window_buffer_size);
  Serial.println(" bytes");

  // Allocate tensor arena in internal SRAM for speed.
  tensor_arena = (uint8_t*)heap_caps_malloc(
    kTensorArenaSize,
    MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
  );

  if (tensor_arena == nullptr) {
    Serial.println("CRITICAL ERROR: Internal SRAM tensor arena allocation failed.");
    Serial.println("Try reducing kTensorArenaSize to 120 * 1024 or 110 * 1024.");
    while (1) {
      delay(1000);
    }
  }

  Serial.print("Tensor arena allocated in internal SRAM: ");
  Serial.print(kTensorArenaSize);
  Serial.println(" bytes");

  Serial.print("Internal SRAM free after allocation: ");
  Serial.println(heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

  // Load model.
  model = tflite::GetModel(g_knevo_boosted_phase_cnn_6phase_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema version mismatch.");
    Serial.print("Model version: ");
    Serial.println(model->version());
    Serial.print("Supported TFLite schema version: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) {
      delay(1000);
    }
  }

  Serial.println("Model schema OK.");

  // Register only the ops your current exported model needs.
  // Keep this matching the model generated by your notebook.
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
    model,
    micro_op_resolver,
    tensor_arena,
    kTensorArenaSize
  );

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: Tensor allocation failed.");
    Serial.println("130 KB may be too small for this model.");
    while (1) {
      delay(1000);
    }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Tensors allocated.");

  Serial.print("Input tensor type: ");
  Serial.println(input->type);

  Serial.print("Output tensor type: ");
  Serial.println(output->type);

  if (input->type != kTfLiteInt8) {
    Serial.println("WARNING: This code expects INT8 input.");
  }

  if (output->type != kTfLiteInt8) {
    Serial.println("WARNING: This code expects INT8 output.");
  }

  next_sample_time_us = micros() + SAMPLE_PERIOD_US;

  Serial.println();
  Serial.println("System ready.");
  Serial.println("Playback/sample loop: 100 Hz");
  Serial.print("CNN inference every ");
  Serial.print(INFERENCE_EVERY_N_SAMPLES);
  Serial.println(" samples");
  Serial.println();
}

// ========================= LOOP =========================

void loop() {
  uint32_t now_us = micros();

  // Non-blocking 100 Hz sampling loop.
  if ((int32_t)(now_us - next_sample_time_us) >= 0) {
    next_sample_time_us += SAMPLE_PERIOD_US;

    float sensor_features[KNEVO_TCN_FEATURE_COUNT];

    // Test playback data.
    // Replace this later with your real 57-feature sensor extraction.
    loadNextPlaybackSample(sensor_features);

    // Quantize only the newest sample once.
    pushQuantizedSampleToWindow(sensor_features);

    if (window_ready) {
      samples_since_inference++;

      if (samples_since_inference >= INFERENCE_EVERY_N_SAMPLES) {
        samples_since_inference = 0;
        runInference();
      }
    }

    // Optional: here you can use last_predicted_phase for control.
    // Example:
    // if (last_predicted_phase >= 0) {
    //   updateMotorReferenceFromPhase(last_predicted_phase);
    // }
  }

  yield();
}
