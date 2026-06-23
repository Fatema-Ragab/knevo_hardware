#include <Arduino.h>
#include <math.h>
#include <string.h>

#include "TFLiteMicro_ArduinoESP32S3.h"
#include "knevo_esp32s3_100hz_6phase_int8_model.h"
#include "knevo_esp32_boosted_phase_cnn_100hz_scaler.h"
#include "test_stride.h"

const uint32_t SAMPLE_PERIOD_US = 10000;     // 100 Hz
const int INFERENCE_EVERY_N_SAMPLES = 5;     // CNN 20 Hz, sensors/control 100 Hz
const int NUM_PHASES = 6;
const int TENSOR_ARENA_SIZE = 130 * 1024;

int8_t quantized_window_buffer[KNEVO_TCN_WINDOW_SIZE * KNEVO_TCN_FEATURE_COUNT];
int current_sample_row = 0;
bool window_ready = false;

float q_mult[KNEVO_TCN_FEATURE_COUNT];
float q_bias[KNEVO_TCN_FEATURE_COUNT];

uint32_t next_sample_time_us = 0;
int samples_since_inference = 0;
int playback_row_counter = 0;

int last_predicted_phase = -1;
float last_probability = 0.0f;

inline int8_t clampInt8(int32_t x) {
  if (x > 127) return 127;
  if (x < -128) return -128;
  return (int8_t)x;
}

inline int8_t fastQuantizeFeature(float raw_val, int feature_index) {
  int32_t q = (int32_t)lrintf(raw_val * q_mult[feature_index] + q_bias[feature_index]);
  return clampInt8(q);
}

void loadNextPlaybackSample(float* sensor_row) {
  for (int i = 0; i < KNEVO_TCN_FEATURE_COUNT; i++) {
    sensor_row[i] = TEST_STRIDE_DATA[playback_row_counter][i];
  }

  playback_row_counter++;
  if (playback_row_counter >= TEST_STRIDE_ROWS) {
    playback_row_counter = 0;
  }
}

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

void copyCircularWindowToInputTensor() {
  const int feature_count = KNEVO_TCN_FEATURE_COUNT;
  const int start_row = current_sample_row;

  int first_rows = KNEVO_TCN_WINDOW_SIZE - start_row;
  int first_count = first_rows * feature_count;

  memcpy(
    TFLMinput->data.int8,
    quantized_window_buffer + (start_row * feature_count),
    first_count * sizeof(int8_t)
  );

  if (start_row > 0) {
    int second_count = start_row * feature_count;

    memcpy(
      TFLMinput->data.int8 + first_count,
      quantized_window_buffer,
      second_count * sizeof(int8_t)
    );
  }
}

void runInference() {
  uint32_t total_start = micros();

  uint32_t prep_start = micros();
  copyCircularWindowToInputTensor();
  uint32_t prep_us = micros() - prep_start;

  uint32_t invoke_start = micros();
  bool ok = TFLMpredict();
  uint32_t invoke_us = micros() - invoke_start;

  uint32_t total_us = micros() - total_start;

  if (!ok) {
    Serial.println("Invoke failed.");
    return;
  }

  int predicted_phase = 0;
  float top_probability = -999.0f;

  for (int i = 0; i < NUM_PHASES; i++) {
    int8_t raw_output = TFLMoutput->data.int8[i];
    float probability = (raw_output - KNEVO_TCN_OUTPUT_ZERO_POINT) * KNEVO_TCN_OUTPUT_SCALE;

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
  Serial.print(prep_us);
  Serial.print(" us");

  Serial.print(" | Invoke: ");
  Serial.print(invoke_us);
  Serial.print(" us");

  Serial.print(" | Total: ");
  Serial.print(total_us);
  Serial.println(" us");
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println();
  Serial.println("Knevo ESP32-S3 100Hz TFLM inference");
  Serial.println("Using converter-generated header + resolver");

  Serial.print("Window size: ");
  Serial.println(KNEVO_TCN_WINDOW_SIZE);

  Serial.print("Feature count: ");
  Serial.println(KNEVO_TCN_FEATURE_COUNT);

  for (int f = 0; f < KNEVO_TCN_FEATURE_COUNT; f++) {
    if (KNEVO_TCN_STD[f] == 0.0f) {
      q_mult[f] = 0.0f;
      q_bias[f] = KNEVO_TCN_INPUT_ZERO_POINT;
    } else {
      q_mult[f] = 1.0f / (KNEVO_TCN_STD[f] * KNEVO_TCN_INPUT_SCALE);
      q_bias[f] = KNEVO_TCN_INPUT_ZERO_POINT - (KNEVO_TCN_MEAN[f] * q_mult[f]);
    }
  }

  memset(quantized_window_buffer, 0, sizeof(quantized_window_buffer));

  TFLMinterpreter = TFLMsetupModel<TFLMnumberOperators, TENSOR_ARENA_SIZE>(
    TFLM_knevo_esp32s3_100hz_6phase_int8_model,
    TFLMgetResolver,
    true
  );

  if (!TFLMinterpreter) {
    Serial.println("Model setup failed. Increase TENSOR_ARENA_SIZE to 180 * 1024 or 250 * 1024.");
    while (1) {
      delay(1000);
    }
  }

  Serial.println("Model setup OK.");

  Serial.print("Input tensor type: ");
  Serial.println(TFLMinput->type);

  Serial.print("Output tensor type: ");
  Serial.println(TFLMoutput->type);

  next_sample_time_us = micros() + SAMPLE_PERIOD_US;

  Serial.println("System ready.");
}

void loop() {
  uint32_t now_us = micros();

  if ((int32_t)(now_us - next_sample_time_us) >= 0) {
    next_sample_time_us += SAMPLE_PERIOD_US;

    float sensor_features[KNEVO_TCN_FEATURE_COUNT];

    // Playback test from test_stride.h.
    // Replace this later with your real 57-feature IMU/FSR extraction.
    loadNextPlaybackSample(sensor_features);

    pushQuantizedSampleToWindow(sensor_features);

    if (window_ready) {
      samples_since_inference++;

      if (samples_since_inference >= INFERENCE_EVERY_N_SAMPLES) {
        samples_since_inference = 0;
        runInference();
      }
    }
  }

  yield();
}
