#include "model.h"
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <MadgwickAHRS.h>

#define WINDOW_SIZE 40
#define NUM_AXES 18 

// NEW 6Hz Butterworth Coefficients for 200Hz Sampling
float b[] = {0.0078, 0.0156, 0.0078};
float a[] = {-1.7347, 0.7660};

// Filter States (v[2] for each axis)
float vT[3][2] = {{0,0}, {0,0}, {0,0}}; // Thigh X, Y, Z
float vS[3][2] = {{0,0}, {0,0}, {0,0}}; // Shank X, Y, Z
float vW[3][2] = {{0,0}, {0,0}, {0,0}}; // Waist X, Y, Z

Adafruit_MPU6050 mpuThigh, mpuShank, mpuWaist;
Madgwick filterT, filterS;
Eloquent::ML::Port::RandomForest model;

float window[WINDOW_SIZE][NUM_AXES];
int bufferIndex = 0;
bool bufferFull = false;

// Modular Butterworth Function (Direct Form II Transposed)
float butterworth(float input, float* v, float b[], float a[]) {
    float output = b[0] * input + v[0];
    v[0] = b[1] * input - a[0] * output + v[1];
    v[1] = b[2] * input - a[1] * output;
    return output;
}

void setup() {
  Serial.begin(115200);
  Wire.begin(4, 5);   // I2C Bus 0
  Wire1.begin(6, 7);  // I2C Bus 1
  
  if(!mpuThigh.begin(0x68, &Wire))  Serial.println("Thigh MPU Failed");
  if(!mpuShank.begin(0x69, &Wire))  Serial.println("Shank MPU Failed");
  if(!mpuWaist.begin(0x68, &Wire1)) Serial.println("Waist MPU Failed");

  filterT.begin(100);
  filterS.begin(100);
  Serial.println("Full System Initialized (All Sensors Filtered @ 6Hz)");
}

void loop() {
  sensors_event_t aT, gT, aS, gS, aW, gW, temp;
  mpuThigh.getEvent(&aT, &gT, &temp);
  mpuShank.getEvent(&aS, &gS, &temp);
  mpuWaist.getEvent(&aW, &gW, &temp);

  // 1. FILTER ALL ACCELEROMETER DATA
  float fAT[3], fAS[3], fAW[3];
  
  // Filter Thigh
  fAT[0] = butterworth(aT.acceleration.x, vT[0], b, a);
  fAT[1] = butterworth(aT.acceleration.y, vT[1], b, a);
  fAT[2] = butterworth(aT.acceleration.z, vT[2], b, a);

  // Filter Shank
  fAS[0] = butterworth(aS.acceleration.x, vS[0], b, a);
  fAS[1] = butterworth(aS.acceleration.y, vS[1], b, a);
  fAS[2] = butterworth(aS.acceleration.z, vS[2], b, a);

  // Filter Waist
  fAW[0] = butterworth(aW.acceleration.x, vW[0], b, a);
  fAW[1] = butterworth(aW.acceleration.y, vW[1], b, a);
  fAW[2] = butterworth(aW.acceleration.z, vW[2], b, a);

  // 2. MADGWICK ORIENTATION (Using Filtered Accel + Raw Gyro)
  filterT.updateIMU(gT.gyro.x * 57.3, gT.gyro.y * 57.3, gT.gyro.z * 57.3, fAT[0], fAT[1], fAT[2]);
  filterS.updateIMU(gS.gyro.x * 57.3, gS.gyro.y * 57.3, gS.gyro.z * 57.3, fAS[0], fAS[1], fAS[2]);
  float kneeAngle = filterT.getPitch() - filterS.getPitch();

  // 3. STORE FILTERED SAMPLES IN WINDOW
  float currentSample[18] = {
    fAT[0], fAT[1], fAT[2], gT.gyro.x, gT.gyro.y, gT.gyro.z,
    fAS[0], fAS[1], fAS[2], gS.gyro.x, gS.gyro.y, gS.gyro.z,
    fAW[0], fAW[1], fAW[2], gW.gyro.x, gW.gyro.y, gW.gyro.z
  };
  
  for(int i=0; i<18; i++) window[bufferIndex][i] = currentSample[i];
  bufferIndex = (bufferIndex + 1) % WINDOW_SIZE;
  if (bufferIndex == 0) bufferFull = true;

  // 4. INFERENCE
  if (bufferFull) {
    float features[36];
    for (int j = 0; j < 18; j++) {
      float sum = 0, sqSum = 0;
      for (int i = 0; i < WINDOW_SIZE; i++) sum += window[i][j];
      float mean = sum / WINDOW_SIZE;
      for (int i = 0; i < WINDOW_SIZE; i++) {
        float diff = window[i][j] - mean;
        sqSum += diff * diff;
      }
      features[j] = mean;
      features[j + 18] = sqrt(sqSum / WINDOW_SIZE);
    }

    int prediction = model.predict(features);

    // Plot Results for Analysis
    Serial.print("KneeAngle:"); Serial.print(kneeAngle); Serial.print(",");
    Serial.print("GaitPhase:"); Serial.println(prediction == 1 ? 40 : 0);
  }

  delay(5);   // 200Hz
}