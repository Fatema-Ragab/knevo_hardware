#include "model.h"
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <MadgwickAHRS.h>

#define WINDOW_SIZE 40
#define NUM_AXES 18 
#define CALIBRATION_SAMPLES 500 

// --- STRUCTURES & GLOBALS ---
struct Offsets {
  float ax, ay, az;
  float gx, gy, gz;
};

Offsets thighOff, shankOff, waistOff;
bool systemRunning = false;

// 6Hz Butterworth Coefficients for 200Hz Sampling
float b[] = {0.0078, 0.0156, 0.0078};
float a[] = {-1.7347, 0.7660};

// Filter States
float vT[3][2] = {{0,0}, {0,0}, {0,0}}; 
float vS[3][2] = {{0,0}, {0,0}, {0,0}}; 
float vW[3][2] = {{0,0}, {0,0}, {0,0}}; 

Adafruit_MPU6050 mpuThigh, mpuShank, mpuWaist;
Madgwick filterT, filterS;
Eloquent::ML::Port::RandomForest model;

float window[WINDOW_SIZE][NUM_AXES];
int bufferIndex = 0;
bool bufferFull = false;

// --- CORE FUNCTIONS ---

float butterworth(float input, float* v, float b[], float a[]) {
    float output = b[0] * input + v[0];
    v[0] = b[1] * input - a[0] * output + v[1];
    v[1] = b[2] * input - a[1] * output;
    return output;
}

void calibrateSensors() {
  Serial.println("\n[CALIBRATION] Starting... KEEP THE EXOSKELETON STILL.");
  sensors_event_t a, g, temp;
  
  float sAT[3] = {0}, sGT[3] = {0};
  float sAS[3] = {0}, sGS[3] = {0};
  float sAW[3] = {0}, sGW[3] = {0};

  for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
    mpuThigh.getEvent(&a, &g, &temp);
    sAT[0]+=a.acceleration.x; sAT[1]+=a.acceleration.y; sAT[2]+=a.acceleration.z;
    sGT[0]+=g.gyro.x; sGT[1]+=g.gyro.y; sGT[2]+=g.gyro.z;

    mpuShank.getEvent(&a, &g, &temp);
    sAS[0]+=a.acceleration.x; sAS[1]+=a.acceleration.y; sAS[2]+=a.acceleration.z;
    sGS[0]+=g.gyro.x; sGS[1]+=g.gyro.y; sGS[2]+=g.gyro.z;

    mpuWaist.getEvent(&a, &g, &temp);
    sAW[0]+=a.acceleration.x; sAW[1]+=a.acceleration.y; sAW[2]+=a.acceleration.z;
    sGW[0]+=g.gyro.x; sGW[1]+=g.gyro.y; sGW[2]+=g.gyro.z;
    
    if(i % 100 == 0) Serial.print(".");
    delay(5); // Match 200Hz timing
  }

  // Calculate averages. Note: For Z-axis, we subtract 9.81 to find the sensor error relative to gravity.
  thighOff = {sAT[0]/500, sAT[1]/500, (sAT[2]/500) - 9.81, sGT[0]/500, sGT[1]/500, sGT[2]/500};
  shankOff = {sAS[0]/500, sAS[1]/500, (sAS[2]/500) - 9.81, sGS[0]/500, sGS[1]/500, sGS[2]/500};
  waistOff = {sAW[0]/500, sAW[1]/500, (sAW[2]/500) - 9.81, sGW[0]/500, sGW[1]/500, sGW[2]/500};

  Serial.println("\n[CALIBRATION] Complete. Offsets stored.");
}

void setup() {
  Serial.begin(115200);
  Wire.begin(4, 5);   
  Wire1.begin(6, 7);  
  
  if(!mpuThigh.begin(0x68, &Wire))  Serial.println("Thigh MPU Failed");
  if(!mpuShank.begin(0x69, &Wire))  Serial.println("Shank MPU Failed");
  if(!mpuWaist.begin(0x68, &Wire1)) Serial.println("Waist MPU Failed");

  Serial.println("\n--- REHAB EXOSKELETON SYSTEM ---");
  Serial.println("Commands: 'c' to Calibrate & Start");

  // Wait for user 'c' input
  while (!systemRunning) {
    if (Serial.available() > 0) {
      if (Serial.read() == 'c') {
        calibrateSensors();
        systemRunning = true;
      }
    }
  }

  filterT.begin(100);
  filterS.begin(100);
  Serial.println("System Active. Data Streaming...");
}

void loop() {
  sensors_event_t aT, gT, aS, gS, aW, gW, temp;
  mpuThigh.getEvent(&aT, &gT, &temp);
  mpuShank.getEvent(&aS, &gS, &temp);
  mpuWaist.getEvent(&aW, &gW, &temp);

  // 1. APPLY OFFSETS & FILTER (Subtracting bias)
  float fAT[3], fAS[3], fAW[3];
  
  fAT[0] = butterworth(aT.acceleration.x - thighOff.ax, vT[0], b, a);
  fAT[1] = butterworth(aT.acceleration.y - thighOff.ay, vT[1], b, a);
  fAT[2] = butterworth(aT.acceleration.z - thighOff.az, vT[2], b, a);

  fAS[0] = butterworth(aS.acceleration.x - shankOff.ax, vS[0], b, a);
  fAS[1] = butterworth(aS.acceleration.y - shankOff.ay, vS[1], b, a);
  fAS[2] = butterworth(aS.acceleration.z - shankOff.az, vS[2], b, a);

  fAW[0] = butterworth(aW.acceleration.x - waistOff.ax, vW[0], b, a);
  fAW[1] = butterworth(aW.acceleration.y - waistOff.ay, vW[1], b, a);
  fAW[2] = butterworth(aW.acceleration.z - waistOff.az, vW[2], b, a);

  // 2. STORE IN WINDOW
  float currentSample[18] = {
    fAT[0], fAT[1], fAT[2], gT.gyro.x - thighOff.gx, gT.gyro.y - thighOff.gy, gT.gyro.z - thighOff.gz,
    fAS[0], fAS[1], fAS[2], gS.gyro.x - shankOff.gx, gS.gyro.y - shankOff.gy, gS.gyro.z - shankOff.gz,
    fAW[0], fAW[1], fAW[2], gW.gyro.x - waistOff.gx, gW.gyro.y - waistOff.gy, gW.gyro.z - waistOff.gz
  };

  // 3. MADGWICK ORIENTATION (Applying Gyro Offsets)
  // Rad/s to Deg/s conversion: 57.3
  filterT.updateIMU((gT.gyro.x - thighOff.gx)*57.3, (gT.gyro.y - thighOff.gy)*57.3, (gT.gyro.z - thighOff.gz)*57.3, fAT[0], fAT[1], fAT[2]);
  filterS.updateIMU((gS.gyro.x - shankOff.gx)*57.3, (gS.gyro.y - shankOff.gy)*57.3, (gS.gyro.z - shankOff.gz)*57.3, fAS[0], fAS[1], fAS[2]);
  
  float kneeAngle = filterT.getPitch() - filterS.getPitch();

  
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

    Serial.print("KneeAngle:"); Serial.print(kneeAngle); Serial.print(",");
    Serial.print("GaitPhase:"); Serial.println(prediction == 1 ? 40 : 0);
  }

  delay(5); // 200Hz Loop
}