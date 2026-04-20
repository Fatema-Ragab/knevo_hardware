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

// NEW: Standing Neutral Alignment (To zero-center the raw IMU data)
float neutralAT[3] = {0}, neutralAS[3] = {0}, neutralAW[3] = {0};
float neutralPitchT = 0, neutralPitchS = 0;

bool systemRunning = false;
bool hardwareDone = false;

unsigned long totalInferenceTime = 0;
unsigned long inferenceCount = 0;

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

// 1. Hardware Calibration: Fixes internal chip Bias/Zero Error (Run while sitting/still)
void calibrateHardware() {
  Serial.println("\n[STEP 1] Hardware Calibration... KEEP STILL.");
  sensors_event_t a, g, temp;
  float sAT[3] = {0}, sGT[3] = {0}, sAS[3] = {0}, sGS[3] = {0}, sAW[3] = {0}, sGW[3] = {0};

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
    delay(5);
  }
  // Subtracting 9.81 on Z to create a Linear Accelerometer baseline
  thighOff = {sAT[0]/500, sAT[1]/500, (sAT[2]/500) - 9.81, sGT[0]/500, sGT[1]/500, sGT[2]/500};
  shankOff = {sAS[0]/500, sAS[1]/500, (sAS[2]/500) - 9.81, sGS[0]/500, sGS[1]/500, sGS[2]/500};
  waistOff = {sAW[0]/500, sAW[1]/500, (sAW[2]/500) - 9.81, sGW[0]/500, sGW[1]/500, sGW[2]/500};
  Serial.println("\nHardware Calibrated.");
}

// 2. Standing Alignment: Zero-centers raw IMU data based on "Straight" pose
void alignNeutralPose() {
  Serial.println("\n[STEP 2] Standing Alignment... STAND STRAIGHT.");
  sensors_event_t aT, gT, aS, gS, aW, gW, temp;
  float sumAT[3]={0}, sumAS[3]={0}, sumAW[3]={0}, pT=0, pS=0;

  for (int i = 0; i < 200; i++) {
    mpuThigh.getEvent(&aT, &gT, &temp);
    mpuShank.getEvent(&aS, &gS, &temp);
    mpuWaist.getEvent(&aW, &gW, &temp);

    // Update filters to find current "Standing" tilt
    filterT.updateIMU((gT.gyro.x-thighOff.gx)*57.3, (gT.gyro.y-thighOff.gy)*57.3, (gT.gyro.z-thighOff.gz)*57.3, aT.acceleration.x-thighOff.ax, aT.acceleration.y-thighOff.ay, aT.acceleration.z-thighOff.az);
    filterS.updateIMU((gS.gyro.x-shankOff.gx)*57.3, (gS.gyro.y-shankOff.gy)*57.3, (gS.gyro.z-shankOff.gz)*57.3, aS.acceleration.x-shankOff.ax, aS.acceleration.y-shankOff.ay, aS.acceleration.z-shankOff.az);
    
    pT += filterT.getPitch(); pS += filterS.getPitch();
    
    // Capture static acceleration (Linear acceleration baseline)
    sumAT[0] += (aT.acceleration.x - thighOff.ax); sumAT[1] += (aT.acceleration.y - thighOff.ay); sumAT[2] += (aT.acceleration.z - thighOff.az);
    sumAS[0] += (aS.acceleration.x - shankOff.ax); sumAS[1] += (aS.acceleration.y - shankOff.ay); sumAS[2] += (aS.acceleration.z - shankOff.az);
    sumAW[0] += (aW.acceleration.x - waistOff.ax); sumAW[1] += (aW.acceleration.y - waistOff.ay); sumAW[2] += (aW.acceleration.z - waistOff.az);
    delay(5);
  }
  neutralPitchT = pT / 200.0;
  neutralPitchS = pS / 200.0;
  for(int i=0; i<3; i++) {
    neutralAT[i] = sumAT[i]/200.0; neutralAS[i] = sumAS[i]/200.0; neutralAW[i] = sumAW[i]/200.0;
  }
  Serial.println("Standing Neutral Aligned.");
}

void setup() {
  Serial.begin(115200);
  Wire.begin(4, 5);   // I2C Bus 0
  Wire1.begin(6, 7);  // I2C Bus 1
  
  if(!mpuThigh.begin(0x68, &Wire))  Serial.println("Thigh MPU Failed");
  if(!mpuShank.begin(0x69, &Wire))  Serial.println("Shank MPU Failed");
  if(!mpuWaist.begin(0x68, &Wire1)) Serial.println("Waist MPU Failed");

  Serial.println("\n--- REHAB EXOSKELETON SYSTEM ---");
  Serial.println("1. Type 'c' to Calibrate Hardware (Sitting/Still)");
  Serial.println("2. Type 'a' to Align Standing Pose (Standing Straight)");

  while (!systemRunning) {
    if (Serial.available() > 0) {
      char cmd = Serial.read();
      if (cmd == 'c') { calibrateHardware(); hardwareDone = true; }
      if (cmd == 'a' && hardwareDone) { alignNeutralPose(); systemRunning = true; }
      else if (cmd == 'a' && !hardwareDone) Serial.println("Run 'c' first!");
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

  // 1. APPLY OFFSETS & FILTER (Subtracting hardware bias AND standing neutral mean)
  float fAT[3], fAS[3], fAW[3];
  
  // Filter Thigh - Centered at 0 for standing
  fAT[0] = butterworth((aT.acceleration.x - thighOff.ax) - neutralAT[0], vT[0], b, a);
  fAT[1] = butterworth((aT.acceleration.y - thighOff.ay) - neutralAT[1], vT[1], b, a);
  fAT[2] = butterworth((aT.acceleration.z - thighOff.az) - neutralAT[2], vT[2], b, a);

  // Filter Shank
  fAS[0] = butterworth((aS.acceleration.x - shankOff.ax) - neutralAS[0], vS[0], b, a);
  fAS[1] = butterworth((aS.acceleration.y - shankOff.ay) - neutralAS[1], vS[1], b, a);
  fAS[2] = butterworth((aS.acceleration.z - shankOff.az) - neutralAS[2], vS[2], b, a);

  // Filter Waist
  fAW[0] = butterworth((aW.acceleration.x - waistOff.ax) - neutralAW[0], vW[0], b, a);
  fAW[1] = butterworth((aW.acceleration.y - waistOff.ay) - neutralAW[1], vW[1], b, a);
  fAW[2] = butterworth((aW.acceleration.z - waistOff.az) - neutralAW[2], vW[2], b, a);

  // 2. STORE IN WINDOW
  // Note: Using Rad/s for raw IMU features as per Camargo dataset
  float currentSample[18] = {
    fAT[0], fAT[1], fAT[2], gT.gyro.x - thighOff.gx, gT.gyro.y - thighOff.gy, gT.gyro.z - thighOff.gz,
    fAS[0], fAS[1], fAS[2], gS.gyro.x - shankOff.gx, gS.gyro.y - shankOff.gy, gS.gyro.z - shankOff.gz,
    fAW[0], fAW[1], fAW[2], gW.gyro.x - waistOff.gx, gW.gyro.y - waistOff.gy, gW.gyro.z - waistOff.gz
  };

  // 3. MADGWICK ORIENTATION
  // Rad/s to Deg/s conversion: 57.3 for library calculation
  filterT.updateIMU((gT.gyro.x - thighOff.gx)*57.3, (gT.gyro.y - thighOff.gy)*57.3, (gT.gyro.z - thighOff.gz)*57.3, fAT[0], fAT[1], fAT[2]);
  filterS.updateIMU((gS.gyro.x - shankOff.gx)*57.3, (gS.gyro.y - shankOff.gy)*57.3, (gS.gyro.z - shankOff.gz)*57.3, fAS[0], fAS[1], fAS[2]);
  
  // Real-time Knee Angle calculation
  float kneeAngle = (filterT.getPitch() - neutralPitchT) - (filterS.getPitch() - neutralPitchS);

  for(int i=0; i<18; i++) window[bufferIndex][i] = currentSample[i];
  bufferIndex = (bufferIndex + 1) % WINDOW_SIZE;
  if (bufferIndex == 0) bufferFull = true;

  // 4. INFERENCE
  if (bufferFull) {
    float features[36];
    // Feature extraction (Mean and Standard Deviation)
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

    // --- Start Inference Timing ---
    unsigned long startTime = micros(); 

    int prediction = model.predict(features);

    unsigned long endTime = micros();
    // --- End Inference Timing ---

    // Calculate Running Average
    unsigned long duration = endTime - startTime;
    totalInferenceTime += duration;
    inferenceCount++;
    float averageInferenceTime = (float)totalInferenceTime / (float)inferenceCount;

    // Convert Numeric Prediction to Text
    // Adjust the logic (prediction == 1) based on how your specific model was labeled
    String gaitLabel = (prediction == 1) ? "Swing" : "Stance";

    // Print Results
    Serial.print("KneeAngle:"); Serial.print(kneeAngle);
    Serial.print(" | Phase: "); Serial.print(gaitLabel); 
    Serial.print(" | Avg Inference: "); Serial.print(averageInferenceTime);
    Serial.println(" us");

    // Optional: Reset average every 1000 samples to prevent variable overflow
    if (inferenceCount > 1000) {
      totalInferenceTime = 0;
      inferenceCount = 0;
    }
  }

  delay(5); // 200Hz Loop
}