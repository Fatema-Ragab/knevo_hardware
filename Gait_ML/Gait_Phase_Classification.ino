#include "model.h"
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <MadgwickAHRS.h>

// Configuration
#define WINDOW_SIZE 40
#define NUM_AXES 18 

unsigned long totalInferenceTime = 0;
unsigned long inferenceCount = 0;

// Hardware Objects
Adafruit_MPU6050 mpuThigh, mpuShank, mpuWaist;
Madgwick filterT, filterS;
Eloquent::ML::Port::RandomForest model; // Changed from SVM to RandomForest

// Data Buffers
float window[WINDOW_SIZE][NUM_AXES];
int bufferIndex = 0;
bool bufferFull = false;

void setup() {
  Serial.begin(115200);
  
  // ESP32-S3 I2C Setup (Standard pins from your scanner)
  Wire.begin(4, 5);   // Bus 0
  Wire1.begin(6, 7);  // Bus 1
  
  // Initialize Sensors
  if(!mpuThigh.begin(0x68, &Wire))  Serial.println("Thigh MPU Failed");
  if(!mpuShank.begin(0x69, &Wire))  Serial.println("Shank MPU Failed");
  if(!mpuWaist.begin(0x68, &Wire1)) Serial.println("Waist MPU Failed");

  filterT.begin(100);
  filterS.begin(100);
  Serial.println("System Initialized...");
  delay(2000);
}

void loop() {
  sensors_event_t aT, gT, aS, gS, aW, gW, temp;
  
  // 1. READ RAW DATA
  mpuThigh.getEvent(&aT, &gT, &temp);
  mpuShank.getEvent(&aS, &gS, &temp);
  mpuWaist.getEvent(&aW, &gW, &temp);

  // 2. MADGWICK (Degrees for Plotter)
  filterT.updateIMU(gT.gyro.x * 57.3, gT.gyro.y * 57.3, gT.gyro.z * 57.3, aT.acceleration.x, aT.acceleration.y, aT.acceleration.z);
  filterS.updateIMU(gS.gyro.x * 57.3, gS.gyro.y * 57.3, gS.gyro.z * 57.3, aS.acceleration.x, aS.acceleration.y, aS.acceleration.z);
  float kneeAngle = filterT.getPitch() - filterS.getPitch();

  // 3. STORE IN SLIDING WINDOW
  float currentSample[18] = {
    aT.acceleration.x, aT.acceleration.y, aT.acceleration.z, gT.gyro.x, gT.gyro.y, gT.gyro.z,
    aS.acceleration.x, aS.acceleration.y, aS.acceleration.z, gS.gyro.x, gS.gyro.y, gS.gyro.z,
    aW.acceleration.x, aW.acceleration.y, aW.acceleration.z, gW.gyro.x, gW.gyro.y, gW.gyro.z
  };
  
  for(int i=0; i<18; i++) window[bufferIndex][i] = currentSample[i];
  bufferIndex = (bufferIndex + 1) % WINDOW_SIZE;
  if (bufferIndex == 0) bufferFull = true;

  // 4. INFERENCE (Only when buffer is full)
  if (bufferFull) {
    float features[36]; // 18 means + 18 stds
    
    for (int j = 0; j < 18; j++) {
      float sum = 0, sqSum = 0;
      for (int i = 0; i < WINDOW_SIZE; i++) sum += window[i][j];
      float mean = sum / WINDOW_SIZE;
      
      for (int i = 0; i < WINDOW_SIZE; i++) {
        sqSum += (window[i][j] - mean) * (window[i][j] - mean);
      }
      
      features[j] = mean;                // 0-17: Means
      features[j + 18] = sqrt(sqSum / WINDOW_SIZE); // 18-35: Stds
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
    //Serial.print("KneeAngle:"); Serial.print(kneeAngle);
    Serial.print(" | Phase: "); Serial.print(gaitLabel); 
    Serial.print(" | Avg Inference: "); Serial.print(averageInferenceTime);
    Serial.println(" us");

    // Optional: Reset average every 1000 samples to prevent variable overflow
    if (inferenceCount > 1000) {
      totalInferenceTime = 0;
      inferenceCount = 0;
    }
  }

  delay(5); // Maintain ~100Hz loop
}