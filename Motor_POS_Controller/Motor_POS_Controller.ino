// ============================================================
// KNEE GAIT FSM + MOTOR CONTROL + REAL ANGLE FEEDBACK
// Serial Plotter shows:
//   1. Gait_percent
//   2. StatusCode x10
//   3. DesiredAngle (what FSM wants)
//   4. ActualAngle  (what motor reports)
// ============================================================

#define MOTOR_RX_PIN 4
#define MOTOR_TX_PIN 5

HardwareSerial MotorSerial(1);

uint8_t cmdOn[5] = {0x3E, 0x88, 0x01, 0x00, 0xC7};

// -----------------------------------------------------------
// FSM TABLE
// -----------------------------------------------------------
struct FSMState {
  int   id;
  float gpStart;
  float gpEnd;
  float angleStart;
  float angleEnd;
  const char* name;
};

const FSMState fsmTable[6] = {
  {1,  0.0f, 10.0f,  5.0f, 15.0f, "LoadingResponse"},
  {2, 10.0f, 30.0f, 15.0f, 10.0f, "MidStance"     },
  {3, 30.0f, 50.0f, 10.0f,  5.0f, "TerminalStance" },
  {4, 50.0f, 60.0f,  5.0f, 20.0f, "PreSwing"       },
  {5, 60.0f, 80.0f, 20.0f, 60.0f, "SwingFlexion"   },
  {6, 80.0f,100.0f, 60.0f,  5.0f, "SwingExtension" }
};

// -----------------------------------------------------------
// VARIABLES
// -----------------------------------------------------------
int   statusCode    = 1;
float desiredAngle  = 5.0f;
float actualAngle   = 0.0f;  // real angle read back from motor
float lastSentAngle = -999.0f;
float Gait_percent  = 0.0f;

unsigned long lastUpdate = 0;
const unsigned long UPDATE_MS = 50;

// -----------------------------------------------------------
// MOTOR FUNCTIONS
// -----------------------------------------------------------
void sendCmd(uint8_t* cmd, int len) {
  while (MotorSerial.available()) MotorSerial.read();
  MotorSerial.write(cmd, len);
  MotorSerial.flush();
}

bool motorOn() {
  for (int attempt = 0; attempt < 20; attempt++) {
    while (MotorSerial.available()) MotorSerial.read();
    sendCmd(cmdOn, 5);
    delay(200);
    uint8_t buf[16];
    int n = 0;
    unsigned long t = millis();
    while (n < 16 && millis() - t < 200) {
      if (MotorSerial.available()) {
        buf[n++] = MotorSerial.read();
        t = millis();
      }
    }
    for (int i = 0; i < n; i++) {
      if (buf[i] == 0x3E) return true;
    }
    delay(100);
  }
  return false;
}

void moveToAngle(float outputDeg) {
  int32_t angleRaw = (int32_t)(outputDeg * 3722.0f);

  uint8_t buf[14];
  buf[0]  = 0x3E;
  buf[1]  = 0xA3;
  buf[2]  = 0x01;
  buf[3]  = 0x08;
  buf[4]  = (0x3E + 0xA3 + 0x01 + 0x08) & 0xFF;
  buf[5]  = angleRaw & 0xFF;
  buf[6]  = (angleRaw >> 8)  & 0xFF;
  buf[7]  = (angleRaw >> 16) & 0xFF;
  buf[8]  = (angleRaw >> 24) & 0xFF;
  buf[9]  = 0x00;
  buf[10] = 0x00;
  buf[11] = 0x00;
  buf[12] = 0x00;
  uint8_t sum = 0;
  for (int i = 5; i < 13; i++) sum += buf[i];
  buf[13] = sum & 0xFF;

  sendCmd(buf, 14);
  delay(50);
  while (MotorSerial.available()) MotorSerial.read();
}

// -----------------------------------------------------------
// READ ACTUAL ANGLE FROM MOTOR
// Uses command 0x92 - Read Multi Loop Angle
// Motor replies with current output shaft angle in 0.01 degree units
// -----------------------------------------------------------
float readActualAngle() {
  // Command 0x92 = Read Multi Loop Angle
  // Checksum = (0x3E + 0x92 + 0x01 + 0x00) & 0xFF = 0xD1
  uint8_t angleCmd[5] = {0x3E, 0x92, 0x01, 0x00, 0xD1};

  while (MotorSerial.available()) MotorSerial.read();
  MotorSerial.write(angleCmd, 5);
  MotorSerial.flush();
  delay(50);

  uint8_t rx[32];
  int n = 0;
  unsigned long t = millis();
  while (n < 32 && millis() - t < 200) {
    if (MotorSerial.available()) {
      rx[n++] = MotorSerial.read();
      t = millis();
    }
  }

  // Find 0x3E header in response
  for (int i = 0; i < n - 8; i++) {
    if (rx[i] == 0x3E && rx[i+1] == 0x92) {
      // Angle is bytes 5-8 (int32 little endian, unit = 0.01 degree)
      int32_t rawAngle = (int32_t)(
        rx[i+5] |
        (rx[i+6] << 8)  |
        (rx[i+7] << 16) |
        (rx[i+8] << 24)
      );
      // Convert from raw units back to degrees
      return (float)rawAngle / 3722.0f;
    }
  }

  // If no valid response — return last known angle
  return actualAngle;
}

// -----------------------------------------------------------
// FSM FUNCTION
// -----------------------------------------------------------
void runFSM(float gp) {
  gp = constrain(gp, 0.0f, 100.0f);

  for (int i = 0; i < 6; i++) {
    if (gp >= fsmTable[i].gpStart && gp <= fsmTable[i].gpEnd) {
      statusCode = fsmTable[i].id;
      float range = fsmTable[i].gpEnd - fsmTable[i].gpStart;
      float t = (range > 0.0f) ? (gp - fsmTable[i].gpStart) / range : 0.0f;
      desiredAngle = fsmTable[i].angleStart + t * (fsmTable[i].angleEnd - fsmTable[i].angleStart);
      return;
    }
  }

  statusCode   = 6;
  desiredAngle = 5.0f;
}

// ===========================================================
// SETUP
// ===========================================================
void setup() {
  Serial.begin(115200);
  delay(3000);

  MotorSerial.begin(115200, SERIAL_8N1, MOTOR_RX_PIN, MOTOR_TX_PIN);

  Serial.println("=== FSM + Motor + Feedback ===");

  if (motorOn()) {
    Serial.println("Motor ON confirmed");
  } else {
    Serial.println("Motor ON failed");
  }

  // Go to start position
  moveToAngle(5.0f);
  delay(3000);

  // Serial Plotter header
  // 4 signals: Gait%, State x10, Desired angle, Actual angle
  Serial.println("Gait_percent,State_x10,Desired_deg,Actual_deg");
}

// ===========================================================
// LOOP
// ===========================================================
void loop() {
  unsigned long now = millis();
  if (now - lastUpdate < UPDATE_MS) return;
  lastUpdate = now;

  // STEP 1: Update Gait_percent - sweeps 0 to 100 then stops
  // Replace with: Gait_percent = getMLOutput();
  if (Gait_percent < 100.0f) {
    Gait_percent += 0.5f;
    if (Gait_percent > 100.0f) Gait_percent = 100.0f;
  }

  // STEP 2: Run FSM → get desiredAngle and statusCode
  runFSM(Gait_percent);

  // STEP 3: Send angle to motor only if changed by >= 0.5 degrees
  if (abs(desiredAngle - lastSentAngle) >= 0.5f) {
    if (motorOn()) {
      moveToAngle(desiredAngle);
      lastSentAngle = desiredAngle;
    }
  }

  // STEP 4: Read actual angle back from motor
  actualAngle = readActualAngle();

  // STEP 5: Print all 4 signals to Serial Plotter
  Serial.print(Gait_percent,    1); Serial.print(",");
  Serial.print(statusCode * 10, 0); Serial.print(",");
  Serial.print(desiredAngle,    1); Serial.print(",");
  Serial.println(actualAngle,   1);
}