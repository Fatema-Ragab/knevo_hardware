#define MOTOR_RX_PIN 4
#define MOTOR_TX_PIN 5

HardwareSerial MotorSerial(1);

// CORRECT Motor ON — confirmed from software TX to turn Motor On
uint8_t cmdOn[5] = {0x3E, 0x88, 0x01, 0x00, 0xC7};

void sendCmd(uint8_t* cmd, int len) {
  while (MotorSerial.available()) MotorSerial.read();
  MotorSerial.write(cmd, len);
  MotorSerial.flush(); // to delete any rx data before 
}

bool motorOn() { // feedback function
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
      if (buf[i] == 0x3E) {
        Serial.println("Motor ON confirmed");
        return true;
      }
    }
    delay(100);
  }
  Serial.println("Motor ON failed");
  return false;
}

void moveToAngle(float outputDeg) {
  // Confirmed scale from software TX log:
  // raw = outputDeg × 3722 (average of all 4 measurements)
  int32_t angleRaw = (int32_t)(outputDeg * 3722.0f);

  Serial.print("Target: "); Serial.print(outputDeg); // angle wanted
  Serial.print(" deg | raw: "); Serial.println(angleRaw);

  uint8_t buf[14];
  buf[0]  = 0x3E;
  buf[1]  = 0xA3;
  buf[2]  = 0x01;
  buf[3]  = 0x08;
  buf[4]  = (0x3E + 0xA3 + 0x01 + 0x08) & 0xFF;
  // angle bytes (confirmed: bytes 5-8)
  buf[5]  = angleRaw & 0xFF;
  buf[6]  = (angleRaw >> 8) & 0xFF;
  buf[7]  = (angleRaw >> 16) & 0xFF;
  buf[8]  = (angleRaw >> 24) & 0xFF;
  // speed bytes — always 0, motor uses default speed
  buf[9]  = 0x00;
  buf[10] = 0x00;
  buf[11] = 0x00;
  buf[12] = 0x00;
  // data checksum
  uint8_t sum = 0;
  for (int i = 5; i < 13; i++) sum += buf[i];
  buf[13] = sum & 0xFF;

  sendCmd(buf, 14);
  delay(150);
  while (MotorSerial.available()) MotorSerial.read();
}

void waitUntilStopped() {
  Serial.print("Moving");
  delay(500);
  unsigned long timeout = millis();
  int zeroCount = 0;
  uint8_t statusCmd[5] = {0x3E, 0x9C, 0x01, 0x00, 0xDB};

  while (millis() - timeout < 15000) {
    while (MotorSerial.available()) MotorSerial.read();
    MotorSerial.write(statusCmd, 5);
    MotorSerial.flush();
    delay(100);

    uint8_t rx[32];
    int n = 0;
    unsigned long t = millis();
    while (n < 32 && millis() - t < 200) {
      if (MotorSerial.available()) {
        rx[n++] = MotorSerial.read();
        t = millis();
      }
    }

    for (int i = 0; i < n - 8; i++) {
      if (rx[i] == 0x3E) {
        int16_t spd = (int16_t)(rx[i+6] | (rx[i+7] << 8));
        Serial.print(".");
        if (abs(spd) < 10) {
          zeroCount++;
          if (zeroCount >= 3) {
            Serial.println(" STOPPED");
            return;
          }
        } else {
          zeroCount = 0;
        }
        break;
      }
    }
    delay(200);
  }
  Serial.println(" TIMEOUT");
}

void setup() {
  Serial.begin(115200);
  delay(3000);
  MotorSerial.begin(115200, SERIAL_8N1, MOTOR_RX_PIN, MOTOR_TX_PIN);
  Serial.println("=== Position Control ===");
  Serial.println("Type angle (0-360) and press Enter");
  Serial.println("Type 'zero' to go to 0");
  motorOn();
  delay(500);

  Serial.println("Going to 0 first...");
  if (motorOn()) {
    moveToAngle(0);
    waitUntilStopped();
  }
  Serial.println("Ready — enter target angle:");
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.equalsIgnoreCase("zero")) {
      Serial.println("Going to 0...");
      if (motorOn()) {
        moveToAngle(0);
        waitUntilStopped();
      }
      Serial.println("Ready:");
      return;
    }

    float targetAngle = input.toFloat();
    if (targetAngle >= 0 && targetAngle <= 360) {
      Serial.print("Moving to: "); Serial.print(targetAngle); Serial.println(" deg");
      if (motorOn()) {
        moveToAngle(targetAngle);
        waitUntilStopped();
        Serial.println("Done. Enter next angle:");
      }
    } else {
      Serial.println("Invalid. Enter 0-360 or 'zero'");
    }
  }
}