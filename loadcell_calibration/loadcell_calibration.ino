/*
  GripVT Stage 1.5 — HX711 Load Cell DAQ
  Target MCU : ESP32-C3 Super Mini
  Library    : HX711_ADC (Olav Kallhovd)

  Behaviour
  ---------
  On boot, the sketch checks EEPROM for a previously saved calibration
  factor. If a valid value is found (non-zero, non-NaN), it loads that
  value and skips the calibration procedure. If none is found, it runs
  the full calibration procedure and saves the result to EEPROM.

  Send 'r' over serial at any time to force re-calibration.
  Send 't' to tare (zero) without re-calibrating.
  Send 'c' to manually enter a known calibration factor.

  Output format (CSV-friendly, one line per reading):
    <timestamp_ms>,<force_g>
  Parse on the Python side and convert to Newtons: F_N = (g / 1000) * 9.80665

  Wiring (ESP32-C3 Super Mini)
  ----------------------------
  HX711 DT  → GPIO 1
  HX711 SCK → GPIO 0
  HX711 VCC → 3.3V (verify AVDD regulator output before wiring)
  HX711 GND → GND
*/

#include <HX711_ADC.h>
#include <EEPROM.h>

// ── Pin definitions ────────────────────────────────────────────────────────────
const int HX711_dout = 1;
const int HX711_sck  = 0;

// ── EEPROM ────────────────────────────────────────────────────────────────────
const int   CAL_EEPROM_ADDR = 0;   // 4 bytes (float)
const int   EEPROM_SIZE     = 16;  // bytes to allocate on ESP32

// ── HX711 ─────────────────────────────────────────────────────────────────────
HX711_ADC LoadCell(HX711_dout, HX711_sck);

// ── Helpers ───────────────────────────────────────────────────────────────────
bool isValidCalFactor(float val) {
  return (!isnan(val) && !isinf(val) && val != 0.0f);
}

void saveCalFactor(float val) {
  EEPROM.begin(EEPROM_SIZE);
  EEPROM.put(CAL_EEPROM_ADDR, val);
  EEPROM.commit();
  EEPROM.end();
}

float loadCalFactor() {
  float val = 0.0f;
  EEPROM.begin(EEPROM_SIZE);
  EEPROM.get(CAL_EEPROM_ADDR, val);
  EEPROM.end();
  return val;
}

// ── Calibration procedure ─────────────────────────────────────────────────────
void runCalibration() {
  Serial.println("=== CALIBRATION ===");
  Serial.println("1. Remove all load from the platform.");
  Serial.println("2. Send 't' to tare.");

  bool tared = false;
  while (!tared) {
    LoadCell.update();
    if (Serial.available()) {
      char c = Serial.read();
      if (c == 't') LoadCell.tareNoDelay();
    }
    if (LoadCell.getTareStatus()) {
      Serial.println("Tare complete.");
      tared = true;
    }
  }

  Serial.println("3. Place known mass on platform.");
  Serial.println("4. Send the mass in grams (e.g. 200.0).");

  float knownMass = 0.0f;
  while (knownMass == 0.0f) {
    LoadCell.update();
    if (Serial.available()) {
      knownMass = Serial.parseFloat();
      if (knownMass != 0.0f) {
        Serial.print("Known mass: ");
        Serial.print(knownMass);
        Serial.println(" g");
      }
    }
  }

  LoadCell.refreshDataSet();
  float newCal = LoadCell.getNewCalibration(knownMass);

  Serial.print("Calibration factor: ");
  Serial.println(newCal, 6);

  saveCalFactor(newCal);
  Serial.println("Saved to EEPROM.");
  Serial.println("=== END CALIBRATION ===");
}

// ── Manual cal factor entry ───────────────────────────────────────────────────
void changeCalFactor() {
  Serial.print("Current cal factor: ");
  Serial.println(LoadCell.getCalFactor(), 6);
  Serial.println("Send new cal factor value:");

  float newVal = 0.0f;
  while (newVal == 0.0f) {
    if (Serial.available()) {
      newVal = Serial.parseFloat();
    }
  }
  LoadCell.setCalFactor(newVal);
  saveCalFactor(newVal);
  Serial.print("Cal factor updated and saved: ");
  Serial.println(newVal, 6);
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\nGripVT HX711 DAQ — booting...");

  LoadCell.begin();

  unsigned long stabilizingTime = 2000;
  LoadCell.start(stabilizingTime, true);  // true = tare on boot

  if (LoadCell.getTareTimeoutFlag() || LoadCell.getSignalTimeoutFlag()) {
    Serial.println("ERROR: HX711 timeout. Check wiring and pin assignments.");
    while (1);
  }

  LoadCell.setSamplesInUse(1);  // single-sample output for Kalman pipeline

  // Try to load saved calibration factor
  float savedCal = loadCalFactor();
  if (isValidCalFactor(savedCal)) {
    LoadCell.setCalFactor(savedCal);
    Serial.print("Loaded cal factor from EEPROM: ");
    Serial.println(savedCal, 6);
    Serial.println("Send 'r' to re-calibrate, 't' to tare, 'c' to set manually.");
  } else {
    Serial.println("No valid cal factor in EEPROM. Running calibration...");
    runCalibration();
  }

  Serial.println("Ready. Output format: timestamp_ms,grams");
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
  static bool newDataReady = false;

  if (LoadCell.update()) newDataReady = true;

  if (newDataReady) {
    float grams = LoadCell.getData();
    Serial.print(millis());
    Serial.print(",");
    Serial.println(grams, 4);
    newDataReady = false;
  }

  if (Serial.available()) {
    char cmd = Serial.read();
    if      (cmd == 't') { LoadCell.tareNoDelay(); Serial.println("Taring..."); }
    else if (cmd == 'r') { runCalibration(); }
    else if (cmd == 'c') { changeCalFactor(); }
  }

  if (LoadCell.getTareStatus()) Serial.println("Tare complete.");
}
