/*
 * servo_control.ino
 * =================
 * Arduino Mega sketch for BCI prosthetic servo driver.
 *
 * Receives JSON commands over Serial (115,200 baud):
 *   {"gesture":"open","angles":{"thumb":180,"index":180,...},"confidence":0.91}
 *
 * Responds with "OK\n" on success, "ERR:<reason>\n" on failure.
 *
 * Servos connected:
 *   Thumb  → Pin 3
 *   Index  → Pin 5
 *   Middle → Pin 6
 *   Ring   → Pin 9
 *   Pinky  → Pin 10
 *
 * Deadman LED: Pin 13 (RED when deadman active)
 */

#include <Servo.h>
#include <ArduinoJson.h>

// ── Servo pin assignments ─────────────────────────────────────────
Servo thumbServo, indexServo, middleServo, ringServo, pinkyServo;
const int THUMB_PIN  = 3;
const int INDEX_PIN  = 5;
const int MIDDLE_PIN = 6;
const int RING_PIN   = 9;
const int PINKY_PIN  = 10;
const int DEADMAN_LED = 13;

// ── Safety constants ──────────────────────────────────────────────
const float MIN_CONFIDENCE = 0.75;
const int   SERIAL_TIMEOUT_MS = 500;  // SAFE_STATE if no command for 500ms

unsigned long lastCommandMs = 0;
bool safeStateActive = false;

void setup() {
    Serial.begin(115200);
    thumbServo.attach(THUMB_PIN);
    indexServo.attach(INDEX_PIN);
    middleServo.attach(MIDDLE_PIN);
    ringServo.attach(RING_PIN);
    pinkyServo.attach(PINKY_PIN);
    pinMode(DEADMAN_LED, OUTPUT);

    // Boot to REST position
    setGesture(90, 90, 90, 90, 90);
    Serial.println("BCI_SERVO_READY");
}

void loop() {
    // ── Watchdog: if no command for 500ms → REST ───────────────────
    if (millis() - lastCommandMs > SERIAL_TIMEOUT_MS && lastCommandMs > 0) {
        if (!safeStateActive) {
            setGesture(90, 90, 90, 90, 90);
            safeStateActive = true;
            digitalWrite(DEADMAN_LED, HIGH);
            Serial.println("SAFE_STATE:timeout");
        }
    }

    // ── Read JSON command from serial ──────────────────────────────
    if (Serial.available() > 0) {
        String line = Serial.readStringUntil('\n');
        line.trim();

        StaticJsonDocument<256> doc;
        DeserializationError err = deserializeJson(doc, line);

        if (err) {
            Serial.print("ERR:json_parse:");
            Serial.println(err.c_str());
            return;
        }

        float confidence = doc["confidence"] | 0.0f;
        if (confidence < MIN_CONFIDENCE) {
            setGesture(90, 90, 90, 90, 90);
            Serial.println("ERR:confidence_too_low");
            return;
        }

        // Extract angles
        JsonObject angles = doc["angles"];
        int thumb  = angles["thumb"]  | 90;
        int index  = angles["index"]  | 90;
        int middle = angles["middle"] | 90;
        int ring   = angles["ring"]   | 90;
        int pinky  = angles["pinky"]  | 90;

        setGesture(thumb, index, middle, ring, pinky);
        lastCommandMs = millis();
        safeStateActive = false;
        digitalWrite(DEADMAN_LED, LOW);
        Serial.println("OK");
    }
}

void setGesture(int thumb, int index, int middle, int ring, int pinky) {
    thumbServo.write(constrain(thumb, 0, 180));
    indexServo.write(constrain(index, 0, 180));
    middleServo.write(constrain(middle, 0, 180));
    ringServo.write(constrain(ring, 0, 180));
    pinkyServo.write(constrain(pinky, 0, 180));
}