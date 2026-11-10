// Beating in Sync: stream the analog pulse sensor on A0 to the Reachy app.
//
// Prints one raw reading per line ("512\n") at 250 Hz, 115200 baud. The laptop app
// detects beats, BPM and finger presence itself (reachy_language_tutor/heart.py), so this
// sketch stays dumb on purpose: tune the detection in Python without reflashing.
//
// Works on an Arduino Uno/Nano (10-bit, 0-1023) or an ESP32 (12-bit, 0-4095); the app
// doesn't care about the scale. The built-in LED mirrors the signal level as a quick
// wiring check: it should flicker with your pulse.

const int PULSE_PIN = A0;
const unsigned long SAMPLE_US = 4000;  // 250 Hz

unsigned long nextSample = 0;
long baseline = -1;  // slow running average, only for the LED check

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  nextSample = micros();
}

void loop() {
  if ((long)(micros() - nextSample) < 0) return;
  nextSample += SAMPLE_US;

  int v = analogRead(PULSE_PIN);
  Serial.println(v);

  if (baseline < 0) baseline = (long)v * 16;
  baseline += v - baseline / 16;  // ~64 ms average, x16 fixed point
  digitalWrite(LED_BUILTIN, (long)v * 16 > baseline + 16 * 8 ? HIGH : LOW);
}
