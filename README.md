🔐 Integrated Face Recognition & Gaze Tracking Door Lock System
An advanced automated door security system based on Face Recognition and Gaze Tracking with built-in anti-spoofing mechanisms.
This system uses TensorFlow Lite, OpenCV, and GPIO control to operate a relay and solenoid lock on a Raspberry Pi.

✨ Key Features
.Real-time Face Recognition powered by TensorFlow Lite.
.Anti-Spoofing Verification with random user commands:
 .Gaze Tracking (via gaze_tracking library) when available.
 .OpenCV fallback for head direction & blink detection.

.Relay & Solenoid Lock Control via Raspberry Pi GPIO (lgpio).
.Interactive UI displaying real-time verification stages.

.Operation Modes:
.Single Verification Mode – one-time verification process to unlock the door.
.Continuous Mode – ongoing verification for multiple users.
.Automatic Timeout & Reset if verification fails or times out.

🛠 Technologies Used
Python 3.9+
TensorFlow Lite
face_recognition
OpenCV
gaze_tracking
lgpio (for Raspberry Pi GPIO control)
NumPy

🔌 Raspberry Pi Wiring to Relay & Solenoid
Raspberry Pi 5V → Relay VCC
Raspberry Pi GND → Relay GND
GPIO 18 → Relay IN
Relay COM → PSU V+
Relay NO → Solenoid +
PSU V- → Solenoid -
