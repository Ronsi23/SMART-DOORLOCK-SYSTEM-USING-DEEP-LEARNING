# 🔐 Smart Door Lock System
### Integrated Face Recognition & Gaze Tracking Security Solution

An advanced automated door security system that combines facial recognition with anti-spoofing gaze tracking technology. Built for Raspberry Pi with real-time verification and smart lock control.

---

## 🌟 Overview

This system provides multi-layered security through:
- **Face Recognition**: TensorFlow Lite-powered facial identification
- **Anti-Spoofing Protection**: Gaze tracking and liveness detection
- **Smart Lock Control**: Automated relay and solenoid operation
- **Interactive Interface**: Real-time status display and user feedback

---

## ✨ Key Features

### 🎯 Security Features
- **Real-time Face Recognition** powered by TensorFlow Lite
- **Advanced Anti-Spoofing** with multiple verification methods:
  - Gaze tracking using specialized library
  - Head movement detection via OpenCV
  - Blink detection for liveness verification
- **Random Challenge System** - dynamic verification commands

### 🔧 Hardware Integration
- **GPIO Control** for Raspberry Pi relay management
- **Solenoid Lock Operation** with automatic timeout
- **LED Status Indicators** (optional)
- **Buzzer Alerts** for system feedback

### 💻 Operation Modes
- **Single Verification Mode**: One-time unlock process
- **Continuous Mode**: Multi-user verification capability
- **Auto-Reset**: Automatic timeout and system reset

### 🎨 User Interface
- Real-time verification stage display
- Visual feedback for each security step
- Status indicators and progress tracking

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Runtime** | Python 3.9+ | Core application |
| **AI/ML** | TensorFlow Lite | Face recognition engine |
| **Computer Vision** | OpenCV | Image processing & fallback detection |
| **Face Processing** | face_recognition | Facial feature extraction |
| **Gaze Detection** | gaze_tracking | Eye movement analysis |
| **Hardware Control** | lgpio | Raspberry Pi GPIO interface |
| **Data Processing** | NumPy | Mathematical operations |

---

## 📋 Prerequisites

### Hardware Requirements
- Raspberry Pi 4 or newer
- USB Camera or Pi Camera Module
- 5V Relay Module
- 12V Solenoid Lock
- External Power Supply Unit (12V)
- Jumper wires and breadboard

### Software Requirements
- Raspberry Pi OS (64-bit recommended)
- Python 3.9 or higher
- Virtual environment (recommended)

---

Raspberry Pi Connections:
├── 5V Pin → Relay VCC
├── GND Pin → Relay GND
├── GPIO 18 → Relay IN
└── Camera → USB Port or CSI Connector
Relay Module Connections:
├── COM → Power Supply V+ (12V)
├── NO → Solenoid Lock (+)
└── Power Supply V- → Solenoid Lock (-)



## 🔌 Hardware Setup

### Wiring Diagram
