# 👁️ REX-47 Vision AI

> **Repository `11`** · Computer vision microservice providing real-time object detection, spatial tracking, and facial recognition using PyTorch and OpenCV.

[![Platform](https://img.shields.io/badge/Platform-AI%2FData-blue)]()
[![Language](https://img.shields.io/badge/Language-Python-3776AB?logo=python)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20YOLOv8-EE4C2C?logo=pytorch)]()
[![Status](https://img.shields.io/badge/Status-Active%20Development-green)]()

---

## 📋 Table of Contents

- [Overview](#-what-is-this-repository)
- [Architecture](#-architecture)
- [Features](#-features)
- [Getting Started](#-getting-started)
- [Model Details](#-model-details)
- [Dependencies](#-dependencies)
- [Related Repositories](#-related-repositories)

---

## 🧭 What Is This Repository?

The **Vision AI** service gives REX-47 the ability to "see" and understand its environment. It ingests the raw MJPEG camera stream from the robot, runs it through deep learning models, and broadcasts the detected objects (bounding boxes, labels, confidences) over WebSockets.

**Key Highlights:**
- ✅ **Real-Time Inference:** Optimized with TensorRT/ONNX to run YOLOv8 at 30+ FPS on edge hardware.
- ✅ **Spatial Awareness:** Calculates the approximate 3D distance of detected objects using camera intrinsics.
- ✅ **Facial Recognition:** Maintains an internal database of known users to greet them by name.

---

## 🏗️ Architecture

### Directory Structure

```
11-rex-vision-ai/
├── src/
│   ├── models/               ← PyTorch weights (.pt) and ONNX files
│   ├── pipelines/            ← Inference loops (Detection, Recognition)
│   ├── stream/               ← RTSP/MJPEG frame grabbers
│   ├── utils/                ← Bounding box drawing, distance estimation
│   └── server.py             ← FastAPI wrapper exposing results via WS
├── .env.example              ← Environment template
├── requirements.txt          ← Python dependencies
└── README.md                 ← This documentation
```

---

## 🎨 Features

### 🔍 **Detection Pipelines**

| Pipeline | Description |
|----------|-------------|
| **Object Detection** | Identifies 80+ common household/industrial items using YOLOv8. |
| **Face Recognition** | Extracts facial embeddings and compares via cosine similarity. |
| **Gesture Recognition** | Detects human hand gestures (e.g., "Stop", "Follow Me"). |

---

## 🚀 Getting Started

### Prerequisites

- **Python** ≥ 3.10
- **NVIDIA GPU** (Optional but highly recommended for CUDA acceleration)

### Installation

```bash
git clone https://github.com/thathsarabandara/11-rex-vision-ai.git
cd 11-rex-vision-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Inference Server

```bash
uvicorn src.server:app --port 5004 --host 0.0.0.0
```

---

## 🔗 Related Repositories

- [03-rex-web-dashbaord](../03-rex-web-dashbaord) — Displays the AI bounding box overlays.
- [04-rex-mobile-app](../04-rex-mobile-app) — Mobile consumer of the AI stream.
- [15-rex-agent-runtime](../15-rex-agent-runtime) — Uses the detected objects to make autonomous decisions.
