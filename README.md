# 📌 Pose Estimation for Human-Computer Interaction

> A real-time system for human keypoint detection, 3D pose reconstruction, and gesture recognition using a hybrid pipeline of classical vision techniques and modern deep learning frameworks.

---

## 🔍 Overview

This project implements a **real-time human pose estimation framework** capable of interpreting body movements and gestures through 2D keypoint detection and monocular 3D pose reconstruction. It integrates:

- **MediaPipe** for landmark detection  
- **MiDaS** for depth estimation  
- **SIFT (Scale-Invariant Feature Transform)** for classical keypoint detection  
- A **rule-based gesture recognition engine** for understanding intuitive actions

The system includes an interactive UI with brightness/contrast control, gesture overlays, and side-by-side visualization of RGB pose tracking and depth maps — designed for performance, robustness, and extensibility.

---

## 🎯 Objectives

- Detect 2D body keypoints in real time
- Extend 2D joints into 3D space using depth estimation
- Recognize gestures like "Hands Up", "T-Pose", "Hands on Hips", and directional pointing
- Benchmark classical SIFT against modern DL models
- Operate reliably in low-light, high-contrast, and occluded scenarios
- Provide GUI-based environmental simulation and tuning
- Capture scenario-specific snapshots automatically

---

## 🧠 System Architecture

```mermaid
graph TD
    A[Webcam Input] --> B[Frame Preprocessing]
    B --> C[SIFT Keypoint Detection]
    B --> D[MediaPipe Pose Estimation]
    D --> E[MiDaS Depth Estimation]
    E --> F[3D Keypoint Mapping]
    F --> G[Gesture Recognition]
    G --> H[Visualization + UI Rendering]
    H --> I[Snapshot Capture]
````

---

## 🧰 Tech Stack

| Component                | Tool / Framework                                    |
| ------------------------ | --------------------------------------------------- |
| Programming Language     | Python 3.8                                          |
| Computer Vision          | OpenCV 4.x                                          |
| Pose Estimation          | MediaPipe Pose (Google)                             |
| Depth Estimation         | MiDaS v3 Small (via PyTorch Hub)                    |
| Deep Learning Framework  | PyTorch                                             |
| Classical Feature Method | SIFT (Scale-Invariant Feature Transform)            |
| UI Controls              | OpenCV Windows + Trackbars                          |
| Hardware Used            | Webcam (640×480), Intel i5 CPU, optional NVIDIA GPU |

---

## ⚙️ Features & Capabilities

### ✅ Pose Estimation

* 33 body landmarks via MediaPipe
* Real-time skeletal overlay
* Pose responsiveness under environmental constraints

### ✅ Depth Estimation

* Monocular depth inference using MiDaS
* Bicubic interpolation for pixel-accurate z-values
* Color-coded (Magma) depth map visualization

### ✅ Gesture Recognition

* Geometric rule-based classification
* Recognizes:

  * 🖐 Hands Up
  * ✋ T-Pose
  * 🤷 Hands on Hips
  * 👉 Pointing (Left/Right)

### ✅ Classical vs DL Comparison

* Overlay of SIFT keypoints for performance benchmarking
* FPS, joint counts, and SIFT stats displayed live

### ✅ Interactive Simulation

* GUI sliders for:

  * Brightness control
  * Contrast adjustment
* Simulated:

  * Low-light conditions
  * Occlusion masking

### ✅ Output & Snapshots

* Side-by-side: Pose tracking + Depth map
* Auto-snapshot logic for:

  * All keypoints detected
  * Gestures activated
  * Max/min brightness & contrast

---

## 🛠️ Installation & Setup

### Prerequisites

Ensure Python ≥ 3.8 with the following libraries:

```bash
pip install opencv-python mediapipe torch torchvision psutil
```

### Clone and Run

```bash
git clone https://github.com/your-username/pose-estimation-hci.git
cd pose-estimation-hci
```

### Run Fast Mode (2D Pose Only)

```bash
python Task_1.py
```

### Run Full Pipeline (3D Pose + Gesture + UI)

```bash
python Task_2.py
```

> ⚠️ Ensure a webcam is connected. A GPU is recommended but not required.

---

## 📁 Repository Structure

```
pose-estimation-hci/
│
├── Task_1.py               # Fast mode: MediaPipe 2D pose only
├── Task_2.py               # Full pipeline: 3D + Gesture + UI + Depth
├── Project_Report.docx     # Editable project report
├── Project_Report.pdf      # Final formatted documentation
├── Project_Statement.pdf   # Assignment specification
└── README.md               # Project overview
```

---

## 📊 Evaluation Metrics

| Metric             | Result (CPU)     |
| ------------------ | ---------------- |
| DL Keypoints       | 33 (MediaPipe)   |
| Classical Features | \~100–150 (SIFT) |
| Depth Resolution   | Full Frame       |
| FPS (CPU)          | \~15–22          |
| FPS (GPU)          | \~28–35          |

Tested under varied:

* Lighting: min, max, simulated occlusion
* Gestures: dynamic arm movements
* Pose completeness: full-body detection

---

## 🔬 Comparative Insights

| Aspect                  | MediaPipe (DL)     | SIFT (Classical)        |
| ----------------------- | ------------------ | ----------------------- |
| Accuracy                | ✅ High             | ❌ Low for body joints   |
| Temporal Consistency    | ✅ Smooth tracking  | ❌ No temporal model     |
| Robustness to Occlusion | ✅ Medium           | ❌ Low                   |
| Processing Speed        | ✅ Optimized        | ✅ Fast (low complexity) |
| Depth Compatibility     | ✅ MiDaS-compatible | ❌ 2D only               |

---

## 🧪 Applications

* Gesture-controlled interfaces (e.g., sign recognition)
* Augmented and Virtual Reality (AR/VR)
* Physical therapy and rehabilitation feedback
* Robotics control using body gestures
* Human activity monitoring for safety/compliance

---

## 🔗 References

* [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/pose)
* [MiDaS Depth Estimation](https://github.com/isl-org/MiDaS)
* [OpenCV](https://opencv.org/)
* [D. Lowe, SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
* [PyTorch Hub](https://pytorch.org/hub/)
* [Google AI Blog on Pose Estimation](https://ai.googleblog.com/2020/05/)

---

## 🤝 Acknowledgments

This project was developed as part of the **CS-474 Computer Vision** course at NUST College of EME, Islamabad. We thank our instructor and peers for guidance throughout.

---
