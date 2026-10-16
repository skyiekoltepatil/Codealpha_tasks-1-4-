# Task 4: Real-Time Object Detection and Tracking

A real-time computer vision system integrating state-of-the-art **YOLOv8** object detection with the **BoT-SORT** multi-object tracking algorithm, rendered through a macOS-styled dark-mode Flask web dashboard.

## ✨ Features

- **YOLOv8 Object Detection**: High-accuracy real-time multi-class object detection. Ultralytics automatically downloads the official weights (`yolov8n.pt` / `yolov8m.pt`) on first run.
- **BoT-SORT Object Tracking**: Maintains persistent tracking IDs across sequential frames to accurately count and follow unique entities.
- **Dynamic Input Fallback**: Tries connecting to the default webcam device (`index 0`). If a camera is unavailable or permissions are restricted, it gracefully falls back to the bundled sample pedestrian video (`sample_video.mp4`).
- **High-Confidence Filter**: Employs a strict confidence threshold (`conf=0.8`) to eliminate false positives.
- **MJPEG Streaming Server**: Low-latency video streaming endpoint utilizing Flask multipart responses (`multipart/x-mixed-replace`).
- **macOS Glass Interface**: Responsive dashboard container with authentic traffic-light controls and ambient styling.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python3 app.py
```
Open [http://127.0.0.1:5002](http://127.0.0.1:5002) in your web browser.

## 🛠️ Tech Stack

- **Computer Vision**: OpenCV (`cv2`), Ultralytics YOLOv8, PyTorch (`torch`, `torchvision`)
- **Web Streaming**: Flask (MJPEG generator)
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), Font Awesome
