# CodeAlpha Internship Projects

[![CI](https://github.com/skyiekoltepatil/Codealpha_tasks-1-4-/actions/workflows/ci.yml/badge.svg)](https://github.com/skyiekoltepatil/Codealpha_tasks-1-4-/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Status: Completed](https://img.shields.io/badge/Status-Completed-success.svg)]()

This repository contains the complete implementation of the four internship tasks for the **CodeAlpha Artificial Intelligence & Web Development Internship**. Each project is crafted with high-performance architectures and a cohesive macOS dark-mode aesthetic.

---

## 🗂️ Tasks Matrix

| Task | Title | Tech Stack | Status | Quick Link |
| :--- | :--- | :--- | :---: | :--- |
| **Task 1** | [Language Translation Tool](Task1_Language_Translation_Tool/) | HTML5, CSS3, ES6 JavaScript, Google Translate API | ✅ Completed | [View Task 1 Guide](Task1_Language_Translation_Tool/README.md) |
| **Task 2** | [AI FAQ Chatbot](Task2_FAQ_Chatbot/) | Python, Flask, NLTK, Scikit-Learn | ✅ Completed | [View Task 2 Guide](Task2_FAQ_Chatbot/README.md) |
| **Task 3** | [Music Generation with AI](Task3_Music_Generation/) | Python, TensorFlow, Keras LSTM, music21, Flask | ✅ Completed | [View Task 3 Guide](Task3_Music_Generation/README.md) |
| **Task 4** | [Real-Time Object Detection & Tracking](Task4_Object_Detection/) | Python, OpenCV, Ultralytics YOLOv8, BoT-SORT, Flask | ✅ Completed | [View Task 4 Guide](Task4_Object_Detection/README.md) |

---

## 🚀 Detailed Tasks Overview

### Task 1: Language Translation Tool (Fluent)
A sleek, responsive translation workspace built using Vanilla HTML5, CSS3, and JavaScript.
- **Features**: Real-time multi-language translation, bi-directional language swapping, instant clipboard copy with toast notifications, live 500-char limiter, and keyboard shortcuts (`Cmd/Ctrl + Enter`).
- **UI**: Frosted glass bento-grid layout with Apple-style typography and dark blur backdrop.

### Task 2: AI FAQ Chatbot
An intelligent NLP chatbot matching customer inquiries against a knowledge base.
- **Features**: WordNet lemmatization, TF-IDF vectorization, cosine similarity ranking, fallback recommendations, and automated unit test suite.
- **UI**: Authentic macOS iMessage window with traffic lights, typing animation indicators, and quick-prompt chips.

### Task 3: Music Generation with AI
A deep learning sequence model generating original musical compositions.
- **Features**: MIDI / Bach chorales tokenization via `music21`, 2-layer stacked LSTM neural network with Dropout, and temperature sampling.
- **UI**: Web dashboard with live `@magenta/music` web MIDI playback, staff sheet notation visualizer, and track download.

### Task 4: Real-time Object Detection and Tracking
A real-time computer vision server streaming annotated video frames over HTTP.
- **Features**: SOTA **YOLOv8** object detection paired with **BoT-SORT** persistent multi-object tracking, confidence thresholding (`conf=0.8`), and automatic camera fallback to bundled sample video.
- **UI**: macOS dark window streaming MJPEG frames in real time.

---

## 🛠️ Quickstart

### Global Dependencies
Install dependencies across all Python tasks:
```bash
pip install -r requirements.txt
```

### Running Individual Tasks

#### Task 1:
```bash
# Open directly in your browser:
open Task1_Language_Translation_Tool/index.html
```

#### Task 2:
```bash
cd Task2_FAQ_Chatbot
pip install -r requirements.txt
python3 app.py
# Server running at: http://127.0.0.1:5000
```

#### Task 3:
```bash
cd Task3_Music_Generation
pip install -r requirements.txt
python3 music_generator.py train --epochs 10   # Optional: re-train model
python3 app.py
# Server running at: http://127.0.0.1:5001
```

#### Task 4:
```bash
cd Task4_Object_Detection
pip install -r requirements.txt
python3 app.py
# Server running at: http://127.0.0.1:5002
```

---

## 🧪 Testing

Run the automated test suite:
```bash
python3 Task2_FAQ_Chatbot/test_chatbot.py
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This repository is distributed under the MIT License. See [LICENSE](LICENSE) for details.
