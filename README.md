# CodeAlpha Internship Projects

This repository contains the completed tasks for the CodeAlpha Artificial Intelligence & Web Development internship. The project spans four distinct AI and software engineering tasks, all designed with a consistent, premium, dark-mode macOS aesthetic.

## 🚀 Tasks Overview

### Task 1: Language Translation Tool
A sleek web-based translation interface built using Vanilla HTML, CSS, and JavaScript. 
- **Features:** Instantly translates text between multiple languages using the Google Translate API.
- **UI:** Features a frosted glass (glassmorphism) design with a dark mode Mac-style window wrapper.
- **Tech Stack:** HTML5, CSS3, JavaScript.

### Task 2: AI FAQ Chatbot
A smart NLP-based FAQ Chatbot capable of matching user questions to the most relevant answers.
- **Features:** Uses Natural Language Processing (NLP) to tokenize, lemmatize, and find the highest cosine similarity (TF-IDF) between the user's input and a pre-defined `faq_data.json` database. If the chatbot doesn't know the answer, it automatically falls back to searching Wikipedia to answer any general knowledge question!
- **UI:** Designed exactly like an iMessage chat window on a Mac.
- **Tech Stack:** Python, Flask, NLTK, Scikit-Learn, Wikipedia API.

### Task 3: Music Generation with AI
A Deep Learning pipeline that trains a neural network to compose its own music.
- **Features:** Parses classical sheet music (Bach) using `music21`, structures the sequences, and trains a 2-layer Long Short-Term Memory (LSTM) Recurrent Neural Network using Keras. It predicts new notes and outputs a brand new `.mid` (MIDI) music file.
- **UI:** A Flask web server that provides a clean UI for downloading the AI-generated music.
- **Tech Stack:** Python, TensorFlow, Keras, music21, Flask.

### Task 4: Real-time Object Detection and Tracking
A real-time computer vision server that tracks objects using your webcam.
- **Features:** Uses the state-of-the-art **YOLOv8** (Medium) object detection model combined with the **BoT-SORT** tracking algorithm. It streams the webcam feed (`cv2.VideoCapture`), detects objects (like people, phones, cups) with a strict confidence threshold (0.8), draws bounding boxes, and assigns permanent tracking IDs to objects across frames.
- **UI:** Streams the annotated video frames to a beautiful Mac-themed web dashboard using Flask MJPEG streaming.
- **Tech Stack:** Python, OpenCV, Ultralytics (YOLOv8), PyTorch, Flask.

---

## 🛠️ How to Run

### Task 1:
Simply open `Task1_Language_Translation_Tool/index.html` in your web browser.

### Task 2:
```bash
cd Task2_FAQ_Chatbot
source venv/bin/activate
python3 app.py
# Open http://127.0.0.1:5000 in your browser
```

### Task 3:
```bash
cd Task3_Music_Generation
source venv/bin/activate
# To train the AI and generate new music:
python3 music_generator.py 
# To start the download server:
python3 app.py
# Open http://127.0.0.1:5001 in your browser
```

### Task 4:
```bash
cd Task4_Object_Detection
source venv/bin/activate
python3 app.py
# Open http://127.0.0.1:5002 in your browser
```

---
*Developed during the CodeAlpha Internship.*
