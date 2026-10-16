# Task 3: Music Generation with AI

A deep learning music generation system that utilizes an LSTM (Long Short-Term Memory) Recurrent Neural Network built with Keras and `music21` to analyze musical corpora and generate original compositions.

## ✨ Features

- **Musical Tokenization**: Parses MIDI and Sheet Music files into composite `pitch_duration` tokens with quarter-length quantization.
- **Corpus Fallback**: Automatically leverages the built-in Bach chorales corpus from `music21` if local MIDI samples are not supplied.
- **Stacked LSTM Architecture**: Two stacked LSTM layers (256 units each) with Dropout regularization (0.3) and Dense softmax output.
- **Temperature Sampling**: Configurable temperature coefficient for balancing deterministic note patterns vs creative variance.
- **Interactive Web Player & Sheet Visualizer**: Built-in `@magenta/music` web MIDI player and staff notation sheet visualizer with audio download capabilities.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python3 music_generator.py train --epochs 15
```

### 3. Generate New Music
```bash
python3 music_generator.py generate --length 300 --temperature 0.9 --out generated_music.mid
```

### 4. Run the Web Dashboard
```bash
python3 app.py
```
Open [http://127.0.0.1:5001](http://127.0.0.1:5001) to listen to and download the AI track.

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow, Keras (Sequential, LSTM, Embedding)
- **Music Theory & Parsing**: `music21`, `midi2audio`
- **Web Interface**: Flask, HTML5, Magenta Core JS, Tone.js
