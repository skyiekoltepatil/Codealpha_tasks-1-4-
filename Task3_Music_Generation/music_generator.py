"""
AI Music Generation with an LSTM (Keras) + music21

Pipeline:
  1. Collect MIDI files into ./midi_data/ (any folder structure, .mid / .midi)
  2. Preprocess with music21 -> sequence of "pitch_duration" tokens
  3. Train a stacked-LSTM next-token model
  4. Generate new sequences with temperature sampling
  5. Convert to MIDI (and optionally render to WAV)

Usage:
  pip install music21 tensorflow numpy midi2audio
  python music_generator.py train    --data midi_data --epochs 50
  python music_generator.py generate --length 300 --temperature 0.9 --out output.mid
  python music_generator.py render   --midi output.mid --soundfont FluidR3_GM.sf2 --wav output.wav
"""
import argparse
import glob
import os
import pickle
import random
from collections import Counter

import numpy as np
from music21 import chord, converter, instrument, note, stream, corpus

# ----------------------------- Config ---------------------------------
SEQ_LEN = 64            # notes of context used to predict the next one
MIN_COUNT = 5           # drop tokens rarer than this (keeps vocab small)
DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]  # quarter-lengths
ARTIFACTS = "artifacts"
TOKENS_FILE = os.path.join(ARTIFACTS, "tokens.pkl")
MODEL_FILE = os.path.join(ARTIFACTS, "model.keras")


def quantize(d):
    """Snap a duration (in quarter lengths) to the nearest allowed value."""
    d = float(d)
    return min(DURATIONS, key=lambda x: abs(x - d))


# ------------------------ 1-2. Data + preprocessing --------------------
def extract_tokens(data_dir):
    """Parse every MIDI file into a flat list of tokens like 'C4_0.5' or '0.4.7_1.0'."""
    files = glob.glob(os.path.join(data_dir, "**", "*.mid"), recursive=True) + \
            glob.glob(os.path.join(data_dir, "**", "*.midi"), recursive=True)
    
    # [ROBUSTNESS ADDITION] If no files are provided by the user, fallback to the built-in Bach corpus
    if not files:
        print(f"No MIDI files found in '{data_dir}'. Falling back to built-in Bach corpus...")
        files = corpus.getComposer('bach')[:10]  # using 10 chorales for a better model
        
    print(f"Found {len(files)} MIDI files")

    songs = []
    for i, f in enumerate(files, 1):
        try:
            # converter.parse works for both local files and music21 corpus internal paths
            midi = converter.parse(f)
            parts = instrument.partitionByInstrument(midi)
            elements = parts.parts[0].recurse() if parts else midi.flat.notes
        except Exception as e:
            print(f"  skipped {f}: {e}")
            continue

        tokens = []
        for el in elements:
            if isinstance(el, note.Note):
                tokens.append(f"{el.pitch.nameWithOctave}_{quantize(el.duration.quarterLength)}")
            elif isinstance(el, chord.Chord):
                pitches = ".".join(str(n) for n in el.normalOrder)
                tokens.append(f"{pitches}_{quantize(el.duration.quarterLength)}")
        if len(tokens) > SEQ_LEN + 1:
            songs.append(tokens)
        
        # Safe printing for corpus paths
        fname = os.path.basename(f) if isinstance(f, str) else "corpus_file"
        print(f"  [{i}/{len(files)}] {fname}: {len(tokens)} events")
    return songs


def build_dataset(songs):
    counts = Counter(t for s in songs for t in s)
    keep = {t for t, c in counts.items() if c >= MIN_COUNT}
    vocab = sorted(keep)
    tok2id = {t: i for i, t in enumerate(vocab)}
    id2tok = {i: t for t, i in tok2id.items()}
    print(f"Vocabulary size: {len(vocab)}")

    X, y = [], []
    for s in songs:
        ids = [tok2id[t] for t in s if t in tok2id]   # drop rare tokens
        for i in range(len(ids) - SEQ_LEN):
            X.append(ids[i:i + SEQ_LEN])
            y.append(ids[i + SEQ_LEN])
    X, y = np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)
    print(f"Training sequences: {len(X)}")
    return X, y, tok2id, id2tok


# ----------------------------- 3. Model --------------------------------
def build_model(vocab_size):
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input(shape=(SEQ_LEN,)),
        layers.Embedding(vocab_size, 128),
        layers.LSTM(256, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(256),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dense(vocab_size, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    return model


# ----------------------------- 4. Train --------------------------------
def train(args):
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    os.makedirs(ARTIFACTS, exist_ok=True)

    songs = extract_tokens(args.data)
    X, y, tok2id, id2tok = build_dataset(songs)
    with open(TOKENS_FILE, "wb") as f:
        pickle.dump({"tok2id": tok2id, "id2tok": id2tok, "X": X}, f)

    model = build_model(len(tok2id))
    model.summary()
    model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        callbacks=[
            ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor="val_loss"),
            EarlyStopping(patience=5, restore_best_weights=True),
        ],
    )
    print(f"Saved best model to {MODEL_FILE}")


# ---------------------------- 5. Generate ------------------------------
def sample(probs, temperature):
    probs = np.log(np.maximum(probs, 1e-9)) / temperature
    probs = np.exp(probs - probs.max())
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)


def generate_tokens(length, temperature):
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_FILE)
    with open(TOKENS_FILE, "rb") as f:
        d = pickle.load(f)
    id2tok, X = d["id2tok"], d["X"]

    window = list(X[random.randrange(len(X))])      # random seed from training data
    out = []
    for _ in range(length):
        probs = model.predict(np.array([window]), verbose=0)[0]
        nxt = sample(probs, temperature)
        out.append(id2tok[nxt])
        window = window[1:] + [nxt]
    return out


def tokens_to_midi(tokens, out_path):
    """Convert 'pitch_duration' tokens back to a MIDI file."""
    offset = 0.0
    events = []
    for t in tokens:
        pitch_part, dur = t.rsplit("_", 1)
        dur = float(dur)
        if "." in pitch_part or pitch_part.isdigit():       # chord (pitch classes)
            notes = []
            for n in pitch_part.split("."):
                nn = note.Note(int(n))
                nn.storedInstrument = instrument.Piano()
                notes.append(nn)
            el = chord.Chord(notes)
        else:                                               # single note
            el = note.Note(pitch_part)
            el.storedInstrument = instrument.Piano()
        el.offset = offset
        el.duration.quarterLength = dur
        events.append(el)
        offset += dur                                       # sequential playback

    s = stream.Stream(events)
    # Output to the root generated_music.mid to keep our Web UI working!
    s.write("midi", fp="generated_music.mid")
    print(f"Saved MIDI to generated_music.mid")


def generate(args):
    tokens = generate_tokens(args.length, args.temperature)
    tokens_to_midi(tokens, args.out)


# --------------------------- 6. MIDI -> audio ---------------------------
def render(args):
    """Needs FluidSynth installed (apt install fluidsynth / brew install fluid-synth)
    and a General-MIDI soundfont (.sf2)."""
    from midi2audio import FluidSynth
    FluidSynth(args.soundfont).midi_to_audio(args.midi, args.wav)
    print(f"Saved audio to {args.wav}")


# ------------------------------- CLI ------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", default="midi_data")
    t.add_argument("--epochs", type=int, default=10) # Set default to 10 for quick testing
    t.add_argument("--batch_size", type=int, default=128)
    t.set_defaults(fn=train)

    g = sub.add_parser("generate")
    g.add_argument("--length", type=int, default=300)
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--out", default="generated_music.mid") # Align with our web UI
    g.set_defaults(fn=generate)

    r = sub.add_parser("render")
    r.add_argument("--midi", default="generated_music.mid")
    r.add_argument("--soundfont", default="/opt/homebrew/share/soundfonts/default.sf2") # Mac brew default
    r.add_argument("--wav", default="output.wav")
    r.set_defaults(fn=render)

    a = p.parse_args()
    a.fn(a)
