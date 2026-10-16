from flask import Flask, render_template, send_file
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download')
def download_music():
    # Serve WAV if available (ideal for Mac)
    if os.path.exists("output.wav"):
        return send_file("output.wav", as_attachment=True)
    # Fallback to MIDI (Native support on Windows)
    elif os.path.exists("generated_music.mid"):
        return send_file("generated_music.mid", as_attachment=True)
    return "Music file not generated yet. Please check the terminal.", 404

@app.route('/play_midi')
def play_midi():
    path = "generated_music.mid"
    if os.path.exists(path):
        return send_file(path, mimetype='audio/midi')
    return "Music file not generated yet.", 404

if __name__ == '__main__':
    # Using Port 5001 so it doesn't conflict with Task 2 (which is on 5000)
    app.run(debug=True, port=5001)
