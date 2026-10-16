import os
from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Prefer lightweight nano model for faster inference, or use medium model if available
weights = 'yolov8n.pt' if os.path.exists('yolov8n.pt') else ('yolov8m.pt' if os.path.exists('yolov8m.pt') else 'yolov8n.pt')
model = YOLO(weights)

def generate_frames():
    # Attempt webcam first, fallback to sample video if webcam is unavailable
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    
    # Check if webcam can be read
    is_working, _ = cap.read() if cap.isOpened() else (False, None)
    if not is_working:
        sample_path = "sample_video.mp4"
        if os.path.exists(sample_path):
            print(f"Webcam not accessible, streaming from {sample_path}")
            cap = cv2.VideoCapture(sample_path)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # Restart video if it reaches the end
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        # Run YOLOv8 tracking, persisting tracks between frames
        # Built-in tracking algorithm: BoT-SORT
        # Added conf=0.8 to only show very confident predictions and stop the "hair drier" false positives
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False, conf=0.8)
        
        # Plot the results onto the frame (boxes, labels, tracking IDs)
        annotated_frame = results[0].plot()
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Multipart HTTP response for streaming video frames
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Running on Port 5002 to avoid conflicts with Tasks 2 & 3
    app.run(host='0.0.0.0', debug=True, port=5002)
