from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load a much smarter YOLOv8 model for higher accuracy (YOLOv8 Medium)
model = YOLO('yolov8m.pt')

def generate_frames():
    # Using webcam (0) for real-time tracking
    video_path = 0
    cap = cv2.VideoCapture(video_path)
    
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
