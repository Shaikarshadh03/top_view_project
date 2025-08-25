from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit, join_room
import cv2
import os
import threading
import time
import json
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
from collections import defaultdict
import uuid
from concurrent.futures import ThreadPoolExecutor
import queue

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
socketio = SocketIO(app, cors_allowed_origins="*")

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

class UltraFastPersonCounter:
    def __init__(self):
        self.confirmed_persons = set()
        self.track_confidence = {}
        self.track_frames_seen = defaultdict(int)
        self.track_last_seen = {}
        self.active_tracks = set()
        self.max_simultaneous_active = 0
        
        # Ultra-aggressive parameters for speed
        self.min_confidence = 0.15  # Lower threshold
        self.min_track_length = 2   # Confirm faster
        self.min_avg_confidence = 0.2
        self.min_person_area = 100  # Smaller minimum area
        self.max_person_area = 50000
        
    def is_valid_detection(self, bbox, confidence):
        """Ultra-fast validation"""
        l, t, r, b = bbox
        width = r - l
        height = b - t
        area = width * height
        
        # Quick checks only
        return (confidence >= self.min_confidence and 
                width >= 8 and height >= 10 and
                area >= self.min_person_area and 
                area <= self.max_person_area and
                height > width * 0.5)  # Basic aspect ratio
    
    def update_tracks(self, tracks, frame_number):
        """Minimal tracking logic for maximum speed"""
        current_active = 0
        
        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                self.track_frames_seen[track_id] += 1
                self.track_last_seen[track_id] = frame_number
                
                # Immediate confirmation for speed
                if self.track_frames_seen[track_id] >= self.min_track_length:
                    if track_id not in self.confirmed_persons:
                        self.confirmed_persons.add(track_id)
                
                # Count as active if seen recently
                if frame_number - self.track_last_seen.get(track_id, 0) <= 10:
                    current_active += 1
        
        # Update max simultaneous
        confirmed_active = sum(1 for tid in self.confirmed_persons 
                             if frame_number - self.track_last_seen.get(tid, 0) <= 10)
        
        if confirmed_active > self.max_simultaneous_active:
            self.max_simultaneous_active = confirmed_active
        
        return confirmed_active

def process_video_ultra_fast(video_path, session_id):
    """Ultra-fast processing with minimal overhead"""
    try:
        counter = UltraFastPersonCounter()
        
        # Load lightweight model
        model = YOLO("yolo11n.pt")  # Nano model for speed
        model.overrides['verbose'] = False
        
        # Minimal tracker setup
        tracker = DeepSort(
            max_age=15,      # Shorter tracking
            n_init=1,        # Immediate initialization
            max_cosine_distance=0.8,
            nn_budget=50     # Smaller budget
        )
        
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Lower resolution for speed
        target_width = min(640, width)  # Max width 640px
        target_height = int(height * (target_width / width))
        
        # Output setup
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{session_id}_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        
        frame_count = 0
        last_update = 0
        
        socketio.emit('processing_started', {
            'total_frames': total_frames,
            'fps': fps,
            'duration': total_frames / fps if fps > 0 else 0
        }, room=session_id)
        
        # Process every Nth frame for speed (skip frames)
        frame_skip = max(1, int(fps / 10))  # Process ~10 FPS max
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for ultra-fast processing
            if frame_count % frame_skip != 0:
                continue
            
            # Resize frame for speed
            if width > target_width:
                frame = cv2.resize(frame, (target_width, target_height))
            
            # Ultra-fast YOLO detection with minimal settings
            results = model(frame, 
                          conf=0.1,           # Very low confidence
                          classes=[0],        # Person only
                          imgsz=320,          # Small image size for speed
                          max_det=50,         # Limit detections
                          agnostic_nms=True,  # Faster NMS
                          verbose=False)
            
            detections = []
            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        if len(box.xyxy) > 0:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            bbox = [x1, y1, x2, y2]
                            
                            if counter.is_valid_detection(bbox, confidence):
                                detections.append((bbox, confidence, 'person'))
            
            # Update tracker
            tracks = tracker.update_tracks(detections, frame=frame)
            current_active = counter.update_tracks(tracks, frame_count)
            
            # Minimal visualization for speed
            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    l, t, r, b = map(int, track.to_ltrb())
                    
                    # Simple colored boxes
                    color = (0, 255, 0) if track_id in counter.confirmed_persons else (0, 165, 255)
                    cv2.rectangle(frame, (l, t), (r, b), color, 2)
                    cv2.putText(frame, f"{track_id}", (l, t-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Minimal stats overlay
            cv2.putText(frame, f"People: {len(counter.confirmed_persons)} | Max: {counter.max_simultaneous_active}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            out.write(frame)
            
            # Send updates every 10 processed frames
            if frame_count - last_update >= 10 * frame_skip:
                progress = (frame_count / total_frames) * 100
                socketio.emit('processing_progress', {
                    'progress': min(progress, 99),  # Cap at 99% until complete
                    'frame': frame_count,
                    'total_frames': total_frames,
                    'confirmed_count': len(counter.confirmed_persons),
                    'max_simultaneous': counter.max_simultaneous_active,
                    'current_active': current_active
                }, room=session_id)
                last_update = frame_count
        
        cap.release()
        out.release()
        
        # Send final results immediately
        results = {
            'total_confirmed': len(counter.confirmed_persons),
            'max_simultaneous': counter.max_simultaneous_active,
            'total_frames_processed': frame_count,
            'output_video': f"{session_id}_output.mp4",
            'processing_complete': True,
            'processing_time': time.time()
        }
        
        socketio.emit('processing_complete', results, room=session_id)
        
    except Exception as e:
        socketio.emit('processing_error', {'error': str(e)}, room=session_id)

# Preload model for faster startup
print("Loading YOLO model...")
global_model = YOLO("yolo11n.pt")
print("Model loaded!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    session_id = str(uuid.uuid4())
    filename = f"{session_id}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    file.save(filepath)
    
    # Start processing immediately in background
    thread = threading.Thread(target=process_video_ultra_fast, args=(filepath, session_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({'session_id': session_id, 'filename': file.filename})

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')

@socketio.on('join_session')
def handle_join_session(data):
    session_id = data['session_id']
    join_room(session_id)
    emit('joined_session', {'session_id': session_id})

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)  # Debug off for speed