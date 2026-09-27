from flask import Flask, request, render_template, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import json

# ==========================================
# FIX: Handle PyTorch weights_only load issue
# ==========================================
import torch
from ultralytics.nn.tasks import DetectionModel

# Allow PyTorch to load YOLO's DetectionModel class safely
try:
    torch.serialization.add_safe_globals([DetectionModel])
except AttributeError:
    # Handle cases where torch < 2.x doesn't have add_safe_globals
    print("Warning: torch.serialization.add_safe_globals not found. Ensure PyTorch is updated.")
# ==========================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load the model
MODEL_PATH = 'best.pt'
try:
    # Model loading now respects the safe globals added above
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
    print(f"Model classes: {model.names}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def filter_fracture_detections(results):
    """Filter out text detections and keep only fracture-related classes"""
    text_keywords = ['text', 'word', 'character', 'letter', 'digit', 'number']
    filtered_detections = []
        
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id].lower()
                                
                # Skip if class name contains text-related keywords
                if not any(keyword in class_name for keyword in text_keywords):
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'confidence': float(box.conf[0]),
                        'class_id': class_id,
                        'class_name': model.names[class_id]
                    }
                    filtered_detections.append(detection)
        
    return filtered_detections

def draw_detections(image, detections, confidence_threshold=0.5):
    """Draw bounding boxes and labels on the image"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
        
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
        
    for i, detection in enumerate(detections):
        if detection['confidence'] >= confidence_threshold:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
                        
            # Choose color based on class
            color = colors[detection['class_id'] % len(colors)]
                        
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        
            # Prepare label text
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
                        
            # Get text bounding box for background
            bbox_text = draw.textbbox((x1, y1-25), label, font=font)
                        
            # Draw background for text
            draw.rectangle(bbox_text, fill=color)
                        
            # Draw text
            draw.text((x1, y1-25), label, fill=(255, 255, 255), font=font)
        
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please ensure best.pt is in the correct path.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
        
    try:
        # Read and process the image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
                
        # Run detection
        results = model(image)
                
        # Filter detections to exclude text
        detections = filter_fracture_detections(results)
                
        # Draw detections on image
        annotated_image = draw_detections(image.copy(), detections)
                
        # Convert image to base64 for display
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
                
        # Prepare detection results
        detection_results = []
        for detection in detections:
            if detection['confidence'] >= 0.5:  # Only include high confidence detections
                detection_results.append({
                    'class': detection['class_name'],
                    'confidence': round(detection['confidence'], 3),
                    'bbox': [round(x, 1) for x in detection['bbox']]
                })
                
        return jsonify({
            'success': True,
            'image': img_base64,
            'detections': detection_results,
            'total_fractures': len(detection_results)
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Fracture Detection Flask App...")
    print("Make sure your 'best.pt' model file is in the same directory as this script.")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
