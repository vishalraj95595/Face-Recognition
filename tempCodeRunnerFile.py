from flask import Flask, render_template, Response, request, jsonify
import cv2
import json
import os
import numpy as np
import logging
from PIL import Image
import time
from settings.settings import CAMERA, FACE_DETECTION, TRAINING, PATHS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
camera = None
face_cascade = None
recognizer = None
capturing = False
recognizing = False
current_name = ""
face_id = 0
capture_count = 0
names = {}

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Directory created: {directory}")

def get_face_id(directory):
    """Get the next available face ID"""
    try:
        if not os.path.exists(directory):
            return 1
            
        existing_files = os.listdir(directory)
        ids = []
        for f in existing_files:
            parts = f.split('-')
            if len(parts) > 1 and parts[0] == "Users":
                try:
                    ids.append(int(parts[1]))
                except ValueError:
                    continue
        return max(ids) + 1 if ids else 1
    except Exception as e:
        logger.error(f"Error getting face ID: {e}")
        return 1

def save_name(face_id, face_name, filename):
    """Save name and ID mapping to JSON file"""
    try:
        names = {}
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                try:
                    names = json.load(file)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse {filename}, creating new file")
        
        names[str(face_id)] = face_name
        
        with open(filename, 'w') as file:
            json.dump(names, file, indent=4)
            
        logger.info(f"Saved name mapping: ID {face_id} -> {face_name}")
    except Exception as e:
        logger.error(f"Error saving name mapping: {e}")
        raise

def load_names(filename):
    """Load name mappings from JSON file"""
    global names
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                try:
                    names = json.load(file)
                    logger.info(f"Loaded {len(names)} names from {filename}")
                    return names
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse {filename}")
                    return {}
        return {}
    except Exception as e:
        logger.error(f"Error loading names: {e}")
        return {}

def initialize_camera():
    """Initialize the camera with error handling"""
    global camera
    try:
        if camera is not None:
            camera.release()
            
        camera = cv2.VideoCapture(CAMERA['index'])
        if not camera.isOpened():
            logger.error("Could not open webcam")
            return False
            
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        return True
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return False

def initialize_recognizer():
    """Initialize the face recognizer"""
    global recognizer
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.exists(PATHS['trainer_file']):
            recognizer.read(PATHS['trainer_file'])
            logger.info("Face recognizer loaded from file")
            return True
        else:
            logger.warning("Trainer file not found")
            return False
    except Exception as e:
        logger.error(f"Error initializing recognizer: {e}")
        return False

def generate_frames():
    """Generate camera frames for streaming"""
    global camera, capturing, face_id, capture_count, recognizing, recognizer, names
    
    if camera is None or not camera.isOpened():
        if not initialize_camera():
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   open('src/static/nocamera.jpg', 'rb').read() + b'\r\n')
            return
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
    if face_cascade.empty():
        logger.error("Error loading cascade classifier")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               open('src/static/error.jpg', 'rb').read() + b'\r\n')
        return
    
    # Load recognizer if needed
    if recognizing and (recognizer is None):
        if not initialize_recognizer():
            recognizing = False
    
    # Load names if not already loaded
    if not names:
        names = load_names(PATHS['names_file'])
    
    while True:
        success, frame = camera.read()
        if not success:
            logger.warning("Failed to grab frame")
            # Try to reinitialize camera
            if initialize_camera():
                continue
            else:
                break
                
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECTION['scale_factor'],
            minNeighbors=FACE_DETECTION['min_neighbors'],
            minSize=FACE_DETECTION['min_size']
        )
        
        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            if capturing and capture_count < TRAINING['samples_needed']:
                # Save face image for training
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Only save images at certain intervals to prevent duplicates
                if capture_count % 3 == 0:
                    face_img = gray[y:y+h, x:x+w]
                    img_path = f'{PATHS["image_dir"]}/Users-{face_id}-{capture_count+1}.jpg'
                    cv2.imwrite(img_path, face_img)
                
                capture_count += 1
                
                progress = f"Capturing: {capture_count}/{TRAINING['samples_needed']}"
                cv2.putText(frame, progress, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                           
                if capture_count >= TRAINING['samples_needed']:
                    capturing = False
                    logger.info(f"Completed capturing {TRAINING['samples_needed']} images")
            
            elif recognizing and recognizer is not None:
                # Recognize face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                try:
                    id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                    
                    # Check confidence and display result
                    if confidence <= 100:
                        name = names.get(str(id), "Unknown")
                        confidence_text = f"{int(100 - confidence)}%"
                        
                        # Display name and confidence
                        cv2.putText(frame, name, (x+5, y-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(frame, confidence_text, (x+5, y+h+25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                except Exception as e:
                    # Don't log every frame error to prevent log flooding
                    pass
            else:
                # Just show rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add help text to the frame
        if not capturing and not recognizing:
            cv2.putText(frame, "Ready: Select operation below", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def train_model():
    """Train the face recognition model"""
    global recognizer
    try:
        logger.info("Starting face recognition training...")
        
        # Check if we have training data
        if not os.path.exists(PATHS['image_dir']) or not os.listdir(PATHS['image_dir']):
            logger.warning("No training data found")
            return False, "No training data found"
        
        # Initialize face recognizer
        local_recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier(PATHS['cascade_file'])
        
        # Get all training images
        image_paths = [os.path.join(PATHS['image_dir'], f) for f in os.listdir(PATHS['image_dir'])]
        
        # Prepare training data
        faces = []
        ids = []
        
        # Process each image
        for image_path in image_paths:
            try:
                # Load image and convert to grayscale
                img = Image.open(image_path).convert('L')
                img_numpy = np.array(img, 'uint8')
                
                # Extract ID from the file name
                id = int(os.path.split(image_path)[-1].split('-')[1])
                
                # Detect faces in the image
                face_rects = detector.detectMultiScale(img_numpy)
                
                # If no face detected, skip this image
                if len(face_rects) == 0:
                    logger.warning(f"No face detected in {image_path}")
                    continue
                
                # Use the first detected face
                (x, y, w, h) = face_rects[0]
                faces.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
                
            except Exception as e:
                logger.warning(f"Error processing {image_path}: {e}")
                continue
        
        if not faces:
            logger.warning("No valid faces found in training data")
            return False, "No valid faces found in training data"
        
        # Train the model
        logger.info(f"Training model with {len(faces)} images")
        local_recognizer.train(faces, np.array(ids))
        
        # Save the model
        local_recognizer.write(PATHS['trainer_file'])
        logger.info(f"Model trained and saved with {len(np.unique(ids))} unique faces")
        
        # Update global recognizer
        recognizer = local_recognizer
        
        return True, f"Model trained with {len(np.unique(ids))} unique faces"
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        return False, f"Training error: {e}"

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    """Start capturing images for training"""
    global capturing, current_name, face_id, capture_count
    
    data = request.get_json()
    name = data.get('name', '')
    
    if not name:
        return jsonify({'success': False, 'message': 'Name cannot be empty'})
    
    # Create directory if needed
    create_directory(PATHS['image_dir'])
    
    # Set up for capture
    current_name = name
    face_id = get_face_id(PATHS['image_dir'])
    save_name(face_id, current_name, PATHS['names_file'])
    
    # Reset counter and start capturing
    capture_count = 0
    capturing = True
    
    return jsonify({'success': True, 'message': f'Starting capture for {name} (ID: {face_id})'})

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    """Stop capturing images"""
    global capturing
    capturing = False
    return jsonify({'success': True, 'message': 'Capture stopped', 'count': capture_count})

@app.route('/train', methods=['POST'])
def train():
    """Train the model with captured images"""
    success, message = train_model()
    return jsonify({'success': success, 'message': message})

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    """Start face recognition"""
    global recognizing, recognizer
    
    if not os.path.exists(PATHS['trainer_file']):
        return jsonify({'success': False, 'message': 'Model not trained. Please train first.'})
    
    # Initialize recognizer if needed
    if recognizer is None:
        if not initialize_recognizer():
            return jsonify({'success': False, 'message': 'Failed to initialize recognizer'})
    
    recognizing = True
    return jsonify({'success': True, 'message': 'Recognition started'})

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    """Stop face recognition"""
    global recognizing
    recognizing = False
    return jsonify({'success': True, 'message': 'Recognition stopped'})

@app.route('/get_status', methods=['GET'])
def get_status():
    """Get current status"""
    return jsonify({
        'capturing': capturing,
        'recognizing': recognizing,
        'captureCount': capture_count,
        'maxCaptures': TRAINING['samples_needed'],
        'currentName': current_name,
        'faceId': face_id,
        'modelTrained': os.path.exists(PATHS['trainer_file'])
    })

@app.route('/get_users', methods=['GET'])
def get_users():
    """Get list of trained users"""
    users = load_names(PATHS['names_file'])
    return jsonify({'success': True, 'users': users})

if __name__ == '__main__':
    # Make sure necessary directories exist
    create_directory(PATHS['image_dir'])
    create_directory('src/static')
    
    # Create placeholder images if they don't exist
    if not os.path.exists('src/static/nocamera.jpg'):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Camera not available", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite('src/static/nocamera.jpg', img)
        
    if not os.path.exists('src/static/error.jpg'):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Error loading resources", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite('src/static/error.jpg', img)
    
    # Initialize camera at startup
    initialize_camera()
    
    # Load names
    load_names(PATHS['names_file'])
    
    # Start the Flask app
    app.run(debug=True)