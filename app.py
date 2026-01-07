from flask import Flask, render_template, Response, request, jsonify
import cv2
import json
import os
import numpy as np
import logging
from PIL import Image
from settings.settings import CAMERA, FACE_DETECTION, TRAINING, PATHS

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
camera = None
recognizer = None
capturing = False
recognizing = False
current_name = ""
capture_count = 0
names = {}  # Will store ID -> Name mapping


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_names(filename):
    global names
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                names = json.load(f)
        except:
            names = {}
    return names


def initialize_camera():
    global camera
    if camera is not None:
        camera.release()

    # Tries to open the camera (Index 0 is default)
    camera = cv2.VideoCapture(CAMERA['index'], cv2.CAP_DSHOW)

    if not camera.isOpened():
        logger.error("Camera not accessible")
        return False

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
    return True


def initialize_recognizer():
    global recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists(PATHS['trainer_file']):
        recognizer.read(PATHS['trainer_file'])
        return True
    return False


def generate_frames():
    global camera, capturing, capture_count, recognizing, recognizer, names, current_name

    if camera is None or not camera.isOpened():
        if not initialize_camera():
            with open('static/nocamera.jpg', 'rb') as f:
                frame = f.read()
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            return

    # Use the fix for finding the cascade file
    face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
    
    if face_cascade.empty():
        with open('static/error.jpg', 'rb') as f:
            frame = f.read()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        return

    if not names:
        load_names(PATHS['names_file'])

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECTION['scale_factor'],
            minNeighbors=FACE_DETECTION['min_neighbors'],
            minSize=FACE_DETECTION['min_size']
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # --- CAPTURE LOGIC ---
            if capturing and capture_count < TRAINING['samples_needed']:
                if capture_count % 3 == 0: # Save every 3rd frame to avoid duplicates
                    face_img = gray[y:y+h, x:x+w]
                    
                    # Sanitize name (replace spaces with underscores to prevent file errors)
                    safe_name = current_name.replace(" ", "_").replace("-", "_")
                    
                    # SAVE AS: Name-Count.jpg (Example: Shivam-1.jpg)
                    img_path = f"{PATHS['image_dir']}/{safe_name}-{capture_count+1}.jpg"
                    cv2.imwrite(img_path, face_img)

                capture_count += 1
                
                # Auto-stop when done
                if capture_count >= TRAINING['samples_needed']:
                    capturing = False
                    logger.info("Capture limit reached.")

                cv2.putText(frame, f"{capture_count}/{TRAINING['samples_needed']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

            # --- RECOGNITION LOGIC ---
            elif recognizing and recognizer is not None:
                try:
                    # Predict gives us an ID (Integer)
                    id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                    
                    # Look up the Name associated with that Integer
                    name = names.get(str(id), "Unknown")
                    
                    # Display Name and Confidence
                    conf_text = f"  {round(100 - confidence)}%"
                    cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, conf_text, (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                except:
                    pass

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'


def train_model():
    """Train the model and automatically generate IDs from Names"""
    global recognizer, names

    try:
        logger.info("Starting face recognition training...")

        if not os.path.exists(PATHS['image_dir']):
            return False, "Image directory not found"

        image_files = os.listdir(PATHS['image_dir'])
        if not image_files:
            return False, "No training images found"

        detector = cv2.CascadeClassifier(PATHS['cascade_file'])
        recognizer_local = cv2.face.LBPHFaceRecognizer_create()

        faces = []
        ids = []
        
        # Temporary dictionaries to map Name <-> ID just for this training session
        name_to_id = {}
        id_to_name = {}
        current_id_counter = 0

        for filename in image_files:
            file_path = os.path.join(PATHS['image_dir'], filename)

            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            # PARSE FILENAME: "Shivam-1.jpg" -> Name: "Shivam"
            # We split by '-' or '_' and take the first part
            try:
                # Assuming format Name-Count.jpg or Name_Count.jpg
                if '-' in filename:
                    name_label = filename.split('-')[0]
                elif '_' in filename:
                    # Fallback if user manually named them with underscores
                    name_label = filename.split('_')[0]
                else:
                    continue # Skip files that don't match format
                
                # If this is a new name we haven't seen in this folder yet, assign a new ID
                if name_label not in name_to_id:
                    current_id_counter += 1
                    name_to_id[name_label] = current_id_counter
                    id_to_name[str(current_id_counter)] = name_label
                
                face_id = name_to_id[name_label]

            except Exception as e:
                logger.warning(f"Could not parse filename {filename}: {e}")
                continue

            try:
                img = Image.open(file_path).convert('L')
                img_np = np.array(img, 'uint8')
            except Exception as e:
                continue

            detected_faces = detector.detectMultiScale(img_np)

            if len(detected_faces) > 0:
                (x, y, w, h) = detected_faces[0]
                faces.append(img_np[y:y+h, x:x+w])
                ids.append(face_id)

        if not faces:
            return False, "No valid face data found"

        # Train the model with the Integers
        recognizer_local.train(faces, np.array(ids))
        recognizer_local.write(PATHS['trainer_file'])
        
        # SAVE THE MAPPING (ID -> Name) for the recognizer to use later
        with open(PATHS['names_file'], 'w') as f:
            json.dump(id_to_name, f, indent=4)
            
        # Update global variable
        names = id_to_name
        recognizer = recognizer_local

        logger.info(f"Training completed. Users: {list(name_to_id.keys())}")
        return True, f"Trained on {len(name_to_id)} users"

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False, f"Error: {str(e)}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capturing, current_name, capture_count

    name = request.get_json().get('name')
    if not name:
        return jsonify(success=False, message="Name required")

    create_directory(PATHS['image_dir'])
    
    # We don't generate an ID here anymore. We just use the name.
    current_name = name
    capture_count = 0
    capturing = True

    return jsonify(success=True, message=f"Capturing for {name}")


@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global capturing
    capturing = False
    return jsonify(success=True)


@app.route('/train', methods=['POST'])
def train():
    success, msg = train_model()
    return jsonify(success=success, message=msg)


@app.route('/get_status')
def get_status():
    global capturing, recognizing, capture_count, names
    model_trained = os.path.exists(PATHS['trainer_file'])
    return jsonify({
        'capturing': capturing,
        'recognizing': recognizing,
        'captureCount': capture_count,
        'maxCaptures': TRAINING['samples_needed'],
        'modelTrained': model_trained
    })


@app.route('/get_users')
def get_users():
    global names
    if not names:
        load_names(PATHS['names_file'])
    return jsonify({'success': True, 'users': names})


@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    global recognizing
    if initialize_recognizer():
        recognizing = True
        return jsonify(success=True)
    return jsonify(success=False, message="Train model first")


@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    global recognizing
    recognizing = False
    return jsonify(success=True)


if __name__ == '__main__':
    create_directory(PATHS['image_dir'])
    create_directory('static')

    # Create dummy images if missing
    if not os.path.exists('static/nocamera.jpg'):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Camera Not Available", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite('static/nocamera.jpg', img)

    if not os.path.exists('static/error.jpg'):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Error loading resources", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite('static/error.jpg', img)

    initialize_camera()
    load_names(PATHS['names_file'])
    app.run(debug=True)