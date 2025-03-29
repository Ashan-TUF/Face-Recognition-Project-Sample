from flask import Flask, request, jsonify, render_template
import face_recognition
import numpy as np
import os
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
KNOWN_FACES_DIR = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

known_faces = []
known_names = []

# **Train all known faces**
def train_faces():
    global known_faces, known_names
    known_faces = []
    known_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(('.jpg', '.png')):
            name = os.path.splitext(filename)[0]  # Extract name
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_faces.append(encodings[0])
                known_names.append(name)
    
    print(f"Trained {len(known_faces)} faces: {known_names}")

# **Train on startup**
train_faces()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register_face():
    if 'imagefile' not in request.files or 'name' not in request.form:
        return jsonify({'error': 'Missing file or name'}), 400

    file = request.files['imagefile']
    name = request.form['name'].strip().replace(" ", "_")  # Clean name

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the image with the provided name
    filename = f"{name}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Retrain with the new image
    train_faces()

    return jsonify({'message': f"Face registered as {name}", 'total_faces': len(known_faces)})

@app.route('/detect', methods=['POST'])
def detect_faces():
    if 'imagefile' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['imagefile']
    image = face_recognition.load_image_file(file)

    face_encodings = face_recognition.face_encodings(image)

    results = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # Find the best match
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_names[best_match_index]

        results.append(name)

    return jsonify({'matches': results, 'face_count': len(results)})

if __name__ == '__main__':
    app.run(debug=True)
