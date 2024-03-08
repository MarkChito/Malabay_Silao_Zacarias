from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import importlib.util

app = Flask(__name__)

# (You can adjust this based on your specific TensorFlow library and version)
pkg = importlib.util.find_spec('tflite_runtime')

from tensorflow.lite.python.interpreter import Interpreter

# Define necessary variables
MODEL_NAME = "model"
GRAPH_NAME = "detect.tflite"
LABELMAP_NAME = "labelmap.txt"
THRESHOLD = 0.5

# Add your specific paths for the model and label map
PATH_TO_CKPT = os.path.join(os.getcwd(), MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(os.getcwd(), MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load your TensorFlow Lite model
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1
outname = output_details[0]['name']

if 'StatefulPartitionedCall' in outname:
    # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Home/Index Page
@app.route('/')
@app.route('/index')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

# Modify the /upload route in your Flask app
@app.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded image
    image_path = os.path.join('static/uploads', file.filename)
    file.save(image_path)

    # Perform object detection
    detections = perform_detection(image_path)

    # Get image dimensions
    image = Image.open(image_path)
    image_width, image_height = image.size

    # Pass the necessary variables to the template
    return render_template('result.html', image_filename=file.filename, detections=detections, image_width=image_width, image_height=image_height)

@app.route("/check_connection")
def check_connection():
    return jsonify({'status': '200'})

def perform_detection(image_path):
    # Load image and resize to expected shape [1xHxWx3]
    image = Image.open(image_path)
    image_rgb = image.convert('RGB')
    image_resized = image_rgb.resize((width, height))

    # Convert image to numpy array
    input_data = np.expand_dims(np.array(image_resized, dtype=np.uint8), axis=0)

    # Normalize pixel values if using a floating model
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    detections = []

    # Loop over all detections and store relevant information
    for i in range(len(scores)):
        if (scores[i] > THRESHOLD) and (scores[i] <= 1.0):
            ymin = int(max(1, (boxes[i][0] * height)))
            xmin = int(max(1, (boxes[i][1] * width)))
            ymax = int(min(height, (boxes[i][2] * height)))
            xmax = int(min(width, (boxes[i][3] * width)))

            object_name = labels[int(classes[i])]
            detections.append({'object_name': object_name, 'confidence': float(scores[i]), 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

    return detections

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)