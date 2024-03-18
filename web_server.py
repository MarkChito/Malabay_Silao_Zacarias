from flask import Flask, render_template, request, jsonify, redirect
import os
import numpy as np
from PIL import Image
import importlib.util
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import shutil

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"]  = "sqlite:///my_database.db"

db = SQLAlchemy(app)

class My_Database(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    object_name = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return "<Name %r>" % self.id

pkg = importlib.util.find_spec('tflite_runtime')

from tensorflow.lite.python.interpreter import Interpreter

MODEL_NAME = "model"
GRAPH_NAME = "detect.tflite"
LABELMAP_NAME = "labelmap.txt"
THRESHOLD = 0.5

PATH_TO_CKPT = os.path.join(os.getcwd(), MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(os.getcwd(), MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']

if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/unlisted_images', methods=["POST", "GET"])
def unlisted_images():
    if request.method == "POST":
        post_object_name = request.form["object_name"]
        file = request.files['object_image']

        new_data = My_Database(object_name=post_object_name)

        image_path = os.path.join('static/uploads/temp', file.filename)
        file.save(image_path)

        try:
            db.session.add(new_data)
            db.session.commit()

            copy_image(post_object_name, file.filename)

            return redirect("/unlisted_images")
        except:
            return "An error has occured.."
    else:
        data = My_Database.query.all()

        my_data = None

        if data is not None:
            my_data = data
            
        return render_template('unlisted_images.html', data=my_data)

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
    image = Image.open(image_path)
    image_rgb = image.convert('RGB')
    image_resized = image_rgb.resize((width, height))

    input_data = np.expand_dims(np.array(image_resized, dtype=np.uint8), axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    detections = []

    for i in range(len(scores)):
        if (scores[i] > THRESHOLD) and (scores[i] <= 1.0):
            ymin = int(max(1, (boxes[i][0] * height)))
            xmin = int(max(1, (boxes[i][1] * width)))
            ymax = int(min(height, (boxes[i][2] * height)))
            xmax = int(min(width, (boxes[i][3] * width)))

            object_name = labels[int(classes[i])]
            detections.append({'object_name': object_name, 'confidence': float(scores[i]), 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

    return detections

def copy_image(filename, filepath):
    base_dir = "static/unlisted_images"

    found_folder = False

    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name == filename:
                found_folder = True
                folder_path = os.path.join(root, dir_name)
                break

    if not found_folder:
        folder_path = os.path.join(base_dir, filename)
        os.makedirs(folder_path)
    
    image_count = 0
    for _, _, files in os.walk(folder_path):
        for file in files:
            if file.startswith("image_"):
                index = int(file.split("_")[-1].split(".")[0])
                image_count = max(image_count, index)

    image_count += 1
    new_image_name = f"image_{image_count:04d}.jpg"
    shutil.copy("static/uploads/temp/" + filepath, os.path.join(folder_path, new_image_name))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)