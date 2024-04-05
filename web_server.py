from flask import Flask, render_template, request, jsonify, redirect
from tensorflow.lite.python.interpreter import Interpreter
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
from PIL import Image

import numpy as np
import shutil
import os
import cv2

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"]  = "sqlite:///my_database.db"

db = SQLAlchemy(app)

class My_Database(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    object_name = db.Column(db.String(255), nullable=False)
    
    def __repr__(self):
        return "<Name %r>" % self.id

with app.app_context():
    db.create_all()

PATH_TO_CKPT = os.path.join(os.getcwd(), "model", "detect.tflite")
PATH_TO_LABELS = os.path.join(os.getcwd(), "model", "labelmap.txt")

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

boxes_idx, classes_idx, scores_idx = 1, 3, 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/result', methods=['POST'])
def result():
    file = request.files['file']

    image_path = os.path.join('static/uploads', file.filename)

    file.save(image_path)

    detections = perform_detection(image_path)

    return render_template('result.html', image_filename=file.filename, detections=detections)

@app.route("/check_connection")
def check_connection():
    return jsonify({'status': '200'})

@app.route('/unlisted_images', methods=["POST", "GET"])
def unlisted_images():
    if request.method == "POST":
        object_name = request.form["object_name"]
        file = request.files['object_image']

        image_path = os.path.join('static/uploads/temp', file.filename)

        file.save(image_path)

        data = My_Database(object_name=object_name)

        db.session.add(data)
        db.session.commit()

        copy_image(object_name, file.filename)

        return redirect("/unlisted_images")
    else:
        data = db.session.query(My_Database).group_by(My_Database.object_name).order_by(My_Database.date_created.desc()).all()

        my_data = None

        if data is not None:
            my_data = data
            
        return render_template('unlisted_images.html', data=my_data)

@app.route("/view_images", methods=["POST"])
def view_images():
    post_image_folder = request.form["folder_name"]
    
    image_folder = 'static/unlisted_images/' + post_image_folder
    filenames = os.listdir(image_folder)

    return render_template('view_images.html', filenames=filenames, title=post_image_folder)

def perform_detection(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    input_data = np.expand_dims(np.array(image_resized, dtype=np.uint8), axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    detections = []

    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            detections.append({'object_name': object_name, 'confidence': int(scores[i] * 100), 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

    image_fn = os.path.basename(image_path)
    image_savepath = os.path.join(os.getcwd(), "static/uploads", image_fn)

    cv2.imwrite(image_savepath, image)
    
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

def insert_db_data(data):
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)