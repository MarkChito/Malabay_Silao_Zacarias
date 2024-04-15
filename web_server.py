from database import Unlisted_Images, Contact_Us_Messages, Newsletter_List, User_Accounts, Upload_History, db, app
from flask import render_template, request, jsonify, redirect
from tensorflow.lite.python.interpreter import Interpreter
from session import Session

import numpy as np
import difflib
import shutil
import cv2
import os

session = Session()

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
    notification = session.get("notification")

    template = render_template('login.html', notification=notification)
    
    session.unset("notification")

    return template

@app.route('/detect')
def detect():
    return render_template('detect.html', notification=None)

@app.route('/result', methods=['POST'])
def result():
    file = request.files['file']

    image_path = os.path.join('static/uploads', file.filename)

    file.save(image_path)

    detections = perform_detection(image_path, "image_detection")

    return render_template('result.html', image_filename=file.filename, detections=detections, notification=None)

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

        data = Unlisted_Images(object_name=object_name)

        data.insert(data)

        copy_image(object_name, file.filename)

        perform_detection(image_path, "unlisted_images")

        copy_image_with_detection(object_name, file.filename)

        os.remove(image_path)

        session.set("notification", {"title": "Success!", "text": "An image has been successfully added to unlisted images.", "icon": "success"})

        return redirect("/unlisted_images")
    
    else:
        data = db.session.query(Unlisted_Images).group_by(Unlisted_Images.object_name).order_by(Unlisted_Images.date_created.desc()).all()

        my_data = None

        if data is not None:
            my_data = data
        
        notification = session.get("notification")

        template = render_template('unlisted_images.html', data=my_data, notification=notification)

        session.unset("notification")

        return template

@app.route("/view_images", methods=["POST"])
def view_images():
    post_image_folder = request.form["folder_name"]
    
    image_folder = 'static/unlisted_images/with_detections/' + post_image_folder
    filenames = os.listdir(image_folder)

    return render_template('view_images.html', filenames=filenames, title=post_image_folder, notification=None)

@app.route("/archive_folder", methods=["POST"])
def archive_folder():
    post_image_folder = request.form["archive_folder_folder_name"]

    db = Unlisted_Images()

    db.delete(post_image_folder)

    source_folder_with_detections = "static/unlisted_images/with_detections/" + post_image_folder
    source_folder_without_detections = "static/unlisted_images/without_detections/" + post_image_folder
    
    destination_folder_with_detections = "static/archived_images/with_detections" 
    destination_folder_without_detections = "static/archived_images/without_detections" 

    shutil.move(source_folder_with_detections, destination_folder_with_detections)
    shutil.move(source_folder_without_detections, destination_folder_without_detections)

    session.set("notification", {"title": "Success!", "text": post_image_folder + " folder has been archived.", "icon": "success"})

    return redirect("/unlisted_images")

@app.route("/update_folder_name", methods=["POST"])
def update_folder_name():
    post_image_folder = request.form["update_folder_name_folder_name"]
    post_old_image_folder = request.form["update_folder_name_old_folder_name"]

    db = Unlisted_Images()

    is_record_available = False

    if post_image_folder != post_old_image_folder:
        is_record_available = db.is_record_available(post_image_folder)

    if not is_record_available:
        db.update(post_old_image_folder, post_image_folder)

        rename_folder(post_old_image_folder, post_image_folder)
        rename_folder_with_detections(post_old_image_folder, post_image_folder)

        session.set("notification", {"title": "Success!", "text": post_old_image_folder + " folder has been renamed to " + post_image_folder + ".", "icon": "success"})
    else:
        session.set("notification", {"title": "Oops...", "text": post_image_folder + " folder is already in the record.", "icon": "error"})

    return redirect("/unlisted_images")

@app.route("/view_images_with_detections", methods=["POST"])
def view_images_with_detections():
    post_image_folder = request.form["folder_name"]
    
    image_folder = 'static/unlisted_images/with_detections/' + post_image_folder
    filenames = os.listdir(image_folder)

    return render_template('view_images_with_detections.html', filenames=filenames, title=post_image_folder, notification=None)

@app.route('/contact_us_message', methods=['POST'])
def contact_us_message():
    contact_us_name = request.form["contact_us_name"]
    contact_us_email = request.form["contact_us_email"]
    contact_us_subject = request.form["contact_us_subject"]
    contact_us_message = request.form["contact_us_message"]

    data = Contact_Us_Messages(name=contact_us_name, email=contact_us_email, subject=contact_us_subject, message=contact_us_message)
    
    data.insert(data)

    session.set("notification", {"title": "Success!", "text": "Your message has been sent!", "icon": "success"})

    return redirect("/#contact")

@app.route('/newsletter_list', methods=['POST'])
def newsletter_list():
    newsletter_email = request.form["newsletter_email"]

    data = Newsletter_List(email=newsletter_email)
    
    data.insert(data)

    session.set("notification", {"title": "Success!", "text": "Thank you for subscribing to our newsletter.", "icon": "success"})

    return redirect("/#contact")

@app.route('/browser_error')
def browser_error():
    notification = session.get("notification")

    template = render_template('browser_error.html', notification=notification)
    
    session.unset("notification")

    return template

@app.route('/check_username', methods=['POST'])
def check_username():
    post_username = request.form["username"]

    db = User_Accounts()

    is_record_available = db.is_record_available(post_username)

    return jsonify(is_record_available)

@app.route('/register', methods=['POST'])
def register():
    post_name = request.form["name"]
    post_username = request.form["username"]
    post_password = request.form["password"]

    db = User_Accounts()

    hashed_password = db.password_hash(post_password)

    data = User_Accounts(name=post_name, username=post_username, password=hashed_password)

    data.insert(data)

    session.set("notification", {"title": "Success!", "text": "Your account has been successfully registered into the database.", "icon": "success"})

    return jsonify(True)

@app.route('/login', methods=['POST'])
def login():
    username = request.form["username"]
    password = request.form["password"]

    db = User_Accounts()

    user_data = db.is_record_available(username)

    if user_data:
        hashed_password = user_data.password

        if db.password_verify(password, hashed_password):
            session.set("notification", {"title": "Success!", "text": "Welcome, " + user_data.name + "!", "icon": "success"})
        else:
            session.set("notification", {"title": "Oops...", "text": "Invalid Username or Password.", "icon": "error"})
    else:
        session.set("notification", {"title": "Oops...", "text": "Invalid Username or Password.", "icon": "error"})

    array_user_data = {
        "id": user_data.id,
        "name": user_data.name,
        "username": user_data.username,
        "password": user_data.password.decode('utf-8'),
    }

    return jsonify(array_user_data)

def perform_detection(image_path, page):
    # ================ Start Image Preprocessing ================ #
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    input_data = np.expand_dims(np.array(image_resized, dtype=np.uint8), axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    # ================ End Image Preprocessing ================ #

    # ================ Start Tensorflow Interpreter ================ #
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    # ================ Start Tensorflow Interpreter ================ #

    # ================ Start Initialization of Detection Process ================ #
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
    # ================ End Initialization of Detection Process ================ #

    detections = []

    # ================ Start Detection Process ================ #
    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
            # ================ Start Detect Bounding Boxes of Trained Images ================ #
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            # ================ End Detect Bounding Boxes of Trained Images ================ #
            
            # ================ Start Intialize Bounding Box to ROI (Region of Interest) ================ #
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            # ================ End Intialize Bounding Box to ROI (Region of Interest) ================ #

            # ================ Start Intialize Labeling the ROI ================ #
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            # ================ End Intialize Labeling the ROI ================ #
            
            # ================ Start Labeling to ROI (Region of Interest) ================ #
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            # ================ End Labeling to ROI (Region of Interest) ================ #

            # ================ Start Set all detection matrix to "detections" variable to be used in the webpage display ================ #
            detections.append({'object_name': object_name, 'confidence': int(scores[i] * 100), 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
            # ================ End Set all detection matrix to "detections" variable to be used in the webpage display ================ #
    # ================ Start Detection Process ================ #

    if page == "image_detection":
        base_dir = "static/uploads"
    else:
        base_dir = "static/uploads/temp"

    image_fn = os.path.basename(image_path)
    image_savepath = os.path.join(os.getcwd(), base_dir, image_fn)

    # ================ Start Save Image to the selected folder ================ #
    cv2.imwrite(image_savepath, image)
    # ================ End Save Image to the selected folder ================ #

    return detections

def copy_image(filename, filepath):
    base_dir = "static/unlisted_images/without_detections"

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

def copy_image_with_detection(filename, filepath):
    base_dir = "static/unlisted_images/with_detections"

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

def similar(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

def rename_folder(old_folder_name, new_folder_name):
    root_dir = "static/unlisted_images/without_detections"

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        
        if os.path.isdir(folder_path):
            similarity = similar(old_folder_name, folder_name)
            
            if similarity >= 0.6:
                new_folder_path = os.path.join(root_dir, new_folder_name)
        
                os.rename(folder_path, new_folder_path)

                return

def rename_folder_with_detections(old_folder_name, new_folder_name):
    root_dir = "static/unlisted_images/with_detections"

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        
        if os.path.isdir(folder_path):
            similarity = similar(old_folder_name, folder_name)
            
            if similarity >= 0.6:
                new_folder_path = os.path.join(root_dir, new_folder_name)
        
                os.rename(folder_path, new_folder_path)

                return

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)