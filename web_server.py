from database import Unlisted_Images, Contact_Us_Messages, Newsletter_List, User_Accounts, Upload_History, Church_Details, db, app
from flask import render_template, request, jsonify, redirect
from tensorflow.lite.python.interpreter import Interpreter
from image_hash import Image_Hash
from session import Session

import numpy as np
import difflib
import shutil
import cv2
import os

session = Session()

PATH_TO_CKPT = os.path.join(os.getcwd(), "model", "detect.tflite")
PATH_TO_LABELS = os.path.join(os.getcwd(), "model", "labelmap.txt")
PATH_TO_HASHED_IMAGES = os.path.join(os.getcwd(), "model", "image_hashes.json")

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

@app.route('/home')
def user_view():
    notification = session.get("notification")

    template = render_template('user_view.html', notification=notification)
    
    session.unset("notification")

    return template

@app.route('/detect')
def detect():
    return render_template('detect.html', notification=None)

@app.route('/result', methods=['POST'])
def result():
    user_id = request.form["result_user_id"]
    file = request.files['file']

    image_path = os.path.join('static/uploads', file.filename)

    file.save(image_path)

    church_code = perform_detection(image_path, "image_detection", user_id)

    image_name = str(user_id) + "_" + file.filename

    church_name = "None"
    location = "None"
    building_capacity = "None"
    date_built = "None"
    short_description = "None"

    if church_code:
        church_data = Church_Details()

        details = church_data.select(church_code)

        church_name = details.church_name
        location = details.location
        building_capacity = details.building_capacity
        date_built = details.date_built
        short_description = details.short_description

    data = Upload_History(user_id=user_id, image_name=image_name, church_name=church_name, location=location, building_capacity=building_capacity, date_built=date_built, short_description=short_description)

    data.insert(data)

    return render_template('result.html', image_filename=image_name, church_name=church_name, location=location, building_capacity=building_capacity, date_built=date_built, short_description=short_description, notification=None)

@app.route("/check_connection")
def check_connection():
    return jsonify({'status': '200'})

@app.route('/unregistered_dataset', methods=["POST", "GET"])
def unregistered_dataset():
    if request.method == "POST":
        user_id = request.form["list_new_image_user_id"]
        object_name = request.form["list_new_image_object_name"]
        location = request.form["list_new_image_location"]
        file = request.files['list_new_image_object_image']

        my_db = Unlisted_Images()

        errors = 0

        data = my_db.is_record_available(object_name)

        if data:
            if data.location != location:
                errors += 1

                session.set("notification", {"title": "Oops...", "text": "This object was initially set to location "+ data.location +". Please contact your develper for assistance.", "icon": "error"})
        
        if errors == 0:
            image_path = os.path.join('static/uploads/temp', file.filename)

            file.save(image_path)

            data = Unlisted_Images(user_id=user_id, object_name=object_name, location=location)

            data.insert(data)

            copy_image(object_name, file.filename)

            perform_detection(image_path, "unlisted_images", user_id)

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
    post_user_id = request.form["user_id"]
    post_image_folder = request.form["folder_name"]
    
    image_folder = 'static/unlisted_images/with_detections/' + post_image_folder
    filenames = os.listdir(image_folder)

    return render_template('view_images.html', filenames=filenames, title=post_image_folder, user_id=post_user_id, notification=None)

@app.route("/upload_history")
def upload_history():
    user_id = request.args.get("user_id")
    
    db = Upload_History()

    data = None

    if user_id == "1":
        data = db.admin_select()
    else:
        data = db.user_select(user_id)

    return render_template('upload_history.html', data=data, notification=None)

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

    response = False

    if (not db.is_record_available(post_username)):
        hashed_password = db.password_hash(post_password)

        data = User_Accounts(name=post_name, username=post_username, password=hashed_password, user_type='student')

        data.insert(data)

        session.set("notification", {"title": "Success!", "text": "Your account has been successfully registered into the database.", "icon": "success"})

        response = True
    else:
        response = False

    return jsonify(response)

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

            array_user_data = {
                "id": user_data.id,
                "name": user_data.name,
                "username": user_data.username,
                "password": user_data.password.decode('utf-8'),
                "user_type": user_data.user_type,
            }

            return jsonify(array_user_data)
        else:
            session.set("notification", {"title": "Oops...", "text": "Invalid Username or Password.", "icon": "error"})

            return jsonify(False)
    else:
        session.set("notification", {"title": "Oops...", "text": "Invalid Username or Password.", "icon": "error"})

        return jsonify(False)

@app.route('/logout', methods=['POST'])
def logout():
    session.set("notification", {"title": "Success!", "text": "You had been signed out!", "icon": "success"})

    return jsonify(True)

@app.route('/invalid_login')
def invalid_login():
    session.set("notification", {"title": "Oops...", "text": "You must login first!", "icon": "error"})

    return redirect("/")

@app.route('/get_upload_data', methods=['POST'])
def get_upload_data():
    user_id = request.form["user_id"]

    db = Upload_History()

    data = db.user_select(user_id)

    if (data):
        return jsonify(True)
    else:
        return jsonify(False)

def perform_detection(image_path, page, user_id):
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

    church_code = None

    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            db = Church_Details()

            model_object_name = labels[int(classes[i])]
            db_object_name = db.select(model_object_name)

            if db_object_name:
                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                object_name = db_object_name.church_name
                label = '%s' % (object_name)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                
                cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                church_code = db_object_name.church_code
            
            else:
                if model_object_name == "vegetation_damage":
                    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2)
                if model_object_name == "water_damage_or_mold":
                    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2)
                if model_object_name == "unfinished_paint":
                    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255, 255, 0), 2)

    if page == "image_detection":
        base_dir = "static/uploads"

        image_fn = str(user_id) + "_" + os.path.basename(image_path)
    else:
        base_dir = "static/uploads/temp"

        image_fn = os.path.basename(image_path)

    image_savepath = os.path.join(os.getcwd(), base_dir, image_fn)

    cv2.imwrite(image_savepath, image)

    return church_code

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