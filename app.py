import sys
import os
from flask import Flask, render_template, request, redirect, url_for
import torch
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Add YOLOv5 directory to the Python path
sys.path.insert(0, r'D:\Panaromic_Xray_Detection.v1i.yolov5pytorch\yolov5')

from utils.general import non_max_suppression, scale_boxes
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv5 model
weights = os.path.join('yolov5', 'runs', 'train', 'exp15', 'weights', 'best.pt')
device = select_device('')
yolo_model = DetectMultiBackend(weights, device=device)
yolo_names = yolo_model.names

# Load MobileNet model
mobilenet_model = load_model(os.path.join('project1_f', 'dentalp (3).h5'))
LABELS = {0: "No Caries", 1: "Caries"}

def prepare_image(image):
    image = image.resize((224, 224))
    image = image.convert('RGB')
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/mobilenet', methods=['GET', 'POST'])
def mobilenet():
    prediction_label = None
    uploaded_image_url = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            uploaded_image_url = os.path.join("static", "upload", file.filename)
            os.makedirs(os.path.dirname(uploaded_image_url), exist_ok=True)
            file.save(uploaded_image_url)

            image = Image.open(file)
            prepared_image = prepare_image(image)
            prediction = mobilenet_model.predict(prepared_image)
            class_index = int(prediction[0][0] > 0.5)
            prediction_label = LABELS[class_index]

    return render_template("mobilenet.html", prediction_label=prediction_label, uploaded_image_url=uploaded_image_url)

@app.route('/yolo', methods=['GET', 'POST'])
def yolo():
    result_image = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img0 = cv2.imread(filepath)
            img = cv2.resize(img0, (640, 640))
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = yolo_model(img)
            pred = non_max_suppression(pred, 0.10, 0.45)

            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        cls_id = int(cls)
                        color = colors[cls_id % len(colors)]
                        label = f'{yolo_names[cls_id]} {conf:.2f}'
                        cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                        cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            output_path = os.path.join(STATIC_FOLDER, 'output.jpg')
            cv2.imwrite(output_path, img0)
            result_image = output_path

    return render_template('yolo.html', result_image=result_image)
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(STATIC_FOLDER, exist_ok=True)
    os.makedirs("static/upload", exist_ok=True)
    app.run(debug=True)