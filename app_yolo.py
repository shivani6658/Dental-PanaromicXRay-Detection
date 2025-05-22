from flask import Flask, render_template, request
import os
import torch
import numpy as np
import cv2
import sys
from PIL import Image

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 🔁 Add path to local yolov5 repo
sys.path.append(r'D:\Panaromic_Xray_Detection.v1i.yolov5pytorch\yolov5')

# 🔁 YOLOv5 local imports
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# ✅ Load model from local path
weights = r'D:\Panaromic_Xray_Detection.v1i.yolov5pytorch\yolov5\runs\train\exp15\weights\best.pt'
device = select_device('')
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt

# Define class colors
class_colors = {
    'Deep Dental Caries': (255, 0, 0),  # Red
    'Dental Caries': (0, 255, 0),       # Green
    'Impacted Teeth': (0, 0, 255),      # Blue
    'Missing Teeth': (255, 255, 0),     # Yellow
    'Restoration': (255, 165, 0)        # Orange
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Read and prepare image
            img0 = cv2.imread(filepath)  # BGR
            img = cv2.resize(img0, (640, 640))
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB -> CHW
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img)
            pred = non_max_suppression(pred, 0.10, 0.45)

            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        
                        # Assign color based on class
                        color = class_colors.get(names[int(cls)], (0, 255, 255))  # Default to Cyan
                        
                        # Draw bounding box and label with the class color
                        cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                        cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save output image
            output_path = os.path.join(STATIC_FOLDER, 'output.jpg')
            cv2.imwrite(output_path, img0)

            return render_template('index.html', result_image='static/output.jpg')

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(STATIC_FOLDER, exist_ok=True)
    app.run(debug=True)
