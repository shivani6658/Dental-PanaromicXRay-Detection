from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
 # Use __name__ (with double underscores)

# Load the trained model
model = load_model("dentalp (3).h5")

# Define labels
LABELS = {0: "Nocaries", 1: "Caries"}

def prepare_image(image):
    """Preprocess the uploaded image for prediction."""
    image = image.resize((224, 224))  # Resize to match model input
    image = image.convert('RGB')     # Ensure 3 channels
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_label = None
    uploaded_image_url = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded.")
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")

        try:
            # Save the uploaded image to the static/uploads directory
            uploaded_image_url = os.path.join("static/upload", file.filename)
            #os.makedirs(os.path.dirname(uploaded_image_url), exist_ok=True)  # Ensure the directory exists
            file.save(uploaded_image_url)

            # Open and preprocess the uploaded image
            image = Image.open(file)
            prepared_image = prepare_image(image)

            # Perform prediction
            prediction = model.predict(prepared_image)
            class_index = int(prediction[0][0] > 0.5)  # Binary classification
            prediction_label = LABELS[class_index]

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html", prediction_label=prediction_label, uploaded_image_url=uploaded_image_url)

if __name__ == "__main__":
    os.makedirs("static/upload", exist_ok=True)  # Ensure the directory exists
    app.run(debug=True)
