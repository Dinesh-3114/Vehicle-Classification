from flask import Flask, render_template, request, send_from_directory
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and processor once
processor = AutoImageProcessor.from_pretrained("malifiahm/vehicle_classification")
model = AutoModelForImageClassification.from_pretrained("malifiahm/vehicle_classification")

# Ensure uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_label = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Load and preprocess image
            image = Image.open(file_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_index = logits.argmax(-1).item()
                predicted_label = model.config.id2label[predicted_class_index]

            # Prepare image URL for HTML
            image_url = f"/uploads/{file.filename}"

    return render_template("index.html", predicted_label=predicted_label, image_path=image_url)

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
