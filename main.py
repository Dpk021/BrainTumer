"""
Flask web app: upload an MRI image and get the predicted tumor class.
Loads the PyTorch checkpoint produced by train.py (models/model.pt).
"""
import os

import torch
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from torchvision import transforms
from werkzeug.utils import secure_filename

from model import IMAGE_SIZE, get_device, load_checkpoint

app = Flask(__name__)

MODEL_PATH = "models/model.pt"
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = get_device()
model, class_labels = load_checkpoint(MODEL_PATH, device)

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # [0, 1], same scaling as training
])


@torch.no_grad()
def predict_tumor(image_path: str):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)  # add batch dimension
    probs = torch.softmax(model(x), dim=1)[0]
    confidence, idx = probs.max(0)
    label = class_labels[idx.item()]
    result = "No Tumor" if label == "notumor" else f"Tumor: {label}"
    return result, confidence.item()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_location = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_location)

            result, confidence = predict_tumor(file_location)
            return render_template(
                "index.html",
                result=result,
                confidence=f"{confidence * 100:.2f}",
                file_path=f"/uploads/{filename}",
            )
    return render_template("index.html", result=None)


@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
