import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from mltu.configs import BaseModelConfigs
from utils import ImageToWordModel

# -----------------------------
#   FLASK + CORS SETUP
# -----------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# -----------------------------
#   MODEL SETUP
# -----------------------------
DEVA_CONFIG_PATH = "Model/Devanagari/202405030509/configs.yaml"

if not os.path.exists(DEVA_CONFIG_PATH):
    raise FileNotFoundError("configs.yaml not found at " + DEVA_CONFIG_PATH)

configs = BaseModelConfigs.load(DEVA_CONFIG_PATH)
model = ImageToWordModel(model_path=configs.model_path, vocab=configs.vocab)

print("📌 Model Loaded Successfully")
print("📌 Vocab Size:", len(configs.vocab))

# Upload folder
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -----------------------------
#   HOME PAGE
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------
#   PREDICT API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    print("\n---- DEBUG ----")
    print("request.files:", request.files)
    print("request.form:", request.form)
    print("----------------")

    # Check for file
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    uploaded_file = request.files["image"]
    file_bytes = uploaded_file.read()

    # Decode image
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # -----------------------------
    # PREPROCESSING
    # -----------------------------
    # Convert to grayscale
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Dilation
    kernel = np.ones((5, 200), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # Ensure grayscale
    if len(img_dilation.shape) >= 3:
        img_dilation = cv2.cvtColor(img_dilation, cv2.COLOR_BGRA2GRAY)

    img_dilation = img_dilation.astype("uint8")

    # -----------------------------
    # FIND CONTOURS
    # -----------------------------
    try:
        contours, _ = cv2.findContours(
            img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    except Exception as e:
        return jsonify({"error": "findContours failed", "details": str(e)}), 500

    # Sort contours top → bottom
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    sorted_contours = [
        c for (_, c) in sorted(zip(bounding_boxes, contours), key=lambda x: x[0][1])
    ]

    # -----------------------------
    # OCR PREDICT
    # -----------------------------
    predicted_texts = []

    for ctr in sorted_contours:
        x, y, w, h = cv2.boundingRect(ctr)
        roi = img[y:y + h, x:x + w]

        # Skip tiny regions (noise)
        if roi.shape[0] <= 20 or roi.shape[1] <= 110:
            continue

        # Ensure ROI is 3-channel BGR
        if roi.ndim == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        elif roi.ndim == 3 and roi.shape[2] == 4:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)

        elif roi.ndim == 3 and roi.shape[2] == 1:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # Model Prediction
        try:
            prediction = model.predict(roi, language="Devanagari")
        except Exception as e:
            return jsonify({"error": "model.predict failed", "details": str(e)}), 500

        predicted_texts.append((x, y, prediction))

    # Sort final text in reading order
    predicted_texts_sorted = sorted(predicted_texts, key=lambda t: (t[1], t[0]))
    final_predictions = [t[2] for t in predicted_texts_sorted]

    return jsonify(final_predictions)


# -----------------------------
#   MAIN
# -----------------------------
if __name__ == "__main__":
    print("🚀 OCR Server running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
