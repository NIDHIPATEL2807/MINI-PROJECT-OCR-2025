from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from PIL import Image
import datetime

from mltu.configs import BaseModelConfigs
from utils import ImageToWordModel

app = Flask(__name__)

# ---------------- MODEL PATHS ---------------- #
MODEL_CONFIG_PATH = "/home/aayush/smart_ocr_rpi/Model/Devanagari/202405030509/configs.yaml"

if not os.path.exists(MODEL_CONFIG_PATH):
    raise FileNotFoundError("configs.yaml missing at: " + MODEL_CONFIG_PATH)

configs = BaseModelConfigs.load(MODEL_CONFIG_PATH)
model = ImageToWordModel(model_path=configs.model_path, vocab=configs.vocab)

print("📌 Model Loaded Successfully")
print("📌 Vocab Size:", len(configs.vocab))

# ---------------- OCR ENDPOINT ---------------- #
@app.route("/ocr", methods=["POST"])
def ocr_image():
    print("\n🟢 OCR Request:", datetime.datetime.now().strftime("%H:%M:%S"))

    data = request.get_json()
    image_path = data.get("image_path")

    print("📥 Image Path:", image_path)

    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 400

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"error": "Failed to load image"}), 500

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Dilation to extract lines
    kernel = np.ones((5, 200), np.uint8)
    dilated = cv2.dilate(thresh, kernel, 1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    sorted_contours = [
        c for (_, c) in sorted(zip(bounding_boxes, contours), key=lambda x: x[0][1])
    ]

    final_texts = []

    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y:y+h, x:x+w]

        # Skip small noise
        if roi.shape[0] < 20 or roi.shape[1] < 110:
            continue

        try:
            text = model.predict(roi, language="Devanagari")
            final_texts.append(text)
        except Exception as e:
            print("⚠️ Prediction error:", e)

    full_result = " ".join(final_texts)
    print("📜 OCR Output:", full_result)

    return jsonify({"text": full_result})


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    print("🚀 Custom Model OCR Server running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
