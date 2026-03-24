# # # ============================================================
# # # Launch interactive Gradio app directly in Colab
# # # ============================================================
# # import gradio as gr

# # def classify_land_use(input_image):
# #     """
# #     Gradio inference function.
# #     input_image: numpy array from Gradio (H, W, 3) uint8
# #     """
# #     if input_image is None:
# #         return "No image provided", {}

# #     # Resize and preprocess
# #     IMG_H, IMG_W = CONFIG['img_size']
# #     img_resized = cv2.resize(input_image, (IMG_W, IMG_H))
# #     img_input = np.expand_dims(img_resized.astype(np.float32), axis=0)

# #     # Predict
# #     probs = model.predict(img_input, verbose=0)[0]
# #     pred_idx = np.argmax(probs)
# #     pred_class = CLASS_NAMES[pred_idx]
# #     confidence = probs[pred_idx]

# #     # Format output
# #     label = f"{pred_class} ({confidence*100:.1f}%)"
# #     prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

# #     return label, prob_dict

# # # Build Gradio interface
# # iface = gr.Interface(
# #     fn=classify_land_use,
# #     inputs=gr.Image(label="Upload Satellite Image", type='numpy'),
# #     outputs=[
# #         gr.Label(label="Predicted Class"),
# #         gr.Label(label="All Class Probabilities", num_top_classes=10)
# #     ],
# #     title="🛰️ LULC Classifier — EfficientNetB3",
# #     description=(
    
# #         "Upload a Sentinel-2 satellite image to classify its Land Use / Land Cover type.\n"
# #         "**Classes**: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, "
# #         "Pasture, PermanentCrop, Residential, River, SeaLake\n"
# #         # f"**Model Accuracy**: {test_acc*100:.2f}%"
# #     ),
# #     theme='soft',
# #     allow_flagging='never'
# # )

# # print("🚀 Launching Gradio app...")
# # iface.launch(share=True, debug=False)

# # new code 

# # import gradio as gr
# # import numpy as np
# # from tensorflow.keras.models import load_model
# # from PIL import Image

# # model = load_model('lulc_efficientnetb3.keras')

# # class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 
# #                'Highway', 'Industrial', 'Pasture', 
# #                'PermanentCrop', 'Residential', 'River', 'SeaLake']

# # def predict(image):
# #     image = image.resize((224, 224))
# #     image = np.array(image) / 255.0
# #     image = np.expand_dims(image, axis=0)

# #     prediction = model.predict(image)
# #     return class_names[np.argmax(prediction)]

# # interface = gr.Interface(
# #     fn=predict,
# #     inputs=gr.Image(type="pil"),
# #     outputs="text",
# #     title="🌍 Land Use Land Cover Classification",
# #     description="Upload satellite image to classify land type"
# # )

# # interface.launch()

# import gradio as gr
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model

# # ==============================
# # Load model
# # ==============================
# model = load_model("lulc_efficientnetb3.keras")

# # ==============================
# # Config (IMPORTANT: define this)
# # ==============================
# CONFIG = {
#     "img_size": (224, 224)   # change if your model uses different size
# }

# CLASS_NAMES = [
#     "AnnualCrop", "Forest", "HerbaceousVegetation",
#     "Highway", "Industrial", "Pasture",
#     "PermanentCrop", "Residential", "River", "SeaLake"
# ]

# # ==============================
# # Prediction Function
# # ==============================
# def classify_land_use(input_image):
#     if input_image is None:
#         return "No image provided", {}

#     IMG_H, IMG_W = CONFIG['img_size']

#     # Resize + preprocess
#     img_resized = cv2.resize(input_image, (IMG_W, IMG_H))
#     img_input = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

#     # Predict
#     probs = model.predict(img_input, verbose=0)[0]
#     pred_idx = np.argmax(probs)
#     pred_class = CLASS_NAMES[pred_idx]
#     confidence = probs[pred_idx]

#     label = f"{pred_class} ({confidence*100:.1f}%)"
#     prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

#     return label, prob_dict

# # ==============================
# # Gradio UI
# # ==============================
# iface = gr.Interface(
#     fn=classify_land_use,
#     inputs=gr.Image(label="Upload Satellite Image", type="numpy"),
#     outputs=[
#         gr.Label(label="Predicted Class"),
#         gr.Label(label="All Class Probabilities", num_top_classes=10)
#     ],
#     title="🛰️ LULC Classifier — EfficientNetB3",
#     description="""
# Upload a Sentinel-2 satellite image to classify its Land Use / Land Cover type.

# Classes:
# AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
# Pasture, PermanentCrop, Residential, River, SeaLake
# """,
#     theme="soft",
#     allow_flagging="never"
# )

# # ==============================
# # Launch (IMPORTANT CHANGE)
# # ==============================
# if __name__ == "__main__":
#     iface.launch()

from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
# from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
import time
from datetime import datetime

app = Flask(__name__)

# Load model
model = load_model("lulc-efficientnetb3_model.h5", compile=False)
# model = load_model("lulc_efficientnetb3.keras", compile=False)

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation",
    "Highway", "Industrial", "Pasture",
    "PermanentCrop", "Residential", "River", "SeaLake"
]

# =========================
# Home Route
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# Predict API
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    start = time.time()
    probs = model.predict(img)[0]
    end = time.time()

    pred_idx = np.argmax(probs)

    return jsonify({
        "predicted_class": CLASS_NAMES[pred_idx],
        "label": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "inference_time_ms": int((end - start) * 1000),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "model_mode": "Production",
        "top5": [
            {
                "label": CLASS_NAMES[i],
                "confidence": float(probs[i]),
                "color": "#06B6D4",
                "emoji": "🌍"
            }
            for i in np.argsort(probs)[::-1][:5]
        ],
        "color": "#06B6D4",
        "emoji": "🌍"
    })

# =========================
# Model Info API
# =========================
@app.route("/api/info")
def info():
    return jsonify({
        "model_ready": True,
        "model_config": {
            "accuracy": 0.94
        }
    })

# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(debug=True)