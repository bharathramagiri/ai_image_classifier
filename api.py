from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import os
import uvicorn

# Standard, modern TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess, decode_predictions as resnet_decode
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobile_preprocess, decode_predictions as mobile_decode

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. INITIALIZE GLOBAL MODEL REGISTRY
print("Pre-loading base architectures...")
models = {
    "resnet50": ResNet50(weights='imagenet'),
    "mobilenetv2": MobileNetV2(weights='imagenet')
}

custom_classes = []
if os.path.exists("my_custom_brain.keras"):
    print("Found custom model file. Loading weights...")
    try:
        models["custom"] = load_model("my_custom_brain.keras")
        
        if os.path.exists("class_names.txt"):
            with open("class_names.txt", "r", encoding="utf-8") as f:
                content = f.read()
                if "," in content:
                    custom_classes = [c.strip() for c in content.split(",") if c.strip()]
                else:
                    custom_classes = [c.strip() for c in content.splitlines() if c.strip()]
        print(f"Custom Brain activated successfully with {len(custom_classes)} target classes.")
    except Exception as e:
        print(f"Failed to load custom model weights: {e}")

active_model = "custom" if "custom" in models else "resnet50"
print(f"System boot complete. Active configuration pipeline: {active_model.upper()}")

# 2. APPLICATION ROUTING
@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "index.html file not found."

@app.post("/switch-model")
async def switch_model(model_name: str = Form(...)):
    global active_model
    if model_name in models:
        active_model = model_name
        return {"status": "success", "active_model": active_model}
    return {"status": "error"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB').resize((224, 224))
        
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        if active_model == "custom":
            predictions = models["custom"].predict(img_array)[0]
            top_index = np.argmax(predictions)
            if custom_classes and top_index < len(custom_classes):
                label = custom_classes[top_index].replace('_', ' ').title()
            else:
                label = f"Class Index {top_index}"
            confidence = float(predictions[top_index]) * 100
            
        elif active_model == "resnet50":
            img_array_processed = resnet_preprocess(img_array.copy())
            predictions = models["resnet50"].predict(img_array_processed)
            decoded = resnet_decode(predictions, top=1)[0]
            label = decoded[0][1].replace('_', ' ').title()
            confidence = float(decoded[0][2]) * 100
            
        else:
            img_array_processed = mobile_preprocess(img_array.copy())
            predictions = models["mobilenetv2"].predict(img_array_processed)
            decoded = mobile_decode(predictions, top=1)[0]
            label = decoded[0][1].replace('_', ' ').title()
            confidence = float(decoded[0][2]) * 100
        
        return {"result": f"{label} ({confidence:.1f}%)", "model_used": active_model.upper()}
    except Exception as e:
        return {"result": f"Analysis Error ({str(e)})", "model_used": active_model.upper()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
