from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. LOAD ALL AI ENGINES (Including your new one!)
print("Loading AI Engines (this might take a minute)...")
models = {
    "resnet50": tf.keras.applications.ResNet50(weights='imagenet'),
    "mobilenetv2": tf.keras.applications.MobileNetV2(weights='imagenet')
}

custom_classes = []
# Check if you successfully trained your custom model
if os.path.exists("my_custom_brain.keras"):
    print("Loading your Custom AI Brain...")
   # --- Change your model loading section to look like this ---

import tf_keras

# Load your custom trained model using the explicit legacy engine
model = tf_keras.models.load_model('my_custom_brain.keras')
    
    # Load the dictionary of words it knows
    with open("class_names.txt", "r") as f:
        custom_classes = f.read().split(",")
else:
    print("Custom Brain not found.")

# Default to your custom model if it exists
active_model = "custom" if "custom" in models else "resnet50"
print(f"All systems ready! Default Engine: {active_model.upper()}")

# 2. HOT-SWAP DOORWAY
@app.post("/switch-model")
async def switch_model(model_name: str = Form(...)):
    global active_model
    if model_name in models:
        active_model = model_name
        return {"status": "success", "active_model": active_model}
    return {"status": "error"}

# 3. CLASSIFICATION DOORWAY
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # If using YOUR custom brain
    if active_model == "custom":
        # Your custom model handles its own math via the layers we built
        predictions = models["custom"].predict(img_array)[0]
        
        # Find the category with the highest score
        top_index = np.argmax(predictions)
        label = custom_classes[top_index].replace('_', ' ').title()
        confidence = float(predictions[top_index]) * 100
        
    # If using the old ImageNet brains
    elif active_model == "resnet50":
        img_array_processed = tf.keras.applications.resnet50.preprocess_input(img_array.copy())
        predictions = models["resnet50"].predict(img_array_processed)
        decoded = tf.keras.applications.resnet50.decode_predictions(predictions, top=1)[0]
        label = decoded[0][1].replace('_', ' ').title()
        confidence = float(decoded[0][2]) * 100
    else:
        img_array_processed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.copy())
        predictions = models["mobilenetv2"].predict(img_array_processed)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
        label = decoded[0][1].replace('_', ' ').title()
        confidence = float(decoded[0][2]) * 100
    
    return {"result": f"{label} ({confidence:.1f}%)", "model_used": active_model.upper()}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # Add this import at the very top of api.py
from fastapi.responses import HTMLResponse

# Add this route right above your "if __name__ == '__main__':" block
@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "index.html file not found."
