from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
import onnxruntime as ort
import torchvision.transforms as transforms
import numpy as np
import json

app = FastAPI()

# Load the ONNX model using ONNX Runtime
ort_session = ort.InferenceSession("./model_updated_256x256.onnx")

# Load class names from JSON file
with open('class_names.json') as f:
    class_names = json.load(f)

# Define the image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Example, adjust to match your training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Transform the image
    image = transform(image).unsqueeze(0).numpy()

    # Make prediction
    outputs = ort_session.run(None, {'input': image})
    predicted = np.argmax(outputs[0], axis=1)
    prediction_index = int(predicted[0])
    
    # Map prediction index to class name
    prediction_name = class_names[prediction_index] if prediction_index < len(class_names) else "Unknown"

    return JSONResponse(content={"prediction": prediction_name})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
