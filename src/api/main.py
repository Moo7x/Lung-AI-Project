from fastapi import FastAPI, UploadFile, File
from typing import List

# This variable MUST be named 'app' because the command uses 'main:app'
app = FastAPI(title="Lung AI Detection API")

# Mock function to simulate AI prediction
def run_mock_prediction(image_bytes: bytes) -> List[dict]:
    # In real version, this will load the ONNX model
    return [
        {
            "label": "cardiomegaly",
            "confidence": 0.92,
            "box": [150, 200, 350, 400]
        },
        {
            "label": "nodule",
            "confidence": 0.45,
            "box": [450, 310, 480, 340]
        }
    ]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    detections = run_mock_prediction(image_bytes)
    return {"filename": file.filename, "detections": detections}