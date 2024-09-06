from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import joblib
import numpy as np
from utils import log_model_evaluation
from model import train_model
import logging

app = FastAPI()

# Load the model when the app starts
try:
    model = joblib.load('model.joblib')
except FileNotFoundError:
    logging.warning("No model found. Retraining...")
    accuracy, saved = train_model()
    log_model_evaluation(accuracy)
    if saved:
        model = joblib.load('model.joblib')

class PredictRequest(BaseModel):
    data: list

# Background task for predictions
def perform_prediction(data):
    logging.info(f"Received data for prediction: {data}")
    reshaped_data = np.array(data).reshape(1, -1)
    prediction = model.predict(reshaped_data)
    logging.info(f"Prediction result: {prediction[0]}")
    return int(prediction[0])

# Prediction endpoint with background task processing
@app.post("/predict/")
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    data = request.data
    background_tasks.add_task(perform_prediction, data)
    return {"message": "Prediction is being processed in the background."}

# Endpoint to retrain model if performance is not satisfactory
@app.get("/retrain/")
async def retrain_model():
    accuracy, saved = train_model()
    log_model_evaluation(accuracy)
    if saved:
        global model
        model = joblib.load('model.joblib')
        return {"message": "Model retrained successfully."}
    else:
        return {"message": "Model retraining failed due to low accuracy."}
