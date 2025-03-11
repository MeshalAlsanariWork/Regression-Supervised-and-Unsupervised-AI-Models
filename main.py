from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Load models
models = {
    "KM_model": joblib.load("KM_model.joblib"),
    "knn_model": joblib.load("knn_model.joblib"),
    "BDSCAN_model": joblib.load("BDSCAN_model.joblib"),
}

# Load the scaler (make sure you have a saved scaler file)
scaler = joblib.load("scaler.joblib")  # Ensure this file exists

# Define input schema for supervised models (KM & KNN)
class ModelInput(BaseModel):
    appearance: int
    minutes_played: float
    award: float

# Define input schema for DBSCAN (unsupervised clustering)
class ClusteringInput(BaseModel):
    data: List[List[float]]  # Expecting a list of lists of float numbers

# Preprocessing function
def preprocessing(input_features: ModelInput):
    """Applies the same preprocessing steps as used during model training."""
    
    dict_f = {
        "appearance": input_features.appearance,
        "minutes_played": input_features.minutes_played,
        "award": input_features.award,
    }
    
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    
    # Scale the input features using the trained scaler
    scaled_features = scaler.transform([features_list])

    return scaled_features

# Prediction function for KM & KNN models
def predict(model, data):
    try:
        preprocessed_data = preprocessing(data)
        prediction = model.predict(preprocessed_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define API endpoints for each model
@app.post("/predict/KM_model")
async def predict_riyadh(input_data: ModelInput):
    return predict(models["KM_model"], input_data)

@app.post("/predict/knn_model")
async def predict_western(input_data: ModelInput):
    return predict(models["knn_model"], input_data)

# API endpoint for DBSCAN model (Clustering)
@app.post("/predict/BDSCAN_model")
async def predict_southern(input_data: ClusteringInput):
    # Convert input data to NumPy array
    data_array = np.array(input_data.data)

    # DBSCAN does NOT support `.predict()`, use `.fit_predict()`
    model = models["BDSCAN_model"]
    labels = model.fit_predict(data_array)

    return {"labels": labels.tolist()}
