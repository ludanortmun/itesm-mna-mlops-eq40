# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from mlops.reproducibility.load_model import load_model  
from sklearn.datasets import load_wine

# Load the model using mlflow 
RUN_ID = "bc8cf7556c204f698695eef704dfaf8b"
model = load_model(RUN_ID)

# Load the target names (class labels)
data = load_wine()
target_names = data.target_names

# Define the input data format for prediction
class WineData(BaseModel):
    features: List[float]

# Initialize FastAPI
app = FastAPI()

# Prediction endpoint
@app.post("/predict")
def predict(wine_data: WineData):
    if len(wine_data.features) != model.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Input must contain {model.n_features_in_} features."
        )
    # Predict
    prediction = model.predict([wine_data.features])[0]
    prediction_name = target_names[prediction]
    return {"prediction": int(prediction), "prediction_name": prediction_name}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Wine classification model API"}
