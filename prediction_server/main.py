# main.py
from mlops.schema.prediction import InputpredictonModel, OutputPredictionModel
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from mlops.reproducibility.load_model import load_model

# Load the model using mlflow ``
RUN_ID = "bc8cf7556c204f698695eef704dfaf8b"
model = load_model(RUN_ID)

# List columns
NUMERIC_COLS = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium',
                'time']
BINARY_COLS = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
ALL_COLS = NUMERIC_COLS + BINARY_COLS



app = FastAPI()


@app.post("/predict")
def predict(input_data: InputpredictonModel):
    # Our model expects a dataframe with named columns
    x = pd.DataFrame([input_data.features.model_dump()])

    # Prediction
    prediction = model.predict(x)[0]
    output = OutputPredictionModel(predict=prediction)
    return output


@app.get("/")
def read_root():
    return {"message": "API de modelo de ML en funcionamiento"}
