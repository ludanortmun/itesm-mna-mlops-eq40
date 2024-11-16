import os
import random

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, List
from uuid import uuid4

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, computed_field


class Sex(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"


class Patient(BaseModel):
    id: str = Field(None, title='ID of the patient', description='ID of the patient')
    name: str = Field(..., title='Name of the patient', description='Name of the patient')
    age: int = Field(..., description="Age of the patient (years)")
    anaemia: bool = Field(..., description="Decrease of red blood cells or hemoglobin (boolean)")
    creatinine_phosphokinase: float = Field(..., description="Level of the CPK enzyme in the blood (mcg/L)")
    diabetes: bool = Field(..., description="If the patient has diabetes (boolean)")
    ejection_fraction: int = Field(...,
                                   description="Percentage of blood leaving the heart at each contraction (percentage)")
    high_blood_pressure: bool = Field(..., description="If the patient has hypertension (boolean)")
    platelets: float = Field(..., description="Platelets in the blood (kiloplatelets/mL)")
    sex: Sex = Field(..., description="Sex of the patient (MALE or FEMALE)")
    serum_creatinine: float = Field(..., description="Level of serum creatinine in the blood (mg/dL)")
    serum_sodium: int = Field(..., description="Level of serum sodium in the blood (mEq/L)")
    smoking: bool = Field(..., description="If the patient smokes or not (boolean)")
    heart_failure_time: datetime = Field(..., description="Time at which the heart failure happened")

    @computed_field(return_type=bool)
    @property
    def is_death_predicted(self):
        return get_death_prediction(self)


app = FastAPI()

# In-memory storage for patients
patients: Dict[str, Patient] = {}



@app.post("/patients/", response_model=Patient)
def create_patient(patient: Patient):
    patient_id = str(uuid4())
    patient.id = patient_id
    patients[patient_id] = patient
    return patient


@app.get("/patients/{patient_id}", response_model=Patient)
def get_patient(patient_id: str):
    if patient_id not in patients:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patients[patient_id]


@app.get("/patients/", response_model=List[Patient])
def list_patients():
    return list(patients.values())


def get_death_prediction(patient: Patient):
    url = "http://localhost:8000/predict"
    data = {
        "age": patient.age,
        "anaemia": 1 if patient.anaemia else 0,
        "creatinine_phosphokinase": patient.creatinine_phosphokinase,
        "diabetes": 1 if patient.diabetes else 0,
        "ejection_fraction": patient.ejection_fraction,
        "high_blood_pressure": 1 if patient.high_blood_pressure else 0,
        "platelets": patient.platelets,
        "sex": 0 if patient.sex == Sex.FEMALE else 1,
        "serum_creatinine": patient.serum_creatinine,
        "serum_sodium": patient.serum_sodium,
        "smoking": 1 if patient.smoking else 0,
        "time": (datetime.now() - patient.heart_failure_time).days
    }

    # For now, we use a dummy prediction
    return random.choice([True, False])

    # response = requests.post(url, json=data)
    #
    # # Extract prediction from response
    # if response.status_code == 200:
    #     print("Prediction:", response.json())
    # else:
    #     print("Failed to get prediction:", response.status_code, response.text)


# Mount the static files directory
app.mount("/static", StaticFiles(directory="sample_client/patient_care"), name="static")


@app.get("/")
def read_root():
    return FileResponse(os.path.join("sample_client/patient_care", "index.html"))
