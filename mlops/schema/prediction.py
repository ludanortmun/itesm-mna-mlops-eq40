
from pydantic import BaseModel


class FeautureModel(BaseModel):
    age: int
    anaemia: int
    creatinine_phosphokinase: float
    diabetes: int
    ejection_fraction: float
    high_blood_pressure: float
    platelets: float
    serum_creatinine: float
    serum_sodium: float
    smoking: int
    time: float
    sex: int
    

class InputpredictonModel(BaseModel):
    features: FeautureModel

class OutputPredictionModel(BaseModel):
    predict: int