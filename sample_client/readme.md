# Patient Care sample

This is a very simple application simulating a health record management system for hearth disease patients. It is intended to be used as a showcase of what the heart failure prediction model can do. 

## How to use

Simply run the application with `uvicorn sample_client.patient_care.main:app --reload --port 80` and navigate to https://localhost/. From there, start filling out "patient" info, and you will see the model predicts whether the patient is likely to die or survive.