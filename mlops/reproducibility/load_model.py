import mlflow
from mlops.model_tracking.connection import setup_default_mlflow_connection


run_relative_path_to_model = 'model'


def load_model(run_id):
    setup_default_mlflow_connection()
    model_uri = f'runs:/{run_id}/{run_relative_path_to_model}'
    return mlflow.sklearn.load_model(model_uri)

