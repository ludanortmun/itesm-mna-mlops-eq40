import mlflow


def setup_default_mlflow_connection():
# TODO: Make both the tracking URI and experiment name configurable
    mlflow.set_tracking_uri("http://13.93.214.226:5000")
    mlflow.set_experiment("Heart_Failure")