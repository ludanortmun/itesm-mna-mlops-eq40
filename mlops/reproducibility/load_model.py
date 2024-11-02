import mlflow

run_relative_path_to_model = 'model'
mlflow.set_tracking_uri("http://13.93.214.226:5000")
mlflow.set_experiment("Heart_Failure")

def load_model(run_id):
    model_uri = f'runs:/{run_id}/{run_relative_path_to_model}'
    return mlflow.sklearn.load_model(model_uri)

