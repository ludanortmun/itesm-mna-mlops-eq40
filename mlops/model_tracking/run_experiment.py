import click
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from lifelines import CoxPHFitter, KaplanMeierFitter
import mlflow
import mlflow.pyfunc
from mlops.model_tracking.load_params import load_params
from mlops.model_tracking.train_and_eval_experiment import train_and_eval_experiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

mlflow.set_tracking_uri("http://13.93.214.226:5000/")

def __parse_model_type(_model_type):
    if _model_type == 'random_forest':
        return RandomForestClassifier(random_state=42)
    elif _model_type == 'logistic_regression':
        return LogisticRegression(random_state=42)
    elif _model_type == 'svm':
        return SVC(random_state=42)
    elif _model_type == 'decision_tree':
        return DecisionTreeClassifier(random_state=42)
    elif _model_type == 'cox':
        return CoxPHFitter()
    else:
        raise ValueError('Invalid model type')

def __remove_high_collinearity(dataframe, threshold=0.8, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []

    corr_matrix = dataframe.corr().abs()
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    to_drop = [
        column for i, column in enumerate(corr_matrix.columns)
        if column not in exclude_columns and any(corr_matrix[column][upper_triangle[:, i]] > threshold)
    ]
    print(f"Eliminando columnas con alta colinealidad adicional: {to_drop}")
    return dataframe.drop(columns=to_drop, errors='ignore')

def plot_kaplan_meier(data, model, duration_col, event_col, save_path=None):
    """Genera y guarda la gráfica de Kaplan-Meier para alto y bajo riesgo."""
    # Calcular riesgos parciales
    data["risk_score"] = model.predict_partial_hazard(data.drop(columns=[duration_col, event_col]))
    median_risk = data["risk_score"].median()

    # Separar en alto riesgo y bajo riesgo
    data["risk_group"] = np.where(data["risk_score"] > median_risk, "High Risk", "Low Risk")

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))

    for group, label in zip(["High Risk", "Low Risk"], ["High Risk", "Low Risk"]):
        subset = data[data["risk_group"] == group]
        kmf.fit(subset[duration_col], event_observed=subset[event_col], label=label)
        kmf.plot_survival_function()

    plt.title("Kaplan-Meier Survival Curve: High vs Low Risk")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

class CoxModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, columns):
        self.model = model
        self.columns = columns

    def predict(self, context, model_input):
        model_input = model_input[self.columns]
        return self.model.predict_partial_hazard(model_input)

@click.command()
@click.option('--model_type', default='random_forest', help='Type of model to use')
@click.option('--x_train_path', required=True, help='Path to the unprocessed training features CSV file')
@click.option('--y_train_path', required=True, help='Path to the unprocessed training labels CSV file')
@click.option('--x_test_path', required=True, help='Path to the unprocessed test features CSV file')
@click.option('--y_test_path', required=True, help='Path to the unprocessed test labels CSV file')
@click.option('--preprocessor_path', required=True, help='Path to the preprocessor object')
@click.option('--params_path', required=True, help='Path to the model parameters YAML file')
@click.option('--model_path', required=True, help='Path in which to save the trained model')
@click.option('--use_cv', is_flag=True, help='Flag to use cross-validation')
def main(model_type, x_train_path, y_train_path, x_test_path, y_test_path, preprocessor_path, params_path, model_path, use_cv):
    mlflow.set_experiment("cox_proportional_hazard_model_experiment")

    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    preprocessor = joblib.load(preprocessor_path)
    params = load_params(params_path)

    if model_type == 'cox':
        with mlflow.start_run():
            x_train_processed = pd.DataFrame(preprocessor.transform(x_train), columns=preprocessor.get_feature_names_out())
            data_train = pd.concat([x_train_processed, y_train.reset_index(drop=True)], axis=1)
            data_train = data_train.dropna().reset_index(drop=True)

            data_train = __remove_high_collinearity(data_train, exclude_columns=['time', 'death_event'])

            model = CoxPHFitter()
            model.fit(data_train, duration_col='time', event_col='death_event', robust=True)

            c_index = model.concordance_index_
            print(f"Concordance index: {c_index}")
            mlflow.log_param('model_type', 'cox')
            mlflow.log_metric('concordance_index', c_index)

            # Generar y registrar gráfica Kaplan-Meier
            km_path = "kaplan_meier_curve_high_low_risk.png"
            plot_kaplan_meier(data_train, model, duration_col='time', event_col='death_event', save_path=km_path)
            mlflow.log_artifact(km_path)

            # Asegurar columnas consistentes
            final_columns = data_train.drop(columns=['time', 'death_event', 'risk_score', 'risk_group']).columns
            x_test_processed = pd.DataFrame(preprocessor.transform(x_test), columns=preprocessor.get_feature_names_out())
            x_test_processed = x_test_processed[final_columns]

            wrapped_model = CoxModelWrapper(model, final_columns)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=wrapped_model,
                input_example=x_test_processed.head(1)
            )
            joblib.dump(model, model_path)
    else:
        model = __parse_model_type(model_type)
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            (f'{model_type}_classifier', model)
        ])
        params = {f'{model_type}_classifier__{k}': v for k, v in params.items()}
        params['model_type'] = model_type
        trained_model = train_and_eval_experiment(model_pipeline, params, x_train, y_train, x_test, y_test, use_cv)
        joblib.dump(trained_model, model_path)

if __name__ == '__main__':
    main()
