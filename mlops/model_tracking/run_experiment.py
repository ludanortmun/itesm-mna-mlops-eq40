import click
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from load_params import load_params
from train_and_eval_experiment import train_and_eval_experiment
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def __parse_model_type(_model_type):
    if _model_type == 'random_forest':
        return RandomForestClassifier(random_state=42)
    elif _model_type == 'logistic_regression':
        return LogisticRegression(random_state=42)
    elif _model_type == 'svm':
        return SVC(random_state=42)
    elif _model_type == 'decision_tree':
        return DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError('Invalid model type')


@click.command()
@click.option('--model_type', default='random_forest', help='Type of model to use')
@click.option('--x_train_path', required=True, help='Path to the training features CSV file')
@click.option('--y_train_path', required=True, help='Path to the training labels CSV file')
@click.option('--x_test_path', required=True, help='Path to the test features CSV file')
@click.option('--y_test_path', required=True, help='Path to the test labels CSV file')
@click.option('--params_path', required=True, help='Path to the model parameters YAML file')
@click.option('--model_path', required=True, help='Path in which to save the trained model')
@click.option('--use_cv', is_flag=True, help='Flag to use cross-validation')
def main(model_type, x_train_path, y_train_path, x_test_path, y_test_path, params_path, model_path, use_cv):
    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    params = load_params(params_path)
    model = __parse_model_type(model_type)

    model = train_and_eval_experiment(model, params, x_train, y_train, x_test, y_test, use_cv)
    joblib.dump(model, model_path)


if __name__ == '__main__':
    main()
