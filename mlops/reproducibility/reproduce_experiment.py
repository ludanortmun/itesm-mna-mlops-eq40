import click
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from load_model import load_model

def __evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'confusion_matrix': cm}

@click.command()
@click.option('--run_id', required=True, help='MLflow run ID')
@click.option('--x_path', required=True, help='Path to the test features CSV file')
@click.option('--y_path', required=True, help='Path to the test labels CSV file')
def main(run_id, x_path, y_path):
    x = pd.read_csv(x_path)
    y = pd.read_csv(y_path).values.ravel()
    loaded_model = load_model(run_id)
    y_pred_loaded = loaded_model.predict(x)
    metrics =  __evaluate_classification(y, y_pred_loaded)
    print(f"Reproduced Metrics: {metrics}")


if __name__ == '__main__':
    main()