import click
import pandas as pd
from mlops.evaluation.classification_metrics import evaluate_classification
from mlops.reproducibility.load_model import load_model


@click.command()
@click.option('--run_id', required=True, help='MLflow run ID')
@click.option('--x_path', required=True, help='Path to the test features CSV file')
@click.option('--y_path', required=True, help='Path to the test labels CSV file')
def main(run_id, x_path, y_path):
    x = pd.read_csv(x_path)
    y = pd.read_csv(y_path).values.ravel()
    loaded_model = load_model(run_id)
    y_pred_loaded = loaded_model.predict(x)
    metrics =  evaluate_classification(y, y_pred_loaded)
    print(f"Reproduced Metrics: {metrics}")


if __name__ == '__main__':
    main()