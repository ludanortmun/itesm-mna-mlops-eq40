import click
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_COLS = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']
BINARY_COLS = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']


def create_preprocessing_pipeline(x_train):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_COLS),
            ('cat', OneHotEncoder(), BINARY_COLS)])
    preprocessor.fit(x_train)
    return preprocessor

@click.command()
@click.option('--x_train_path', type=click.Path(exists=True), help='Path to the training data CSV file.')
@click.option('--preprocessor_path', type=click.Path(), help='Path to save the preprocessor object.')
def main(x_train_path, preprocessor_path):
    x_train = pd.read_csv(x_train_path)
    preprocessor = create_preprocessing_pipeline(x_train)
    joblib.dump(preprocessor, preprocessor_path)
    click.echo(f'Preprocessor saved to {preprocessor_path}')


if __name__ == '__main__':
    main()
