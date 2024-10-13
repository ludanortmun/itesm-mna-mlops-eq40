import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_COLS = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']
BINARY_COLS = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']


def load(path):
    heart_failure = pd.read_csv(path)
    x = heart_failure.drop(columns='DEATH_EVENT')
    y = heart_failure['DEATH_EVENT']
    return x, y


def preprocess(x):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_COLS),
            ('cat', OneHotEncoder(), BINARY_COLS)])

    return preprocessor.fit_transform(x)


def split(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42)


if __name__ == '__main__':
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    X, y = load(data_path)
    X = preprocess(X)
    x_train, x_test, y_train, y_test = split(X, y)

    pd.DataFrame(x_train).to_csv(output_train_features, index=False)
    pd.DataFrame(x_test).to_csv(output_test_features, index=False)
    pd.DataFrame(y_train).to_csv(output_train_target, index=False)
    pd.DataFrame(y_test).to_csv(output_test_target, index=False)