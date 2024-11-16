import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def load(path):
    heart_failure = pd.read_csv(path)
    x = heart_failure.drop(columns='DEATH_EVENT')
    y = heart_failure['DEATH_EVENT']
    return x, y


def split(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42)


def load_and_split(path):
    x, y = load(path)
    return split(x, y)

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    x_train, x_test, y_train, y_test = load_and_split(data_path)

    pd.DataFrame(x_train).to_csv(output_train_features, index=False)
    pd.DataFrame(x_test).to_csv(output_test_features, index=False)
    pd.DataFrame(y_train).to_csv(output_train_target, index=False)
    pd.DataFrame(y_test).to_csv(output_test_target, index=False)