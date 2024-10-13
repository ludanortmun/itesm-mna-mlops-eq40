import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}


# Used GridSearchCV to find the best hyperparameters for the RandomForestClassifier
def train_model(x_train_path, y_train_path):
    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5,
                               scoring='accuracy')
    grid_search.fit(x_train, y_train)

    # Returns the best estimator obtained from the grid search
    return grid_search.best_estimator_


if __name__ == '__main__':
    x_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_path = sys.argv[3]

    model = train_model(x_train_path, y_train_path)
    joblib.dump(model, model_path)
