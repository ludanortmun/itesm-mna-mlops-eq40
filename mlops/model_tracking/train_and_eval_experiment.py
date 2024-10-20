import mlflow
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Heart_Failure")

def _train(model, params, x_train, y_train):
    mlflow.log_params(params)
    model.set_params(**params)
    model.fit(x_train, y_train)

    return model

def _train_cv(model, param_grid, x_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    _report_cv_params(grid_search.cv_results_)
    mlflow.log_params(grid_search.best_params_)

    return grid_search.best_estimator_

# Use this to report each set of hyperparameters and their corresponding scores during GridSearchCV
def _report_cv_params(cv_results):
    n = len(cv_results['params'])

    for i in range(n):
        mlflow.start_run(nested=True)
        mlflow.log_params(cv_results['params'][i])
        mlflow.log_metrics({
            'mean_test_score': cv_results['mean_test_score'][i],
            'std_test_score': cv_results['std_test_score'][i]
        })
        mlflow.end_run()


def _evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    mlflow.log_metrics({'accuracy': accuracy, 'precision': precision, 'recall': recall})

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
    disp.plot()
    plt.savefig('artifacts/confusion_matrix.png')
    mlflow.log_artifact('artifacts/confusion_matrix.png')
    plt.close()


def train_and_eval_experiment(model, params, x_train, y_train, x_test, y_test, use_cv=False):
    with mlflow.start_run():
        if use_cv:
            trained_model = _train_cv(model, params, x_train, y_train)
        else:
            trained_model = _train(model, params, x_train, y_train)

        _evaluate(trained_model, x_test, y_test)
        mlflow.sklearn.log_model(trained_model, 'model', input_example=x_train.head(1))