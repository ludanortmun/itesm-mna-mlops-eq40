import mlflow
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

from mlops.evaluation.classification_metrics import evaluate_classification
from mlops.model_tracking.connection import setup_default_mlflow_connection


def __exclude_model_type(params):
    return {k: v for k, v in params.items() if k != 'model_type'}


def _train(model, params, x_train, y_train):
    mlflow.log_params(params)
    model.set_params(**__exclude_model_type(params))
    model.fit(x_train, y_train)

    return model


def _train_cv(model, param_grid, x_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=__exclude_model_type(param_grid), cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    _report_cv_params(grid_search.cv_results_, param_grid['model_type'])
    mlflow.log_params({**grid_search.best_params_, 'model_type': param_grid['model_type']})

    return grid_search.best_estimator_


# Use this to report each set of hyperparameters and their corresponding scores during GridSearchCV
def _report_cv_params(cv_results, model_type):
    n = len(cv_results['params'])

    for i in range(n):
        mlflow.start_run(nested=True)
        iteration_params = {**cv_results['params'][i], 'model_type': model_type}
        mlflow.log_params(iteration_params)
        mlflow.log_metrics({
            'mean_test_score': cv_results['mean_test_score'][i],
            'std_test_score': cv_results['std_test_score'][i]
        })
        mlflow.end_run()


def _evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    metrics = evaluate_classification(y_test, y_pred)
    mlflow.log_metrics({'accuracy': metrics.accuracy, 'precision': metrics.precision, 'recall': metrics.recall})
    disp = ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix,
                                  display_labels=model.classes_)
    disp.plot()
    plt.savefig('artifacts/confusion_matrix.png')
    mlflow.log_artifact('artifacts/confusion_matrix.png')
    plt.close()


def train_and_eval_experiment(model, params, x_train, y_train, x_test, y_test, use_cv=False):
    setup_default_mlflow_connection()
    with mlflow.start_run():
        if use_cv:
            trained_model = _train_cv(model, params, x_train, y_train)
        else:
            trained_model = _train(model, params, x_train, y_train)

        _evaluate(trained_model, x_test, y_test)
        mlflow.sklearn.log_model(trained_model, 'model', input_example=x_train.head(1))
