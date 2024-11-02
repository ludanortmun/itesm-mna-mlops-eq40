from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


class ClassificationMetrics:
    """A simple data class to store the relevant classification metrics for this problem."""

    def __init__(self, accuracy, precision, recall, cm):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.confusion_matrix = cm

    def __str__(self):
        return f'Accuracy: {self.accuracy}, Precision: {self.precision}, Recall: {self.recall}'


def evaluate_classification(y_true, y_pred):
    """Centralized evaluation function for classification tasks."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return ClassificationMetrics(accuracy, precision, recall, cm)