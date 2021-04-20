import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    """

    return [1 if y >= t else 0 for y in y_scores]


def print_accuracy(y_train, y_train_pred, y_test, y_test_pred):
    """Prints the training and test set accuracy of the classifier"""
    print(f'training set accuracy {round(accuracy_score(y_train, y_train_pred), 4)}')
    print(f'test set accuracy {round(accuracy_score(y_test, y_test_pred), 4)}')


def labelled_confusion_matrix(y, y_pred, prop=False):
    """Returns a labelled confusion matrix"""
    matrix = pd.DataFrame(confusion_matrix(y, y_pred))
    matrix.columns = ['Predicted:0', 'Predicted:1']
    matrix['Total'] = matrix['Predicted:0'] + matrix['Predicted:1']
    matrix = matrix.append(matrix.sum(), ignore_index=True)
    matrix.index = ['Actual:0', 'Actual:1', 'Total']

    if prop is True:
        matrix = round(matrix / matrix.iloc[2, 2], 4)

    return matrix


def print_metrics(y_train, y_train_pred, y_test, y_test_pred):
    """Prints performance precision, recall, and f1 scores"""
    print('training set performance:')
    print(f'training precision score: {round(precision_score(y_train, y_train_pred), 4)}')
    print(f'training recall score: {round(recall_score(y_train, y_train_pred), 4)}')
    print(f'training f1 score: {round(f1_score(y_train, y_train_pred), 4)}')

    print('\ntest set performance:')
    print(f'testing precision score: {round(precision_score(y_test, y_test_pred), 4)}')
    print(f'testing recall score: {round(recall_score(y_test, y_test_pred), 4)}')
    print(f'testing f1 score: {round(f1_score(y_test, y_test_pred), 4)}')
