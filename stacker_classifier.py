import numpy as np

class StackedClassifier:
    """
    Builds a stacked classifier
    """

    def __init__(self, estimators, random_seed=42):
        self.estimators = estimators
        self.n_estimators = len(estimators)
        self.random_seed = random_seed

    def fit(self, x, y, test_pct=0.2, l1_pct=0.7, verbose=False):
        """Fit models"""

        n_train = int(x.shape[0] * (1 - test_pct))
        l1_train = int(n_train * l1_pct)

        y = np.array(y)

        # Take copies and shuffle the data
        x_copy = x.copy()
        y_copy = y.copy()

        np.random.seed(self.random_seed)
        np.random.shuffle(x_copy)
        np.random.seed(self.random_seed)
        np.random.shuffle(y_copy)

        # Create datasets
        x_train_l1 = x_copy[:l1_train]
        y_train_l1 = y_copy[:l1_train]
        x_train_l2 = x_copy[l1_train:n_train]
        y_train_l2 = y_copy[l1_train:n_train]
        x_test = x_copy[n_train:]
        y_test = y_copy[n_train:]

        for i, estimator in enumerate(self.estimators):
            if i < self.n_estimators - 1:
                estimator.fit(x_train_l1, y_train_l1)
                train_l1_pred = estimator.predict(x_train_l1)
                train_l2_pred = estimator.predict(x_train_l2)

                if verbose:
                    print(f'Model {i} performance:')
                    print_metrics(y_train_l1, train_l1_pred, y_train_l2, train_l2_pred)

            else:
                l2_train_preds = self.create_l2_predictions(x_train_l2)
                l2_test_preds = self.create_l2_predictions(x_test)

                self.estimators[i].fit(l2_train_preds, y_train_l2)
                train_pred = self.estimators[i].predict(l2_train_preds)
                test_pred = self.estimators[i].predict(l2_test_preds)

                if verbose:
                    print(f'Meta model performance:')
                    print_metrics(y_train_l2, train_pred, y_test, test_pred)

    def create_l2_predictions(self, x_l2):
        l2_preds = np.zeros((x_l2.shape[0], self.n_estimators - 1))

        for i, estimator in enumerate(self.estimators[:-1]):
            l2_preds[:, i] = estimator.predict(x_l2)

        return l2_preds

    def predict(self, x):
        """Predict on new data"""
        preds = np.zeros(x.shape[0])
        l2_preds = self.create_l2_predictions(x)
        preds = self.estimators[-1].predict(l2_preds)
        preds = np.rint(preds)

        return preds
