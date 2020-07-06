import copy
import numpy as np


class Adaboost:
    '''Adaboost Classifier.

    Note that this class only support binary classification.
    '''

    def __init__(self,
                 base_learner,
                 n_estimator,
                 seed=2020):
        '''Initialize the classifier.

        Args:
            base_learner: the base_learner should provide the .fit() and .predict() interface.
            n_estimator (int): The number of base learners in RandomForest.
            seed (int): random seed
        '''
        np.random.seed(seed)
        self.base_learner = base_learner
        self.n_estimator = n_estimator
        self._estimators = [copy.deepcopy(self.base_learner) for _ in range(self.n_estimator)]
        self._alphas = [1 for _ in range(n_estimator)]

    def fit(self, X, y):
        """Build the Adaboost according to the training data.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).
        """
        # YOUR CODE HERE
        # begin answer
        sample_weights = np.ones(len(y)) / len(y)
        for i in range(self.n_estimator):
            self._estimators[i].fit(X, y, sample_weights)
            pred = self._estimators[i].predict(X)
            I = pred != y
            err = np.sum(I * sample_weights)
            self._alphas[i] = 0.5 * np.log((1 - err) / err)
            sample_weights *= np.exp(self._alphas[i] * (2 * err - 1))
            sample_weights /= np.sum(sample_weights)
        # end answer
        return self

    def predict(self, X):
        """Predict classification results for X.

        Args:
            X: testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        N = X.shape[0]
        y_pred = np.zeros(N)
        # YOUR CODE HERE
        # begin answer
        tmp = np.zeros((N, self.n_estimator))
        for i in range(self.n_estimator):
            tmp[:, i] = self._estimators[i].predict(X)
        tmp = tmp * np.array(self._alphas).reshape((1, -1))
        y_pred = np.sum(tmp, axis=1) > 0.5
        y_pred[y_pred == True] = 1
        y_pred[y_pred == False] = 0
        # end answer
        return y_pred
