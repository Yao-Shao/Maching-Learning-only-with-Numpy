import copy
import numpy as np


class GBDT:
    '''GDBT Classifier.

    Note that this class only support binary classification.
    '''

    def __init__(self,
                 base_learner,
                 n_estimator,
                 learning_rate=0.1,
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
        self.lr = learning_rate


    def fit(self, X, y):
        """Build the Adaboost according to the training data.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).
        """
        # YOUR CODE HERE
        # begin answer
        def _gredient(y, p):
            tmp = np.clip(p, 1e-12, 1-1e-12)
            return -(y/tmp) + (1-y)/(1-tmp)

        self._estimators[0].fit(X, y)
        pred = self._estimators[0].predict(X)
        for i in range(1, self.n_estimator):
            s_r = _gredient(y, pred)
            self._estimators[i].fit(X, s_r)
            pred -= self.lr * self._estimators[i].predict(X)
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
        # YOUR CODE HERE
        # begin answer
        pred = self._estimators[0].predict(X)
        for i in range(1, self.n_estimator):
            pred -= self.lr * self._estimators[i].predict(X)
        y_pred = pred > 0.5
        y_pred[y_pred == True] = 1
        y_pred[y_pred == False] = 0
        # end answer
        return y_pred
