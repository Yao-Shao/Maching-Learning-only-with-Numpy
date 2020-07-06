import copy
import numpy as np
from scipy.stats import mode
import multiprocessing as mp


class RandomForest:
    '''Random Forest Classifier.

    Note that this class only support binary classification.
    '''

    def __init__(self,
                 base_learner,
                 n_estimator,
                 thread_num = 1,
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
        self.thread_num = thread_num

    def _get_bootstrap_dataset(self, X, y):
        """Create a bootstrap dataset for X.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).

        Returns:
            X_bootstrap: a sampled dataset, of shape (N, D).
            y_bootstrap: the labels for sampled dataset.
        """
        # YOUR CODE HERE
        # TODO: re‐sample N examples from X with replacement
        # begin answer
        idx = np.random.randint(0, len(y), len(y))
        X_bootstrap = X.iloc[idx]
        y_bootstrap = y.iloc[idx]
        return X_bootstrap, y_bootstrap
        # end answer

    def fit(self, X, y):
        """Build the random forest according to the training data.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).
        """
        # YOUR CODE HERE
        # begin answer
        if self.thread_num == 1:
            for i in range(self.n_estimator):
                XX, yy = self._get_bootstrap_dataset(X, y)
                self._estimators[i].fit(XX, yy)
        else:
            p = mp.Pool(self.thread_num)
            p_l = []
            for i in range(self.n_estimator):
                p_l.append(p.apply_async(self.worker, args=(self._estimators[i], X, y)))
            p.close()
            p.join()
            for i in range(self.n_estimator):
                self._estimators[i] = p_l[i].get()
        # end answer
        return self

    def worker(self, est, X, y):
        idx = np.random.randint(0, len(y), len(y))
        est.fit(X.iloc[idx], y.iloc[idx])
        return est

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
        tmp = np.zeros((X.shape[0], self.n_estimator))
        if self.thread_num > 1:
            p = mp.Pool(self.thread_num)
            p_l = []
            for i in range(self.n_estimator):
                p_l.append(p.apply_async(self._estimators[i].predict, args=(X,)))
            p.close()
            p.join()
            for i in range(self.n_estimator):
                tmp[:, i] = p_l[i].get()
        else:
            for i in range(self.n_estimator):
                tmp[:, i] = self._estimators[i].predict(X)
        y_pred = mode(tmp, axis=1)[0].reshape(1, -1)[0]
        # end answer
        return y_pred


