import numpy as np

class DecisionTree:
    '''Decision Tree Classifier.

    Note that this class only supports binary classification.
    '''

    def __init__(self,
                 criterion,
                 max_depth,
                 min_samples_leaf,
                 sample_feature=False,
                 continuous=False):
        '''Initialize the classifier.

        Args:
            criterion (str): the criterion used to select features and split nodes.
            max_depth (int): the max depth for the decision tree. This parameter is
                a trade-off between underfitting and overfitting.
            min_samples_leaf (int): the minimal samples in a leaf. This parameter is a trade-off
                between underfitting and overfitting.
            sample_feature (bool): whether to sample features for each splitting. Note that for random forest,
                we would randomly select a subset of features for learning. Here we select sqrt(p) features.
                For single decision tree, we do not sample features.
        '''
        if criterion == 'infogain_ratio':
            self.criterion = self._information_gain_ratio
        elif criterion == 'entropy':
            self.criterion = self._information_gain
        elif criterion == 'gini':
            self.criterion = self._gini_purification
        else:
            raise Exception('Criterion should be infogain_ratio or entropy or gini')
        self._tree = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sample_feature = sample_feature
        self.continuous = continuous
        self.crit_THRESH = 0.05

    def fit(self, X, y, sample_weights=None):
        """Build the decision tree according to the training data.

        Args:
            X: (pd.Dataframe) training features, of shape (N, D). Each X[i] is a training sample.
            y: (pd.Series) vector of training labels, of shape (N,). y[i] is the label for X[i], and each y[i] is
            an integer in the range 0 <= y[i] <= C. Here C = 1.
            sample_weights: weights for each samples, of shape (N,).
        """
        if sample_weights is None:
            # if the sample weights is not provided, then by default all
            # the samples have unit weights.
            sample_weights = np.ones(X.shape[0]) / X.shape[0]
        else:
            sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        feature_names = X.columns.tolist()
        X = np.array(X)
        y = np.array(y)
        self._tree = self._build_tree(X, y, feature_names, depth=1, sample_weights=sample_weights)
        return self

    @staticmethod
    def entropy(y, sample_weights):
        """Calculate the entropy for label.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the entropy for y.
        """
        entropy = 0.0
        # begin answer
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        y_set = np.unique(y)
        prob = np.zeros(y_set.shape)
        S = np.sum(sample_weights)
        for i in range(len(y_set)):
            prob[i] = np.sum(np.multiply(y == y_set[i], sample_weights)) / S
        # end answer
        return entropy

    def _information_gain(self, X, y, index, sample_weights):
        """Calculate the information gain given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the information gain calculated.
        """
        info_gain = 0
        # YOUR CODE HERE
        # begin answer
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        en = DecisionTree.entropy(y, sample_weights)
        sub_en = 0
        if self.continuous:
            x_set = np.unique(X[:, index])
            best_sub_en = np.Inf
            best_x = None
            for x in x_set:
                for c_idx in range(2):
                    sub_X, sub_y, sub_sample_weights = self._split_dataset(X, y, index, x, sample_weights, c_idx)
                    sub_en += np.sum(sub_sample_weights) / np.sum(sample_weights) * DecisionTree.entropy(sub_y,
                                                                                                         sub_sample_weights)
                    if sub_en < best_sub_en:
                        best_sub_en = sub_en
                        best_x = x
        else:
            thresh = np.unique(X[:, index])
            for i in range(len(thresh)):
                sub_X, sub_y, sub_sample_weights = self._split_dataset(X, y, index, thresh[i], sample_weights)
                sub_en += np.sum(sub_sample_weights)/np.sum(sample_weights) * DecisionTree.entropy(sub_y, sub_sample_weights)
        info_gain = en - sub_en
        # end answer
        return info_gain, best_x

    def _information_gain_ratio(self, X, y, index, sample_weights):
        """Calculate the information gain ratio given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the information gain ratio calculated.
        """
        info_gain_ratio = 0
        split_information = 0.0
        # YOUR CODE HERE
        # begin answer
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        en = DecisionTree.entropy(y, sample_weights)
        sub_en = 0
        sub_info = 0
        if self.continuous:
            x_set = np.unique(X[:, index])
            best_sub_info = np.Inf
            best_x = None
            for x in x_set:
                for c_idx in range(2):
                    sub_X, sub_y, sub_sample_weights = self._split_dataset(X, y, index, x, sample_weights, c_idx)
                    coef = np.sum(sub_sample_weights) / np.sum(sample_weights)
                    sub_en += coef * DecisionTree.entropy(sub_y, sub_sample_weights)
                    sub_info = coef * np.log2(coef)
                    if sub_info < best_sub_info:
                        best_sub_info = sub_info
                        best_x = x
        else:
            thresh = np.unique(X[:, index])
            for i in range(len(thresh)):
                sub_X, sub_y, sub_sample_weights = self._split_dataset(X, y, index, thresh[i], sample_weights)
                coef = np.sum(sub_sample_weights) / np.sum(sample_weights)
                sub_en += coef * DecisionTree.entropy(sub_y, sub_sample_weights)
                sub_info = coef * np.log2(coef)
        info_gain = en - sub_en
        if sub_info > 1e-10:
            info_gain_ratio = info_gain / sub_info
        else:
            info_gain_ratio = np.Inf
        # end answerself
        return info_gain_ratio, best_x

    @staticmethod
    def gini_impurity(y, sample_weights):
        """Calculate the gini impurity for labels.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the gini impurity for y.
        """
        gini = 1
        # YOUR CODE HERE
        # begin answer
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        y_set = np.unique(y)
        prob = np.zeros(y_set.shape)
        gini = 1 - np.sum(prob ** 2)
        # end answer
        return gini

    def _gini_purification(self, X, y, index, sample_weights):
        """Calculate the resulted gini impurity given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the resulted gini impurity after splitting by this feature.
        """
        new_impurity = 1
        # YOUR CODE HERE
        # begin answer
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        gini = DecisionTree.gini_impurity(y, sample_weights)
        sub_gini = 0
        if self.continuous:
            x_set = np.unique(X[:, index])
            best_sub_gini = np.Inf
            best_x =None
            for x in x_set:
                for c_idx in range(2):
                    sub_X, sub_y, sub_sample_weights = self._split_dataset(X, y, index, x, sample_weights, c_idx)
                    coef = np.sum(sub_sample_weights) / np.sum(sample_weights)
                    sub_gini += coef * DecisionTree.gini_impurity(sub_y, sub_sample_weights)
                    if sub_gini < best_sub_gini:
                        best_sub_gini = sub_gini
                        best_x = x
        else:
            thresh = np.unique(X[:, index])
            for i in range(len(thresh)):
                sub_X, sub_y, sub_sample_weights = self._split_dataset(X, y, index, thresh[i], sample_weights)
                coef = np.sum(sub_sample_weights) / np.sum(sample_weights)
                sub_gini += coef * DecisionTree.gini_impurity(sub_y, sub_sample_weights)
        new_impurity = gini - sub_gini
        # end answer
        return new_impurity, best_x

    def _split_dataset(self, X, y, index, value, sample_weights, c_idx):
        """Return the split of data whose index-th feature equals value.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for splitting.
            value: the value of the index-th feature for splitting.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (np.array): the subset of X whose index-th feature equals value.
            (np.array): the subset of y whose index-th feature equals value.
            (np.array): the subset of sample weights whose index-th feature equals value.
        """
        sub_X, sub_y, sub_sample_weights = X, y, sample_weights
        # YOUR CODE HERE
        # Hint: Do not forget to remove the index-th feature from X.
        # begin answer
        if self.continuous:
            if c_idx == 1:
                idx = np.where(X[:, index] > value)[0]
            else:
                idx = np.where(X[:, index] <= value)[0]
            sub_X = X[idx, :]
        else:
            idx = np.where(X[:, index] == value)[0]
            sub_X = np.delete(X[idx, :], index, axis=1)
        sub_y = y[idx]
        sub_sample_weights = sample_weights[idx]
        # end answer
        return sub_X, sub_y, sub_sample_weights

    def _choose_best_feature(self, X, y, sample_weights):
        """Choose the best feature to split according to criterion.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the index for the best feature
        """
        best_feature_idx = 0
        # YOUR CODE HERE
        # Note that you need to implement the sampling feature part here for random forest!
        # Hint: You may find `np.random.choice` is useful for sampling.
        # begin answer
        best_score = 0
        sub_fea = np.arange(X.shape[1])
        if self.sample_feature:
            sub_fea = np.random.choice(np.arange(X.shape[1]), int(np.sqrt(X.shape[1])))
        for i in sub_fea:
            cur_score, best_x = self.criterion(X, y, i, sample_weights)
            if cur_score > best_score:
                best_score = cur_score
                best_feature_idx = i
        # end answer
        return best_feature_idx, best_x

    @staticmethod
    def majority_vote(y, sample_weights=None):
        """Return the label which appears the most in y.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the majority label
        """
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        majority_label = y[0]
        # YOUR CODE HERE
        # begin answer
        y_set = np.unique(y)
        num = np.zeros(y_set.shape)
        for i in range(len(y_set)):
            num[i] = np.sum((y == y_set[i]) * sample_weights)
        majority_label = y_set[np.argmax(num)]
        # end answer
        return majority_label

    def _build_tree(self, X, y, feature_names, depth, sample_weights):
        """Build the decision tree according to the data.

        Args:
            X: (np.array) training features, of shape (N, D).
            y: (np.array) vector of training labels, of shape (N,).
            feature_names (list): record the name of features in X in the original dataset.
            depth (int): current depth for this node.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (dict): a dict denoting the decision tree. 
            Example:
                The first best feature name is 'title', and it has 5 different values: 0,1,2,3,4. For 'title' == 4, the next best feature name is 'pclass', we continue split the remain data. If it comes to the leaf, we use the majority_label by calling majority_vote.
                mytree = {
                    'titile': {
                        0: subtree0,
                        1: subtree1,
                        2: subtree2,
                        3: subtree3,
                        4: {
                            'pclass': {
                                1: majority_vote([1, 1, 1, 1]) # which is 1, majority_label
                                2: majority_vote([1, 0, 1, 1]) # which is 1
                                3: majority_vote([0, 0, 0]) # which is 0
                            }
                        }
                    }
                }
        """
        mytree = dict()
        # YOUR CODE HERE
        # TODO: Use `_choose_best_feature` to find the best feature to split the X. Then use `_split_dataset` to
        # get subtrees.
        # Hint: You may find `np.unique` is useful.
        # begin answer
        # Todo prune, early stop
        if depth <= self.max_depth and X.shape[0] >= self.min_samples_leaf:
            fea_idx, best_thresh = self._choose_best_feature(X, y, sample_weights)
            fea_name = feature_names[fea_idx]
            sub_fea_names =feature_names[:fea_idx] + feature_names[fea_idx+1:]
            if self.continuous:
                mytree[(fea_name, best_thresh)] = {}
                for c_idx in range(2):
                    sub_X, sub_y, sub_sample_weights = self._split_dataset(X, y, fea_idx, best_thresh, sample_weights, c_idx)
                    if len(sub_y) > 0:
                        mytree[(fea_name, best_thresh)][c_idx] =  self._build_tree(sub_X, sub_y, sub_fea_names, depth+1, sub_sample_weights)
            else:
                mytree[fea_name] = {}
                fea_set = np.unique(X[:, fea_idx])
                for i in range(len(fea_set)):
                    sub_X, sub_y, sub_sample_weights = self._split_dataset(X, y, fea_idx, fea_set[i], sample_weights)
                    mytree[fea_name][fea_set[i]] = self._build_tree(sub_X, sub_y, sub_fea_names, depth+1, sub_sample_weights)
        else:
            mytree = self.majority_vote(y, sample_weights)
        # end answer
        return mytree

    def predict(self, X):
        """Predict classification results for X.

        Args:
            X: (pd.Dataframe) testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        def _classify(tree, x):
            """Classify a single sample with the fitted decision tree.

            Args:
                x: ((pd.Dataframe) a single sample features, of shape (D,).

            Returns:
                (int): predicted testing sample label.
            """
            # YOUR CODE HERE
            # begin answer
            while isinstance(tree, dict):
                fea_name = list(tree.keys())[0]
                if isinstance(fea_name, tuple):
                    if x[fea_name[0]] < x[fea_name[1]]:
                        idx = 0
                    else:
                        idx = 1
                else:
                    idx = x[fea_name]
                if idx not in tree[fea_name]:
                    idx = np.random.choice(list(tree[fea_name].keys()))
                tree = tree[fea_name][idx]
            return tree
            # end answer

        # YOUR CODE HERE
        # begin answer
        pred = np.zeros(len(X))
        for i in range(len(X)):
            pred[i] = _classify(self._tree, X.iloc[i])
        return pred
        # end answer

    def show(self):
        """Plot the tree using matplotlib
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        import tree_plotter
        tree_plotter.createPlot(self._tree)
