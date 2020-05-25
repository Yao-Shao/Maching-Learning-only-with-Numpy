"""
This is the starter code and some suggested architecture we provide you with. 
But feel free to do any modifications as you wish or just completely ignore 
all of them and have your own implementations.
"""
import numpy as np
import scipy.io
from scipy import stats
import random
import time
import math

####################### config ########################
MAX_DEPTH = 3
NODE_PUTITY_THRESH = 0 # 0.5 - 1.0
IG_THRESH = 0 # 0.001 - 0.9, mostly less than 0.1
#######################################################

class DecisionTree:
    label_list = [0, 1]

    def __init__(self, features, max_depth=3, npt=0, igt=0):
        """
        TODO: initialization of a decision tree
        """
        # hyper_params
        self.NODE_PUTITY_THRESH = npt
        self.IG_THRESH = igt
        self.max_depth = max_depth

        self.features = features

        self.left = None
        self.right = None
        self.split_id = None
        self.thresh = None
        self.data = None
        self.labels = None
        self.pred = None
        
    @staticmethod
    def entropy(y):
        """
        TODO: implement a method that calculates the entropy given all the labels
        """
        if y.shape[0] == 0:
            return 0
        num = np.sum(y < 0.5)
        p = num / y.shape[0]
        if p < 1e-10 or 1-p < 1e-10:
            return 0
        res = -p * math.log(p, 2) - (1-p) * math.log(1-p,2)
        return res

    @staticmethod
    def information_gain(X, y, thresh, total_entr):
        """
        TODO: implement a method that calculates information gain given a vector of features
        and a split threshold
        """
        y0 = y[np.where(X < thresh)[0]]
        p0 = y0.size / y.size
        y1 = y[np.where(X >= thresh)[0]]
        p1 = y1.size / y.size
        sub_entr = p0*DecisionTree.entropy(y0) + p1*DecisionTree.entropy(y1)
        return total_entr - sub_entr

    @staticmethod
    def gini_impurity(y):
        """
        TODO: implement a method that calculates the gini impurity given all the labels
        """
        if y.shape[0] == 0:
            return 0
        res = 1
        for label in DecisionTree.label_list:
            p = np.sum(y == label) / y.shape[0]
            res -= p ** 2
        return res

    @staticmethod
    def gini_purification(X, y, thresh):
        """
        TODO: implement a method that calculates reduction in impurity gain given a vector of features
        and a split threshold
        """
        total_gini = DecisionTree.gini_impurity(y)
        y0 = y[np.where(X < thresh)[0]]
        p0 = y0.size / y.size
        y1 = y[np.where(X >= thresh)[0]]
        p1 = y1.size / y.size
        sub_gini = p0 * DecisionTree.gini_impurity(y0) + p1 * DecisionTree.gini_impurity(y1)
        return total_gini - sub_gini

    def split(self, X, y, idx, thresh):
        """
        TODO: implement a method that return a split of the dataset given an index of the feature and
        a threshold for it
        """
        Xi = X[:, idx]
        X0 = X[np.where(Xi < thresh)[0], :]
        y0 = y[np.where(Xi < thresh)[0], :]
        X1 = X[np.where(Xi >= thresh)[0], :]
        y1 = y[np.where(Xi >= thresh)[0], :]
        return X0, y0, X1, y1
    
    def segmenter(self, X, y):
        """
        TODO: compute entropy gain for all single-dimension splits,
        return the feature and the threshold for the split that
        has maximum gain
        """
        max_id = 0
        max_thresh = 0
        max_ig = 0
        total_entr = DecisionTree.entropy(y)
        for i in range(X.shape[1]):
            Xi = X[:, i]
            for thresh in np.unique(Xi):
                ig = DecisionTree.information_gain(Xi,y,thresh, total_entr)
                if ig > max_ig:
                    max_id = i
                    max_thresh = thresh
                    max_ig = ig
        return max_id, max_thresh, max_ig
    
    
    def train(self, X, y):
        """
        TODO: fit the model to a training set. Think about what would be 
        your stopping criteria
        """
        if self.max_depth > 0:
            self.split_id , self.thresh, max_ig = self.segmenter(X,y)
            X0, y0, X1, y1 = self.split(X,y,self.split_id, self.thresh)
            node_purity = max(y0.size,y1.size) / y.size
            # print("np: {}".format(node_purity))
            # print("ig: {}".format(max_ig))
            if X0.size > 0 and X1.size > 0 \
                    and node_purity > self.NODE_PUTITY_THRESH \
                    and max_ig > self.IG_THRESH:
                self.left = DecisionTree(self.features, self.max_depth-1, self.NODE_PUTITY_THRESH, self.IG_THRESH)
                self.left.train(X0, y0)
                self.right = DecisionTree(self.features, self.max_depth-1, self.NODE_PUTITY_THRESH, self.IG_THRESH)
                self.right.train(X1, y1)
            else:
                self.data = X
                self.labels = y
                self.pred = stats.mode(y).mode[0]
                self.max_depth = 0
        else:
            self.data = X
            self.labels = y
            self.pred = stats.mode(y).mode[0]

    def segmenter_bag(self, X, y, attr_list):
        max_id = 0
        max_thresh = 0
        max_ig = 0
        total_entr = DecisionTree.entropy(y)
        for i in attr_list:
            Xi = X[:, i]
            for thresh in np.unique(Xi):
                ig = DecisionTree.information_gain(Xi, y, thresh, total_entr)
                if ig > max_ig:
                    max_id = i
                    max_thresh = thresh
                    max_ig = ig
        return max_id, max_thresh, max_ig


    def train_bag(self, X, y, attr_list):
        if self.max_depth > 0:
            self.split_id , self.thresh, max_ig = self.segmenter_bag(X,y, attr_list)
            X0, y0, X1, y1 = self.split(X,y,self.split_id, self.thresh)
            node_purity = max(y0.size,y1.size) / y.size
            # print("np: {}".format(node_purity))
            # print("ig: {}".format(max_ig))
            if X0.size > 0 and X1.size > 0 \
                    and node_purity > self.NODE_PUTITY_THRESH \
                    and max_ig > self.IG_THRESH:
                self.left = DecisionTree(self.features, self.max_depth-1, self.NODE_PUTITY_THRESH, self.IG_THRESH)
                self.left.train(X0, y0)
                self.right = DecisionTree(self.features, self.max_depth-1, self.NODE_PUTITY_THRESH, self.IG_THRESH)
                self.right.train(X1, y1)
            else:
                self.data = X
                self.labels = y
                self.pred = stats.mode(y).mode[0]
                self.max_depth = 0
        else:
            self.data = X
            self.labels = y
            self.pred = stats.mode(y).mode[0]

    def predict(self, X):
        """
        TODO: predict the labels for input data 
        """
        if self.max_depth > 0:
            id0 = np.where(X[:,self.split_id] < self.thresh)[0]
            id1 = np.where(X[:,self.split_id] >= self.thresh)[0]
            X0 = X[id0, :]
            X1 = X[id1, :]
            y_hat = np.zeros((X.shape[0],1))
            y_hat[id0] = self.left.predict(X0)
            y_hat[id1] = self.right.predict(X1)
        else:
            y_hat = self.pred * np.ones((X.shape[0],1))
        return y_hat

    def __repr__(self):
        """
        TODO: one way to visualize the decision tree is to write out a __repr__ method
        that returns the string representation of a tree. Think about how to visualize 
        a tree structure. You might have seen this before in CS61A.
        """
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (
                self.features[self.split_id], self.thresh,
                self.left.__repr__(), self.right.__repr__())


class RandomForest():
    
    def __init__(self, num_dt, num_data_bag, num_attr_bag, features, max_depth=3, npt=0, igt=0):
        """
        TODO: initialization of a random forest
        """
        self.num_dt = num_dt
        self.num_data_bag = num_data_bag
        self.num_attr_bag = num_attr_bag
        self.dt_list = [DecisionTree(features, max_depth, npt, igt) for i in range(num_dt)]

    def fit(self, X, y):
        """
        TODO: fit the model to a training set.
        """
        for dt in self.dt_list:
            np.random.seed(np.random.randint(0, self.num_dt*100))
            Xy = np.column_stack((X, y))
            np.random.shuffle(Xy)
            XX = Xy[:, :X.shape[1]]
            yy = Xy[:, X.shape[1]:]
            XX = XX[:self.num_data_bag, :]
            yy = yy[:self.num_data_bag, :]

            attr_list = np.arange(X.shape[1])
            np.random.shuffle(attr_list)
            attr_list = attr_list[:self.num_attr_bag]
            dt.train_bag(XX, yy, attr_list)
    
    def predict(self, X):
        """
        TODO: predict the labels for input data 
        """
        yy_hat = np.zeros((X.shape[0], self.num_dt))
        for i, cur_dt in enumerate(self.dt_list):
            yy_hat[:,i] = np.concatenate(cur_dt.predict(X))
        y_hat = stats.mode(yy_hat, axis=1).mode
        return y_hat

if __name__ == "__main__":

   
    features = [
        "pain", "private", "bank", "money", "drug", "spam", "prescription",
        "creative", "height", "featured", "differ", "width", "other",
        "energy", "business", "message", "volumes", "revision", "path",
        "meter", "memo", "planning", "pleased", "record", "out",
        "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
        "square_bracket", "ampersand"
    ]
    assert len(features) == 32

    # Load spam data
    path_train = 'datasets/spam-dataset/spam_data.mat'
    data = scipy.io.loadmat(path_train)
    X = data['training_data']
    y = np.squeeze(data['training_labels']).reshape((-1,1))
    class_names = ["Ham", "Spam"]
     

    """
    TODO: train decision tree/random forest on different datasets and perform the tasks 
    in the problem
    """

    # shuffle and split dataset
    np.random.seed(0)
    Xy = np.column_stack((X,y))
    np.random.shuffle(Xy)
    X = Xy[:,:X.shape[1]]
    y = Xy[:,X.shape[1]:]
    num_train = int(X.shape[0] * 0.8)
    X_train = X[:num_train,:]
    y_train = y[:num_train,:]
    X_valid = X[num_train:,:]
    y_valid = y[num_train:,:]

    best_acc = 0
    depth = 11
    npt_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    ig_list = [0.001, 0.01, 0.05, 0.1, 0.5, 0.9]
    for npt in npt_list:
        for ig in ig_list:
            dt = DecisionTree(features, depth, npt, ig)
            dt.train(X_train, y_train)
            y_hat_tr = dt.predict(X_train)
            y_hat_va = dt.predict(X_valid)
            acc_tr = np.sum(y_hat_tr == y_train) / y_train.size * 100
            acc_va = np.sum(y_hat_va == y_valid) / y_valid.size * 100
            print(
                "depth: {:5}, npt: {:5}, ig: {:5}, training acc: {:.3f}%, validation acc: {:.3f}%".format(dt.max_depth,
                                                                                                          npt, ig,
                                                                                                          acc_tr,
                                                                                                          acc_va))
            if acc_va > best_acc:
                best_npt = npt
                best_ig = ig
                best_acc = acc_va
    print("\nThe best hparams: ")
    print("\nNode purity thresh: {} \nInfomation gain thresh: {}\nBest validation accuracy: {:.3f}".format(best_npt,
                                                                                                           best_ig,
                                                                                                           best_acc))

    # hyper_params
    """
    dt = DecisionTree(features, 1, NODE_PUTITY_THRESH, IG_THRESH)
    dt.train(X_train, y_train)
    y_hat_tr = dt.predict(X_train)
    y_hat_va = dt.predict(X_valid)
    acc_tr = np.sum(y_hat_tr == y_train) / y_train.size * 100
    acc_va = np.sum(y_hat_va == y_valid) / y_valid.size * 100
    print("depth: {}, training acc: {:.5f}%, validation acc: {:.5f}%".format(dt.max_depth, acc_tr, acc_va))
    print(dt)
    """

    # train decision tree of diff depth
    """
    for depth in range(1,41):
        dt = DecisionTree(features, depth, NODE_PUTITY_THRESH, IG_THRESH)
        dt.train(X_train, y_train)
        y_hat_tr = dt.predict(X_train)
        y_hat_va = dt.predict(X_valid)
        acc_tr = np.sum(y_hat_tr == y_train) / y_train.size * 100
        acc_va = np.sum(y_hat_va == y_valid) / y_valid.size * 100
        print("depth: {}, training acc: {:.3f}%, validation acc: {:.3f}%".format(dt.max_depth, acc_tr, acc_va))
    """

    # train random forest
    """
    print('*'*100)
    print("Training Random Forest")
    num_dt = 7
    depth = 11
    num_data = 2800
    num_attr = 18
    rf = RandomForest(num_dt, num_data, num_attr, features, depth, NODE_PUTITY_THRESH, IG_THRESH)
    rf.fit(X_train, y_train)
    y_hat_tr = rf.predict(X_train)
    y_hat_va = rf.predict(X_valid)
    acc_tr = np.sum(y_hat_tr == y_train) / y_train.size * 100
    acc_va = np.sum(y_hat_va == y_valid) / y_valid.size * 100
    print("num_dt: {}, num_data: {}, num_attr: {}, training acc: {:.3f}%, validation acc: {:.3f}%".format(rf.num_dt,
                                                                                                          rf.num_data_bag,
                                                                                                          rf.num_attr_bag,
                                                                                                          acc_tr,
                                                                                                          acc_va))
    """



