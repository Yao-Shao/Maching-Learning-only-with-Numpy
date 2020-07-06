import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE

    # begin answer
    dists = np.sum(x**2, axis=1).reshape((-1,1)) + np.sum(x_train**2, axis=1) - 2 * x @ x_train.T
    idx = np.argsort(dists, axis=1)[:,:k]
    yy_t = np.tile(y_train, (idx.shape[0],1))
    # tmp = y_train[idx]
    y = scipy.stats.mode(y_train[idx], axis=1)[0].flatten()
    # end answer

    return y
