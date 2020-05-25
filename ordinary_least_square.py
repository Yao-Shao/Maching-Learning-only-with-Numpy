import numpy as np
import copy

def linear_regression(X, y, k = 2):
    '''
    LINEAR_REGRESSION Linear Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.
            k: # of classes

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    yy = copy.deepcopy(y)

    X = np.vstack((np.ones((1, X.shape[1])), X))
    yy[yy==1] = False
    yy[yy==-1] = True
    yy = yy.astype(np.int32)
    Y = np.eye(k)[yy[0]]
    assert(Y.shape == (N,k))

    ww = np.linalg.pinv(X @ X.T) @ X @ Y

    w = (ww[:,0] - ww[:,1]).reshape((-1,1))

    assert(w.shape == (P+1,1))
    return w
