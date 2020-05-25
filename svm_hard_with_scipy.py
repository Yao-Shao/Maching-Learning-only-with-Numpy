import numpy as np
from scipy import optimize

def obj(w):
    return 0.5 * np.sum(w ** 2)

def get_num(X, y, w):
	'''
	return # of suport vectors
	'''
    eps = 1e-5
    return np.sum(w.T @ X - y - 1 < eps)

def cons_fun(w, X, y):
    return np.squeeze(w.T @ X * y - 1)

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num = 0
    XX = np.vstack((np.ones((1, X.shape[1])), X))
    constrs = {'type': 'ineq', 'fun': cons_fun, 'args': (XX, y)}
    opt1 = optimize.minimize(obj, w, constraints=constrs)
    w = opt1.x
    num = get_num(XX, y, w)

    return w, num
