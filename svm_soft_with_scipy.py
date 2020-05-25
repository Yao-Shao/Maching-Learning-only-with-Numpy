import numpy as np
from scipy import optimize

def soft_svm(X, y, C = 1):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    theta = np.zeros((P + 1 + N, 1))
    num = 0
    def obj(theta):
        w = theta[:P+1]
        kesi = theta[P+1:]
        return 0.5 * np.sum(w ** 2) + C * np.sum(kesi)

    def cons_fun(theta, X, y):
        w = theta[:P + 1]
        kesi = theta[P + 1:]
        return np.hstack((np.squeeze(w.T @ X * y - 1 + kesi), kesi))

    XX = np.vstack((np.ones((1, X.shape[1])), X))
    constrs = {'type': 'ineq', 'fun': cons_fun, 'args': (XX, y)}
    opt1 = optimize.minimize(obj, theta, constraints=constrs)
    w = opt1.x[:P+1]
    return w
