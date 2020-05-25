import numpy as np

def perceptron(X, y, lr = 0.001, th = 1e-7, max_it = 1000):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P+1, 1))
    iters = 0
    X = np.vstack((np.ones((1, X.shape[1])), X))
    while(iters < max_it):
        z = np.sign(np.matmul(w.T, X))
        pos = z != y
        delta = lr * np.sum(np.multiply(X[:,pos[0]], y[0, pos[0]]), axis=1).reshape((-1,1))
        dd = np.linalg.norm(delta)
        w = w + delta
        iters += 1
    return w, iters