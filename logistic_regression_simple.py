import numpy as np

def h(X, w):
    return  1 / (np.exp(-w.T @ X) + 1)

def get_loss(X, y, w):
    z_tr = h(X, w) > 0.5
    z_tr = z_tr.astype(np.float)
    z_tr[z_tr == 0] = -1
    return np.sum(z_tr != y) / X.shape[1]

def logistic(X, y, lr = 0.01, th = 0.001, max_it = 1000):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    XX = np.vstack((np.ones((1, X.shape[1])), X))
    it = 0
    loss = 1

    while(it < max_it and loss >= th):
        coe = y - h(XX, w)
        if(np.sum(coe) == 0):
            break;
        delta = (1 / N * np.average(XX.T, axis=0, weights=coe[0]) * np.sum(coe)).reshape((-1, 1))
        w = w + lr * delta
        it += 1

        loss = get_loss(XX, y, w)
    
    return w, it
