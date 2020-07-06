import numpy as np
import time

def kmeans(x, k, max_it=20):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
                err  - total square distance
    '''
    # YOUR CODE HERE

    # begin answer
    np.random.seed(int(time.time()))
    N = x.shape[0]
    p = x.shape[1]
    ctrs = x[np.random.randint(N, size=k)]
    iter_ctrs = np.zeros((max_it, k, p))
    idx = np.zeros(N, dtype=np.int64)
    dist = np.zeros((N, k))
    cnt = 0
    for i in range(max_it):
        # print(i)
        for j, ctr in enumerate(ctrs):
            dist[:,j] = np.linalg.norm(x-ctr, axis=1)
        # some dist may be NAN, so must use nanargmin to ignore these values.
        iidx = np.nanargmin(dist, axis=1)
        if((iidx == idx).all()):
            break
        idx = iidx
        for j in range(k):
            ctrs[j,:] = np.mean(x[idx == j], 0)
        iter_ctrs[i] = ctrs
        cnt += 1
    for j, ctr in enumerate(ctrs):
        dist[:, j] = np.linalg.norm(x - ctr, axis=1)
    err = np.sum(np.min(dist, axis=1))
    iter_ctrs = iter_ctrs[:cnt,:]
    # end answer
    return idx, ctrs, iter_ctrs, err
