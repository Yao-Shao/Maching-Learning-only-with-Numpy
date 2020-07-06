import numpy as np
from kmeans import kmeans

def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    D = np.diag(np.sum(W > 0, axis=1))
    D_2 = np.diag(np.sum(W > 0, axis=1) ** (-1/2))
    
    L = D - W
    L_nml = np.dot(np.dot(D_2, L), D_2)
    
    eig_value, eig_vec = np.linalg.eig(L_nml)
    index = np.argsort(eig_value)[:1]
    eig_vec = eig_vec[:, index]
    
    return kmeans(eig_vec, k)
    # end answer
