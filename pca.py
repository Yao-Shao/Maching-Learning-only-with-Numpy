import numpy as np

def PCA(data):
    '''
    PCA	Principal Component Analysis

    Input:
      data      - Data numpy array. Each row vector of fea is a data point.
    Output:
      eigvector - Each column is an embedding function, for a new
                  data point (row vector) x,  y = x*eigvector
                  will be the embedding result of x.
      eigvalue  - The sorted eigvalue of PCA eigen-problem.
    '''

    # YOUR CODE HERE
    # Hint: you may need to normalize the data before applying PCA
    # begin answer
    # normalize
    data = data - np.mean(data, axis=1, keepdims=True)

    C = np.cov(data.T)
    eig_value, eig_vec = np.linalg.eig(C)
    idx = np.argsort(eig_value)[::-1]

    return eig_value[idx], eig_vec[:, idx]
    # end answer