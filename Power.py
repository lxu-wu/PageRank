import numpy as np

def PageRankPower(A: np.matrix, alpha: float=0.9, v: np.array=None):
    x = v
    for _ in range(A.shape[0]):
        P = np.dot(A, x.T)
        G = alpha * P + (1 - alpha) * v.T
        if np.allclose(G, x):  
            break
        x = G
    return x / sum(x)