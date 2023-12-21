import numpy as np

def pageRankPower(A: np.matrix, alpha: float=0.9, v: np.array=None):
    x = v
    for _ in range(A.shape[0]):
        P = np.dot(A, x.T)
        G = alpha * P + (1 - alpha) * v.T
        if np.allclose(G, x):  
            break
        x = G
    return x / sum(x)

if __name__ == '__main__':

    A= np.matrix([[0,0,1],[2,0,1],[0,2,0]])
    alpha = 0.9
    v= np.array([1,2,3])

    result = pageRankPower(A,alpha,v)

    print(result)
