import numpy as np

def normalize(v):
    return v / np.sum(v)

def googleMatrix(A: np.matrix, alpha: float, v: np.array):
    n = A.shape[0]
    P = A / A.sum(axis=0)
    e = np.ones(n)
    G = alpha * P + (1 - alpha) * np.outer(e, v/sum(v))*n
    return G

def pageRankPower(A : np.matrix, alpha : float, v : np.array):
    tol=1e-8
    n = A.shape[0]
    G = googleMatrix(A,alpha,v)
    x = np.ones(n) / n
    while 1:
        x_new = np.matmul(G,x)
        x_new = normalize(x_new)
        if np.linalg.norm(x_new - x, 1) < tol:
            break
        x = np.array(x_new).flatten()
    return np.array(x_new).flatten()

if __name__ == '__main__':

    A= np.matrix([[0,0,1],[2,0,1],[0,2,0]])
    alpha = 0.9
    v= np.array([1,2,3])

    result = pageRankPower(A,alpha,v)

    print(result)