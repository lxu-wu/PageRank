import numpy as np

def calculofG(A: np.matrix, alpha: float, v: np.array):
    P = A / A.sum(axis=1)
    e = np.ones(len(A))
    if v is None:
        v = e / len(A)
    G = alpha * P + (1 - alpha) * np.outer(e, v/sum(v))
    return G


def pageRankPower(A: np.matrix, alpha= 0.9, v = None):
    print("Matrice d'adjacense\n", A) # Imprimer la matrice d'adjacense A
    tolerance = 1e-8
    G = calculofG(A,alpha,v)
    x = (np.sum(A, axis=0) / np.sum(np.sum(A, axis=0))).reshape(-1, 1) # initialiser les scores par le le degré entrant (indegree) de chaque noeud 
    x = x / np.sum(x) # normaliser les scores
    while (True):
        x_new = G.T @ x # P^T * x
        x_new /= np.sum(np.abs(x_new)) # x/||x||
        if np.linalg.norm(x_new - x, ord=1) < tolerance:
            return np.array(x_new).flatten()
        x = x_new

# def pageRankPower(A: np.matrix, alpha: float=0.9, v: np.array=None):
#     x = v
#     for _ in range(A.shape[0]):
#         P = np.dot(A, x.T)
#         G = alpha * P + (1 - alpha) * v.T
#         if np.allclose(G, x):  
#             break
#         x = G
#     return x / sum(x)

if __name__ == '__main__':

    A= np.matrix([[0,0,1],[2,0,1],[0,2,0]])
    alpha = 0.9
    v= np.array([1,2,3])

    result = pageRankPower(A,alpha,v)

    print(result)
