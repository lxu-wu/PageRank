import numpy as np


def PageRankLinear(A: np.matrix ,alpha: float ,v:np.array):
    M_norm = A / A.sum(axis=1)

    pageRank_One = np.identity(A.shape[0])-np.multiply(alpha,M_norm)
    pageRank_Two = np.multiply((1-alpha),np.linalg.inv(pageRank_One))
    x = np.matmul(v,np.array(pageRank_Two))
    return x / x.sum()

def PageRankLinearSys(A, alpha, v):
     transitionProbabilityMatrix = A / A.sum(axis=1)
     lenght = transitionProbabilityMatrix.shape[0]

     A = (np.identity(lenght) - np.multiply(alpha,transitionProbabilityMatrix)).transpose()
     G = np.linalg.solve(A,v.transpose())
     G = G/G.sum()

     return G
