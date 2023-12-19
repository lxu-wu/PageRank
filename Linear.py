import numpy as np


def PageRankLinear(A,alpha,v):
    M_norm = A / A.sum(axis=1)
    pageRank_One = np.identity(A.shape[0])-np.multiply(alpha,M_norm)
    pageRank_Two = np.multiply((1-alpha),np.linalg.inv(pageRank_One))
    x = np.matmul(v,pageRank_Two)
    return x / x.sum()


A= np.matrix([[0,0,1],[2,0,1],[0,2,0]])

v= np.array([1,2,3])

print(PageRankLinear(A,0.9,v))

N = 4  # Number of nodes
A = np.matrix([[0, 1, 1, 0], 
               [1, 0, 1, 1], 
               [0, 0, 0, 1], 
               [0, 0, 1, 0]])  # Example adjacency matrix
alpha = 0.9  # Teleportation parameter
v = np.array([1/N] * N)  # Uniform personalization vector

pageRankScores = PageRankLinear(A, alpha, v)
print(pageRankScores)
