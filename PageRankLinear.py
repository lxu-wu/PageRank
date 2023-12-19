import numpy as np

def PageRankLinear(A, alpha, v):
     transitionProbabilityMatrix = A / A.sum(axis=1)
     lenght = transitionProbabilityMatrix.shape[0]

     A = (np.identity(lenght) - np.multiply(alpha,transitionProbabilityMatrix)).transpose()
     G = np.linalg.solve(A,v.transpose())
     G = G/G.sum()

     return G


A = np.matrix([[0,0,1],[2,0,1],[0,2,0]])

v= np.array([1,2,3])

print(PageRankLinear(A,0.9,v))