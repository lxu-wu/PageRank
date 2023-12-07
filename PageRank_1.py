import numpy as np


def PageRankLinear(A,alpha,v):
    normalized_A_col = A / A.sum(axis=0)
    pageRank_One = np.identity(A.shape[0])-np.multiply(alpha,normalized_A_col)
    pageRank_Two = np.multiply((1-alpha),np.linalg.inv(pageRank_One))
    pageRank_Final = np.matmul(v,pageRank_Two)
    return pageRank_Final


print(PageRankLinear(np.matrix([[0,0,1/3,0],[1/2,0,1/3,0],[1/2,0,0,1],[0, 1, 1/3, 0]]),0.85,(0.2,0.5,0.3,0.2)))
