import numpy as np


def PageRankLinear(A,alpha,v):
    normalized_A_col = A / A.sum(axis=0)
    pageRank_One = np.identity(3)-np.multiply(0.85,normalized_A_col)
    pageRank_Two = np.multiply((1-alpha),np.linalg.inv(pageRank_One))
    pageRank_Final = np.matmul(v,pageRank_Two)
    return pageRank_Final


print(PageRankLinear(np.matrix([[0,1,1],[1,0,0],[0,1,0]]),0.85,(0.2,0.5,0.3)))
