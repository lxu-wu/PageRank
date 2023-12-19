import unittest
import numpy as np
from Linear import PageRankLinear
from Power import PageRankPower


class TestPageRankLinearFunction(unittest.TestCase):

    def test_pageRankLinear_1(self):
        # Test avec des valeurs par défaut (alpha=0.9, pas de vecteur de personnalisation)
        A= np.matrix([[0,0,1],[2,0,1],[0,2,0]])
        alpha = 0.9
        v= np.array([1,2,3])
        
        result = PageRankLinear(A,alpha,v)
        expected_result = np.array([0.24248634, 0.37636612, 0.38114754])
        np.testing.assert_array_almost_equal(result, expected_result)
    
    def test_pageRankLinear_2(self):
        N = 4  # Number of nodes
        A = np.matrix([[0, 1, 1, 0], 
               [1, 0, 1, 1], 
               [0, 0, 0, 1], 
               [0, 0, 1, 0]]) 
        alpha = 0.9 
        v = np.array([1/N] * N)  # Uniform personalization vector
        
        result = PageRankLinear(A,alpha,v)
        expected_result = np.array([0.03757225, 0.04190751, 0.46470946, 0.45581077])
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_pageRankPower_1(self):
        # Test avec des valeurs par défaut (alpha=0.9, pas de vecteur de personnalisation)
        A= np.matrix([[0,0,1],[2,0,1],[0,2,0]])
        alpha = 0.9
        v= np.array([1,2,3])
        
        result = PageRankPower(A,alpha,v)
        expected_result = np.array([0.24248634, 0.37636612, 0.38114754])
        np.testing.assert_array_almost_equal(result, expected_result)
    
    def test_PageRankPower_2(self):
        N = 4  # Number of nodes
        A = np.matrix([[0, 1, 1, 0], 
               [1, 0, 1, 1], 
               [0, 0, 0, 1], 
               [0, 0, 1, 0]]) 
        alpha = 0.9 
        v = np.array([1/N] * N)  # Uniform personalization vector
        
        result = PageRankLinear(A,alpha,v)
        expected_result = np.array([0.03757225, 0.04190751, 0.46470946, 0.45581077])
        np.testing.assert_array_almost_equal(result, expected_result)
    


if __name__ == '__main__':
    unittest.main()
