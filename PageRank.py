import numpy as np

def pageRankPower(A, alpha=0.9, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    
    # Initialiser le vecteur de probabilités
    x = np.ones((n, 1)) / n
    
    # Calculer le vecteur propre dominant avec la power method
    for i in range(max_iter):
        x_next = alpha * np.dot(A.T, x) + (1 - alpha) / n
        if np.linalg.norm(x_next - x) < tol:
            break
        x = x_next

    return x_next

# Exemple d'utilisation avec une matrice d'adjacence simple
adjacency_matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 1, 0]])
pagerank_scores = pageRankPower(adjacency_matrix)

print("Scores PageRank:", pagerank_scores)