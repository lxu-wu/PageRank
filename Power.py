import numpy as np

def normalize_adjacency_matrix(A):
    """
    Normalise la matrice d'adjacence A pour qu'elle soit stochastique par colonnes.

    :param A: Matrice d'adjacence.
    :return: Matrice d'adjacence normalisée.
    """
    column_sums = np.sum(A, axis=0)
    # Pour éviter la division par zéro, remplacer les zéros par des uns (ce qui ne change pas les colonnes nulles)
    column_sums[column_sums == 0] = 1
    return A / column_sums

def normalize_vector(v):
    """ Normalise un vecteur pour que sa somme soit égale à 1. """
    return v / np.sum(v)

def pageRankPower(A : np.matrix, alpha : float, v : np.array):
    """
    Calcule le PageRank en utilisant la méthode de puissance.

    :param A: Matrice d'adjacence (doit être stochastique par colonnes).
    :param alpha: Le facteur de téléportation.
    :param v: Vecteur de personnalisation.
    :param tol: La tolérance pour la convergence.
    :param max_iter: Le nombre maximal d'itérations.
    :return: Le vecteur des scores PageRank.
    """
    tol=1e-8
    max_iter=100
    
    v = normalize_vector(v)
    A = normalize_adjacency_matrix(A)

    # Nombre de noeuds dans le graphe
    n = A.shape[0]
    
    # Construire la matrice Google G
    G = alpha * A + (1 - alpha) * np.outer(np.ones(n), v)
    
    # Initialiser le vecteur PageRank x
    x = np.ones(n) / n
    
    print(G)
    print(x)

    # Boucle d'itération de la méthode de puissance
    for _ in range(2):
        x_new = G @ x  # Calculer le nouveau vecteur PageRank
        x_new = normalize_vector(x_new)  # Normaliser avec la norme L1
        
        # Vérifier la convergence
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


import numpy as np

# def pageRankPower(A: np.matrix, alpha= 0.9, v = None):
#     print("Matrice d'adjacense\n", A) # Imprimer la matrice d'adjacense A
#     tolerance = 1e-8
#     maxiterations = 500
#     G = calculofG(A,alpha,v)
#     x = (np.sum(A, axis=0) / np.sum(np.sum(A, axis=0))).reshape(-1, 1) # initialiser les scores par le le degré entrant (indegree) de chaque noeud 
#     x = x / np.sum(x) # normaliser les scores
#     for i in range(maxiterations):
#         x_next = G.T @ x # P^T * x
#         x_next /= np.sum(np.abs(x_next)) # x/||x||
#         if i <= 3:
#             print("Itérations de power methode N°",i, "\n" , np.array(x_next).flatten()) # Impression des trois premières itérations de la power method (vecteur de scores)
#         if np.linalg.norm(x_next - x, ord=1) < tolerance:
#             return np.array(x_next).flatten()
#         x = x_next
#     return np.array(x_next).flatten()

# def calculofG(A: np.matrix, alpha: float, v: np.array):
#     P = A / A.sum(axis=1)
#     print("Matrice de Probabilité (P)\n", P) # Imprimer la matrice de probabilité P
#     e = np.ones(len(A))
#     if v is None:
#         v = e / len(A)
#     G = alpha * P + (1 - alpha) * np.outer(e, v/sum(v))
#     print("Matrice Google (G)\n", G) # Imprimer la matrice Google G
#     return G