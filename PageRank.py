import numpy as np

def solve_linear_system(A, B):
    """
    Résout un système d'équations linéaires Ax = B en utilisant l'élimination de Gauss.

    Paramètres :
    - A : np.matrix, matrice des coefficients du système d'équations.
    - B : np.array, vecteur de termes constants.

    Retourne :
    - np.array, vecteur solution x.
    """
    n = len(B)
    
    # Étape d'élimination
    for i in range(n):
        # Normalisation de la ligne courante
        A[i, :] /= A[i, i]
        B[i] /= A[i, i]

        # Élimination des éléments au-dessus et au-dessous de l'élément pivot
        for j in range(n):
            if i != j:
                ratio = A[j, i]
                A[j, :] -= ratio * A[i, :]
                B[j] -= ratio * B[i]

    # Rétro-substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = B[i] - np.dot(A[i, i + 1:], x[i + 1:])

    return x

def pageRankLinear(A, alpha, v):
    """
    Calcule les scores PageRank en résolvant un système d'équations linéaires.

    Paramètres :
    - A : np.matrix, matrice d'adjacence d'un graphe dirigé, pondéré et régulier.
    - alpha : float, paramètre de téléportation compris entre 0 et 1.
    - v : np.array, vecteur de personnalisation.

    Retourne :
    - np.array, vecteur x contenant les scores d'importance des nœuds.
    """
    n = len(A)
    I = np.identity(n)
    G = alpha * A + (1 - alpha) * np.outer(v, np.ones(n))
    
    # Résolution du système d'équations linéaires sans utiliser np.linalg.solve
    x = solve_linear_system(I - G, np.ones(n))
    
    return x
