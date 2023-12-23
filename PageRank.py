import numpy as np

def PageRankLinear(A: np.matrix ,alpha: float ,v:np.array):
     """
    Calcule le PageRank d'un graphe représenté par une matrice d'adjacence.

    
    :param A (np.matrix): Matrice d'adjacence du graphe. Chaque élément A[i, j] représente un lien du nœud i au nœud j.
    :param alpha (float): Facteur de téléportation 
    :param v (np.array): Vecteur de personnalisation. Il doit avoir la même longueur que le nombre de nœuds dans le graphe.
    :return np.array: Vecteur de PageRank normalisé pour le graphe, représentant l'importance de chaque nœud.

    """
     # Calcul de la matrice de transition de probabilité P à partir de la matrice d'adjacence A
     P = A / A.sum(axis=1)
     # Création de la matrice identité I de la même taille que P
     I = np.identity(P.shape[0])
     
     # Formation de la matrice pour le système linéaire, en soustrayant alpha * P de I, puis en transposant le résultat
     # Résolution du système linéaire pour obtenir le vecteur de PageRank
     A = (I - np.multiply(alpha,P)).transpose()
     PR = np.linalg.solve(A,v.transpose())

     # Normalisation du vecteur de PageRank
     # Retour du vecteur de PageRank normalisé 
     PR = PR/PR.sum()
     return PR.transpose().flatten()


def googleMatrix(A: np.matrix, alpha: float, v: np.array):
    """
    Calcule la valeur de la matrice google.
    :param A (np.matrix): Matrice d'adjacence du graphe. Chaque élément A[i, j] représente un lien du nœud i au nœud j.
    :param alpha (float): Facteur de téléportation 
    :param v (np.array): Vecteur de personnalisation. Il doit avoir la même longueur que le nombre de nœuds dans le graphe.
    :return np.array: Matrice google normalisée
    """
    n = len(A) # = len(A) taille de A (nxm matrice)
    P = A / A.sum(axis=1)
    print(f"Matrice de probabilités de transition P: \n{P}\n")
    e = np.ones((n,1)) # vecteur-colonne de 1
    G = alpha * P + (1 - alpha) * np.outer(e, np.transpose(v)) # calcule la matrice google (G)
    return np.transpose(G)

def pageRankPower(A : np.matrix, alpha : float, v : np.array):
    """
    Calcule le PageRank d'un graphe représenté par une matrice d'adjacence.
    :param A (np.matrix): Matrice d'adjacence du graphe. Chaque élément A[i, j] représente un lien du nœud i au nœud j.
    :param alpha (float): Facteur de téléportation 
    :param v (np.array): Vecteur de personnalisation. Il doit avoir la même longueur que le nombre de nœuds dans le graphe.
    :return np.array: Vecteur de PageRank normalisé pour le graphe, représentant l'importance de chaque nœud.
    """
    tol=1e-8 # tolérance
    n = A.shape[0]
    G = googleMatrix(A,alpha,v)
    i = 0
    x = np.ones(n) / n
    print(f"Matrice adjacence A: \n{A}\n")
    print(f"Matrice Google G: \n{G}\n")
    i = 0 # nombre d'itérations de l'algorithme PageRank 
    while 1:
        i+=1 # nouvelle l'itération
        x_new = np.matmul(G,x) # G*x
        x_new = x_new / np.sum(x_new) # normalise x
        if i <= 3:
            print(f"Itérations de PageRank N°{i}\n" , np.array(x_new).flatten(), "\n")
        if np.linalg.norm(x_new - x) < tol:
            # si la distance euclidienne entre x et x_new
            # est strictement plus petite que la tolérence, 
            # alors l'algorithme est stable et le calcul est fini.
            break
        x = np.array(x_new).flatten() # Réassigne x a x_new (nouveau x)
    print("PageRank final obtenu de la méthode power: ")
    return np.array(x_new).flatten()
