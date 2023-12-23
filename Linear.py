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
     return PR.transpose()


if __name__ == '__main__':

    A= np.matrix([[0,0,1],[2,0,1],[0,2,0]])
    alpha = 0.9
    v= np.array([1,2,3])

    result = PageRankLinear(A,alpha,v)

    print(result)

