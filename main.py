import csv
import numpy as np
from PageRank import PageRankLinear, pageRankPower
# from Linear import PageRankLinear
# from Power import pageRankPower

matrix_csv = 'matrice_adgacence.csv'
vector_csv = 'G15-personalisation_vector.csv'


matrix = []
vector = []
with open(matrix_csv, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        matrix.append([int(val) for val in row])

with open(vector_csv, 'r') as file:
    csv_reader = csv.reader(file)
    l = 0
    for row in csv_reader:
        if l ==1:
            vector.append([float(val) for val in row])
        l+=1


matrix_np = np.matrix(matrix)
vector_np = np.array(vector)


print("PageRank les informations pour la méthode power: \n")
print(pageRankPower(matrix_np, 0.9, vector_np), "\n")

print("PageRank final obtenu de manière linéaire: ")
print(PageRankLinear(matrix_np, 0.9, vector_np))