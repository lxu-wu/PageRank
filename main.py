import csv
import numpy as np
from Linear import PageRankLinearSys
from Power import pageRankPower

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


matrix_np = np.array(matrix)
vector_np = np.array(vector)

print(matrix_np)


print("PageRank final obtenu de manière linéaire: ")
print(PageRankLinearSys(matrix_np, 1, vector_np))
