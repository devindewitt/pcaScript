"""
Author: Devin DeWitt
File: pcaScript.py

Purpose: Accept a data set with dimension M and project it onto dimension D
"""

import sys
import os
import csv
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

# Uncomment code below to generate random data to test pca with
# Building normal distribution and generating random data
# rng = np.random.RandomState(1)
# mu, sigma = 6, 4
# #Y = np.dot(rng.rand(baseD, baseD), rng.normal(mu, sigma, (baseD, numData))).T

# Get file path from user
if len(sys.argv) != 2:
    file_path = input("Enter the path of your file: ")
else:
    file_path = str(sys.argv[1])

assert os.path.exists(file_path), "File path, " + str(file_path) + ", not found."

# Read csv file into list
data = []
with open(file_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data.append(row)
        line_count += 1


# Variables to define: feature vector dimension, projected dimension, and data set size
projD = 3
numData = len(data)
baseD = len(row)
Y = np.zeros((numData, baseD))
X = np.zeros((projD, numData))

for i in range(numData):
    temp = data[i]
    for j in range(baseD):
        Y[i][j] = temp[j]

# Normalizing the mean in the data
Yhat = []
for i in range(baseD):
    Yhat.append(1 / numData * np.sum(Y[:, i]))

Ynorm = Y - Yhat

# Building covariance matrix
C = 1 / numData * np.matmul(Ynorm.T, Ynorm)

# Find eigenvalue/eigenvector pairs
w, v = la.eig(C)

# Creating indexed matrix of eigenvalue/eigenvector pairs
dtype = [('Eigenvalue', float), ('Eigenvector', np.ndarray)]
values = []
for i in range(baseD):
    values.append((w[i], v[:, i]))
eigMat = np.array(values, dtype=dtype)

# Sorting matrix in descending order of eigenvalues
eigMat = np.sort(eigMat, order='Eigenvalue')
eigMat[::-1].sort()

# Projecting onto chosen dimensions
for i in range(projD):
    X[i] = np.matmul(Y, eigMat[i]['Eigenvector'])

# Projecting data to 1, 2, and 3 dimensions
fig2d = plt.figure(1)
plt.scatter(X[0, :], X[1, :])
plt.axis('equal')
fig2d.show()

fig3d = plt.figure(2)
ax = fig3d.gca(projection='3d')
ax.plot_trisurf(X[0, :], X[1, :], X[2, :], linewidth=0.2, antialiased=True)
fig3d.show()

fig1d = plt.figure(3)
zero = np.zeros((1, numData))
plt.scatter(X[0, :], zero)
fig1d.show()

plt.show()
