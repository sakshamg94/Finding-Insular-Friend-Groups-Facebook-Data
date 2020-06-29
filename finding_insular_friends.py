# code written by Ray Iyer and Saksham Gakhar
# conceptual documentation at
# http://web.stanford.edu/~sakshamg/portfolio/socialnetworks/

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from collections import defaultdict
from random import sample

# read data and create the adjacency matrix(A) for Graph G =(V,E)
adj_matrix = np.zeros((1495, 1495))
with open('data.csv') as f:
    lines = f.readlines()
    for line in lines:
        i, j = line.split(',')
        adj_matrix[int(i)-1, int(j)-1] += 1

# calculate degree (D) of each of the vertices V
degrees = np.sum(adj_matrix, axis=0)

# calculate Laplacian: L = D - A
laplacian = adj_matrix * -1
for i in range(1495):
    laplacian[i, i] += degrees[i]

# calculate laplacian's eignevalues and vectors
eigvals, eigvecs = la.eig(laplacian)

for eigval in reversed(sorted(eigvals)):
    print(f'{round(float(eigval), 3)}')

# calculate the size of the connected components in G
# using spectral graph theory results
# see documentation
compSizes = []
EPSILON = 1e-12 
for i in range(1495):
    eigval = eigvals[i]
    eigvec = eigvecs[:, i]
    eigvec = np.round(eigvec, 6)
    if eigval <= EPSILON:
        counts = defaultdict(int)
        for val in eigvec:
            counts[val] += 1
        print(sorted(counts.values()))

# Finding insular groups
def separateNodesByThreshold(used, eigvector, mean, std):
    """
    Function to calculate the node indices that are within a group
    and ones that are outside the group. The within and outside group
    critera are based on whether elements of the eigenvector have
    values within +/- std.dev of the mean. If they do, then we call
    them within a group vertices else outside the group vertices
    
    We are forcing groups that have 150<=size<=750 and conductance 0.1
    Conductance is a measure of how disjoint the group is from the rest
    
    args:
        used: members of the group cached
        eigvector : the eigenvector of Laplaian under consideration
        mean: the mean value that vertices will have in their corresponding
                eigvector location to be considered a group
        std: standard deviation that acts as a bound on deviation away from mean
             for vertices to still be considered part of a group
    """
    withinSetIndices = []
    outsideSetIndices = []
    for i, val in enumerate(eigvector):
        if val >= mean - std and val <= mean + std:
            withinSetIndices.append(i)
        else:
            outsideSetIndices.append(i)

    print(f'SIZE: {len(withinSetIndices)}')
    assert(len(withinSetIndices) < 750 and len(withinSetIndices) >= 150)
    assert(len(used.intersection(set(withinSetIndices)))/float(1 if len(used) == 0 else len(used)) < .3)
    withinSetIndices = np.array(withinSetIndices)
    outsideSetIndices = np.array(outsideSetIndices)
    sumDegreeWithin = np.sum(laplacian[withinSetIndices, withinSetIndices])
    sumDegreeOutside = np.sum(laplacian[outsideSetIndices, outsideSetIndices])

    cumulative = 0
    for i in withinSetIndices:
        for j in outsideSetIndices:
            cumulative += adj_matrix[i, j]
    
    conductance = float(cumulative)/min(sumDegreeWithin, sumDegreeOutside)
    assert(conductance < .1)
    print(f'CONDUCTANCE: {conductance}')
    print(f'TEN RANDOM MEMBERS: {sample(list(withinSetIndices), 10)}')
    print('-' * 10)
    return withinSetIndices, outsideSetIndices


# Plot of node vs corresponding element of eigenvector
# Key point: nodes that are ~ in a group will have values
# at the eigenvector positions corresponding to their index
sorted_eigval_indices = np.argsort(eigvals)
cands = []
for i in cands:
    plt.figure()
    eigvec = eigvecs[:,sorted_eigval_indices[i]]
    plt.scatter(np.arange(1495), eigvec)
    plt.title(f'Plot of node vs corresponding element of eigenvector {i}')
    plt.xlabel('Node i')
    plt.ylabel('ith element of eigenvector')
    plt.show()


# 3 insular groups that have 150<=size<=750 and conductance 0.1
# these were found using trial and error approach by making the plots below
solutions = [(6, .042, .005), (29, .014, .003), (9, -.027, .002)]
cache = set()
for eig_index, mean, std in solutions:
    eigvec = eigvecs[:, sorted_eigval_indices[eig_index]]
    used_in_curr, outside = separateNodesByThreshold(cache, eigvec, mean, std)
    plt.figure()
    plt.scatter(np.arange(1495), eigvec)
    plt.title(f'Plot of Vector Corresponding to {eig_index + 1}th Smallest Eigenvalue')
    plt.xlabel('Entry Index')
    plt.ylabel('Value')
    plt.hlines(mean, 1, 1495, colors='r', linestyles='dashed')
    plt.hlines(mean-std, 1, 1495, colors='r', linestyles='dotted')
    plt.hlines(mean+std, 1, 1495, colors='r', linestyles='dotted')
    cache = cache.union(set(used_in_curr))

plt.show()

