import matplotlib.pyplot as plt
import random
import math
import networkx as nx
import numpy as np
from numpy.linalg import eigh

def Laplacian(A):
    n = len(A)
    adj = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                adj[i,j] = sum(A[i])
            else:
                adj[i,j] = -1*A[i,j]
    return adj


def mat_maze(vec):
    n = len(vec)
    mod = int(np.sqrt(n))
    vec = np.asarray(vec)
    adj = np.zeros((n,n))
    for i in range(n):
        if i==0: #Top left corner
            adj[i, i+mod] = vec[i]*vec[i+mod]
            adj[i, i+1] = vec[i]*vec[i+1]
        elif i == mod: #Bottom left corner
            adj[i,i-1] = vec[i]*vec[i-1]
            adj[i, i+1] = vec[i]*vec[i+1]
        elif i == (mod-1)*mod: #Top right corner
            adj[i,i-mod] = vec[i]*vec[i-mod]
            adj[i, i+1] = vec[i]*vec[i+1]
        elif i== mod**2-1: #Bottom right corner
            adj[i,i-mod] = vec[i]*vec[i-mod]
            adj[i,i-1] = vec[i]*vec[i-1]
        elif int(i/mod) == 0: #left border
            adj[i, i+mod] = vec[i]*vec[i+mod]
            adj[i, i+1] = vec[i]*vec[i+1]
            adj[i,i-1] = vec[i]*vec[i-1]
        elif (i%mod) == 0: #top border
            adj[i, i+mod] = vec[i]*vec[i+mod]
            adj[i, i+1] = vec[i]*vec[i+1]
            adj[i, i-1] = vec[i]*vec[i-1]
        elif (i%mod) == mod-1: #bottom border
            adj[i, i+mod] = vec[i]*vec[i+mod]
            adj[i, i-mod] = vec[i]*vec[i-mod]
            adj[i, i-1] = vec[i]*vec[i-1]
        elif int(i/mod) == mod-1:#right border
            adj[i, i-mod] = vec[i]*vec[i-mod]
            adj[i, i+1] = vec[i]*vec[i+1]
            adj[i, i-1] = vec[i]*vec[i-1]
        else: #Interior points
            adj[i, i+mod] = vec[i]*vec[i+mod]
            adj[i, i-mod] = vec[i]*vec[i-mod]
            adj[i, i+1] = vec[i]*vec[i+1]
            adj[i, i-1] = vec[i]*vec[i-1]
    return adj


def maze_sol(phi, tol):
    conv = 1
    old_val = 0
    vec = np.asarray(phi)
    while conv >tol:
        adj = mat_maze(vec)
        n = len(vec)
        L = Laplacian(adj)
        M = np.diag(vec)
        lamb, eigv = np.linalg.eigh(L)

        for i in range(n):
            if eigv[:,i].transpose()@M@eigv[:,i] == 1:
                new_val = np.ones(n).transpose()@vec
                conv = np.abs(new_val - old_val)
                old_val = new_val
                vec = vec+eigv[:,i]
                vec = vec.astype(int)
                break
    return vec

def maze_path(phi):
    vec = np.asarray(phi)
    adj = mat_maze(vec)
    n = len(vec)
    L = Laplacian(adj)
    M = np.diag(vec)
    _, eigv = np.linalg.eigh(L)
    for i in range(n):
            if eigv[:,i].transpose()@M@eigv[:,i] == 1:
                idx = i
                break   
    m = int(np.sqrt(n))
    plt.imshow(np.reshape(eigv[:,i], (m,m), order='F')*100, cmap='viridis', interpolation='nearest')

    # Add a color bar to show the mapping of values to colors
    plt.colorbar()

def scale_01(vec):
    n = len(vec)
    for i in range(n): 
        dist1 = np.abs(1-vec[i])
        dist0 = np.abs(0-vec[i])
        if dist1<dist0:
            vec[i] = 1
        else:
            vec[i] = 0
    return vec
        

def maze_iter(vec):
    vec = np.asarray(vec)
    adj = mat_maze(vec)
    n = len(vec)
    L = Laplacian(adj)
    M = np.diag(vec)
    lamb, eigv = np.linalg.eigh(L)
    for i in range(n):
        if eigv[:,i].transpose()@M@eigv[:,i] == 1:
            idx = i
            break   
    vec = scale_01(((1/np.abs(min(eigv[:,idx])))*eigv[:,idx])+vec)
    return vec

####### Run This ##########

# vec = np.asarray(phi121)
# adj = mat_maze(vec)
# n = len(vec)
# L = Laplacian(adj)
# M = np.diag(vec)
# _, eigv = np.linalg.eigh(L)
# for i in range(n):
#         if eigv[:,i].transpose()@M@eigv[:,i] == 1:
#             idx = i
#             break   
# vec = scale_01(((1/np.abs(min(eigv[:,idx])))*eigv[:,idx])+vec)
# maze(vec)