import math
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp
from scipy.optimize import minimize


# global variables
C = 1
N = 5
x = np.zeros(N)
t = np.zeros(N)
p = np.zeros((N, N))
alphas = np.zeros(N)


# equation 4
def objective(alphas):
    global p
    return (1/2)*(np.dot(alphas, np.dot(alphas, p))) - np.sum(alphas)


# equality constraint of (10)
def zerofun(alphas):
    return np.dot(alphas, t)


# linear kernel
def linear_kernel(p1, p2):
    return np.dot(p1.transpose(), p2)


# polynomial kernel
def polynomial_kernel(p1, p2):
    p = 2
    return (np.dot(p1.transpose(), p2) + 1) ** p


# radial basis function kernel
def rbf_kernel(p1, p2):
    sigma = 1
    return math.e ** (-1 * (la.norm(p1-p2)**2) / (2*sigma**2))


def calculatematrixp():
    global p
    for i in range(0, t.size):
        for j in range(0, t.size):
            p[i][j] = t[i] * t[j] * linear_kernel(x[i], x[j])


# equation 6
def indicator(s):
    result = 0
    for i in range(N):
        result += alphas[i]*t[i]*linear_kernel(s, x[i])
    return (result - b)


def main():

    # alphas = np.zeros(N)
    constraint = {'type': 'eq', 'fun': zerofun}
    bounds = [(0, C) for b in range(N)]

    calculatematrixp()

    ret = minimize(objective, alphas, bounds, constraint)
    alpha = ret['x']
    if (ret['success']):
        print("Success!")


if __name__ == "__main__":
    main()
