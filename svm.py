import math
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp
from scipy.optimize import minimize

# type of kernel to be used
#   1 - linear
#   2 - polynomial
#   3 - rbf
kernel_type = 1


# global variables
C = 1
N = 5
x = np.zeros(N)
t = np.zeros(N)
p = np.zeros((N, N))
alphas = np.zeros(N)

nonzero_x = []
nonzero_t = []
nonzero_p = []

# bias, along with s and t_s used to calculate it
b = 0
s = 0
t_s = 0


# equation 4: objective function
def objective(alphas):
    global p
    return (1/2)*(np.sum(alphas, np.dot(alphas, p))) - np.sum(alphas)


# equation 6: indicator function
def ind(s):
    return np.dot(alphas, np.dot(t, [kernel(s, x[i]) for i in range(x.size)])) - b


# equation 7: threshold function to calculate bias
def bias():
    global s
    global t_s
    while (s == 0):
        index = random.randint(0, alphas.size)
        s = alphas[index]
        t_s = t[index]
    b = np.dot(alphas, np.dot(t, [kernel(s, x[i])
                                 for i in range(x.size)])) - t_s


# equality constraint of (10)
def zerofun(alphas):
    return np.dot(alphas, t)


# switcher function for different kernel types
def kernel(p1, p2):
    switcher = {
        1: linear_kernel,
        2: polynomial_kernel,
        3: rbf_kernel
    }
    func = switcher.get(kernel_type, linear_kernel)
    return func(p1, p2)


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


# precompute p
def calculatematrixp():
    global p
    for i in range(0, t.size):
        for j in range(0, t.size):
            p[i][j] = t[i] * t[j] * kernel(x[i], x[j])


def main():

    constraint = {'type': 'eq', 'fun': zerofun}
    bounds = [(0, C) for b in range(N)]

    calculatematrixp()
    bias()

    ret = minimize(objective, alphas, bounds, constraint)
    alpha = ret['x']
    if (ret['success']):
        print("Success!")


if __name__ == "__main__":
    main()
