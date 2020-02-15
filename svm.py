import math
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp

from numpy.random import randn
from scipy.optimize import minimize


# type of kernel to be used
#   1 - linear
#   2 - polynomial
#   3 - rbf
kernel_type = 1


# global variables
classA = []
classB = []
inputs = []
targets = []
permute = []


C = 1
N = 0
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


# generate the test data
def generate_data():
    global classA
    global classB
    global inputs
    global targets
    global permute
    global N

    # numpy.random.seed(100)”
    classA = np.concatenate(
        (randn(10, 2) * 0.2 + [1.5, 0.5],
         randn(10, 2) * 0.2 + [-1.5, 0.5]))
    classB = randn(20, 2) * 0.2 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate(
        (np.ones(classA.shape[0]),
         -np.ones(classB.shape[0])))

    N = inputs.shape[0]

    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

# plot the data


def plot_data():
    plt.plot([p[0] for p in classA],
             [p[1] for p in classA],
             'b.')

    plt.plot([p[0] for p in classB],
             [p[1] for p in classB],
             'r.')

    plt.axis('equal')
    plt.savefig('svmplot.pdf')
    plt.show()


def main():

    constraint = {'type': 'eq', 'fun': zerofun}
    bounds = [(0, C) for b in range(N)]

    generate_data()
    calculatematrixp()
    bias()

    ret = minimize(objective, alphas, bounds, constraint)
    alpha = ret['x']
    if (ret['success']):
        print("Success!")


if __name__ == "__main__":
    main()
