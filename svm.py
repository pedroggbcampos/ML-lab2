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
p = []
alphas = []

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
    return (1/2)*(np.sum(np.dot(np.dot(alphas, p), alphas))) - np.sum(alphas)


# equation 6: indicator function
def indicator(s):
    return np.dot(np.multiply(alphas, targets), [kernel(s, inputs[i]) for i in range(len(inputs))]) - b


# equation 7: threshold function to calculate bias
def bias():
    global b
    global s
    global t_s
    global alphas
    global inputs
    while s == 0:
        index = random.randint(0, len(alphas) - 1)
        s = alphas[index]
        t_s = targets[index]
    b = np.dot(np.multiply(alphas, targets), [kernel(s, inputs[i]) for i in range(len(inputs))]) - t_s


# equality constraint of (10)
def zerofun(alphas):
    return np.dot(alphas, targets)


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
    return np.dot(p1, p2)

'''transposing a 1D array returns the array unchanged'''


# polynomial kernel
def polynomial_kernel(p1, p2):
    p = 2
    return (np.dot(p1, p2) + 1) ** p


# radial basis function kernel
def rbf_kernel(p1, p2):
    sigma = 1
    return math.e ** (-1 * (la.norm(p1-p2)**2) / (2*sigma**2))


# precompute p
def calculatematrixp():
    global p
    p = np.zeros((len(targets), len(targets)))
    for i in range(0, len(targets)):
        for j in range(0, len(targets)):
            p[i][j] = targets[i] * targets[j] * kernel(inputs[i], inputs[j])


# generate the test data
def generate_data():
    global classA
    global classB
    global inputs
    global targets
    global permute
    global N

    # numpy.random.seed(100)
    classA = np.concatenate(
        (randn(10, 2) * 0.2 + [1.5, 0.5],
         randn(10, 2) * 0.2 + [-1.5, 0.5]))
    classB = randn(20, 2) * 0.2 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate(
        (np.ones(classA.shape[0]),
         -np.ones(classB.shape[0])))

    N = inputs.shape[0]  # number of rows (samples)

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

    plt.axis('equal')  # force same scale on both axes
    plt.savefig('svmplot.pdf')  # save a copy in a file
    plt.show()  # show the plot on the screen


# plot the decsion boundary
def plot_decision_boundary():
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)

    grid = np.array([[indicator([x, y])
                      for x in xgrid]
                     for y in ygrid])

    print(grid)
    plt.contour(xgrid, ygrid, grid,
                (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'),
                linewidths=(1, 3, 1))


def main():

    global alphas, inputs, C
    cts = {'type': 'eq', 'fun': zerofun}

    generate_data()
    alphas = np.zeros(len(inputs))
    bds = [(0, C) for b in range(len(inputs))]

    calculatematrixp()

    ret = minimize(objective, alphas, bounds=bds, constraints=cts)
    alphas = ret['x']
    if (ret['success']):
        print("Success!")

    bias()
    plot_data()
    plot_decision_boundary()


if __name__ == "__main__":
    main()
