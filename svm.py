import math
import random

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from numpy.random import randn
from scipy.optimize import minimize

# type of kernel to be used
#   1 - linear
#   2 - polynomial
#   3 - rbf
kernel_type = 3
p = 2
sigma = 1
variance = 0.7

# global variables
classA = []
classB = []
inputs = []
targets = []
permute = []

C = 10
N = 100
p = []
alphas = []

nonzero_x = []
nonzero_t = []
nonzero_p = []
nonzero_alphas = []

# bias, along with s and t_s used to calculate it
b = 0
s = 0
t_s = 0


# equation 4: objective function
def objective(alphas):
    global p
    return (1 / 2) * (np.sum(np.dot(np.dot(alphas, p), alphas))) - np.sum(alphas)


# equation 6: indicator function
def indicator(s):
    global nonzero_alphas, nonzero_t, nonzero_x
    return np.dot(np.multiply(nonzero_alphas, nonzero_t), [kernel(s, nonzero_x[i]) for i in range(len(nonzero_x))]) - b


# equation 7: threshold function to calculate bias
def bias():
    global b
    global s
    global t_s
    global nonzero_alphas
    global nonzero_x
    global nonzero_t
    while s == 0:
        index = random.randint(0, len(nonzero_alphas) - 1)
        s = nonzero_alphas[index]
        t_s = nonzero_t[index]
    s = nonzero_x[index]
    b = np.dot(np.multiply(nonzero_alphas, nonzero_t), [kernel(s, nonzero_x[i]) for i in range(len(nonzero_x))]) - t_s


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
    return (np.dot(p1, p2) + 1) ** p


# radial basis function kernel
def rbf_kernel(p1, p2):
    return math.e ** (-1 * (la.norm(p1 - p2) ** 2) / (2 * sigma ** 2))


# pre compute p
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
        (randn(int(N / 4), 2) * variance + [1.5, 0.5],
         randn(int(N / 4), 2) * variance + [-1.5, 0.5]))
    classB = randn(int(N / 2), 2) * variance + [0, -0.5]

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
    '''plt.show()  # show the plot on the screen'''


# plot the decsion boundary
def plot_decision_boundary():
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)

    grid = np.array([[indicator([x, y])
                      for x in xgrid]
                     for y in ygrid])

    plt.contour(xgrid, ygrid, grid,
                (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'),
                linewidths=(1, 3, 1))
    plt.show()


def main():
    global alphas, inputs, C
    cts = {'type': 'eq', 'fun': zerofun}

    generate_data()
    alphas = np.zeros(len(inputs))
    bds = [(0, C) for b in range(len(inputs))]

    calculatematrixp()

    ret = minimize(objective, alphas, bounds=bds, constraints=cts)
    alphas = ret['x']
    if ret['success']:
        print("Success!")

    for i in range(0, len(alphas)):
        if 10 ** (-5) < alphas[i] < C:
            nonzero_alphas.append(alphas[i])
            nonzero_x.append(inputs[i])
            nonzero_t.append(targets[i])

    bias()
    plot_data()
    plot_decision_boundary()


if __name__ == "__main__":
    main()
