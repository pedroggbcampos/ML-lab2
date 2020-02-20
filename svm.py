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

# for data generation
classA = np.empty((0, 0))
classB = np.empty((0, 0))
inputs = np.empty((0, 0))
targets = np.empty((0, 0))
permute = ()

C = 1
N = 0  # number of points
b = 0  # bias
data_points = np.empty(0)
t = np.empty(0)
p = np.empty((0, 0))
alphas = np.empty(0)

nonzero_data_points = []
nonzero_t = []
nonzero_p = []

# bias, along with s and t_s used to calculate it
b = 0
s = 0
t_s = 0


# equation 4: objective function
def objective(alphas):
    global p
    return (1 / 2) * (np.sum(alphas, np.dot(alphas, p))) - np.sum(alphas)


# equation 6: indicator function
def indicator(s):
    # def indicator():
    return np.dot(alphas, np.dot(t, [kernel(s, data_points[i]) for i in range(data_points.size)])) - b


# equation 7: threshold function to calculate bias
def bias():
    global s
    global t_s
    while s == 0:
        index = random.randint(0, alphas.size)
        s = alphas[index]
        t_s = t[index]
    b = np.dot(alphas, np.dot(t, [kernel(s, data_points[i])
                                  for i in range(data_points.size)])) - t_s


# equality constraint of (10)
# def zerofun(alphas):
def zerofun():
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
    p = 2  # controls degree of polynomial, the higher the more complex the shape
    return (np.dot(p1.transpose(), p2) + 1) ** p


# radial basis function kernel
def rbf_kernel(p1, p2):
    sigma = 1
    return math.e ** (-1 * (la.norm(p1 - p2) ** 2) / (2 * (sigma ** 2)))


# generate the test data
def generate_data():
    global classA, classB, inputs, targets, permute, N

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


# initialize containers
def init():
    global data_points, t, p, alphas
    data_points = np.zeros(N)
    t = np.zeros(N)
    p = np.zeros((N, N))
    alphas = np.zeros(N)


# precompute p
def calculatematrixp():
    global p
    for i in range(0, t.size):
        for j in range(0, t.size):
            p[i][j] = t[i] * t[j] * kernel(data_points[i], data_points[j])


# plot the data
def plot_data():
    plt.plot([pt[0] for pt in classA],
             [pt[1] for pt in classA],
             'b.')

    plt.plot([pt[0] for pt in classB],
             [pt[1] for pt in classB],
             'r.')

    plt.axis('equal')  # force same scale on both axes
    plt.savefig('svmplot.pdf')  # save a copy in a file
    plt.show()  # show the plot on the screen


# plot the decsion boundary
def plot_decision_boundary():
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)

    grid = np.array([[indicator(x, y)
                      for x in xgrid]
                     for y in ygrid])

    plt.contour(xgrid, ygrid, grid,
                (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'),
                linewidths=(1, 3, 1))


def main():
    generate_data()
    init()
    calculatematrixp()
    bias()

    constraint = {'type': 'eq', 'fun': zerofun}
    bounds = [(0, C) for b in range(N)]

    ret = minimize(objective, alphas, bounds, constraint)
    alpha = ret['x']
    if ret['success']:
        print("Success!")

    plot_data()
    plot_decision_boundary()


if __name__ == "__main__":
    main()
