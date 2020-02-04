import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def objective(alphas):
    global p
    return (1/2)*(numpy.dot(alphas, numpy.dot(alphas, p))) - numpy.sum(alphas)


def zerofun(alphas):
    global t
    return numpy.dot(alphas, t)


def linearkernel(p1, p2):
    return numpy.dot(p1, p2)


def calculatematrixp():
    global t
    global x
    p = numpy.zeros((N, N))
    for i in range(0, t.length):
        for j in range(0, t.length):
            p[i][j] = t[i]*t[j]*linearkernel(x[i], x[j])
    return p


C = 1
N = 5

x = numpy.zeros(N)
alphas = numpy.zeros(N)
t = numpy.zeros(N)
constraint = {'type': 'eq', 'fun': zerofun}
bounds = [(0, C) for b in range(N)]

p = calculatematrixp()

ret = minimize(objective, alphas, bounds, constraint)
alpha = ret['x']
