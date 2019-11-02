import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import optimization as op

# c = 10

def main():
    # x0 = np.ones(n)
    # x0 = np.zeros(n)
    # x0 = np.array([-1.2, 1])
    x0 = np.array([2, -2])
    # x0 = np.array([1, -1])
    res = op.minimize(f, x0, method='grad-des', jac=f_der, hess=f_hess)
    print("The result is: [%.5f, %.5f]" % (res[0], res[1]))
    print("and fnc value is: %.5f" %f(res)) #comment


def f(x):
    # fx = np.dot(x.T, x)
    # fx = x[0] ** 2 + 2 * x[1] ** 2 - 2 * x[0] * x[1] - 2 * x[1]
    # fx = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    fx = (x[0] + x[1]) ** 4 + x[1] ** 2
    # fx = c*(x[0]**2 + x[1]**2 - 1/4)**2 + (x[0] - 1)**2 + (x[1] - 1)**2
    return fx


def f_der(x):
    # der = np.dot(2, x)
    # der = np.array([2*x[0]-2*x[1], 4*x[1]-2*x[0]-2])
    # der = np.array([2*x[0] - 400*x[0]*(- x[0] ** 2 + x[1]) - 2, - 200*x[0] ** 2 + 200*x[1]])
    der = np.array([4*(x[0] + x[1]) ** 3, 2*x[1] + 4*(x[0] + x[1]) ** 3])
    # der = np.array([2*x[0] + 4*c*x[0]*(x[0]**2 + x[1]**2 - 1/4) - 2, 2*x[1] + 4*c*x[1]*(x[0]**2 + x[1]**2 - 1/4) - 2])
    der.reshape(1, len(x))
    return der


def f_hess(x):
    # h = np.identity(len(x)) * 2
    # h = np.array([[2, -2], [-2, 4]])
    # h = np.array([[1200*x[0] ** 2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200]])
    h = np.array([[12*(x[0] + x[1]) ** 2, 12*(x[0] + x[1]) ** 2], [12*(x[0] + x[1]) ** 2, 12*(x[0] + x[1]) ** 2 + 2]])
    # h = np.array([[12*c*x[0]**2 + 4*c*x[1]**2 - c + 2, 8*c*x[0]*x[1]], [8*c*x[0]*x[1], 4*c*x[0]**2 + 12*c*x[1]**2 - c
    #                                                                     + 2]])
    return h


if __name__ == '__main__':
    main()