import numpy as np
from numpy import linalg as LA
import pandas as pd

def minimize (f, x0, method, jac, hess):
    x = x0
    Imax = 1000

    if method == 'grad-des':
        eps = 1e-5
        err = 1
        i = 0
        # df = pd.DataFrame(columns=["x", "f", "p", "alpha"])
        while (i < Imax) & (err > eps):
            p = -jac(x)
            alpha = 1
            # d = {"x": x, "f": f(x), "p": p, "alpha": alpha}
            # df1 = pd.DataFrame(d)
            # df.append(df1)
            x_next = x + alpha*p
            while f(x_next) > f(x):
                alpha /= 2
                x_next = x + alpha * p
            x = x_next
            err = LA.norm(p)/(1+LA.norm(p))
            i += 1
            print("step %d: [%.5f, %.5f]" % (i, x[0], x[1]))

    elif method == 'Newton':
        eps = 1e-5
        err = 1
        i = 0
        # df = pd.DataFrame(columns=["x", "f", "p"])
        while (i < Imax) & (err > eps):
            p = -np.dot(LA.inv(hess(x)), jac(x))
            # d = {"x": x, "f": f(x), "p": p}
            # df1 = pd.DataFrame(d)
            # df.append(df1)
            x = x + p
            err = LA.norm(jac(x)) / (1 + LA.norm(jac(x)))
            i += 1
            print("step %d: [%.5f, %.5f]" % (i, x[0], x[1]))

    elif method == 'BFGS':
        b = np.identity(len(x))
        eps = 1e-8
        err = 1
        i = 0
        # df = pd.DataFrame(columns=["x", "f", "p", "alpha"])
        while (i < Imax) & (err > eps):
            p = -np.dot(LA.inv(b), jac(x))
            alpha = 1
            # d = {"x": x, "f": f(x), "p": p, "alpha": alpha}
            # df1 = pd.DataFrame(d)
            # df.append(df1)
            x_next = x + alpha*p
            print(x_next)
            while f(x_next) > f(x):
                alpha /= 2
                x_next = x + alpha * p
            s = x_next - x + 1e-20
            y = jac(x_next) - jac(x) + 1e-20
            ww = (y-np.dot(b, s))
            w = np.dot(ww, 1/LA.norm(ww))
            c = (LA.norm(ww) ** 2) / np.dot(ww.T, s)
            delta_b = np.dot(w, w.T)
            delta_b = np.dot(c, delta_b)
            b = b + delta_b
            x = x_next
            err = LA.norm(p)/(1+LA.norm(p))
            i += 1
            print("step %d: [%.5f, %.5f]" % (i, x[0], x[1]))

    elif method == 'Quasi-Newton':
        b = np.identity(len(x))
        eps = 1e-8
        err = 1
        i = 0
        # df = pd.DataFrame(columns=["x", "f", "p", "alpha"])
        while (i < Imax) & (err > eps):
            p = -np.dot(LA.inv(b), jac(x))
            alpha = 1
            # d = {"x": x, "f": f(x), "p": p, "alpha": alpha}
            # df1 = pd.DataFrame(d)
            # df.append(df1)
            x_next = x + alpha * p
            while f(x_next) > f(x):
                alpha /= 2
                x_next = x + alpha * p
            s = x_next - x + 1e-20
            y = jac(x_next) - jac(x) + 1e-20
            ww = np.dot(b, s)
            delta_b = -np.dot(ww, ww.T)/np.dot(s.T, ww) + np.dot(y, y.T)/np.dot(y.T, s)
            b = b + delta_b
            x = x_next
            err = LA.norm(p) / (1 + LA.norm(p))
            i += 1
            print("step %d: [%.5f, %.5f]" % (i, x[0], x[1]))
    return x

def line_search(p, q):
    # p += 1e-12
    # num = np.dot(p.T, p)
    # den = np.dot(np.dot(p.T, q).T, p)
    # alpha = num/den
    alpha = 1e-5
    return alpha