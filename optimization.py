import numpy as np
from numpy import linalg as LA
import pandas as pd
# changesss

def minimize (f, x0, method, jac, hess):
    x = x0
    Imax = 1000
    if method == 'grad-des':
        # df = pd.DataFrame()
        eps = 1e-5
        err = LA.norm(jac(x))/(1+np.absolute(f(x)))
        i = 0
        if len(x) == 2:
            print("\\textbf{%d} & [%0.3g, %0.3g] & ... & ... & %0.3g & %0.3g \\\\" % (0, x[0], x[1], f(x), err))
        elif len(x) == 3:
            print("\\textbf{%d} & [%0.3g, %0.3g, %.3g] & ... & ... & %0.3g & %0.3g \\\\" % (0, x[0], x[1], x[2], f(x), err))
        while (i < Imax) & (err > eps):
            p = -jac(x)
            alpha = linsearch(f, x, p, jac, method="division")
            x = x + np.dot(alpha,p)
            # d = {"x": x, "f": f(x), "p": p, "alpha": alpha}
            # x = x_next
            # if i == 0:
            #     df = pd.DataFrame(d)
            # elif i > 0:
            #     df1 = pd.DataFrame(d)
            #     df = df.append(df1, ignore_index=True)
            err = LA.norm(jac(x))/(1+np.absolute(f(x)))
            i += 1
            if len(x) == 2:
                print("\\textbf{%d} & [%0.3g, %0.3g] & [%0.2f, %.2f] & %0.2g & %0.3g & %0.3g \\\\" % (i, x[0], x[1], p[0], p[1], alpha, f(x), err))
            elif len(x) == 3:
                print("\\textbf{%d} & [%0.3g, %0.3g, %0.3g] & [%0.2f, %.2f, %.2f] & %0.2g & %0.3g & %0.3g \\\\" % (i, x[0], x[1], x[2], p[0], p[1], p[2], alpha, f(x), err))

    elif method == 'Newton':
        eps = 1e-5
        err = LA.norm(jac(x)) / (1 + np.absolute(f(x)))
        i = 0
        if len(x) == 2:
            print("\\textbf{%d} & [%0.3g, %0.3g] & ... &  %0.3g & %0.3g \\\\" % (i, x[0], x[1], f(x), err))
        elif len(x) == 3:
            print("\\textbf{%d} & [%0.3g, %0.3g, %0.3g] & ... & %0.3g & %0.3g \\\\" % (i, x[0], x[1], x[2], f(x), err))
        while (i < Imax) & (err > eps):
            p = -np.dot(LA.inv(hess(x)), jac(x))
            x = x + p
            err = LA.norm(jac(x))/(1+np.absolute(f(x)))
            i += 1
            if len(x)==2:
                print("\\textbf{%d} & [%0.3g, %0.3g] & [%0.2f, %.2f] &  %0.3g & %0.3g \\\\" % (i, x[0], x[1], p[0], p[1], f(x), err))
            elif len(x) == 3:
                print("\\textbf{%d} & [%0.3g, %0.3g, %0.3g] & [%0.2f, %.2f, %.2f] & %0.3g & %0.3g \\\\" % (i, x[0], x[1], x[2], p[0], p[1], p[2], f(x), err))

    elif method == 'bfgs':
        b = np.identity(len(x))
        eps = 1e-8
        err = LA.norm(jac(x))/(1+np.absolute(f(x)))
        i = 0
        if len(x) == 2:
            print("\\textbf{%d} & [%0.3g, %0.3g] & ... & ... & %0.3g & %0.3g \\\\" % (0, x[0], x[1], f(x), err))
        elif len(x) == 3:
            print("\\textbf{%d} & [%0.3g, %0.3g, %.3g] & ... & ... & %0.3g & %0.3g \\\\" % (0, x[0], x[1], x[2], f(x), err))
        while (i < Imax) & (err > eps):
            p = -np.dot(LA.inv(b), jac(x))
            alpha = linsearch(f, x, p, jac, method="division")
            x_next = x + alpha * p
            s = x_next - x
            y = jac(x_next) - jac(x) + 1e-20
            ww = np.dot(b, s)
            delta_b = -np.outer(ww, ww)/np.dot(s.T, ww) + np.outer(y, y)/np.dot(y.T, s)
            b = b + delta_b
            x = x_next
            err = LA.norm(jac(x))/(1+np.absolute(f(x)))
            i += 1
            if len(x)==2:
                print("\\textbf{%d} & [%0.3g, %0.3g] & [%0.2f, %.2f] & %0.2g & %0.3g & %0.3g \\\\" % (i, x[0], x[1], p[0], p[1], alpha, f(x), err))
            elif len(x) == 3:
                print("\\textbf{%d} & [%0.3g, %0.3g, %0.3g] & [%0.2f, %.2f, %.2f] & %0.2g & %0.3g & %0.3g \\\\" % (i, x[0], x[1], x[2], p[0], p[1], p[2], alpha, f(x), err))

    elif method == 'conj-grad':
        d0 = -jac(x)
        eps = 1e-5
        err = LA.norm(jac(x))/(1+np.absolute(f(x)))
        i = 0
        if len(x) == 2:
            print("\\textbf{%d} & [%0.3g, %0.3g] & ... & ... & %0.3g & %0.3g \\\\" % (0, x[0], x[1], f(x), err))
        elif len(x) == 3:
            print("\\textbf{%d} & [%0.3g, %0.3g, %.3g] & ... & ... & %0.3g & %0.3g \\\\" % (0, x[0], x[1], x[2], f(x), err))
        while (i < Imax) & (err > eps):
            alpha = linsearch(f, x, d0, jac, method="division")
            x_next = x + alpha * d0
            bet = np.dot(jac(x_next).T, jac(x_next)) / np.dot(jac(x).T, jac(x))
            d = -jac(x_next) + np.dot(bet, d0)
            d0 = d
            x = x_next
            err = LA.norm(jac(x))/(1+np.absolute(f(x)))
            i += 1
            if len(x) == 2:
                print("\\textbf{%d} & [%0.3g, %0.3g] & [%0.2f, %.2f] & %0.2g & %0.3g & %0.3g \\\\" % (i, x[0], x[1], d[0], d[1], alpha, f(x), err))
            elif len(x) == 3:
                print("\\textbf{%d} & [%0.3g, %0.3g, %0.3g] & [%0.2f, %.2f, %.2f] & %0.2g & %0.3g & %0.3g \\\\" % (i, x[0], x[1], x[2], d[0], d[1], d[2], alpha, f(x), err))

    return x


def linsearch(f, x, p, jac, method):
    alpha = 1
    mew = 0.3
    der = np.array(jac(x))
    if method == "golden-rule":
        a0 = 0
        b0 = 1
        row = 0.302
        while (f(x + np.dot(alpha, p)) - (f(x) + np.dot(mew * alpha, np.dot(p.T, der)))) > 0:
            b1 = a0 + (1-row) * (b0 - a0)
            a1 = a0 + row * (b0 - a0)
            if f(x + np.dot(a1, p)) <= f(x + np.dot(b1, p)):
                a0 = a0
                b0 = b1
            else:
                a0 = a1
                b0 = b0
            alpha = b0
    elif method == "division":
        while (f(x + np.dot(alpha, p)) - (f(x) + np.dot(mew * alpha, np.dot(p.T, der)))) > 0:
            alpha = 0.5 * alpha
    return alpha
