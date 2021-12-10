import os
import sys
import time
import warnings

from matplotlib import pyplot as plt
from numpy import (array, where, empty, mean, log, pi, zeros, ones,
                   sqrt, eye, set_printoptions, prod, delete,
                   logspace, log10, trace, dot, flip, sign, arange)
from numpy.linalg import pinv
from numpy.random import default_rng, normal, multivariate_normal
from scipy.optimize import minimize, show_options
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lasso_path
from sklearn.utils._testing import ignore_warnings

set_printoptions(suppress=True)
a_font = {'fontname': 'Arial', 'fontsize': 12}
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12


start = time.perf_counter()

n, p, q = 1000, 5, 1
sigma_b, sigma_e = 2, 3     # so var_b = 4, var_e = 9
rnge = default_rng(777)
E, S, Z1 = rnge.normal(0, sigma_e, n), eye(p*q), empty((n,))
Z1 = E.copy()

rngx, rngb = default_rng(42), default_rng(69)
# originally it was X_1 B_1 + ... + X_p B_p
# each B_i is q*1, each X_i is n*q
# now merge all the X_i and B_i together
X1 = rngx.multivariate_normal(zeros((p*q,)), S, n)
B1 = rngb.normal(0, sigma_b, p*q)
Z1 += X1 @ B1

sigma_u = 2.5
rngu = default_rng(81)
U4 = rngu.normal(0, sigma_u, p*q)
V4 = B1 / U4
X4, B4, Z4 = X1.copy(), B1.copy(), Z1.copy()

# Standard EM


def STEM(Y, X, s_b, s_e, MF=False):

    global n, p, q
    lLs_st, inlLs_st = [], []
    varb_old, vare_old = s_b, s_e

    while True:

        print("The existing parameters:")
        for values in [varb_old, vare_old]:
            print(round(values, 6), end=" ")
        print("")

        Mb = pinv((vare_old / varb_old) * eye(p * q) +
                      X.transpose() @ X) @ X.transpose() @ Y
        Sb = pinv(varb_old ** -1 * eye(p * q) +
                  vare_old ** -1 * X.transpose() @ X)

        # the incomplete log likelihood
        # ln p(Y | B, var_e)

        E_error_old = (Y - X @ Mb).transpose() @ (
                Y - X @ Mb) + trace(X.transpose() @ X @ Sb)
        inc_ll_old = -(n * log(2 * pi * vare_old) +
                       E_error_old / vare_old) / 2
        inlLs_st.append(inc_ll_old)

        # parameter update

        varb_new = (Mb.transpose() @ Mb + trace(Sb)) / (p * q)
        vare_new = sqrt(E_error_old / n)

        Mb_new = pinv((vare_new / varb_new) * eye(p * q) +
                      X.transpose() @ X) @ X.transpose() @ Y

        E_error_new = (Y - X @ Mb).transpose() @ (
                Y - X @ Mb) + trace(X.transpose() @ X @ Sb)
        inc_ll_new = -(n * log(2 * pi * vare_new) +
                       E_error_new / vare_new) / 2

        # refresh

        varb_old, vare_old = varb_new, vare_new

        print("Incomplete log-likelihood difference")
        print(inc_ll_new - inc_ll_old)

        if inc_ll_new - inc_ll_old < 10 ** -6:
            inlLs_st.append(inc_ll_new)
            print("Incomplete log-likelihood looks converged!")
            print("Final EMed parameters: var_b & var_e")
            for values in [varb_new, vare_new]:
                print(round(values, 6), end=" ")
            print("\nThe EMed mean of B:")
            print(pinv((vare_new / varb_new) ** 2 * eye(p * q) +
                       X.transpose() @ X) @ X.transpose() @ Y)
            break

    print("Standard EM done!")
    return inlLs_st


def MFEM(Y, X, s_b, s_e):

    global n, p, q
    lLs_st, lLs_px, inlLs_st, inlLs_px = [], [], [], []
    g_old, varb_old, vare_old = 1, s_b, s_e

    while True:

        print("The existing parameters:")
        for values in [g_old, varb_old, vare_old]:
            print(round(values, 6), end=" ")
        print("")

        Mb_old = pinv((vare_old / varb_old) * eye(p * q) +
                      X.transpose() @ X) @ X.transpose() @ Y
        Sb = pinv(varb_old ** -1 * eye(p * q) +
                  vare_old ** -1 * X.transpose() @ X)

        # the incomplete log likelihood
        # ln p(Y | B, var_e)
        E_error_old = (Y - g_old * X @ Mb).transpose() @ (
                Y - g_old * X @ Mb) + g_old ** 2 * trace(
            X.transpose() @ X @ Sb)
        inc_ll_old = -(n * log(2 * pi * vare_old) +
                       E_error_old / vare_old) / 2

        # parameter update
        C = (Mb.transpose() @ Mb + trace(Sb)) / (p * q)
        if PX:
            g_new, varb_new = C / varb_old, C / g_old ** 2
        else:
            g_new, varb_new = 1, C

        E_error_new = (Y - g_new * X @ Mb).transpose() @ (
                Y - g_new * X @ Mb) + g_new ** 2 * trace(
            X.transpose() @ X @ Sb)

        vare_new = sqrt(E_error_old / n)
        inc_ll_new = -(n * log(2 * pi * vare_new) +
                       E_error_new / vare_new) / 2

        if PX:
            # R step
            varb_old, vare_old = varb_new * g_new ** 2, vare_new
        else:
            varb_old, vare_old = varb_new, vare_new

        inlLs_st.append(inc_ll_old)
        if not inlLs_px:
            inlLs_px.append(inc_ll_old)
        inlLs_px.append(inc_ll_new)
        #print(inlLs_px)
        #input("wait")

#        if PX:
#            print("The calculated parameters γ:")
#            print(g_new)


        # Sth is wrong as we deduce g_new
        # and calculate the incomplete lLs
        print("Incomplete log-likelihood difference")
        if PX:
            print(inc_ll_new - inlLs_px[-2])
        else:
            print(inc_ll_new - inc_ll_old)

        Mb_new = pinv((vare_new / varb_new) ** 2 * eye(p * q) +
                      X.transpose() @ X) @ X.transpose() @ Y

        if PX and abs(inc_ll_new - inlLs_px[-2]) < 10 ** -6:
            print("Incomplete log-likelihood looks converged!")
            print("Final EMed parameters: gamma, var_b & var_e")
            for values in [g_new, varb_new, vare_new]:
                print(round(values, 6), end=" ")
            print("\nThe EMed mean of B:")
            print(Mb_new)
            break
        elif not PX and inc_ll_new - inc_ll_old < 10 ** -6:
            inlLs_st.append(inc_ll_new)
            print("Incomplete log-likelihood looks converged!")
            print("Final EMed parameters: gamma, var_b & var_e")
            for values in [g_new, varb_new, vare_new]:
                print(round(values, 6), end=" ")
            print("\nThe EMed mean of B:")
            print(Mb_new)
            break

    if PX:
        print("PXEM done!")
        return inlLs_px
    else:
        print("Standard EM done!")
        return inlLs_st


inlLs_out = STEM(Z1, X1, s_b=1, s_e=1)

figg, axeg = plt.subplots(figsize=(4, 4), constrained_layout=True)
axeg.plot(inlLs_out)
axeg.set_xlabel("Takes", **a_font)
axeg.set_title("Incomplete log-likelihood", **a_font)

end = time.perf_counter()
print("This part takes {:.2f} seconds!".format(end - start))

plt.show()

