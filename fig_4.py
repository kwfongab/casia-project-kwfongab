from numpy.random import normal, default_rng
from numpy import (array, logspace, log, where, isclose, arange, empty,
                   linspace as lins, ones, concatenate as concat)
from sklearn.linear_model import lasso_path
from math import e
from matplotlib import pyplot as plt
from statistics import mean
from scipy.stats import sem

a_font = {'fontname': 'Arial', 'fontsize': 12}
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

X, B = array([[0, 1], [2, -5]]), array([-6, -1])
sige, max_iter = 0.3, 1000
vare = sige ** 2

# Y_tru = [-1 -7]
Y_tru = X @ B

lasso_a = logspace(-4, 5, 46, base=e)
lasso_b = logspace(-4, 5, 4 * len(lasso_a) - 3, base=e)

_, c_lasso, _ = lasso_path(X, Y_tru, alphas=lasso_a)
neg_log_a_lasso = log(lasso_a)
coef_ls = c_lasso.tolist()


def check_0s(lasso_coeffs, lasso_values):

    all_b1_is_0 = where(isclose(lasso_coeffs[0], 0.0))[0]
    all_b2_is_0 = where(isclose(lasso_coeffs[1], 0.0))[0]
    # all_b2_is_not_0 = list(set(lins(0, len(lasso_a) - 1, len(lasso_a))) - set(all_b2_is_0))
    # all_b2_is_not_0 = array([int(j) for j in all_b2_is_not_0])

    not_0_count = 2 * ones((len(lasso_values),))
    for i in lins(0, len(lasso_values) - 1, len(lasso_values)):
        if int(i) in all_b1_is_0:
            not_0_count[int(i)] -= 1
        if int(i) in all_b2_is_0:
            not_0_count[int(i)] -= 1

    return not_0_count


all_b2_is_0 = where(isclose(c_lasso[1], 0.0))[0]

for i in range(len(all_b2_is_0) - 1):
    if all_b2_is_0[i + 1] - all_b2_is_0[i] != 1:
        b2_is_0 = all_b2_is_0[(i + 1):]

neg_log_b_lasso = log(lasso_b)
not_0_collects = empty((max_iter, len(lasso_b)))

for i in range(max_iter):
    # Y = Y_tru + E
    rnge = default_rng(42 * (i + 1))
    Y = Y_tru + rnge.normal(0, sige, 2)
    _, c_lasso_noised, _ = lasso_path(X, Y, alphas=lasso_b)
    not_0_collects[i, :] = check_0s(c_lasso_noised, lasso_b)

all_not_0 = not_0_collects.T

# DF for penalized form lasso = E(no. of not 0 coefficients)
# DF for constraint form lasso = the above - 1

mu_all_not_0_list, se_all_not_0_list = [], []
for rows in all_not_0:
    mu_all_not_0_list.append(mean(rows))
    # this should give the 2 (Monte Carlo) SE
    se_all_not_0_list.append(2 * sem(rows))

from matplotlib.gridspec import GridSpec

figa = plt.figure(figsize=(9, 4), tight_layout=True)
gs = GridSpec(2, 7)
axea = figa.add_subplot(gs[0, :4])
axeb = figa.add_subplot(gs[1, :4])
axec = figa.add_subplot(gs[:, 4:])

axea.axhline(y=0, color='k', linestyle='--')
for coef_l in coef_ls:
    axea.plot(neg_log_a_lasso, coef_l, label=r"$\beta_{{{}}}$".format(
        1 + coef_ls.index(coef_l)))
axeb.errorbar(x=neg_log_b_lasso[::4], y=mu_all_not_0_list[::4],
              yerr=se_all_not_0_list[::4], capsize=3, c="g")
axec.errorbar(x=neg_log_b_lasso[120:141], y=mu_all_not_0_list[120:141],
              yerr=se_all_not_0_list[120:141], capsize=5, marker="o", c="g")

try:
    min_a_0_b2 = neg_log_a_lasso[min(b2_is_0)]
    max_a_0_b2 = neg_log_a_lasso[max(b2_is_0)]
    for ax in [axea, axeb, axec]:
        ax.axvline(x=min_a_0_b2, color='k', linestyle='--')
        ax.axvline(x=max_a_0_b2, color='k', linestyle='--')
except NameError:
    pass

for ax in [axea, axeb]:
    ax.set_xticks(arange(min(neg_log_a_lasso), max(neg_log_a_lasso) + 1, 1.0))


axea.set_ylabel("coefficients", **a_font)
axea.legend(loc="lower left")
for ax in [axeb, axec]:
    ax.set_ylabel("DoF = E(not 0 predictors)", **a_font)
    ax.set_xlabel(r"-ln(λ)", **a_font)

plt.savefig('Section-3B-Lasso.eps', format='eps', bbox_inches='tight')

