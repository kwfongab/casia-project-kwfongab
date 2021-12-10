from numpy import (cov, ones, zeros, where, empty, arange)
from numpy.random import default_rng, normal, multivariate_normal
from sklearn.linear_model import LinearRegression as LR
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from how_to_find_the_best_subset import best_subset


def workflow(X, B, est, max_iter, max_ps,
             sige=1, gsr=True, fsr=True, ntf=100):

    sample_size = X.shape[0]
    if gsr:
        dfs_gsr, eps_gsr = empty((max_iter, max_ps)), empty((max_iter, max_ps))
    if fsr:
        dfs_fsr = empty((max_iter, max_ps))

    for j in range(1, max_iter + 1):

        # Y = XB + E
        E = default_rng(69 * j).normal(0, sige, sample_size)
        Y = X @ B + E

        if gsr:
            lr_gss_ids = best_subset(est, X, Y, max_size=max_ps)
            for g in lr_gss_ids:
                # initialize b_hat
                B_pred_gs = zeros((B.shape[0],))
                lrg = est.fit(X[:, g], Y)
                for var_g, coef_g in zip(g, lrg.coef_):
                    B_pred_gs[var_g] = coef_g
                # yhat = x * (bhat = lr.coef_)
                eps_gsr[j - 1, len(g) - 1] = max(E)
                dfs_gsr[j - 1, len(g) - 1] = ((Y - X @ B).T @ (X @ B_pred_gs))

        if fsr:
            lr_fss_ids = []
            for i in range(1, max_ps):
                lr_fss = SFS(est, n_features_to_select=i).fit(X, Y)
                lr_fss_ids.append(list(where(lr_fss.support_)[0]))
            lr_fss_ids.append(list(arange(0, max_ps, 1)))
            for f in lr_fss_ids:
                # initialize b_hat
                B_pred_fs = zeros((B.shape[0],))
                lrf = est.fit(X[:, f], Y)
                for var_f, coef_f in zip(f, lrf.coef_):
                    B_pred_fs[var_f] = coef_f
                # yhat = x * (bhat = lr.coef_)
                dfs_fsr[j - 1, len(f) - 1] = ((Y - X @ B).T @ (X @ B_pred_fs))

        try:
            if j % ntf == 0:
                print("Take {} done!".format(j))
        except TypeError:
            pass

    return_what = []
    if gsr:
        return_what.append(dfs_gsr)
        return_what.append(eps_gsr)
    if fsr:
        return_what.append(dfs_fsr)
    return return_what
