from numpy import linspace as lins, matrix, zeros, logspace as logs,\
    identity as I, argmin
from numpy.random import normal, default_rng
from numpy.linalg import inv, pinv
import numpy as np

from sklearn.model_selection import train_test_split as TTS,\
    RepeatedKFold as RKF

from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler as SS

from sklearn.model_selection import cross_validate as CVal,\
    cross_val_score as CVScore

from numpy import where

import math

import time

start = time.perf_counter()

# workflow
# going to repeat for 200 times
m = 40 # change to 200 later
tr = 0.25 # test to total ratio
vr = tr / (1 - tr) # valid to train+valid ratio

# set up the cross-validation operations for Ridge
# with hyperparameters to be tuned
# 5 fold now repeated all 5 times
cvs = RKF(n_splits=5, n_repeats=5, random_state=42)

alphas = logs(-4, 4, 91) # from 0.001 to 1000



def workflow(repeats=m, test_ratio=tr, valid_ratio=vr,
             alphas_fit=alphas, cv_fit=cvs):

    n = 400  # sample size

    # cross validation number
    cv = 5

    # for k-fold
    # original 400 => training 300 => 60 for cv each time
    thesize = int(n*(1-tr)/cv) # 60 now

    mu_x, sigma_x, mu_b, sigma_b, mu_e, sigma_e = 0, 1, 0, 1, 0, 5
    var_x, var_b, var_e = sigma_x ** 2, sigma_b ** 2, sigma_e ** 2

    bias_train, vars_train, bias_test, vars_test = [0], [0], [0], [0]

    mse_valids_r, mean_vmse_r = [], [0]*len(alphas)
    mse_tests_r, mean_tmse_r = [], [0]*len(alphas)

    rng = default_rng(3000)

    X = rng.normal(mu_x, var_x, (n, 5))  # create n * p matrix X
    B = rng.normal(mu_b, var_b, (5, 1))  # Beta
    R = rng.normal(mu_e, var_e, (n, 1))  # eRRoR

    # Underlying model
    Y = X @ B + R  # matrix product

    for i in range(repeats):

        # Train test split
        X_train, X_test, Y_train, Y_test = TTS(
            X, Y, test_size=tr, random_state=i)

        # MSE lists for valid & test sets, flow first, ridge next
        mse_vs_f, mse_ts_f = [], []
        mse_vs_r, mse_ts_r = [], []

        # manually record the MSE per train-valid-test split given each al
        for al in alphas:

            hes, has = [], []

            for fold in range(cv):
                X_valid = np.concatenate((X_train[fold * thesize:(fold + 1) * thesize - 1, :],
                                          X_train[(fold - 4) * thesize:(fold - 4) * thesize + 1, :]), axis=0)
                Y_valid = np.concatenate((Y_train[fold * thesize:(fold + 1) * thesize - 1, :],
                                          Y_train[(fold - 4) * thesize:(fold - 4) * thesize + 1, :]), axis=0)

                XV_rows = X_valid.view([('', X_valid.dtype)] * X_valid.shape[1])
                XT_rows = X_train.view([('', X_train.dtype)] * X_train.shape[1])
                X_based = np.setdiff1d(XT_rows, XV_rows).view(X_valid.dtype).reshape(-1, X_valid.shape[1])
                YV_rows = Y_valid.view([('', Y_valid.dtype)] * Y_valid.shape[1])
                YT_rows = Y_train.view([('', Y_train.dtype)] * Y_train.shape[1])
                Y_based = np.setdiff1d(YT_rows, YV_rows).view(Y_valid.dtype).reshape(-1, Y_valid.shape[1])

                ri_fold = Ridge(alpha=al, random_state=42).fit(
                    X_based, Y_based.ravel())
                Y_pred_fold = ri_fold.predict(X_valid)
                Y_pred_test = ri_fold.predict(X_test)

                he = MSE(Y_pred_fold, Y_valid)
                ha = MSE(Y_pred_test, Y_test)
                #print(he)
                #input("wait")
                hes.append(he)
                has.append(ha)

            mse_vs_r.append(sum(hes) / len(hes))
            mse_ts_r.append(sum(has) / len(has))

            ri_train = Ridge(alpha=al, random_state=42).fit(
                X_train, Y_train.ravel())

            ## Train validation split
            # X_based, X_valid, Y_based, Y_valid = TTS(
            #    X_train, Y_train, test_size=vr, random_state=42)

            # fit then predict on the validation set
            # ri = RidgeCV(alpha=al, cv=cvs).fit(X_based, Y_based.ravel())

            # Do (4-fold repeated 4 times actually) CV given the al
            # then get the -MSE score
            # what = CVScore(Ridge(alpha=al, random_state=42), X_train, Y_train,
            #               scoring="neg_mean_squared_error", cv=cvs)

            # print(-sum(what)/len(what))
            # input("wait")

            Y_pred_t_r = ri_train.predict(X_test)

            # B_pred_r = pinv(X_based.transpose() @ X_based + n * al * I(
            #    X_based.shape[1])) @ X_based.transpose() @ Y_based

            # Y_pred_v_r = X_valid @ B_pred_r
            # Y_pred_t_r = X_test @ B_pred_r

            # record the MSE on the validation set on the list
            # mse_valid[where(alphas==al)[0][0]].append(MSE(Y_pred_valid, Y_valid))

            # print(MSE(Y_pred_v_r, Y_valid))
            # input("wait")

            # mse_vs_r.append(-sum(what)/len(what))
            # mse_ts_r.append(MSE(Y_pred_t_r, Y_test))

        mse_valids_r.append(mse_vs_r)
        mse_tests_r.append(mse_ts_r)

    # short circuits at shortest nested list if table is jagged:
    valids_mse = list(map(list, zip(*mse_valids_r)))
    tests_mse = list(map(list, zip(*mse_tests_r)))

    #print(len(valids_mse))
    #input("wait")

    #print("valids_mse")
    #print(valids_mse)
    #input("wait")

    # print(where(alphas==al)[0][0]) # should be 6
    # print(valids_mse[where(alphas==al)[0][0]]) # should be the last list

    for al in alphas:

        ind = where(alphas == al)[0][0]
        mean_vmse_r[ind] = sum(valids_mse[ind])/repeats
        mean_tmse_r[ind] = sum(tests_mse[ind])/repeats

    #print(mean_valid_mse)
    #print(mean_test_mse)
    #input("wait")

    return mean_vmse_r, mean_tmse_r

# test if it works
mean_valid_mse, mean_test_mse = workflow()

end = time.perf_counter()

print("The workflow takes {} seconds!".format(end - start))

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

lambs = lins(-4, 4, 91)

hi, bye = lambs[argmin(mean_valid_mse)], lambs[argmin(mean_test_mse)]

figa, axea = plt.subplots()

this = min(min(mean_valid_mse), min(mean_test_mse))
that = max(max(mean_valid_mse), max(mean_test_mse))

axea.scatter(lambs, mean_valid_mse, label="Validation")
axea.scatter(lambs, mean_test_mse, label="Test")
axea.plot([hi, hi], [this - 1, that + 1], 'b--')
axea.plot([bye, bye], [this - 1, that + 1], 'r--')
axea.legend()
axea.set_xlabel('$\log_{10} \lambda$')
axea.set_ylabel('Validation or Test MSE')
plt.show()