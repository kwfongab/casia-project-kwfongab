from sklearn.ensemble import (RandomForestRegressor as RFR,
                              RandomForestClassifier as RFC,
                              BaggingRegressor as BR, BaggingClassifier as BC)

from numpy import empty, zeros, ones, concatenate, geomspace as geoms, eye,\
    array, mean, std, argmin, set_printoptions, linspace as lins
from numpy.random import default_rng, normal, multivariate_normal
import time
from math import sqrt, sin, pi


# TODO: why the MSE I get is much larger?


def get_cov_and_coeffs(sizes_dict):

    global rho_1
    p, s = sizes_dict['p'], sizes_dict['s']

    # S_1 is p * p covariance matrix
    S_1 = empty((p, p))
    for j in range(p):
        for i in range(p):
            S_1[i, j] = rho_1 ** abs(i - j)

    # B_1 is p * 1 coefficient matrix
    B_1 = concatenate((ones((s, )), zeros((p - s, ))))

    return S_1, B_1


def get_lin_models(sizes_dict, S, B, v, i=0, train=True):

    global rho_1
    n, p = sizes_dict['n'], sizes_dict['p']
    # p. 11
    sigma_e = sqrt(B.transpose() @ S @ B / v)

    if train:

        rngx = default_rng(42)
        X_train = rngx.multivariate_normal(zeros((p,)), S, n)
        rnge = default_rng(69)
        E_train = rnge.normal(0, sigma_e, n)

        # create the train set using 42 & 69
        X_out, Y_out = X_train, X_train @ B + E_train

    else:

        rngx = default_rng(42 * (i + 1))
        X_test = rngx.multivariate_normal(zeros((p,)), S, n)
        rnge = default_rng(69 * (i + 1))
        E_test = rnge.normal(0, sigma_e, n)

        # create the test set using the fixed random states
        X_out, Y_out = X_test, X_test @ B + E_test

    return X_out, Y_out


def get_nonlin_models(sizes_dict, v, i=0, train=True):

    global rho_1
    n, p = sizes_dict['n'], sizes_dict['p']
    # how find the sigma_e from v for the nonlinear model?
    # Y = 10 sin(πX1X2) + 20(X3 − 0.05)2 + 10X4 + 5X5 + E
    # for convenience try var(Y) = 10**2 + 20**2 + 10**2 + 5**2 = 625
    sigma_e = 25 / sqrt(v)
    #print(sigma_e)

    if train:

        rngx = default_rng(42)
        # sampled independently from Unif(0, 1)
        train_X = rngx.uniform(size=(n, p))
        #print(train_X)
        #input("wait")
        rnge = default_rng(69)
        E_train = rnge.normal(0, sigma_e, n)
        #print(E_train)
        #input("wait")
        X_train = train_X.transpose()

        #print(E_train.shape)
        #print(X_train.shape)
        #print(X_train[3].shape)
        #print(array([sin(pi * x0 * x1) for (x0, x1) in zip(
        #    X_train[0], X_train[1])]).shape)
        #print(array([5 * x4 for x4 in X_train[4]]).shape)
        #print(E_train.shape)

        Y_train = array([10 * sin(pi * x0 * x1) for (x0, x1) in zip(
            X_train[0], X_train[1])]) + array([20 * (
                x2 - 0.05) ** 2 for x2 in X_train[2]]) +\
                  array([10 * x3 for x3 in X_train[3]]) +\
                  array([5 * x4 for x4 in X_train[4]]) + E_train

        # create the train set using 42 & 69
        X_out, Y_out = train_X, Y_train

    else:

        rngx = default_rng(42 * (i + 1))
        # sampled independently from Unif(0, 1)
        test_X = rngx.uniform(size=(n, p))
        rnge = default_rng(69 * (i + 1))
        E_test = rnge.normal(0, sigma_e, n)
        X_test = test_X.transpose()

        Y_test = array([10 * sin(pi * x0 * x1) for (x0, x1) in zip(
            X_test[0], X_test[1])]) + array([20 * (
                x2 - 0.05) ** 2 for x2 in X_test[2]]) + \
                  array([10 * x3 for x3 in X_test[3]]) + \
                  array([5 * x4 for x4 in X_test[4]]) + E_test

        # create the test set using fixed random states
        X_out, Y_out = test_X, Y_test

    return X_out, Y_out


def workflow_q4a(vs, params_lin, params_msn):

    global test_repeats, test_n

    # should be done 1 time for each dataset setting
    S_in, B_in = get_cov_and_coeffs(params_lin)

    params_lin_test, params_msn_test = params_lin.copy(), params_msn.copy()
    params_lin_test['n'], params_msn_test['n'] = test_n, test_n

    results_lin, results_msn = [], []

    print("Sum of MSEs from {} test sets".format(test_repeats))
    print("""using Bagging & Random Forest respectively
    with the linear model Y = XB + E first
    and the MARS nonlinear model in section 4.1 next:
    SNR    sum(MSE(Bag)) sum(MSE(RF)) mean(diff) mean(diff, %)
    ==========================================================""")

    for v in vs:

        X_train_lin, train_Y_lin = get_lin_models(params_lin, S_in, B_in, v)
        X_train_msn, train_Y_msn = get_nonlin_models(params_msn, v)
        Y_train_lin, Y_train_msn = train_Y_lin.ravel(), train_Y_msn.ravel()
        brmses_lin, rfrmses_lin, brmses_msn, rfrmses_msn = 0, 0, 0, 0

        for j in range(1, test_repeats + 1):
            # for the linear model Y = XB + E
            X_test_lin, test_Y_lin = get_lin_models(
                params_lin_test, S_in, B_in, v, j, train=False)
            Y_test_lin = test_Y_lin.ravel()
            # fit using bagging (the regressor one)
            Y_br_lin = BR(random_state=(777 * j)).fit(
                X_train_lin, Y_train_lin).predict(X_test_lin)
            # fit using the random forest with only 1 / 3 features fed in
            # 1 / 3 c.f. the default value in the R packages
            Y_rfr_lin = RFR(random_state=(3000 * j), max_features=(
                    1 / 3)).fit(X_train_lin, Y_train_lin).predict(X_test_lin)
            brmses_lin += sum((Y_br_lin - Y_test_lin) ** 2) / test_n
            rfrmses_lin += sum((Y_rfr_lin - Y_test_lin) ** 2) / test_n

            # for MARS nonlinear model
            X_test_msn, test_Y_msn = get_nonlin_models(
                params_msn_test, v, j, train=False)
            Y_test_msn = test_Y_msn.ravel()
            # fit using bagging (the regressor one)
            Y_br_msn = BR(random_state=(777 * j)).fit(
                X_train_msn, Y_train_msn).predict(X_test_msn)
            # fit using the random forest with only 1 / 3 features fed in
            # 1 / 3 c.f. the default value in the R packages
            Y_rfr_msn = RFR(random_state=(3000 * j), max_features=(
                    1 / 3)).fit(X_train_msn, Y_train_msn).predict(X_test_msn)
            brmses_msn += sum((Y_br_msn - Y_test_msn) ** 2) / test_n
            rfrmses_msn += sum((Y_rfr_msn - Y_test_msn) ** 2) / test_n

        result_lin = (brmses_lin - rfrmses_lin) / test_repeats
        per_lin = 100 * result_lin * test_repeats / brmses_lin
        print("{:.4f} {:13.4f} {:12.4f} {:10.4f} {:14.4f}".format(
            v, brmses_lin, rfrmses_lin, result_lin, per_lin))
        results_lin.append(result_lin)

        result_msn = (brmses_msn - rfrmses_msn) / test_repeats
        per_msn = 100 * result_msn * test_repeats / brmses_msn
        print("{:.4f} {:13.4f} {:12.4f} {:10.4f} {:14.4f}".format(
            v, brmses_msn, rfrmses_msn, result_msn, per_msn))
        results_msn.append(result_msn)

    # for convenience
    print("Then the MSE differences from bagging and random forests")
    print("based on 10 different values of SNR ratio from the linear model are")
    print(results_lin)
    print("While the same quantities from the MARS nonlinear model are")
    print(results_msn)

    return results_lin, results_msn


# workflow_q4b -- find
# 1. the very most optimal mtry value (least MSE on the test set)
#### for both linear and MARS nonlinear model [x 2]
#### on each v value [x 10]
#### across the 500 test sets [x 500]
#### as we train the forest [x 1]
#### while the training (and testing) set has n = 50 or 500 [x 2]
# 2. and the distribution on the optimal mtry value
#### from each of the 500 test sets
#### on both linear and MARS nonlinear model
#### with the training set has n = 50 or 500
#### so the distribution on each v value
#### should have n = 500


def workflow_q4b(ms, vs, params_lin, params_msn):

    global test_repeats

    # should be done 1 time for each dataset setting
    S_in, B_in = get_cov_and_coeffs(params_lin)

    params_lin_test, params_msn_test = params_lin.copy(), params_msn.copy()
    test_n = params_lin_test['n']

    al_lin, means_lin, stds_lin, mins_mse_lin, argmins_lin = [], [], [], [], []
    al_msn, means_msn, stds_msn, mins_mse_msn, argmins_msn = [], [], [], [], []

    print("Summary from {} test sets".format(test_repeats))
    print("""using Random Forest with different SNR and mtry values
with the linear model Y = XB + E first
and the MARS nonlinear model in section 4.1 next:
SNR     mean(mtry)    SD(mtry)    min(MSE) argmin(MSE)
======================================================""")

    for v in vs:

        X_train_lin, train_Y_lin = get_lin_models(params_lin, S_in, B_in, v)
        X_train_msn, train_Y_msn = get_nonlin_models(params_msn, v)
        Y_train_lin, Y_train_msn = train_Y_lin.ravel(), train_Y_msn.ravel()
        rfrmses_lin, rfrmses_msn = [], []
        rfr_ms_lin, rfr_ms_msn = [], []

        for j in range(1, test_repeats + 1):

            this_lin, this_msn = [], []

            # for the linear model Y = XB + E
            X_test_lin, test_Y_lin = get_lin_models(
                params_lin_test, S_in, B_in, v, j, train=False)
            Y_test_lin = test_Y_lin.ravel()

            # for MARS nonlinear model
            X_test_msn, test_Y_msn = get_nonlin_models(
                params_msn_test, v, j, train=False)
            Y_test_msn = test_Y_msn.ravel()

            for m in ms:

                # linear model
                # fit using the random forest with only 1 / 3 features fed in
                # 1 / 3 c.f. the default value in the R packages
                Y_rfr_lin = RFR(random_state=(3000 * j), max_features=m).fit(
                    X_train_lin, Y_train_lin).predict(X_test_lin)
                this_lin.append(sum((Y_rfr_lin - Y_test_lin) ** 2) / test_n)

                # MARS nonlinear model
                # fit using the random forest with only 1 / 3 features fed in
                # 1 / 3 c.f. the default value in the R packages
                Y_rfrmses_msn = RFR(random_state=(3000 * j), max_features=m).fit(
                    X_train_msn, Y_train_msn).predict(X_test_msn)
                this_msn.append(sum((Y_rfrmses_msn - Y_test_msn) ** 2) / test_n)

            # change the list to numpy array
            this_lin, this_msn = array(this_lin), array(this_msn)
            # then find the index of the argmin with respect to m
            # e.g. min_ind_lin = argmin_(ind of m) (this_lin)
            min_ind_lin, min_ind_msn = argmin(this_lin), argmin(this_msn)
            min_lin, min_msn = min(this_lin), min(this_msn)
            # append the corresponding minimized MSE and m value
            [x.append(y) for (x, y) in zip(
                [rfrmses_lin, rfrmses_msn, rfr_ms_lin, rfr_ms_msn],
                [min_lin, min_msn, ms[min_ind_lin], ms[min_ind_msn]])]

        #print(rfr_ms_lin)
        #print(rfrmses_lin)

        #input("wait")

        hi = 0

        # should give the mean and sample SD of the linear model
        # for each of the test_n test sets
        mean_lin, std_lin = mean(rfr_ms_lin), std(rfr_ms_lin, ddof=1)
        min_mse_lin, argmin_lin = min(
            rfrmses_lin), rfr_ms_lin[argmin(rfrmses_lin)]
        print("{:.4f} {:11.4f} {:11.4f} {:11.4f} {:11.4f}".format(
            v, mean_lin, std_lin, min_mse_lin, argmin_lin))
        [x.append(y) for (x, y) in zip(
            [al_lin, means_lin, stds_lin, mins_mse_lin, argmins_lin],
            [rfr_ms_lin, mean_lin, std_lin, min_mse_lin, argmin_lin])]

        # should give the mean and sample SD of the linear model
        # for each of the test_n test sets
        mean_msn, std_msn = mean(rfr_ms_msn), std(rfr_ms_msn, ddof=1)
        min_mse_msn, argmin_msn = min(
            rfrmses_msn), rfr_ms_msn[argmin(rfrmses_msn)]
        print("{:.4f} {:11.4f} {:11.4f} {:11.4f} {:11.4f}".format(
            v, mean_msn, std_msn, min_mse_msn, argmin_msn))
        [x.append(y) for (x, y) in zip(
            [al_msn, means_msn, stds_msn, mins_mse_msn, argmins_msn],
            [rfr_ms_msn, mean_msn, std_msn, min_mse_msn, argmin_msn])]

    # for convenience
    print("""Then the mean, sample SD for the optimal mtry values (n = {}),
and the smallest MSE value and the corresponding optimal mtry value
out of {} test sets are""".format(test_repeats, test_repeats))
    print(means_lin, stds_lin, mins_mse_lin, argmins_lin)
    print("The distribution (n = {}) of the mtry values is".format(test_repeats))
    print(al_lin)
    print("While the same quantities from the MARS nonlinear model are")
    print(means_msn, stds_msn, mins_mse_msn, argmins_msn)
    print("And the mtry values distribution (n = {}) is".format(test_repeats))
    print(al_msn)

    input("wait")

    # expected results_lin:
    # list of 10 lists [per v]
    # of 500 values [per test set]
    # of argmin(MSE on each test set) [choices from 0.05 to 1]
    results_lin = [al_lin, means_lin, stds_lin, mins_mse_lin, argmins_lin]
    results_msn = [al_msn, means_msn, stds_msn, mins_mse_msn, argmins_msn]

    return results_lin, results_msn


start = time.perf_counter()

# Crucial for comparison
# set max_features for RFR to 1/3 or RFC to "sqrt"


# n = training sample size
# p = no. of generated features
# s = no. of features with nonzero coefficient thus considered signal

# Part A
# The linear Y = XB + E
# Low: n = 100, p = 10, s = 5
low = {'n': 100,        'p': 10,    's': 5}
# Medium: n = 500, p = 100, s = 5
medium = {'n': 500,     'p': 100,   's': 5}
# High-10: n = 100, p = 1000, s = 10
high_10 = {'n': 100,    'p': 1000,  's': 10}

# MARS nonlinear model
low_marsb = {'n': 200,  'p': 5, 's': 5}
med_marsb = {'n': 500,  'p': 5, 's': 5}
hig_marsb = {'n': 1000, 'p': 5, 's': 5}

# Part B
# The linear Y = XB + E
# p = 20, s = 10
# small: n = 50, large: n = 500
smo_lin = {'n': 50,     'p': 20, 's': 10}
lar_lin = {'n': 500,    'p': 20, 's': 10}

# the MARS nonlinear model: p = 5, s = 5
smo_marsb = {'n': 50,   'p': 5, 's': 5}
lar_marsb = med_marsb.copy()

# Section 4 how to create the test set
# change to 1000 and 500 later
# the test_n here only for Part A
# Part B use the same n as the training sets
test_n, test_repeats = 500, 25
v_list = geoms(0.05, 6.0, 10)

# for part b, mtry ranges from lins(0.05, 1, 20)
mtry_list = lins(0.05, 1, 20)

# S_1 and B_1 should only depend on p and s
rho_1 = 0.35

#lin_low, msn_low = workflow_q4a(v_list, low)
#lin_med, msn_med = workflow_q4a(v_list, medium)
#lin_h10, msn_h10 = workflow_q4a(v_list, high_10)

#lin_smo, msn_smo = workflow_q4b(mtry_list, v_list, smo_lin, smo_marsb)
#lin_lar, msn_lar = workflow_q4b(mtry_list, v_list, lar_lin, lar_marsb)

#dist_lin_smo, dist_msn_smo = lin_smo[0], msn_smo[0]
#dist_lin_lar, dist_msn_lar = lin_lar[0], msn_lar[0]
#ml_min_smo, mm_min_smo = lin_smo[-1], msn_smo[-1]
#ml_min_lar, mm_min_lar = lin_lar[-1], msn_lar[-1]

# the indices for plotting the graphs
inds = lins(1, len(v_list), len(v_list))
v_strings = [round(v, 2) for v in v_list]
inds_corr = [x - 1 for x in inds]
rights = [x - 0.85 for x in inds]
lefts = [x - 1.15 for x in inds]

from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

#figa, [axel, axem, axeh] = plt.subplots(
#    figsize=(12, 3.5), ncols=3, sharex=True)

figa, [axea, axeb] = plt.subplots(figsize=(10, 4.5), ncols=2)

#results_low = [19.665690453715143, 11.003969537406393, 5.987559731950805, 3.1641938076099585, 2.108849740749688, 1.1731256961893086, 0.6759477066173781, 0.5400392396925008, 0.3985270527537961, 0.26983126031851495]
#results_med = [16.232511282168307, 10.395192426858266, 6.295370596633575, 3.7465763133667678, 2.3476536843205396, 1.6346521715445101, 1.122763120337254, 0.763424912043148, 0.5848014962209607, 0.4182039068485034]
#results_h10 = [32.24126874549162, 19.33573588995281, 11.667472713421539, 7.193315194925108, 5.240099130206481, 3.3429207360315196, 1.5239305846158822, 0.9477091550712612, 0.815466775151549, 0.3301287363685651]

lin_low = [19.60976315454147, 10.929936436400531, 5.874601257590821, 3.364670678794491, 2.2294849573414424, 1.1546375429387172, 0.6070446371000195, 0.45097419586262505, 0.3761364378345525, 0.1972457074124344]
msn_low = [1237.819218955387, 701.0556116726797, 431.04776612733724, 237.88937891517475, 167.73753522598855, 103.2220007548784, 75.60085726141887, 45.52544840508875, 28.764293038074683, 19.331517537093095]
lin_med = [17.121659753999158, 10.509305723733178, 6.32903170681885, 3.7466954518542934, 2.7481758388291975, 1.6058666377368853, 1.1528224216693035, 0.6922987340413624, 0.5900208950634931, 0.41279645641129037]
msn_med = [1134.004768933754, 672.8272235284198, 385.56417767010106, 234.36641211819892, 146.0412947033372, 81.14318521251815, 46.27486182575798, 26.362350025777342, 19.10287275888015, 8.469019040703206]
lin_h10 = [31.829436738572113, 22.07128217243149, 12.597484650809456, 5.638583114070206, 4.9180337342964595, 2.7278476432642766, 1.5368379558997458, 0.7889394043507173, 0.9000762277588911, 0.338133525965327]
msn_h10 = [955.0487342324969, 547.5293916239345, 316.4617972584511, 205.5098723073263, 123.53881141102029, 75.87852773179591, 42.44430993467286, 21.112449740076045, 9.889361367934399, 3.923186664984132]

# Create empty plot with blank marker containing the extra label
axea.plot([], [], ' ', label="Setting")
axea.plot(inds, lin_low, 'o-', label="Low")
axea.plot(inds, lin_med, 'o-', label="Medium")
axea.plot(inds, lin_h10, 'o-', label="High-10")
axeb.plot([], [], ' ', label="Setting")
axeb.plot(inds, msn_low, 'o-', label="Low")
axeb.plot(inds, msn_med, 'o-', label="Medium")
axeb.plot(inds, msn_h10, 'o-', label="High-10")

figc, [axec, axed] = plt.subplots(figsize=(10, 4.5), ncols=2, sharey=True)

# Mean
#mean_lin_smo = [0.24, 0.266, 0.30400000000000005, 0.40199999999999997, 0.426, 0.4099999999999999, 0.5619999999999999, 0.5920000000000001, 0.628, 0.56]
#mean_msn_smo = [0.098, 0.11399999999999999, 0.11399999999999999, 0.17999999999999997, 0.174, 0.24400000000000002, 0.158, 0.32599999999999996, 0.368, 0.36199999999999993]
#mean_lin_lar = [0.18799999999999997, 0.204, 0.26799999999999996, 0.296, 0.40399999999999997, 0.54, 0.5820000000000001, 0.66, 0.78, 0.72]
#mean_msn_lar = [0.11399999999999999, 0.07200000000000001, 0.146, 0.20199999999999999, 0.174, 0.21599999999999997, 0.184, 0.222, 0.25999999999999995, 0.40199999999999997]

# SD
#sd_lin_smo = [0.2581988897471611, 0.2418332759016702, 0.2617728022541685, 0.30224162519414827, 0.290516780926679, 0.28504385627478446, 0.2825774230188958, 0.30505463991444765, 0.27691755692504105, 0.21984843263788198]
#sd_msn_smo = [0.17587874611030557, 0.18903262505010432, 0.18903262505010432, 0.2553592241007427, 0.27238147269347573, 0.31135456744147283, 0.24095296913159903, 0.2965496023714526, 0.3331916365496989, 0.3301767203584569]
#sd_lin_lar = [0.19218047073866096, 0.16765540054926156, 0.22355088906108156, 0.19467922333931786, 0.2908034846191955, 0.29119008682760245, 0.24869660230891774, 0.24324199198877375, 0.1513825177048746, 0.17320508075688773]
#sd_msn_lar = [0.14966629547095767, 0.11, 0.20912516188477492, 0.26711420778386163, 0.2630430889924057, 0.2748939189335891, 0.232163735324878, 0.24709984486707662, 0.22776083947860748, 0.2518597493315145]

# Min MSE
# min_mse_lin_smo = [279.50714169991056, 160.40741691826955, 97.04417480321554, 58.17536394526206, 36.82360254487074, 24.808837273168365, 15.5627687641653, 9.81396360640336, 7.336321206561521, 6.279453426761705]
# min_mse_msn_smo = [9969.92790109849, 5875.314400593707, 3430.4960383354933, 1996.3102745853464, 1185.4065904177485, 680.9775710872967, 409.07888065132136, 243.2743067169257, 149.85024799453413, 88.40274582032939]
# min_mse_lin_lar = [352.2330237151014, 206.41965316345784, 123.16926439149944, 74.009523393518, 44.234409673246994, 27.260554157000183, 17.171996417323136, 11.4533877312012, 7.885215115701107, 5.871089946024349]
# min_mse_msn_lar = [12045.272306108667, 7063.373221842845, 4139.176730367141, 2427.635412216465, 1439.6074225585262, 851.5241468153, 503.36900595544347, 298.649593282223, 176.6628100365925, 106.46917256697006]

bye = 0

# Corr. optimal mtry
ml_min_smo = [0.25, 0.25, 0.3, 0.95, 0.35, 0.45, 0.7, 0.7, 0.7, 1.0]
mm_min_smo = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.85]
ml_min_lar = [0.05, 0.05, 0.1, 0.2, 0.1, 0.3, 0.65, 0.45, 0.75, 0.55]
mm_min_lar = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.45]

import seaborn as sns
from pandas import DataFrame as DF

#dict_lin_smo = {
#    "v": [str(v) for v in v_list],
#    "Mean": mean_lin_smo,
#    "SD": sd_lin_smo}
#df_lin_smo = DF(dict_lin_smo)
#dict_msn_smo = {
#    "v": [str(v) for v in v_list],
#    "Mean": mean_msn_smo,
#    "SD": sd_msn_smo}
#df_msn_smo = DF(dict_msn_smo)
#dict_lin_lar = {
#    "v": [str(v) for v in v_list],
#    "Mean": mean_lin_lar,
#    "SD": sd_lin_lar}
#df_lin_lar = DF(dict_lin_lar)
#dict_msn_lar = {
#    "v": [str(v) for v in v_list],
#    "Mean": mean_msn_lar,
#    "SD": sd_msn_lar}
#df_msn_lar = DF(dict_msn_lar)

dist_lin_smo = [
    0.15, 0.65, 0.1, 0.05, 0.5499999999999999, 0.05, 0.2, 0.15, 0.15, 0.9, 0.1, 0.1, 0.05, 0.15, 0.1, 0.05, 0.05, 0.6, 0.85, 0.05, 0.35, 0.1, 0.2, 0.25, 0.05, 0.2, 0.6, 0.1, 0.6, 0.3, 0.05, 0.15, 0.35, 0.1, 0.9, 0.2, 0.1, 0.05, 0.1, 0.25, 0.35, 0.05, 0.05, 0.75, 0.05, 0.05, 0.6, 0.2, 0.25, 0.25, 0.05, 0.7, 0.1, 0.6, 0.3, 0.05, 0.1, 0.6, 0.44999999999999996, 0.9, 0.05, 0.1, 0.05, 0.15, 0.25, 0.35, 0.1, 0.5499999999999999, 0.25, 0.1, 0.05, 0.25, 0.6, 0.75, 0.15, 0.1, 0.85, 0.2, 0.25, 0.3, 0.05, 0.35, 0.75, 0.44999999999999996, 0.85, 0.15, 0.1, 0.05, 0.1, 0.2, 0.65, 0.1, 0.6, 0.9, 0.35, 0.1, 0.7, 0.35, 0.95, 0.6, 0.1, 0.6, 0.35, 0.6, 0.2, 0.05, 0.5499999999999999, 0.6, 0.44999999999999996, 0.75, 0.75, 0.05, 0.15, 0.3, 0.1, 0.65, 0.1, 0.95, 0.35, 0.75, 0.1, 0.2, 1.0, 0.35, 0.6, 0.2, 0.6, 0.1, 0.85, 0.1, 0.05, 0.35, 0.7, 0.2, 0.75, 0.7, 0.15, 0.15, 0.3, 0.35, 0.44999999999999996, 0.25, 0.15, 0.15, 0.75, 0.35, 0.35, 0.35, 0.95, 0.95, 0.44999999999999996, 0.9, 0.35, 0.7, 0.15, 0.9, 0.85, 0.35, 0.15, 0.85, 0.7, 0.35, 0.15, 1.0, 0.2, 0.7, 0.7, 0.35, 0.15, 0.7, 0.65, 0.6, 0.7, 1.0, 0.44999999999999996, 0.75, 0.95, 0.3, 0.95, 0.15, 0.05, 0.85, 0.35, 0.35, 0.95, 0.15, 0.9, 0.65, 0.9, 0.2, 0.7, 0.95, 0.85, 0.35, 0.6, 0.44999999999999996, 0.85, 0.3, 0.85, 0.44999999999999996, 0.75, 0.85, 0.1, 1.0, 0.25, 1.0, 0.25, 1.0, 0.3, 0.5499999999999999, 0.65, 0.5499999999999999, 0.75, 0.85, 0.2, 0.7, 0.95, 0.85, 0.85, 0.85, 0.44999999999999996, 0.65, 0.3, 0.6, 0.44999999999999996, 0.6, 0.7, 0.65, 0.25, 0.65, 1.0, 0.25, 0.35, 0.3, 0.6, 0.65, 0.6, 0.65, 0.6, 0.35, 1.0, 0.7, 0.9, 0.44999999999999996, 0.65, 0.44999999999999996, 0.6, 0.3, 0.5499999999999999, 0.2]
dist_lin_lar = [
    0.05, 0.05, 0.3, 0.05, 0.05, 0.15, 0.3, 0.05, 0.35, 0.35, 0.05, 0.3, 0.1, 0.7, 0.2, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.05, 0.75, 0.05, 0.05, 0.05, 0.2, 0.1, 0.1, 0.05, 0.1, 0.25, 0.65, 0.44999999999999996, 0.3, 0.05, 0.35, 0.6, 0.35, 0.2, 0.1, 0.3, 0.1, 0.1, 0.15, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.2, 0.1, 0.05, 0.1, 0.6, 0.25, 0.5499999999999999, 0.15, 0.15, 0.75, 0.1, 0.6, 0.35, 0.2, 0.1, 0.3, 0.25, 0.2, 0.15, 0.15, 0.15, 0.85, 0.25, 0.2, 0.15, 0.6, 0.5499999999999999, 0.25, 0.1, 0.5499999999999999, 0.2, 0.6, 0.15, 0.75, 0.2, 0.1, 0.6, 0.3, 0.1, 0.05, 0.3, 0.25, 0.2, 0.25, 0.15, 0.2, 0.25, 0.35, 0.1, 0.2, 0.2, 0.6, 0.9, 0.25, 0.35, 0.44999999999999996, 0.3, 0.25, 0.7, 0.25, 0.1, 0.85, 0.75, 0.75, 0.1, 0.95, 0.25, 0.2, 0.15, 0.3, 0.2, 0.85, 0.1, 0.3, 0.65, 0.5499999999999999, 0.95, 0.35, 0.25, 0.35, 0.85, 0.7, 0.95, 0.75, 0.95, 0.15, 0.9, 0.5499999999999999, 0.3, 0.15, 0.75, 0.5499999999999999, 0.2, 0.15, 0.25, 0.95, 0.7, 0.3, 0.2, 0.85, 0.25, 0.75, 0.3, 0.35, 0.65, 0.85, 0.95, 0.6, 1.0, 0.35, 0.35, 0.95, 0.65, 0.65, 0.7, 0.65, 0.3, 0.65, 0.3, 0.85, 0.65, 0.44999999999999996, 0.3, 0.7, 0.7, 0.7, 1.0, 0.75, 0.35, 0.85, 0.7, 0.7, 0.35, 1.0, 0.3, 0.9, 0.7, 0.85, 0.44999999999999996, 0.44999999999999996, 0.95, 0.95, 0.9, 0.2, 0.7, 0.35, 0.65, 0.35, 0.85, 0.85, 0.85, 0.75, 0.75, 0.6, 0.85, 0.65, 1.0, 0.44999999999999996, 0.85, 0.65, 0.95, 1.0, 1.0, 0.75, 0.85, 0.7, 0.65, 0.85, 0.95, 0.85, 0.75, 0.5499999999999999, 0.5499999999999999, 0.6, 1.0, 0.6, 0.6, 1.0, 0.35, 0.95, 0.85, 0.9, 0.65, 0.75, 0.44999999999999996, 0.7, 0.75, 0.5499999999999999, 0.5499999999999999, 0.5499999999999999, 0.9, 0.75, 0.95, 0.65, 0.7, 0.85, 0.75, 0.65]
dist_msn_smo = [
    0.05, 0.05, 0.05, 0.05, 0.05, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.44999999999999996, 0.6, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.05, 0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 1.0, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.05, 0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 1.0, 0.6, 0.05, 0.05, 0.05, 0.85, 0.05, 0.05, 0.44999999999999996, 0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.44999999999999996, 0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 1.0, 0.44999999999999996, 0.05, 0.05, 0.6, 0.05, 0.05, 0.85, 0.44999999999999996, 0.44999999999999996, 0.6, 0.05, 0.44999999999999996, 0.05, 0.6, 0.05, 0.05, 0.44999999999999996, 0.6, 0.6, 0.6, 0.05, 0.05, 0.05, 0.44999999999999996, 0.85, 0.85, 0.05, 0.05, 0.85, 0.05, 0.05, 0.6, 0.6, 0.44999999999999996, 0.44999999999999996, 0.05, 0.05, 0.05, 0.85, 0.05, 0.05, 0.85, 0.6, 0.6, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.85, 0.44999999999999996, 0.05, 0.05, 0.6, 0.05, 0.05, 0.85, 0.85, 0.6, 0.6, 0.05, 0.05, 0.6, 0.85, 0.05, 0.05, 0.44999999999999996, 0.85, 0.44999999999999996]
dist_msn_lar = [
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.44999999999999996, 0.44999999999999996, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.44999999999999996, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.85, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.44999999999999996, 0.05, 0.44999999999999996, 0.05, 0.05, 0.05, 0.6, 0.05, 0.05, 0.6, 0.05, 0.05, 0.05, 0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.6, 0.05, 0.05, 0.85, 0.05, 0.05, 0.6, 0.05, 0.44999999999999996, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.05, 1.0, 0.05, 0.44999999999999996, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.85, 0.05, 0.44999999999999996, 0.05, 0.05, 0.85, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.44999999999999996, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.6, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.44999999999999996, 0.6, 0.05, 0.6, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.05, 0.05, 0.85, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.44999999999999996, 0.05, 0.44999999999999996, 0.05, 0.05, 0.6, 0.05, 0.05, 0.05, 0.44999999999999996, 0.44999999999999996, 0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.44999999999999996, 0.44999999999999996, 0.05, 0.6, 0.44999999999999996, 0.44999999999999996, 0.05, 0.44999999999999996, 0.44999999999999996, 0.05, 0.44999999999999996, 0.05, 0.6, 0.6, 0.44999999999999996, 0.44999999999999996, 0.05, 0.05, 0.44999999999999996, 0.6, 0.05, 0.44999999999999996, 0.05, 0.05, 0.44999999999999996, 0.44999999999999996, 0.44999999999999996, 0.6, 0.44999999999999996, 0.44999999999999996, 0.6, 0.6, 1.0, 0.6]

dict_lin = {"v": [y for x in v_list for y in (x,)*(test_repeats)] * 2,
            "y": dist_lin_smo + dist_lin_lar,
            "n": ['50'] * (len(mean_lin_smo) * test_repeats) +
                 ['500'] * (len(mean_lin_lar) * test_repeats)}
df_lin = DF(dict_lin)

dict_msn = {"v": [y for x in v_list for y in (x,)*(test_repeats)] * 2,
            "y": dist_msn_smo + dist_msn_lar,
            "n": ['50'] * (len(mean_msn_smo) * test_repeats) +
                 ['500'] * (len(mean_msn_lar) * test_repeats)}
df_msn = DF(dict_msn)

# Plot Mean +/- 1SD
# Plot the optimal mtry values against v values for Y = XB + E
sns.pointplot(x="v", y="y", hue="n", data=df_lin, dodge=0.3, ax=axec,
              ci=68, markers="x", scale=0.8).set(xlabel=None)
# Close the legend in axec
axec.legend([], [], frameon=False)
# Plot the optimal mtry values against v values for MARS nonlinear model
sns.pointplot(x="v", y="y", hue="n", data=df_msn, dodge=0.3, ax=axed,
              ci=68, markers="x", scale=0.8).set(xlabel=None, ylabel=None)

axec.scatter(rights, ml_min_smo, label="n = 50")
axec.scatter(lefts, ml_min_lar, label="n = 500")
axed.scatter(rights, mm_min_smo, label="n = 50")
axed.scatter(lefts, mm_min_lar, label="n = 500")

plt.setp([axea, axeb],
         xticks=inds, xticklabels=v_strings)

plt.setp([axec, axed],
         xticks=inds_corr, xticklabels=v_strings)

for ax in [axea, axeb, axec, axed]:
    ax.grid()

axea.set_ylabel('MSE(Bag) - MSE(Forest)')
axec.set_ylabel('Optimal mtry value')

axea.set_title("Linear model", fontsize=12)
axeb.set_title("The MARS model (Section 4)", fontsize=12)
axec.set_title("Linear model", fontsize=12)
axed.set_title("The MARS model (Section 4)", fontsize=12)

figa.supxlabel("Signal-noise ratio", fontsize=12)
axeb.legend()

figc.supxlabel("Signal-noise ratio", fontsize=12)

end = time.perf_counter()
print("This part takes {:.2f} seconds!".format(end - start))

plt.show()
