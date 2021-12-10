import matplotlib.pyplot as plt
from sklearn.ensemble import (RandomForestRegressor as RFR,
                              RandomForestClassifier as RFC,
                              BaggingRegressor as BR, BaggingClassifier as BC)

from numpy import empty, zeros, ones, concatenate, geomspace as geoms, eye,\
    array, mean, std, argmin, set_printoptions, linspace as lins, around
from numpy.random import default_rng, normal, multivariate_normal
import time
from math import sqrt, sin, pi

from numpy import var, sum, log2
from sklearn.datasets import fetch_openml
from pandas import DataFrame as DF, Series, to_numeric
from sklearn.model_selection import cross_val_score as CVS, KFold
from sklearn.preprocessing import LabelEncoder as LE


# TODO: why the MSE I get is much larger?


# Abalone Age [abalone] (Waugh, 1995) 8 4177
# Bike Sharing [bike] (Fanaee-T and Gama, 2014) 11 731 (name='Bike_sharing_demand', version=3)
# Conventional and Social Movie [csm] (Ahmed et al., 2015) 10 187 (name='CSM', version=2)
# # mustdo
# Parkinsons Telemonitoring [parkinsons] (Tsanas et al., 2009) 20 5875
# Molecular Descriptor Influencing Melting Point [mtp2] (Bergstr¨om et al., 2003) 1142 274
# Weighted Holistic Invariant Molecular Descriptor [pah] (Todeschini et al., 1995) 112 80 # mustdo
# Adrenergic Blocking Potencies [phen] (Cammarata, 1972) 110 22
# PDGFR Inhibitor [pdgfr] (Guha and Jurs, 2004)
#
# CSV
# Aquatic Toxicity [AquaticTox] (He and Jurs, 2005) 468 322

# Call the datasets
mtp2_b = fetch_openml(name="mtp2")
pah_b = fetch_openml(name="pah")
phen_b = fetch_openml(name="phen")
pdgfr_b = fetch_openml(name="pdgfr")

aba_b = fetch_openml(name="abalone", version=1)
bike_b = fetch_openml(name='Bike_sharing_demand', version=3)
park_b = fetch_openml(name="parkinsons")


def bunch_to_df(datasets):
    for dataset in datasets:
        yield DF(dataset.data, columns=dataset.feature_names)


def target_to_df(datasets):
    for dataset in datasets:
        yield Series(dataset.target)


# low dim datasets
aba_x, bike_x, park_x = bunch_to_df([aba_b, bike_b, park_b])
aba_y, bike_y, park_y = target_to_df([aba_b, bike_b, park_b])

#print(aba_x.describe(include="category"))
#print(park_x.describe(include="all"))
#print(aba_y)
#print(park_y)
#input("wait")

#print(bike_x.describe(include="category"))
#print(bike_x[["season"]])
#le_szn = LE()
bike_x[["season"]] = LE().fit_transform(bike_x[["season"]])
#print(le_szn.classes_)
#input("wait")
bike_x["season"].replace([0, 1, 2], [2, 0, 1], inplace=True)
#input("wait")
#print(bike_x[["weather"]])
bike_x[["weather"]] = LE().fit_transform(bike_x[["weather"]])
#print(bike_x[["weather"]])
bike_x["weather"].replace([1, 2, 3], [3, 1, 2], inplace=True)
#print(bike_x[["weather"]])
#print(bike_x.describe(include="all"))
#input("wait")

# high dim datasets
#mtp2_x, pah_x, phen_x, pdgfr_x = bunch_to_df([mtp2_b, pah_b, phen_b, pdgfr_b])
#mtp2_y, pah_y, phen_y, pdgfr_y = target_to_df([mtp2_b, pah_b, phen_b, pdgfr_b])

# abalone is classification of 29 holes
# bike is still a regression problem about how many will lend the bike
# parkinsons is either have the disease or not (boolean)

# change to 500 later
test_repeats = 10
alphas = [0.01, 0.05, 0.1, 0.25, 0.5]
alphas_all = [0.0] + alphas
#def add_noise(target, alphas_in):
#    for a in alphas_in:
#        yield target + default_rng(42).normal(0, a * var(target), len(target))
# successful
#mtp2_y_0, mtp2_y_1, mtp2_y_5, mtp2_y_10, mtp2_y_25, mtp2_y_50 = add_noise(mtp2_y, alphas)


def workflow_q4c(X_in, Y_in, alphas_in, reg=True):

    global test_repeats

    #print(len(Y_in))

    #print("Is regression? {}".format(str(reg)))

    if reg:
        therfs = RFR(random_state=689, max_features=(1 / 3))
        thebs = BR(random_state=777)
    else:
        therfs = RFC(random_state=310, max_features='sqrt')
        thebs = BC(random_state=222)

    # change n_splits to 10 later
    folds = KFold(n_splits=5)
    nmse = "neg_mean_squared_error"
    mses_diffs_0 = []
    Y_in = to_numeric(Y_in)
    #print(Y_what)
    #input("wait")
    pY = Y_in / Y_in.sum()
    shan = -sum(pY * log2(pY))
    #print(shan)

    for take in range(test_repeats):

        print("Take {} starts!".format(take + 1))

        if reg:
            rfran = RFR(random_state=(689 * (take + 2)), max_features=(1 / 3))
            bran = BR(random_state=(777 * (take + 2)))
        else:
            rfran = RFC(random_state=(310 * (take + 2)), max_features="sqrt")
            bran = BC(random_state=(222 * (take + 2)))

        mse_rfr0 = CVS(estimator=rfran, X=X_in, y=Y_in, scoring=nmse, cv=folds)
        mse_br0 = CVS(estimator=bran, X=X_in, y=Y_in, scoring=nmse, cv=folds)

        # both errors are set negative so rfr0-br0 is expected to >= 0
        # get the RTE error when there are no additional noise
        if reg:
            mses_diffs_0.append((mse_rfr0 - mse_br0).mean() / var(Y_in) * 100)
        else:
            mses_diffs_0.append((mse_rfr0 - mse_br0).mean() / shan * 100)

    print("The CV MSEs for the not noised dataset are")
    print(mses_diffs_0)

    mses_diffs = [mean(mses_diffs_0), mses_diffs_0]

    for a in alphas_in:

        mses_diff = []

        print("Alpha {} starts!".format(a))

        for take in range(1, test_repeats + 1):

            print("Take {} starts!".format(take))

            if reg:

                Y_noised = Y_in + default_rng(
                    42 * take).normal(0, a * var(Y_in), len(Y_in))
                mse_rfr = CVS(estimator=therfs, X=X_in, y=Y_noised,
                              scoring="neg_mean_squared_error", cv=folds)
                mse_br = CVS(estimator=thebs, X=X_in, y=Y_noised,
                             scoring="neg_mean_squared_error", cv=folds)
                mses_diff.append((mse_rfr - mse_br).mean() / var(Y_in) * 100)

            else:

                # if the dataset is for classification
                # make the noise +/- 1 or some classes
                # but also bound the noise
                ymin, ymax = min(Y_in), max(Y_in)
                Y_noised = Y_in + around(default_rng(
                    42 * take).normal(0, a * shan, len(Y_in)))
                Y_noised[Y_noised > ymax] = ymax
                Y_noised[Y_noised < ymin] = ymin

                mse_rfr = CVS(estimator=therfs, X=X_in, y=Y_noised,
                              scoring=nmse, cv=folds)
                mse_br = CVS(estimator=thebs, X=X_in, y=Y_noised,
                             scoring=nmse, cv=folds)
                mses_diff.append((mse_rfr - mse_br).mean() / shan * 100)

        print("The CV MSEs for the noised dataset "
              "with alpha = {} are".format(a))

#        if reg:
#            print("The CV MSEs for the noised dataset "
#                  "with alpha = {} are".format(a))
#        else:
#            print("The CV MSEs for the noised dataset "
#                  "with alpha = {} are".format(a))

        print(mses_diff)

        mses_diffs.append(mses_diff)

    return mses_diffs


start = time.perf_counter()


#mtp2_mses = workflow_q4c(mtp2_x, mtp2_y, alphas_in=alphas)
#pah_mses = workflow_q4c(pah_x, pdgfr_y, alphas_in=alphas)
#phen_mses = workflow_q4c(phen_x, pdgfr_y, alphas_in=alphas)
#pdgfr_mses = workflow_q4c(pdgfr_x, pdgfr_y, alphas_in=alphas)

#aba_mses = workflow_q4c(aba_x, aba_y, alphas_in=alphas, reg=False)
bike_mses = workflow_q4c(bike_x, bike_y, alphas_in=alphas)
#park_mses = workflow_q4c(park_x, park_y, alphas_in=alphas, reg=False)

mu_aba, mu_bike, mu_park = aba_mses[0], bike_mses[0], park_mses[0]

#aba_flat = [item for sublist in aba_mses[1] for item in sublist]
bike_flat = [item for sublist in bike_mses[1] for item in sublist]
#park_flat = [item for sublist in park_mses[1] for item in sublist]

#mu_mtp2, mu_pah = mtp2_mses[0], pah_mses[0]
#mu_phen, mu_pdgfr = phen_mses[0], pdgfr_mses[0]

mtp2_0 = [9.307043392923275, 1.0108258132965087, 6.675743773648255, 8.701126166609574, 7.01338046550626, 7.183439572062171, 9.461774841964735, 7.807815441337783, 12.518698760836369, 4.928940783765528]
mtp2_1 = [10.771970584698423, 6.6227011949139385, 7.495401869776977, 9.87337052561723, 9.493923545334008, 10.18148148142913, 10.200191231146109, 9.483778189859812, 6.705383271759491, 9.361784652370424]
mtp2_5 = [9.940129811321182, 9.44239981077946, 10.456702309109364, 9.526457197326396, 8.083101067330293, 9.765986634368627, 13.407954808103103, 9.895421071730487, 8.44794100370224, 8.403164647977706]
mtp2_10 = [9.28615166330274, 12.07193974799741, 8.068960593440016, 10.814265367962507, 15.01895199965307, 9.578973013428993, 15.86206146022018, 8.593238920281252, 13.143377176301623, 7.406546184566228]
mtp2_25 = [13.61962417984369, 10.91186101976275, 6.636662791146721, 11.538034350704606, 15.741884511609323, 7.936268333082194, 11.557538321774558, 7.965212827135339, 10.918806829254038, 9.929999381938357]
mtp2_50 = [12.249767844369874, 12.492486694146303, 6.411271368017192, 8.332016483123382, 12.371679481291393, 4.741417231823561, 9.385659644195451, 8.127969400002913, 7.033257316552678, 6.83513629673462]
mu_mtp2 = mean(mtp2_0)
mtp2_flat = mtp2_0+mtp2_1+mtp2_5+mtp2_10+mtp2_25+mtp2_50

fige, [axee, axef] = plt.subplots(figsize=(5, 5), ncols=2)

import seaborn as sns

dict_mtp2 = {"a": [y for x in alphas_all for y in (x,)*test_repeats],
             "y": [y - mu_mtp2 for y in mtp2_flat],
             "n": ['mtp2'] * ((len(alphas_all)) * test_repeats)}
df_mtp2 = DF(dict_mtp2)

dict_aba = {"a": [y for x in alphas_all for y in (x,)*test_repeats],
            "y": [y - mu_aba for y in aba_flat],
            "n": ['mtp2'] * ((len(alphas_all)) * test_repeats)}
df_aba = DF(dict_aba)
dict_bike = {"a": [y for x in alphas_all for y in (x,)*test_repeats],
             "y": [y - mu_bike for y in bike_flat],
             "n": ['mtp2'] * ((len(alphas_all)) * test_repeats)}
df_bike = DF(dict_bike)
dict_park = {"a": [y for x in alphas_all for y in (x,)*test_repeats],
             "y": [y - mu_park for y in park_flat],
             "n": ['mtp2'] * ((len(alphas_all)) * test_repeats)}
df_park = DF(dict_park)


# Plot Mean +/- 1SD
sns.lineplot(x="a", y="y", hue="n", data=df_mtp2, ax=axef,
             ci='sd', style="n", markers=["o"],
             err_style="bars").set(xlabel=None)
#sns.lineplot(x="a", y="y", hue="n", data=df_pah, ax=axef,
#             ci='sd', style="n", markers=["o"],
#             err_style="bars").set(xlabel=None)
#sns.lineplot(x="a", y="y", hue="n", data=df_phen, ax=axef,
#             ci='sd', style="n", markers=["o"],
#             err_style="bars").set(xlabel=None)
#sns.lineplot(x="a", y="y", hue="n", data=df_pdgfr, ax=axef,
#             ci='sd', style="n", markers=["o"],
#             err_style="bars").set(xlabel=None)

sns.lineplot(x="a", y="y", hue="n", data=df_aba, ax=axee,
             ci='sd', style="n", markers=["o"],
             err_style="bars").set(xlabel=None)
sns.lineplot(x="a", y="y", hue="n", data=df_bike, ax=axee,
             ci='sd', style="n", markers=["o"],
             err_style="bars").set(xlabel=None)
sns.lineplot(x="a", y="y", hue="n", data=df_park, ax=axee,
             ci='sd', style="n", markers=["o"],
             err_style="bars").set(xlabel=None)

axee.set_ylabel('Shifted relative test error (%)')
axef.set_title("Low dim datasets", fontsize=12)
axef.set_title("High dim datasets", fontsize=12)
fige.supxlabel("var(noise)/var(original target)", fontsize=12)

plt.setp([axee, axef],
         xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5])

axee.grid()
axef.grid()

end = time.perf_counter()
print("This part takes {:.2f} seconds!".format(end - start))

plt.show()
