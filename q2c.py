"""
Let's go
"""

import os
import sys
import time
import warnings
from math import log
from matplotlib import pyplot as plt
from numpy import (linspace as lins, exp, array, where, zeros, ones, argmax,
                   set_printoptions, empty, full)
from pandas import read_csv, DataFrame as DF
from seaborn import pairplot, lineplot as slplot, set_context
from statistics import mean, stdev

from sklearn.ensemble import (GradientBoostingClassifier as GBC,
                              AdaBoostClassifier as ADBC)
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import roc_auc_score, balanced_accuracy_score as bal_acc
from sklearn.model_selection import (train_test_split as TTS,
                                     GridSearchCV as GSCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler as SS, PowerTransformer as PT
from sklearn.tree import DecisionTreeClassifier as TC

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    # Also affect subprocesses

set_printoptions(precision=3, suppress=True)

start = time.perf_counter()

# configs about the plots

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
#h_font = {'fontname': 'Helvetica', 'fontsize': 12}
a_font = {'fontname': 'Arial', 'fontsize': 12}

# load the csv file as a data frame
# and add the names to the columns

glass_df = read_csv('glass.csv', header=None)
glass_df.set_axis(['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe',
                   'Type'], axis=1, inplace=True)

# visualize each column first by itself

figa, axea = plt.subplots(figsize=(3, 3))
glass_df['Type'].hist(grid=False, bins=[1, 2, 3, 4, 5, 6, 7], ax=axea)
axea.set_ylabel("Count", **a_font)
axea.set_title("Type code of glass", **a_font)

figb, axeb = plt.subplots(ncols=3, nrows=3, figsize=(6, 5.85))
glass_df.iloc[:, :9].hist(grid=False, ax=axeb)
figb.supylabel("Count", **a_font)

# since we observe quite skewed and even bimodal distributions
# we do the power transform and standardization in the following
# before feeding the data into the classifiers

# retrieve numpy array
# split into input and output elements

glass_np = glass_df.values
X, Y = glass_np[:, :-1], glass_np[:, -1]

# sample size M = 214; the no. of classes K = 7

M, K = X.shape[0], int(max(Y))
class_iter = lins(1, K, K)

# max(iter_rs) = 20, max(iter_boost) = 3000

iter_rs, iter_boost = lins(1, 20, 20), lins(1, 25, 25)
pipe_all = [Pipeline([('power', PT()), ('scaler', SS()), ('model', TC(
        random_state=i))]) for i in range(1, int(len(iter_boost)) + 1)]


mbc_train_es, mbc_test_es, gbc_train_es, gbc_test_es = [], [], [], []

# =====================================================================

print("Iterations start!")

for h in iter_rs:

    X_train, X_test, Y_train, Y_test = TTS(
        X, Y, test_size=(64 / 214), random_state=int(h))

    train_e_mbc, test_e_mbc, train_e_gbc, test_e_gbc = [], [], [], []

    for l in iter_boost:

        # now l weak learners

        learners = pipe_all[:int(l)]
        M_train, M_test = int(X_train.shape[0]), int(X_test.shape[0])
        # N_learn = len(learner) = l

        # initialize the weights for each data entry

        w_sample_in = full((M_train,), fill_value=(1 / M_train))
        w_learn = zeros((int(l),))

        # sort the weak learners by error

        error = [0] * int(l)
        for id, learner in enumerate(learners):
            for (X_in, Y_in) in zip(X_train, Y_train):
                weak = learner.fit(X_train, Y_train)
                error[id] += int(int(weak.predict([X_in])[0]) != int(Y_in))

        sort_learners = [r for (r, e) in sorted(
            zip(learners, error), key=(lambda pair: pair[1]))]

        # boost

        for learn_id, learner in enumerate(sort_learners):
            
            # compute weighted error

            is_wrong = zeros((M_train,))
            stronger = learner.fit(X_train, Y_train)
            for X_id, X_in in enumerate(X_train):
                if int(stronger.predict([X_in])) != int(Y_train[X_id]):
                    is_wrong[X_id] = 1
            w_learner_e = sum(is_wrong * w_sample_in) / sum(w_sample_in)

            # compute alpha, if the learner is not qualified, set to 0
            # + 10 ** -8 to avoid numerical error

            w_learn[learn_id] = max(0.0, log(
                1 / (w_learner_e + 10 ** -8) - 1) + log(K - 1))
            alpha_arr = full((M_train,), fill_value=w_learn[learn_id])

            # update entry weights, prediction made by unqualified learner
            # will not update the entry weights

            w_sample_in *= exp(alpha_arr * is_wrong)
            w_sample_in = w_sample_in / sum(w_sample_in)

        # normalize the learner weights
        
        w_learn = w_learn / sum(w_learn)

        # really do the multiclass prediction

        mbc_pred_train, mbc_pred_test = [], []
        train_pools, train_pooled = zeros((M_train, K)), zeros((M_train, K))
        test_pools, test_pooled = zeros((M_test, K)), zeros((M_test, K))

        for idx, learner in enumerate(sort_learners):

            strongest = learner.fit(X_train, Y_train)

            for i in range(X_train.shape[0]):
                pred = [-1 / (K - 1)] * K
                pred[int(strongest.predict([X_train[i]])) - 1] = 1
                train_pooled[i, :] += array([w_learn[idx] * p for p in pred])

            train_pools += train_pooled

            for j in range(X_test.shape[0]):
                pred = [-1 / (K - 1)] * K
                pred[int(strongest.predict([X_test[j]])) - 1] = 1
                test_pooled[j, :] += array([w_learn[idx] * p for p in pred])

            test_pools += test_pooled

        for train_pool in train_pools:
            mbc_pred_train.append(argmax(train_pool) + 1)

        for test_pool in test_pools:
            mbc_pred_test.append(argmax(test_pool) + 1)

        mbc_train_score = sum([int(p == a) for (p, a) in zip(
            mbc_pred_train, Y_train)]) / M_train
        mbc_test_score = sum([int(p == a) for (p, a) in zip(
            mbc_pred_test, Y_test)]) / M_test

        train_mbc_e = (exp(-(K - 1) ** -1) * mbc_train_score +
                       exp((K - 1) ** -2) * (1 - mbc_train_score))
        test_mbc_e = (exp(-(K - 1) ** -1) * mbc_test_score +
                      exp((K - 1) ** -2) * (1 - mbc_test_score))

        train_e_mbc.append(train_mbc_e)
        test_e_mbc.append(test_mbc_e)

        print("{} boosting iteration done...".format(int(l)))

        # The standard gradient boosting by sklearn
        # after the power transform and standardization
        # on the distributions of those RI & chemical weights

        gbc = Pipeline([('power', PT()), ('scaler', SS()), ('model', GBC(
            random_state=0, n_estimators=int(l)))]).fit(X_train, Y_train)

        # print([x == y for (x, y) in zip(gbc.predict(X_test), Y_test)])
        # print(gbc.score(X_train, Y_train))
        # print(gbc.score(X_test, Y_test))

        # gbc.score for GBC already counts the portion of correct classification
        # so no need divide the count by total sample size in the train/test sets

        train_gbc_e = (exp(-(K - 1) ** -1) * gbc.score(X_train, Y_train) +
                       exp((K - 1) ** -2) * (1 - gbc.score(X_train, Y_train)))
        test_gbc_e = (exp(-(K - 1) ** -1) * gbc.score(X_test, Y_test) +
                      exp((K - 1) ** -2) * (1 - gbc.score(X_test, Y_test)))

        train_e_gbc.append(train_gbc_e)
        test_e_gbc.append(test_gbc_e)
        #train_acc.append(1 - gbc.score(X_train, Y_train))
        #test_acc.append(1 - gbc.score(X_test, Y_test))

    print("Take {} done!".format(int(h)))

    #train_accs.append(train_acc)
    #test_accs.append(test_acc)
    mbc_train_es.append(train_e_mbc)
    mbc_test_es.append(test_e_mbc)
    gbc_train_es.append(train_e_gbc)
    gbc_test_es.append(test_e_gbc)


fige, [axee, axef] = plt.subplots(figsize=(10, 5), ncols=2, sharey=True)


def liststats(lists):

    for item in lists:
        out = array(item).transpose()
        out_mean, out_sd = [mean(s) for s in out], [stdev(s) for s in out]
        yield [out_mean, out_sd]


#[accs_train_mean, accs_train_sd], [accs_test_mean, accs_test_sd], [
#    expe_train_mean, expe_train_sd], [expe_test_mean, expe_test_sd] = liststats(
#    [train_accs, test_accs, train_exps, test_exps])
[mbc_train_mean, mbc_train_sd], [mbc_test_mean, mbc_test_sd], [
    gbc_train_mean, gbc_train_sd], [gbc_test_mean, gbc_test_sd] = liststats(
    [mbc_train_es, mbc_test_es, gbc_train_es, gbc_test_es])

# Looks good but not useful now

#accs_df = DF({"Boosting iterations": iter_boost,
#              "Train mean": accs_train_mean, "Test mean": accs_test_mean,
#              "Train sd": accs_train_sd, "Test sd": accs_test_sd})

#expe_df = DF({"Boosting iterations": iter_boost,
#              "Train mean": expe_train_mean, "Test mean": expe_test_mean,
#              "Train sd": expe_train_sd, "Test sd": expe_test_sd})

#figg, [axeg, axeh] = plt.subplots(figsize=(10, 5), ncols=2, sharex=True)


def setdfthenplot(y_mean, y_sd, axe):

    global iter_boost

    thedf = DF({"Boosting iterations": iter_boost, "Accuracy": y_mean})
    slplot(data=thedf.astype({'Boosting iterations': 'str'}),
           x=thedf["Boosting iterations"], y=thedf["Accuracy"], ax=axe)
    axe.fill_between(iter_boost, y1=[
        max(0, mu - sigma) for (mu, sigma) in zip(y_mean, y_sd)], y2=[
        mu + sigma for (mu, sigma) in zip(y_mean, y_sd)], alpha=.3)
    axe.set(xlabel=None, ylabel=None)


setdfthenplot(gbc_train_mean, gbc_train_sd, axee)
setdfthenplot(gbc_test_mean, gbc_test_sd, axee)
setdfthenplot(mbc_train_mean, mbc_train_sd, axef)
setdfthenplot(mbc_test_mean, mbc_test_sd, axef)
#setdfthenplot(accs_train_mean, accs_train_sd, axef)
#setdfthenplot(accs_test_mean, accs_test_sd, axef)

for ax in [axee, axef]:
    ax.legend(handles=ax.lines[:], labels=["Train", "Test"])
axee.set_title("Gradient Boosting Classifier from sklearn", **a_font)
axef.set_title("DIY Multiclass Boosting Classifier", **a_font)

fige.supxlabel("Boosting Iterations", **a_font)
fige.supylabel("Incorrectness or exponential error", **a_font)

end = time.perf_counter()
print("This part takes {:.2f} seconds!".format(end - start))

plt.tight_layout()
#fige.subplots_adjust(bottom=0.12)
plt.show()