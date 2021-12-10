from numpy import (cov, eye, ones, zeros, where, array,
                   inf, linspace as lins, empty, trace, meshgrid)
from numpy.random import default_rng, normal, multivariate_normal
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split as TTS
from itertools import chain, combinations
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import cross_val_score as CVS
from sklearn.datasets import make_regression

rnge = default_rng(42)
sige = 1
vare = sige ** 2

# Y = XB + E

n = 100
tts_ratio = 0.2

X_in = ones((n, 1))
x_means, y_means = lins(-10, 10, 101), lins(-10, 10, 101)


def best_subset(estimator, X, Y, max_size=1, cv=5):

    n_features = X.shape[1]
    subsets = (combinations(range(n_features), k + 1)
               for k in range(min(n_features, max_size)))

    best_size_subset = []
    for subsets_k in subsets:  # for each list of subsets of the same size
        best_score, best_subset = -inf, None
        for subset in subsets_k:  # for each subset
            ES = estimator.fit(X[:, list(subset)], Y)
            # get the subset with the best score among subsets of the same size
            score = ES.score(X[:, list(subset)], Y)
            if score > best_score:
                best_score, best_subset = score, subset
        # to compare subsets of different sizes we must use CV
        # first store the best subset of each size
        best_size_subset.append(best_subset)

    # compare best subsets of each size
    best_score, best_subset = -inf, None
    list_scores = []
    for subset in best_size_subset:
        score = CVS(estimator, X[:, list(subset)], Y, cv=cv).mean()
        list_scores.append(score)
        if score > best_score:
            best_score, best_subset = score, subset

    return best_subset, best_score, best_size_subset, list_scores


df_tests = empty((len(x_means), len(y_means)))
X_ind = lins(0, int(X_in.shape[0] - 1), int(X_in.shape[0]))

print("Start")

for y_mean in y_means:

    print(y_mean)

    for x_mean in x_means:

        Y_in = rnge.multivariate_normal([x_mean, y_mean], eye(2), n)

        X_train_ind, X_test_ind, Y_train, Y_test = TTS(
            X_ind, Y_in, test_size=tts_ratio, random_state=42)

        X_train = empty((int((1 - tts_ratio) * n), 2))
        for ind in X_train_ind:
            X_train[where(X_train_ind == ind), :] = X_in[int(ind), :]

        X_test = empty((int(tts_ratio * n), 2))
        for ind in X_test_ind:
            X_test[where(X_test_ind == ind), :] = X_in[int(ind), :]

        hi1, _, _, _ = best_subset(LR(fit_intercept=False), X_train, Y_train)

        X_train_sel = X_train.copy()
        X_test_sel = X_test.copy()
        for i in range(int(X_train.shape[1])):
            if i not in hi1:
                X_train_sel[:, i] = zeros((int(X_train.shape[0]),))
                X_test_sel[:, i] = zeros((int(X_test.shape[0]),))

        lr_sel = LR(fit_intercept=False).fit(X_train_sel, Y_train)
        Y_pred_lr = lr_sel.predict(X_test_sel)

        df_test = 0
        for j in range(int(X_test.shape[0])):
            df_test += cov(Y_pred_lr[j], Y_test[j])[0][1] / vare

        df_tests[where(x_means == x_mean), where(y_means == y_mean)] = df_test

from matplotlib import pyplot as plt

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data
plot_X, plot_Y = meshgrid(x_means, y_means)
Z = df_tests

# Plot the surface.
surf = ax.plot_surface(plot_X, plot_Y, Z,
                       linewidth=0, antialiased=False)

#ax._axis3don = False

plt.show()
