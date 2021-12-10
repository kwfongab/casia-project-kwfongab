from numpy import inf
from sklearn.model_selection import cross_val_score as CVS
from itertools import combinations


def best_subset(est, X, Y, max_size, cvs=5, ofall=False):

    """
    est: the (sklearn) estimator with .fit() and .score() methods
    X: the design matrix, should be a numpy array
    Y: the outcome, should be a numpy array
    max_size: upper bound, should be a natural number
    cvs: cross validation operator, makes sense only when ofall=True
    ofall: find the best subset of all subset size
    """

    n_features = X.shape[1]
    subsets = (combinations(range(n_features), k + 1)
               for k in range(min(n_features, max_size)))

    best_size_subset = []
    for subsets_k in subsets:  # for each list of subsets of the same size
        best_score, subset_best = -inf, None
        for subset in subsets_k:  # for each subset
            ES = est.fit(X[:, list(subset)], Y)
            # get the one with the best score among subsets of the same size
            score = ES.score(X[:, list(subset)], Y)
            # this if faster than collect everything then get the argmax
            if score > best_score:
                best_score, subset_best = score, subset
        # to compare subsets of different sizes we must use CV
        # first store the best subset of each size
        best_size_subset.append(subset_best)

    if ofall:
        # compare best subsets of each size
        high_score, best_sub = -inf, None
        list_scores = []
        for subset in best_size_subset:
            ano_score = CVS(est, X[:, list(subset)], Y, cv=cvs).mean()
            list_scores.append(ano_score)
            if ano_score > high_score:
                high_score, best_sub = ano_score, subset
        return best_sub, high_score, best_size_subset, list_scores
    else:
        return best_size_subset
