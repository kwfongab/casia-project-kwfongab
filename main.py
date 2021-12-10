from numpy import linspace as lins
from numpy.random import normal, default_rng
from numpy.linalg import pinv
from sklearn.model_selection import train_test_split as TTS
import time

start = time.perf_counter()

# workflow
# going to repeat for 200 times
m = 50
tr = 0.4 # test to total ratio

def workflow(repeats=m, test_ratio=tr):

    n = 20  # sample size
    ps = lins(1, 2 * n, 2 * n)  # endpoint=True

    mu_x, mu_b, mu_e, sigma_x, sigma_b, sigma_e = 0, 0.2, 0, 0.2, 0, 1
    var_x, var_b, var_e = sigma_x ** 2, sigma_b ** 2, sigma_e ** 2

    bias_train, vars_train, bias_test, vars_test =\
        [0] * len(ps), [0] * len(ps), [0] * len(ps), [0] * len(ps)

    for p in ps:

        gamma = p / n
        train_bias, train_vars, test_bias, test_vars = [], [], [], []

        for i in range(repeats):

            rng = default_rng(i+1)
            X = rng.normal(mu_x, var_x, (n, int(p)))  # create n * p matrix X
            B = rng.normal(mu_b, var_b, (int(p), 1))
            R = rng.normal(mu_e, var_e, (n, 1))

            # Underlying model
            Y = X @ B + R  # matrix product

            # Train test split
            X_train, X_test, Y_train, Y_test = TTS(
                X, Y, test_size = tr, random_state = 42)

            # Estimate the parameter beta_i in B
            if gamma <= 1:  # i.e. n >= p
                B_pred = pinv(X_train.transpose() @ X_train) @\
                         X_train.transpose() @ Y_train
            else:           # i.e. n < p
                B_pred = X_train.transpose() @ pinv(X_train @ X_train.transpose()) @ Y_train

            Y_train_pred = X_train @ B_pred
            Y_test_pred = X_test @ B_pred

            train_bias.append((sum((Y_train_pred - Y_train) ** 2) /
                               ((1 - tr) * n))[0])
            train_vars.append((sum((Y_train_pred - sum(Y_train_pred) /
                                    repeats) ** 2) / ((1 - tr) * n))[0])
            test_bias.append((sum((Y_test_pred - Y_test) ** 2) / (tr * n))[0])
            test_vars.append((sum((Y_test_pred - sum(Y_test_pred) /
                                    repeats) ** 2) / (tr * n))[0])

        # for each p
        # bias = expected value of difference btw predicted & actual values
        bias_train[int(p)-1] = sum(train_bias) / (repeats)
        bias_test[int(p)-1] = sum(test_bias) / (repeats)
        vars_train[int(p)-1] = sum(train_vars) / (repeats)
        vars_test[int(p)-1] = sum(test_vars) / (repeats)


    return bias_train, bias_test, vars_train, vars_test

# test if it works
bias_train, bias_test, vars_train, vars_test = workflow()

end = time.perf_counter()

print("The workflow takes {} seconds!".format(end - start))

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

theps = lins(1, 40, 40)/20

figa, axea = plt.subplots()
axea.set_xlabel('p/n')
axea.set_ylabel('Bias squared or Variance')
axea.plot(theps, bias_train, label="Train bias sq")
axea.plot(theps, bias_test, label="Test bias sq")
axea.plot(theps, vars_train, label="Train vars")
axea.plot(theps, vars_test, label="Test vars")
axea.plot([1 - tr, 1 - tr], [0, max(
    max(bias_train), max(bias_test), max(vars_train), max(vars_test))],
          'k--', label="Train to all ratio")
axea.legend()
plt.xlim(min(theps), max(theps))
plt.ylim(0, 10)
plt.show()