from numpy.random import normal, default_rng
from numpy.linalg import pinv, norm
import time
from math import sqrt
from numpy import identity as I
from numpy import empty

def q2b(repeats):

    mu_x = [0, 0]
    cov_x = [[1, -0.5], [-0.5, 1]]  # diagonal covariance
    # Diagonal covariance means that points are oriented along x or y-axis:

    mu_e = [0, 0]
    cov_e = [[1, 0], [0, 1]]
    mu_f = [0, 0]
    cov_f = [[0.5 ** 2, 0], [0, 0.5 ** 2]]

    mse_mle_ns, mse_mle_ys = empty((repeats, 2)), empty((repeats, 2))
    mse_js_ns, mse_js_ys = empty((repeats, 2)), empty((repeats, 2))

    for i in range(repeats):
        rng = default_rng(int(99*(i+1)))

        # the distribution x, with mean mu_x
        x = rng.multivariate_normal(mu_x, cov_x, n)

        # 2 different residue config
        y = x + rng.multivariate_normal(mu_e, cov_e, n)
        z = y + rng.multivariate_normal(mu_f, cov_f, n)

        # and partition for graphing
        y1, y2, z1, z2 = y[:, 0], y[:, 1], z[:, 0], z[:, 1]

        # if the model is misspecified
        # MLE estimate & the MSE by MLE
        mu_mle_n = z[:, :]
        mse_mle_n = sum((mu_mle_n - x) ** 2) / n
        # JS estimate & the MSE by JS
        # how to find norm(z)? z.transpose() @ z should be symmetric
        mu_js_n = (1 - (n - 2) / norm(z[:, :].transpose() @ z[:, :])) * z[:, :]
        mse_js_n = sum((mu_js_n - x) ** 2) / n

        # if the model is specified
        # MLE estimate & the MSE by MLE
        mu_mle_y = y[:, :]
        mse_mle_y = sum((mu_mle_y - x) ** 2) / n
        # JS estimate & the MSE by JS
        # how to find norm(z)? z.transpose() @ z should be symmetric
        mu_js_y = (1 - (n - 2) / norm(y[:, :].transpose() @ y[:, :])) * y[:, :]
        mse_js_y = sum((mu_js_y - x)**2)/n

        mse_js_ns[i, :] = mse_js_n
        mse_js_ys[i, :] = mse_js_y
        mse_mle_ns[i, :] = mse_mle_n
        mse_mle_ys[i, :] = mse_mle_y


    return y1, y2, z1, z2,\
           mse_mle_ns.transpose(), mse_js_ns.transpose(),\
           mse_mle_ys.transpose(), mse_js_ys.transpose()


start = time.perf_counter()

n = 200
times = 100

y1, y2, z1, z2, msemns, msejns, msemys, msejys = q2b(repeats=times)

end = time.perf_counter()

print("The workflow takes {} seconds!".format(end - start))

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame as df

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

fige, [axee, axef] = plt.subplots(nrows=1, ncols=2, figsize=(8.5, 4),
                                  sharex=True, sharey=True)
sns.scatterplot(x=y1, y=y2, marker='x', ax=axee)
sns.scatterplot(x=z1, y=z2, marker='x', ax=axef)
axee.set_title("Group A")
axef.set_title("Group B")

mlena, mlenb = msemns[0, :].tolist(), msemns[1, :].tolist()
jsna, jsnb = msejns[0, :].tolist(), msejns[1, :].tolist()
mleya, mleyb = msemys[0, :].tolist(), msemys[1, :].tolist()
jsya, jsyb = msejys[0, :].tolist(), msejys[1, :].tolist()

my_dict_n = {'MLE, A': mlena, 'MLE, B': mlenb, 'JS, A': jsna, 'JS, B': jsnb}
my_dict_y = {'MLE, A': mleya, 'MLE, B': mleyb, 'JS, A': jsya, 'JS, B': jsyb}

datan = df(my_dict_n)
datay = df(my_dict_y)

flierprops = dict(marker='o')

figg, [axeg, axeh] = plt.subplots(nrows=1, ncols=2, figsize=(11, 5),
                                  sharey=True)
sns.boxplot(data=datan, ax=axeg, flierprops=flierprops)
sns.boxplot(data=datay, ax=axeh, flierprops=flierprops)
axeg.set_title("Misspec by adding irrelavent variables")
axeh.set_title("Correct model spec")

eff_an, eff_bn = sum(mlena)/sum(jsna), sum(mlenb)/sum(jsnb)
eff_ay, eff_by = sum(mleya)/sum(jsya), sum(mleyb)/sum(jsyb)
list_n, list_y = sum(msemns)/sum(msejns), sum(msemys)/sum(msejys)
eff_n, eff_y = sum(list_n)/len(list_n), sum(list_y)/len(list_y)

print(eff_an, eff_ay, eff_bn, eff_by, eff_n, eff_y)

print("The MSE ratio of MLE to JSE when the model is misspecified "
      "and the model is correct are "
      "{:.3f} and {:.3f} respectively.".format(eff_n, eff_y))

print("By focusing on group A, the above ratios become "
      "{:.3f} and {:.3f} respectively.".format(eff_an, eff_ay))

print("By focusing on group B, the above ratios become "
      "{:.3f} and {:.3f} respectively.".format(eff_bn, eff_by))

plt.show()
