from numpy import eye, ones
from numpy.random import default_rng, normal
from sklearn.linear_model import LinearRegression as LR
from statistics import mean
from scipy.stats import sem
import time
from datetime import datetime
from the_monte_carlo import workflow

# Section 3.3

# print when it starts
print(datetime.now().time())

p, A, max_iter = 2, 10 ** 4, 10 ** 5
X, B = A * eye(p), ones((p, ))

start = time.perf_counter()

[dfs_gsr, eps_gsr] = workflow(X, B, LR(fit_intercept=False), max_iter,
                              max_ps=1, sige=1, gsr=True, fsr=False, ntf=None)

print("The (Monte Carlo) mean of max(e1, e2) is {:.6f}, and the (Monte Carlo) "
      "mean and SE of the DoF is {:.3f} and {:.3f} respectively.".format(
        mean(eps_gsr[:, 0]), mean(dfs_gsr[:, 0]), sem(dfs_gsr[:, 0], ddof=0)))

end = time.perf_counter()
print("{:.2f}s elapsed for Section 3.3.".format(end - start))
