# import libraries
import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import scipy.stats as stats
#import numdifftools as ndt
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from matplotlib import pyplot as plt

from numpy.random import normal, default_rng
from numpy import ones, zeros, identity as I

# set the random seed
rng = default_rng(42)

# generate data
N = 100
x = np.linspace(0, 20, N) # 1 D though
e = rng.normal(loc = 0.0, scale = 5.0, size = N)
y = 3*x + e
df = pd.DataFrame({'y':y, 'x':x})
df['constant'] = 1

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# plot
sns.regplot(df.x, df.y)
plt.show()

def workflow_3a():

    mu_x, sigma_x, mu_b, sigma_b, mu_e, sigma_e = 0, 1, 0, 1, 0, 5
    var_x, var_b, var_e = sigma_x ** 2, sigma_b ** 2, sigma_e ** 2

    bias_train, vars_train, bias_test, vars_test = [0], [0], [0], [0]

    mse_valids_r, mean_vmse_r = [], [0]*len(alphas)
    mse_tests_r, mean_tmse_r = [], [0]*len(alphas)

    rng = default_rng(3000)

    # W_u = random effect by school             # to be filled up
    # W_n = random effect by subject            # to be filled up
    # u follows N_30(zeros(30, 1), var_u * I)   # not know
    # n follows N_15(zeros(15, 1), var_n * I)   # not know
    # e follows N_450(zeros(450, 1), var_e * I) # not know

    _1 = ones((30 * 15, 1)) # 450 school-subject combinations bruh
    W_u = zeros((30 * 15, 30)) # create 450*30 empty matrice first
    W_n = zeros((30 * 15, 15))  # create 450*15 empty matrice first

    X = rng.normal(mu_x, var_x, (n, 5))  # create n * p matrix X
    B = rng.normal(mu_b, var_b, (5, 1))  # Beta
    R = rng.normal(mu_e, var_e, (n, 1))  # eRRoR

    # Underlying model
    Y = X @ B + R  # matrix product

    return Y

Y = workflow_3a()

print(Y)
