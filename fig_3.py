from numpy import ones, linspace as lins, arange
from numpy.random import default_rng, normal
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import StandardScaler as SS
from statistics import mean
from scipy.stats import sem
from matplotlib import pyplot as plt
import time
from datetime import datetime
from the_monte_carlo import workflow

# Figure 3, section 3.1

# print when it starts
print(datetime.now().time())

# looks quite close to the plots the authors have
rngx = default_rng(1)
sige = 1
vare = sige ** 2
# yes 3000 is smaller than 5000 but time doesn't permit
n, p, snr, max_iter = 50, 15, 7, 3000

X = (SS().fit_transform(rngx.normal(0, sige, (n * p, 1)))[:, 0]).reshape(n, p)
# after the standard scaler X should have mean 0, population SD 1
B = snr * ones((p, ))

start = time.perf_counter()

[dfs_gsr, _, dfs_fsr] = workflow(X, B, LR(fit_intercept=False), max_iter,
                                 max_ps=p, sige=1, gsr=True, fsr=True, ntf=100)

me_gss, se_gss, me_fss, se_fss = [], [], [], []
for i in range(p):
    me_gss.append(mean(dfs_gsr[:, i]))
    se_gss.append(2 * sem(dfs_gsr[:, i], ddof=0))
    me_fss.append(mean(dfs_fsr[:, i]))
    se_fss.append(2 * sem(dfs_fsr[:, i], ddof=0))

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

figa, [axea, axeb] = plt.subplots(figsize=(6, 6), ncols=2, sharey=True)

a_font = {'fontname': 'Arial', 'fontsize': 12}
subset_size = lins(1, p, p)

axea.errorbar(x=subset_size, y=me_gss, yerr=se_gss, capsize=5, marker="o")
axea.set_title("Best Subset\n(Linear) Regression", **a_font)
axeb.errorbar(x=subset_size, y=me_fss, yerr=se_fss, capsize=5, marker="o")
axeb.set_title("Forward Stepwise\n(Linear) Regression", **a_font)

for ax in [axea, axeb]:
    ax.plot(subset_size, [p] * p, 'k--')
    ax.plot(subset_size, subset_size, 'k--')
    ax.set_xticks(arange(1, p + 1, 2))
    ax.set_xlabel("Subset size", **a_font)

axea.set_ylabel("DoF", **a_font)
plt.tight_layout()

end = time.perf_counter()
print("{}s elapsed before showing the figure 3.1".format(end - start))

plt.savefig('Section-3A-BSR-FSR.eps', format='eps', bbox_inches='tight')
