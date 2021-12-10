from pandas import read_csv
from scipy.stats import ttest_1samp

data = read_csv('data_asm4.txt', sep="\t", header=None, skiprows=1)
data.columns = ["H", "h", "s_G", "s_g"]
data["var_G"] = data["s_G"] ** 2
data["var_g"] = data["s_g"] ** 2

H, h, var_G, var_g = data["H"], data["h"], data["var_G"], data["var_g"]
s_G, s_g = data["s_G"], data["s_g"]

tau_sq = sum([h_i ** 2 for h_i in h]) / (h.shape[0] - 2) - sum(
    [v for v in var_g]) / h.shape[0]

# the less likelihood estimate of g and G
g_LLE, G_LLE = [tau_sq * h_i / (tau_sq + v) for v, h_i in zip(var_g, h)], H[:]
b_LLE = [G/g for G, g in zip(G_LLE, g_LLE)]

b_equals_0 = ttest_1samp(b_LLE, 0)
print("The 2-sided p-value for the null hypoethesis that b = 0:")
print(b_equals_0.pvalue)

