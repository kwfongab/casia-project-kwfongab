from numpy import where, empty, mean, log, pi, zeros, sqrt, exp, linspace
from numpy.random import default_rng, normal

# set up everything to get the M and Z
var = 4
a = var ** -1
# mu_i ~ N(0, sigma^2 = a^(-1)), so a here = 1/4
n = 10000
rngm = default_rng(42)
M = rngm.normal(0, sqrt(var), n)    # (mean, sd = 1/sqrt(2), sample size)
Z = empty(n, )
for m in M:
    m_in = where(M == m)[0][0]
    rngz = default_rng(69*(m_in + 1))
    Z[m_in, ] = rngz.normal(m, 1, 1)

Z_norm_sq = sum([z ** 2 for z in Z])
# guess sigma^2 = 1/(alpha) = 0.5 first, i.e. guess alpha = 1.25
var_old = 9
ilL_old = -Z_norm_sq / (2 * (1 + var_old)) - n * log(
    2 * pi * (1 + var_old)) / 2
var_list, ilL_list = [var_old], [ilL_old]

while True:

    # M-step
    b_old = var_old / (1 + var_old)
    var_new = b_old + b_old ** 2 * (Z_norm_sq / n)

    # after that get the new incomplete log likelihood
    ilL_new = -Z_norm_sq / (2 * (1 + var_new)) - n * log(
        1 + var_new) / 2 - n * log(2 * pi) / 2

    var_list.append(var_new)
    ilL_list.append(ilL_new)

    if ilL_new - ilL_old < 10 ** -12:
        var_est = var_new
        break

    # reassign the value to alpha
    var_old, ilL_old = var_new, ilL_new

# See the final EM-fitting result;
# when n = 10000, hat(a) should be close to the actual a
print("The estimated value of a is {}".format(1/var_est))

# eventually after the EM loop is completed
M_EM = Z * var_est / (1 + var_est)
M_MLE = Z.copy()
M_JS = (1 - (Z.shape[0] - 2)/(Z.transpose() @ Z)) * Z

print("The SSE by EM is {}".format(
    sum([(p - t) ** 2 for (p, t) in zip(M, M_EM)])))
print("The SSE by MLE is {}".format(
    sum([(p - t) ** 2 for (p, t) in zip(M, M_MLE)])))
print("The SSE by JS is {}".format(
    sum([(p - t) ** 2 for (p, t) in zip(M, M_JS)])))

from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# plot the incomplete log-likelihood, and the alpha
fige, [axee, axef, axeg] = plt.subplots(figsize=(10.5, 4), ncols=3)
axee.plot(ilL_list)
axee.set_title("Incomplete log-likelihood", fontsize=12)
axef.plot([1 / var for var in var_list])
axef.set_title(r"$\alpha = 1/\sigma^2$", fontsize=12)
repeats = linspace(0, len(var_list) - 1, len(var_list))
#plt.setp([axee], xticks=repeats,
#         xticklabels=[int(i+1) for i in repeats])
axee.set_xlabel("Takes", fontsize=12)
axee.set_xlabel("Takes", fontsize=12)


# Plot all the (mu_i)s
figg, axeg = plt.subplots(figsize=(5, 5))
count, bins, ignored = axeg.hist(M, 100, density=True)
axeg.plot(bins, 1/(sqrt(2 * pi * var_est)) * exp(
    -(bins)**2 / (2 * var_est)), linewidth=2, color='r')
axeg.set_title(("All the " + r"$\mu_i$" +
                r"from $N(0, 1/\alpha), n = {}$".format(n)), fontsize=12)
axeg.set_xlabel(r"$\mu_i$")

plt.show()
