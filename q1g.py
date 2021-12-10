from numpy import array, where, empty, mean, log, pi, zeros, sqrt, exp
from numpy.random import default_rng, normal

# set up everything to get the M and Z
a = 2
# mu_i ~ N(0, sigma^2 = a^(-1))
# so a here = 2
n = 100
rngm = default_rng(42)
M = rngm.normal(0, sqrt(1 / a), n)    # (mean, sd = 1/sqrt(2), sample size)
Z = empty(n, )
for m in M:
    m_in = where(M == m)[0][0]
    rngz = default_rng(69*(m_in + 1))
    Z[m_in, ] = rngz.normal(m, 1, 1)


Z_norm_sq = sum([z ** 2 for z in Z])
# guess sigma^2 = 1/(alpha) = 0.5 first, i.e. guess alpha = 1.25
a_old = 1.25
#print(-Z_norm_sq / (2 * (1 + 1 / a_old)))
#print(n * log(2 * pi * (1 + 1 / a_old)) / 2)
#input("wait")
ilL_old = -Z_norm_sq / (2 * (1 + 1 / a_old)) - n * log(
    2 * pi * (1 + 1 / a_old)) / 2
#print("Initial incomplete log likelihood")
#print(ilL_old)

a_list, ilL_list = [a_old], [ilL_old]

while True:

    # M-step
    a_new = (1 + a_old) ** 2 / (1 + a_old + Z_norm_sq / n)

    # after that get the variable part of the new incomplete log likelihood
    ilL_new = -Z_norm_sq / (2 * (1 + 1 / a_new)) - n * log(
        1 + 1 / a_new) / 2 - n * log(2 * pi) / 2

    a_list.append(a_new)
    ilL_list.append(ilL_new)

    if ilL_new - ilL_old < 10 ** -15:
        a_est = a_new
        break

    # reassign the value to alpha
    a_old, ilL_old = a_new, ilL_new

# See the final EM-fitting result
print("The estimated value of a is {}".format(a_est))

print(len(a_list))

# eventually after the EM loop is completed
M_EM = Z / (1 + a_est)
M_MLE = Z.copy()
M_JS = (1 - (Z.shape[0] - 2)/(Z.transpose() @ Z)) * Z

print("The SSE by EM is {}".format(sum([(p - t) ** 2 for (p, t) in zip(M, M_EM)])))
print("The SSE by MLE is {}".format(sum([(p - t) ** 2 for (p, t) in zip(M, M_MLE)])))
print("The SSE by JS is {}".format(sum([(p - t) ** 2 for (p, t) in zip(M, M_JS)])))

from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

figa, [axea, axeb] = plt.subplots(figsize=(8, 4), ncols=2)
axea.plot(ilL_list)
axea.set_title("Incomplete log-likelihood", fontsize=12)
axeb.plot(a_list)
axeb.set_title("The alpha = 1/(sigma^2)", fontsize=12)
figc, axec = plt.subplots(figsize=(5, 5))
# plot the generated (mu_i)s
count, bins, ignored = axec.hist(M, 25, density=True)
axec.plot(bins, 1/(sqrt(2 * pi / a_est)) * exp(
    -(bins - mean(Z))**2 / (2 / a_est)),
          linewidth=2, color='r')
axec.set_title(("The distribution of all the " + r"$\mu_i$"), fontsize=12)
plt.show()
