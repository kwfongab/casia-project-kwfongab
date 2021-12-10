from numpy import array, where, empty, mean, log, pi, zeros, sqrt
from numpy.random import default_rng, normal

# set up everything to get the M and Z
a = 2
# mu_i ~ N(0, sigma^2 = a^(-1))
# so a here = 2
n = 50
rngm = default_rng(42)
M = rngm.normal(0, sqrt(1 / a), n)    # (mean, sd = 1/sqrt(2), sample size)
Z = empty(n, )
for m in M:
    m_in = where(M == m)[0][0]
    rngz = default_rng(69*(m_in + 1))
    Z[m_in, ] = rngz.normal(m, 1, 1)

# guess sigma^2 = 1/(alpha) = 0.5 first, i.e. guess alpha = 1.25
var_old = 0.8
beta_old = var_old / (1 + var_old)
M_old = Z * beta_old
const = n * log(2 * pi) / 2
ilL_old = -sum([(z - u) ** 2 for (z, u) in zip(Z, M_old)]) / 2 - const
print("Initial incomplete log likelihood")
print(ilL_old)

while True:

    print("Old sigma^2 = old 1/alpha")
    print(var_old)

    # E-step (?)
    beta_old = var_old / (1 + var_old)
    M_old = Z * beta_old

    # M-step
    var_new = (beta_old * n + beta_old ** 2 * sum([z ** 2 for z in Z])) / n
    print("New sigma^2 = new 1/alpha")
    print(var_new)

    # after that get the variable part of the new incomplete log likelihood
    beta_new = var_new / (1 + var_new)
    M_new = Z * beta_new
    ilL_new = -sum([(z - u) ** 2 for (z, u) in zip(Z, M_new)]) / 2 - const

    print("New incomplete log likelihood")
    print(ilL_new)
    #input("wait")

    if abs(ilL_new - ilL_old) < 10 ** -8:
        var_est = var_new
        break

    # reassign the value to alpha
    var_old, ilL_old, beta_old = var_new, ilL_new, beta_new

# See the final EM-fitting result
print("The estimated value of a is {}".format(1/var_est))

# eventually after the EM loop is completed
M_EM = Z * var_est / (1 + var_est)
M_MLE = Z.copy()
M_JS = (1 - (Z.shape[0] - 2)/(Z.transpose() @ Z)) * Z

print("The SSE by EM is {}".format(sum([(p - t) ** 2 for (p, t) in zip(M, M_EM)])))
print("The SSE by MLE is {}".format(sum([(p - t) ** 2 for (p, t) in zip(M, M_MLE)])))
print("The SSE by JS is {}".format(sum([(p - t) ** 2 for (p, t) in zip(M, M_JS)])))
