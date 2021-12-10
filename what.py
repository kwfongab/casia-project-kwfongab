# X4, B4 and Z4 are from the synthetic model I made
# X4 has size (n x p)
# you decide the value of n and p
# you define what alpha (lambda) values to try
for a in lasso_a:
    # the main dish -- minimize on both U and V
    # not sure how to lambda on u, v so I manually input all the entries here
    fun_uv = lambda b: sum([z ** 2 for z in Z4 - X4 @ array(
        [u[0] * v[0], u[1] * v[1], etc])]) / n + a * sum([
        bi ** 2 for bi in b])
    cons_uv = ({'type': 'eq', 'fun': lambda b: B4 - array([
        u[0] * v[0], u[1] * v[1], etc])})
    # but rmb to convert u[...] and v[...] to b[...]
    res_uv = minimize(fun_uv, # (2p-tuple here;
                      # you decide how to align the entries of u and v),
                      constraints=cons_uv, method='Nelder-Mead',
                      options={"maxiter": 10000, 'xatol': 10 ** -15})
