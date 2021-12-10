from numpy import cov, eye, where, array, meshgrid, linspace as lins, empty
from sklearn.linear_model import LinearRegression as LR
from statistics import mean
import time
from datetime import datetime
from the_monte_carlo import workflow
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap as LC, BoundaryNorm as BN
from matplotlib.colorbar import ColorbarBase

# Figure 1 of the manuscript
# The idea behind this figure is conveyed through other figures

# print when it starts
print(datetime.now().time())

x1_means, x2_means = lins(-10, 10, 101), lins(-10, 10, 101)
dfs_all = empty((len(x1_means), len(x2_means)))

start = time.perf_counter()

for x2_mean in x2_means:

    for x1_mean in x1_means:

        [dfs_gsr, _] = workflow(
            eye(2), array([x1_mean, x2_mean]), LR(fit_intercept=False),
            max_iter=1000, max_ps=1, sige=1, gsr=True, fsr=False, ntf=None)

        dfs_all[where(x1_means == x1_mean), where(
            x2_means == x2_mean)] = mean(dfs_gsr[:, 0])


figa, axea = plt.subplots(figsize=(6, 5))

# Make data
plot_X, plot_Y = meshgrid(x1_means, x2_means)

# set color map
cmap = LC([(1., (1 - i / 5), 0.) for i in range(5)])
cmap.set_over((1., 0., 0.))
cmap.set_under((1., 1., 1.))
bounds = [1, 2, 3, 4, 5, 6, 7]
norm = BN(bounds[1:-1], cmap.N)

# Plot the surface.
surf = axea.contourf(plot_X, plot_Y, dfs_all, cmap=cmap, norm=norm,
                     levels=bounds, extend='max')
axea.axhline(y=0, color='k', linestyle='--')
axea.axvline(x=0, color='k', linestyle='--')
axeb = plt.colorbar(surf, ax=axea)

end = time.perf_counter()
print("{}s elapsed before showing the figure 3.1".format(end - start))

plt.tight_layout()
plt.show()
