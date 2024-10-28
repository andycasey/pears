import numpy as np
import read_mist_models
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor


isochrones = read_mist_models.ISOCMD("MIST_iso_671ba94bb956f.iso.cmd")

index = np.array(isochrones.ages).searchsorted(9) # 1e9 years
isochrone = isochrones.isocmds[index]

use = (
    # MS to CHeB (inclusive)
    (0 <= isochrone["phase"]) * (isochrone["phase"] <= 3)
)
isochrone = isochrone[use]


fig, ax = plt.subplots()

ax.scatter(
    10**isochrone["log_Teff"], 
    isochrone["log_g"],
    c=isochrone["phase"],
)
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_ylim(ax.get_ylim()[::-1])






np.random.seed(42)

n_lines = 250
n_stars = 100
n_pairs = n_stars // 2

assert n_stars % 2 == 0
σ_line = 0.3 # standard deviation of the per-line systematics wrt teff
μ = 0.03
σ_intrinsic = 0.003
σ_random = 0.001
σ_teff = 100
σ_log = 0.08


star_indices = np.random.choice(len(isochrone), n_stars, replace=False)


kernel = σ_line**2 * RBF(length_scale=3000)
gp = GaussianProcessRegressor(kernel=kernel)


# Compute the per-line systematics
teff_true = 10**isochrone["log_Teff"][star_indices]
logg_true = isochrone["log_g"][star_indices]

teff_sort_indices = np.argsort(teff_true)

y = gp.sample_y(teff_true.reshape(-1, 1), n_lines)

fig, ax = plt.subplots()
for i in range(n_lines):
    ax.plot(teff_true[teff_sort_indices], y[teff_sort_indices, i], lw=1, alpha=0.5, c="k")



# Simulate the per-star abundances
X_true = np.random.normal(μ, σ_intrinsic, size=n_stars).reshape(-1, 1)

X_obs = X_true - y + np.random.normal(0, σ_random, size=(n_stars, n_lines)) * np.random.randn(n_stars, n_lines)

teff_obs = teff_true + np.random.normal(0, σ_teff, size=n_stars)
teff_obs_err = np.abs(np.random.normal(0, σ_teff, size=n_stars))
logg_obs = logg_true + np.random.normal(0, σ_log, size=n_stars)
logg_obs_err = np.abs(np.random.normal(0, σ_log, size=n_stars))

fig, (ax, ax_fe_h) = plt.subplots(1, 2, figsize=(9, 4))
ax.errorbar(
    teff_obs,
    logg_obs,
    xerr=teff_obs_err,
    yerr=logg_obs_err,
    fmt="None",
    zorder=-1,
    lw=0.5,
    c='#666666'
)
scat = ax.scatter(
    teff_obs,
    logg_obs,
    c=np.mean(X_obs, axis=1),
    cmap="copper",
    s=25
)
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_aspect(np.ptp(ax.get_xlim()) / np.ptp(ax.get_ylim()))

ax.set_xlabel(r"$T_\mathrm{eff}$ [K]")
ax.set_ylabel(r"$\log{g}$")
from matplotlib.ticker import MaxNLocator
ax.yaxis.set_major_locator(MaxNLocator(4))
cbar = fig.colorbar(scat)
cbar.set_label("[Fe/H] [dex]")


X_star = np.mean(X_obs, axis=1)
X_star_err = np.std(X_obs, axis=1) / np.sqrt(n_lines)

σ_x_naive = np.std(X_star)


# Pair wise stars 
pair_indices = np.argsort(star_indices).reshape((-1, 2))
# randomize

X_diff = np.diff(X_obs[pair_indices], axis=1)[:, 0] 
# Randomize A-B or B-A
X_diff *= np.sign(np.random.normal(size=n_stars // 2)).reshape((-1, 1))
Y = np.mean(X_diff, axis=1)

σ_x_from_y = np.std(Y) * (1/np.sqrt(2))

print(f"True cluster dispersion: {σ_intrinsic:.4f}")
print(f"Naive dispersion measured by [Fe/H]: {σ_x_naive:.4f} (+/- {σ_x_naive / np.sqrt(n_stars):.4f})")
print(f"Dispersion measured by pair-wise differences: {σ_x_from_y:.4f} (+/- {σ_x_from_y / np.sqrt(n_pairs):.4f})")


naive_color = "red"
pairwise_color = "k"
from scipy import stats

ax_fe_h.axvline(σ_intrinsic, c="tab:blue", label="Truth", lw=2, marker=None, zorder=100)
xi = np.linspace(0, 0.01, 1000)

ax_fe_h.fill_between(
    xi,
    np.zeros_like(xi),
    stats.norm.pdf(xi, σ_x_naive, σ_x_naive / np.sqrt(n_stars)),
    facecolor=naive_color,
    alpha=0.2,
    zorder=1
)
ax_fe_h.plot(
    xi,
    stats.norm.pdf(xi, σ_x_naive, σ_x_naive / np.sqrt(n_stars)),
    c=naive_color,
    zorder=1,
    label=f"Naive ({n_stars} stars)",
)


yi = stats.norm.pdf(xi, σ_x_from_y, σ_x_from_y / np.sqrt(n_pairs))
    
ax_fe_h.fill_between(
    xi,
    np.zeros_like(xi),
    yi,
    facecolor=pairwise_color,
    alpha=0.2,
    zorder=10
)
ax_fe_h.plot(xi, yi, c=pairwise_color, label=f"Pair-wise ({n_pairs} pairs)", zorder=10)


ax_fe_h.set_ylim(0, ax_fe_h.get_ylim()[1])


ax_fe_h.set_xlim(0, 0.01)

ax_fe_h.set_aspect(np.ptp(ax_fe_h.get_xlim()) / np.ptp(ax_fe_h.get_ylim()))

ax_fe_h.legend(loc="upper right", frameon=False)
ax_fe_h.set_yticks([])
for spine in ("top", "right"):
    ax_fe_h.spines[spine].set_visible(False)

ax_fe_h.set_ylabel("Probability density")
ax_fe_h.set_xlabel(f"Cluster homogeneity $\sigma_{{\mathrm{{[Fe/H]}}}}$ [dex]")

for i in range(10):
    fig.tight_layout() # amazing

fig.savefig("simulation.pdf", dpi=300)