import numpy as np
import matplotlib.pyplot as plt

T = 5777

chi_samples = np.linspace(0, 5, 100)
delta_T_samples = np.linspace(0, 250, 99)

chi, delta_T = np.meshgrid(chi_samples, delta_T_samples)

delta_log_A = 5040 * chi / (T - delta_T) - 5040 * chi / T

fig, ax = plt.subplots()
image = ax.imshow(
    delta_log_A, 
    extent=(*chi_samples[[0, -1]], *delta_T_samples[[0, -1]]),
    origin='lower', 
    aspect='auto'
)
ax.set_xlabel("$\chi$ [eV]")
ax.set_ylabel("$\Delta{}T_\mathrm{eff}$ [K]")
cbar = plt.colorbar(image)
cbar.set_label("$\Delta\log{A}$ [dex]")
fig.savefig("bad_stellar_parameters.png", dpi=300)