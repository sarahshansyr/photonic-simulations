import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
from tidy3d import web

# ---- frequencies ----
lda0 = 1.55
ldas = np.linspace(1.50, 1.60, 1001)          # dense grid: resonance is narrow
freq0 = td.C_0 / lda0
freqs = td.C_0 / ldas
fwidth = 0.5 * (np.max(freqs) - np.min(freqs))

# ---- materials ----
n_diamond = 2.41
diamond = td.Medium(permittivity=n_diamond**2)
air = td.Medium(permittivity=1.0)

# ---- geometry parameters ----
w = 0.5            # waveguide width (um)
h_si = 0.22        # thickness
l = 12             # total waveguide length
slot_w = 0.08      # air-slot mirror width (um)  -> sets mirror reflectivity
L_cav = 7       # mirror separation; resonance at m*lam/(2*neff), m=10, neff~1.56

# ---- simulation domain ----
Lx = l + 4 * lda0
Ly = 2 * w + lda0
Lz = 9 * h_si

# waveguide (later structures override earlier ones, so the air slots carve it)
waveguide = td.Structure(
    geometry=td.Box(center=(0, 0, 0), size=(td.inf, w, h_si)),
    medium=diamond,
)
# two air-slot mirrors at x = +/- L_cav/2
slots = [
    td.Structure(
        geometry=td.Box(center=(xs, 0, 0), size=(slot_w, w, h_si)),
        medium=air,
    )
    for xs in (-L_cav / 2, +L_cav / 2)
]

# ---- mode specs ----
mode_spec = td.ModeSpec(num_modes=1, target_neff=n_diamond, filter_pol="te")

# ---- source ----
mode_source = td.ModeSource(
    center=(-l / 2 - lda0 / 4, 0, 0),
    size=(0, 6 * w, 6 * h_si),
    source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    direction="+",
    mode_spec=mode_spec,
    mode_index=0,
)

# ---- monitors ----
mode_monitor_reflection = td.ModeMonitor(
    center=(-l / 2 - lda0 / 2, 0, 0),
    size=mode_source.size,
    freqs=freqs,
    mode_spec=mode_spec,
    name="reflection",
)
mode_monitor_through = td.ModeMonitor(
    center=(l / 2 + lda0 / 4, 0, 0),
    size=mode_source.size,
    freqs=freqs,
    mode_spec=mode_spec,
    name="through",
)

mode_monitor_cavity = td.ModeMonitor(
    center=(-L_cav / 2, 0, 0),
    size=mode_source.size,
    freqs=freqs,
    mode_spec=mode_spec,
    name="cavity",
)

field_monitor = td.FieldMonitor(
    center=(0, 0, 0), size=(td.inf, td.inf, 0),
    freqs=[freq0], name="field",
)

run_time = 5e-12   # must exceed the cavity ringdown time tau = Q/omega0

sim = td.Simulation(
    center=(0, 0, 0),
    size=(Lx, Ly, Lz),
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=25, wavelength=lda0),
    structures=[waveguide, *slots],
    sources=[mode_source],
    monitors=[mode_monitor_reflection, mode_monitor_through, mode_monitor_cavity, field_monitor],
    run_time=run_time,
    boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    medium=air,
    symmetry=(0, 0, 1),
)

ax = sim.plot(z=0)
plt.show()

fp_simulation = web.run(
    simulation=sim,
    task_name="fabry_perot_cavity3",
    path="data/simulation_fabry_perot3.hdf5",
)

T = np.abs(fp_simulation["through"].amps.sel(direction="+").values) ** 2
R = np.abs(fp_simulation["reflection"].amps.sel(direction="-").values) ** 2

plt.plot(ldas, T, label="T (through)")
plt.plot(ldas, R, label="R (reflection)")
plt.xlabel("Wavelength (um)"); plt.ylabel("Power")
plt.legend(); plt.grid(alpha=0.3)
plt.show()

# resonance: transmission peak (through-coupled FP), reflection dip
i_pk = np.argmax(T)
lam_res = ldas[i_pk]
half = (T.max() + T.min()) / 2
above = np.where(T > half)[0]
fwhm = ldas[above[-1]] - ldas[above[0]]          # linewidth in um
Q = lam_res / fwhm
FSR = lam_res**2 / (2 * 1.56 * L_cav)            # free spectral range (um)
print(f"resonance: {lam_res:.4f} um, T_peak = {T.max():.3f}, "
      f"extinction at dip: {1 - T.min():.3f}, Q ~ {Q:.0f}, FSR ~ {FSR*1e3:.1f} nm")

ax = fp_simulation.plot_field(
    field_monitor_name="field",
    field_name="Ey",
    val="real",
    f=freq0,
)

ax.set_xlim(-L_cav, L_cav)
ax.set_title(r"Real part of the electric field $\mathrm{Re}(E_y)$")
plt.show()

# Get Ey from the field monitor
Ey_data = fp_simulation["field"].Ey

# Select the frequency closest to freq0
Ey_data = Ey_data.sel(f=freq0, method="nearest")

# Select the waveguide centerline y=0
Ey_line = Ey_data.sel(y=0, method="nearest")

# Remove the singleton z dimension, if present
if "z" in Ey_line.dims:
    Ey_line = Ey_line.isel(z=0)

Ey_line = Ey_line.squeeze()

# Coordinates and complex field values
x = Ey_line.coords["x"].values
Ey = Ey_line.values

# Normalize intensity
intensity = np.abs(Ey)**2
intensity /= np.max(intensity)

# Rotate the arbitrary complex phase so the largest field is positive
phase = -np.angle(Ey[np.argmax(np.abs(Ey))])
Ey_real = np.real(Ey * np.exp(1j * phase))
Ey_real /= np.max(np.abs(Ey_real))

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

# Real field
axes[0].plot(x, Ey_real, linewidth=1.5)
axes[0].axhline(0, color="black", linewidth=0.7)
axes[0].set_ylabel(r"$\mathrm{Re}(E_y)$")
axes[0].set_title("Fabry–Pérot standing-wave pattern")
axes[0].grid(alpha=0.3)

# Intensity
axes[1].plot(x, intensity, color="red", linewidth=1.5)
axes[1].set_xlabel("x position (µm)")
axes[1].set_ylabel(r"$|E_y|^2$")
axes[1].set_title("Intracavity intensity")
axes[1].grid(alpha=0.3)

# Mark the two mirror positions
for ax in axes:
    ax.axvline(-L_cav / 2, color="gray", linestyle="--", label="mirror centers")
    ax.axvline(+L_cav / 2, color="gray", linestyle="--")

axes[0].legend()
plt.tight_layout()
plt.show()