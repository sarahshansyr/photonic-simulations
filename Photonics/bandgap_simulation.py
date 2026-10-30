# attempt to recreate Kevin Chen's thesis fig 4.1

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import tidy3d as td
from tidy3d import web


# ============================================================
# User settings
# ============================================================

# Geometry from the reference figure
a = 0.201      # lattice period [m]
W = 0.269      # beam width [m]
H = 0.202      # beam thickness [m]
r = 0.058       # air-hole radius [m]

# Assumed core material.
# The reference geometry is commonly associated with diamond/SnV devices.
n_core = 2.40

# Target SnV wavelength shown in the reference figure
lambda_snv = 0.62
f_snv = td.C_0 / lambda_snv

# Frequency range of the plotted band diagram
fmin = 300e12
fmax = 800e12

# Broadband source
f_source = 550e12
fwidth = 500e12

# Simulation duration.
# Longer duration improves frequency resolution but costs more.
run_time = 2.0e-12

# Air padding around the nanobeam in y and z
air_buffer = 5*W

# Use only nine k-points to reduce credits.
# 0.40 is deliberately included as the pilot point.
k_norms = np.linspace(0.30, 0.50, 9)

# The pilot simulation is used before the full sweep.
pilot_k = 0.40
pilot_key = f"k_{pilot_k:.3f}".replace(".", "p")

# Set this to True only after inspecting the pilot spectrum.
RUN_FULL_SWEEP = False

# Output directory
DATA_DIR = "tidy3d_bandstructure_data"
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# Materials and geometry
# ============================================================

core = td.Medium(permittivity=n_core**2)
air = td.Medium(permittivity=1.0)


def make_structures():
    """
    Create one unit cell:
        - dielectric nanobeam
        - cylindrical air hole through the thickness
    """

    beam = td.Structure(
        geometry=td.Box(
            center=(0, 0, 0),
            size=(a, W, H),
        ),
        medium=core,
        name="dielectric_beam",
    )

    hole = td.Structure(
        geometry=td.Cylinder(
            center=(0, 0, 0),
            radius=r,
            length=H + 1e-9,
            axis=2,       # cylinder axis along z
        ),
        medium=air,
        name="air_hole",
    )

    # The hole is listed after the beam so it overrides the dielectric
    # in the overlapping cylindrical region.
    return [beam, hole]


# ============================================================
# Sources and monitors
# ============================================================

def make_sources():
    """
    Use an Ey-oriented point dipole inside the dielectric.

    The hole is centered at x=0. Since r=58 nm and 0.4*a=80.4 nm,
    x=0.4*a lies in the dielectric rather than inside the hole.
    """

    source = td.PointDipole(
        center=(0.40 * a, 0, 0),
        polarization="Ey",
        source_time=td.GaussianPulse(
            freq0=f_source,
            fwidth=fwidth,
        ),
        name="Ey_dipole",
    )

    return [source]


def make_monitors():
    """
    The monitor is placed away from the source to reduce direct-source
    contamination in the spectrum.
    """

    monitor = td.FieldTimeMonitor(
        fields=["Ey"],
        center=(-0.40 * a, 0, 0),
        size=(0, 0, 0),
        start=0,
        name="field_time",
    )

    return [monitor]


# ============================================================
# Simulation construction
# ============================================================

def make_simulation(k_norm):
    """
    Construct the FDTD simulation for a normalized Bloch wavevector

        k_norm = k_x / (2*pi/a).

    Tidy3D's Boundary.bloch() expects this normalized value directly.
    """

    boundary_spec = td.BoundarySpec(
        x=td.Boundary.bloch(k_norm),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    sim = td.Simulation(
        center=(0, 0, 0),
        size=(
            a,
            W + 2 * air_buffer,
            H + 2 * air_buffer,
        ),
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=20,
        ),
        structures=make_structures(),
        sources=make_sources(),
        monitors=make_monitors(),
        boundary_spec=boundary_spec,
        run_time=run_time,
        courant=0.99,
    )

    return sim


# ============================================================
# No-credit local validation and visualization
# ============================================================

pilot_sim = make_simulation(pilot_k)

print("Pilot simulation:")
print(f"  normalized kx = {pilot_k:.3f}")
print(f"  physical kx   = {pilot_k * 2*np.pi/a:.4e} rad/m")
print(f"  target f      = {f_snv/1e12:.2f} THz")

# This validates the simulation object locally.
# It does not submit a cloud job.
pilot_sim.validate_pre_upload()

# Plot geometry before using any credits.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

pilot_sim.plot(z=0, ax=axes[0])
axes[0].set_title("Unit cell: xy plane, z = 0")

pilot_sim.plot(y=0, ax=axes[1])
axes[1].set_title("Unit cell: xz plane, y = 0")

plt.tight_layout()
plt.show()

# ============================================================
# One-credit pilot simulation
# ============================================================

print("Submitting one pilot simulation only...")

pilot_data_path = os.path.join(
    DATA_DIR,
    f"pilot_{pilot_key}.hdf5",
)

pilot_data = web.run(
    simulation=pilot_sim,
    task_name=f"bandstructure_pilot_{pilot_key}",
    path=pilot_data_path,
)

print("Pilot completed.")

# ============================================================
# Spectrum extraction
# ============================================================

def extract_time_trace(sim_data, monitor_name="field_time", field_name="Ey"):
    """
    Extract a one-dimensional field-time trace from a point FieldTimeMonitor.

    Compatible with Tidy3D versions where FieldTimeData does not have
    a top-level .time attribute.
    """

    monitor_data = sim_data[monitor_name]

    # Get the field component, e.g. monitor_data.Ey
    field_da = getattr(monitor_data, field_name)

    # The time coordinate belongs to the field DataArray
    if "t" not in field_da.coords:
        raise RuntimeError(
            f"No 'time' coordinate found. "
            f"Available coordinates: {list(field_da.coords)}"
        )

    time = np.asarray(field_da.coords["t"].values)

    # Convert DataArray to NumPy array
    values = np.asarray(field_da.values)

    # Find which axis corresponds to time
    if "t" not in field_da.dims:
        raise RuntimeError(
            f"No 'time' dimension found. "
            f"Available dimensions: {field_da.dims}"
        )

    time_axis = field_da.dims.index("t")

    # Move time to axis 0
    values = np.moveaxis(values, time_axis, 0)

    # Collapse all remaining singleton/spatial dimensions.
    # For a point monitor, this should produce shape (Nt,).
    values = values.reshape(len(time), -1)

    if values.shape[1] != 1:
        print(
            f"Warning: monitor contains {values.shape[1]} spatial samples. "
            "Using the first one."
        )

    signal = values[:, 0]

    if len(time) != len(signal):
        raise RuntimeError(
            f"Time and signal lengths differ: "
            f"{len(time)} versus {len(signal)}"
        )

    return time, signal


def calculate_spectrum(sim_data):
    """
    Calculate a positive-frequency spectrum from Ey(t).
    """

    time, signal = extract_time_trace(sim_data)

    # Remove DC offset
    signal = signal - np.mean(signal)

    # Window the time signal to reduce FFT sidelobes.
    # This works for real or complex Bloch-field data.
    window = np.hanning(len(signal))
    signal_windowed = signal * window

    dt = np.median(np.diff(time))
    fft_values = np.fft.fft(signal_windowed)
    fft_freqs = np.fft.fftfreq(len(signal), d=dt)

    positive = (
        (fft_freqs > fmin) &
        (fft_freqs < fmax)
    )

    freqs = fft_freqs[positive]
    spectrum = np.abs(fft_values[positive])

    # Normalize for plotting and peak detection
    if np.max(spectrum) > 0:
        spectrum = spectrum / np.max(spectrum)

    return freqs, spectrum


pilot_freqs, pilot_spectrum = calculate_spectrum(pilot_data)

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(
    pilot_freqs / 1e12,
    pilot_spectrum,
    color="navy",
    linewidth=1.2,
)

ax.axvline(
    f_snv / 1e12,
    color="crimson",
    linestyle="--",
    linewidth=1.5,
    label=r"$\lambda=620$ nm",
)

ax.set_xlabel("Frequency (THz)")
ax.set_ylabel(r"Normalized $|\mathcal{F}\{E_y(t)\}|$")
ax.set_title(rf"Pilot spectrum at $k_x/(2\pi/a)={pilot_k:.2f}$")
ax.set_xlim(fmin / 1e12, fmax / 1e12)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.25)
ax.legend()

plt.tight_layout()
plt.show()