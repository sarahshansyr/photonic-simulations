import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
from tidy3d import web
from scipy.signal import find_peaks

# ============================================================
# Basic parameters
# ============================================================

# All distances are in meters
a = 0.5           # lattice constant
r_hole = 0.28 * a      # air-hole radius
slab_thickness = 0.22
n_slab = 3.4

slab_medium = td.Medium(permittivity=n_slab**2)
air = td.Medium(permittivity=1.0)

# Frequency range to scan
fmin = 160e12
fmax = 300e12
f0 = 0.5 * (fmin + fmax)
fwidth = fmax - fmin

# Simulation dimensions in z
air_buffer_z = 1
sim_z_size = slab_thickness + 2 * air_buffer_z

# Time-domain simulation duration
run_time = 3e-11
time_interval = 1

# ============================================================
# Unit-cell geometry
# ============================================================

slab = td.Structure(
    geometry=td.Box(
        center=(0, 0, 0),
        size=(a, a, slab_thickness),
    ),
    medium=slab_medium,
)

# One air hole at the center of the unit cell
air_hole = td.Structure(
    geometry=td.Cylinder(
        center=(0, 0, 0),
        radius=r_hole,
        length=slab_thickness,
        axis=2,
    ),
    medium=air,
)

structures = [slab, air_hole]

# ============================================================
# Bloch wavevector path: Gamma -> X -> M -> Gamma
# ============================================================

def interpolate_k(k_start, k_end, number_of_points):
    """Interpolate between two reduced wavevectors."""
    return np.linspace(
        np.array(k_start),
        np.array(k_end),
        number_of_points,
        endpoint=False,
    )


points_per_segment = 10

k_path = np.vstack([
    interpolate_k((0.0, 0.0), (0.5, 0.0), points_per_segment),
    interpolate_k((0.5, 0.0), (0.5, 0.5), points_per_segment),
    interpolate_k((0.5, 0.5), (0.0, 0.0), points_per_segment),
    np.array([[0.0, 0.0]]),
])

k_labels = ["Γ", "X", "M", "Γ"]

# ============================================================
# One FDTD simulation for a particular Bloch wavevector
# ============================================================

def make_band_simulation(k_reduced):
    """
    Build one unit-cell simulation.

    k_reduced = (kx, ky), in units of 2*pi/a.
    """

    kx_reduced, ky_reduced = k_reduced

    # Physical Bloch wavevector in inverse meters
    kx = 2 * np.pi * kx_reduced / a
    ky = 2 * np.pi * ky_reduced / a

    source = td.PointDipole(
        center=(0.35 * a, 0, 0),
        source_time=td.GaussianPulse(
            freq0=f0,
            fwidth=fwidth,
        ),
        polarization="Ey",   # corrected
    )

    monitor = td.FieldTimeMonitor(
        center=(-0.35 * a, 0, 0),
        size=(0, 0, 0),
        fields=["Ey"],
        start=0,
        stop=run_time,
        interval=time_interval,
        name="band_time",
    )

    boundary_spec = td.BoundarySpec(
        x=td.Boundary.bloch(bloch_vec=kx),
        y=td.Boundary.bloch(bloch_vec=ky),
        z=td.Boundary.pml(),
    )

    simulation = td.Simulation(
        center=(0, 0, 0),
        size=(a, a, sim_z_size),
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=20,
            wavelength=td.C_0 / f0,
        ),
        structures=structures,
        sources=[source],
        monitors=[monitor],
        run_time=run_time,
        boundary_spec=boundary_spec,
        medium=air,
    )

    return simulation

# ============================================================
# Run the band-structure sweep
# ============================================================

band_frequencies = []
band_k_indices = []

for index, k_reduced in enumerate(k_path):

    print(
        f"Running k-point {index + 1}/{len(k_path)}: "
        f"k = ({k_reduced[0]:.3f}, {k_reduced[1]:.3f})"
    )

    band_sim = make_band_simulation(k_reduced)

    band_data = web.run(
        simulation=band_sim,
        task_name=f"pc_band_{index:03d}",
    )

    # Get the time-domain field
    Ey_time = band_data["band_time"].Ey.squeeze()

    t = np.asarray(Ey_time.coords["t"].values)

    signal = np.asarray(Ey_time.values)
    signal = np.squeeze(signal)
    signal = np.real(signal).astype(float)

    if signal.ndim != 1:
        raise ValueError(
            f"Expected a 1D signal, but got shape {signal.shape}"
        )

    signal = signal - np.mean(signal)

    window = np.hanning(len(signal))
    signal_windowed = signal * window

    dt = np.mean(np.diff(t))

    spectrum = np.abs(
        np.fft.rfft(signal_windowed)
    )

    frequencies_fft = np.fft.rfftfreq(
        len(signal_windowed),
        d=dt,
    )

    # Keep only the frequency range of interest
    frequency_mask = (
        (frequencies_fft >= fmin)
        & (frequencies_fft <= fmax)
    )

    frequencies_plot = frequencies_fft[frequency_mask]
    spectrum_plot = spectrum[frequency_mask]

    # Normalize this k-point's spectrum
    if np.max(spectrum_plot) > 0:
        spectrum_plot = spectrum_plot / np.max(spectrum_plot)

    # Identify spectral peaks
    peaks, properties = find_peaks(
        spectrum_plot,
        prominence=0.05,
        distance=4,
    )

    peak_frequencies = frequencies_plot[peaks]

    # Store all detected bands
    band_frequencies.extend(peak_frequencies)
    band_k_indices.extend(
        [index] * len(peak_frequencies)
    )

# Convert to arrays
band_frequencies = np.array(band_frequencies)
band_k_indices = np.array(band_k_indices)

# ============================================================
# Plot the band diagram
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    band_k_indices,
    band_frequencies / 1e12,
    s=18,
    color="navy",
)

# High-symmetry point locations
segment_boundaries = [
    0,
    points_per_segment,
    2 * points_per_segment,
    3 * points_per_segment,
]

for boundary in segment_boundaries:
    plt.axvline(
        boundary,
        color="gray",
        linestyle="--",
        linewidth=0.8,
    )

plt.xticks(
    segment_boundaries,
    k_labels,
)

plt.xlabel("Wavevector path")
plt.ylabel("Frequency (THz)")
plt.title("Photonic-crystal slab band structure")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()