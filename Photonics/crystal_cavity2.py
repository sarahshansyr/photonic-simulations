import numpy as np
import matplotlib.pyplot as plt

import tidy3d as td
from tidy3d import web

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from matplotlib.colors import LogNorm


# ============================================================
# Photonic-crystal slab band-structure simulation
# ============================================================
#
# Geometry units:
#     micrometers
#
# Frequency units:
#     Hz
#
# Band path:
#     Gamma -> X -> M -> Gamma
#
# Boundaries:
#     Bloch-periodic in x and y
#     PML in z
#
# The band structure is estimated by:
#
#     1. Exciting each Bloch unit cell with a point dipole.
#     2. Recording the time-domain electric field.
#     3. Fourier transforming the time signal.
#     4. Plotting spectral power versus Bloch wavevector.
#
# The heatmap is the primary result. Bright continuous lines
# correspond to candidate photonic bands.
# ============================================================


# ============================================================
# User settings
# ============================================================

# Tidy3D geometric lengths are in micrometers.
a = 0.5                  # lattice constant: 500 nm
r_hole = 0.28 * a        # air-hole radius: 140 nm
slab_thickness = 0.22    # slab thickness: 220 nm
n_slab = 3.4             # slab refractive index

# Frequency range
fmin = 160e12            # Hz
fmax = 300e12            # Hz

# Pulse parameters
f0 = 0.5 * (fmin + fmax)
fwidth = 0.5 * (fmax - fmin)

# Vertical air padding around the slab
air_buffer_z = 1.5       # micrometers
sim_z_size = slab_thickness + 2 * air_buffer_z

# Longer runtime improves frequency resolution.
# Approximate FFT resolution is 1 / run_time.
run_time = 5e-12

# FieldTimeMonitor.interval is an integer number of FDTD steps.
monitor_interval_steps = 1

# Number of PML layers above and below the slab
num_pml_layers = 20

# Grid resolution
min_steps_per_wavelength = 30

# Number of wavevector samples per high-symmetry segment
points_per_segment = 10

# Excite both in-plane polarizations.
# For a faster test, use:
# source_polarizations = ("Ey",)
source_polarizations = ("Ex", "Ey")

# Monitor all electric-field components.
recorded_fields = ("Ex", "Ey", "Ez")

# Set False if you only want to inspect the script.
RUN_SIMULATIONS = True

# Peak extraction is disabled by default because the heatmap
# is more reliable for interpreting the bands.
EXTRACT_PEAKS = False


# ============================================================
# Materials
# ============================================================

air = td.Medium(
    permittivity=1.0
)

slab_medium = td.Medium(
    permittivity=n_slab**2
)


# ============================================================
# Unit-cell geometry
# ============================================================

# High-index dielectric slab
slab = td.Structure(
    geometry=td.Box(
        center=(0, 0, 0),
        size=(a, a, slab_thickness),
    ),
    medium=slab_medium,
)

# Air hole in the center of the unit cell
air_hole = td.Structure(
    geometry=td.Cylinder(
        center=(0, 0, 0),
        radius=r_hole,
        length=slab_thickness,
        axis=2,
    ),
    medium=air,
)

# Later structures override earlier structures,
# so the air cylinder carves the hole in the slab.
structures = [
    slab,
    air_hole,
]


# ============================================================
# Print and verify physical parameters
# ============================================================

lambda_center = td.C_0 / f0
lambda_min = td.C_0 / fmax
lambda_max = td.C_0 / fmin

print("Photonic-crystal band simulation")
print("---------------------------------")
print(f"Lattice constant       : {a:.4f} um")
print(f"Hole radius            : {r_hole:.4f} um")
print(f"Slab thickness         : {slab_thickness:.4f} um")
print(f"Vertical simulation    : {sim_z_size:.4f} um")
print(f"Frequency range        : {fmin/1e12:.1f} - {fmax/1e12:.1f} THz")
print(f"Center frequency       : {f0/1e12:.1f} THz")
print(f"Center wavelength      : {lambda_center:.4f} um")
print(f"Shortest wavelength    : {lambda_min:.4f} um")
print(f"Longest wavelength     : {lambda_max:.4f} um")
print(f"Runtime                : {run_time:.2e} s")
print(f"Points per segment     : {points_per_segment}")
print(f"Source polarizations   : {source_polarizations}")
print()


# ============================================================
# Brillouin-zone path
# ============================================================

def interpolate_k(k_start, k_end, number_of_points):
    """
    Interpolate reduced wavevectors.

    Reduced wavevectors are expressed in units of 2*pi/a.
    """
    k_start = np.asarray(k_start, dtype=float)
    k_end = np.asarray(k_end, dtype=float)

    return np.linspace(
        k_start,
        k_end,
        number_of_points,
        endpoint=False,
    )


# Square lattice:
#
# Gamma = (0, 0)
# X     = (1/2, 0)
# M     = (1/2, 1/2)
#
k_path = np.vstack(
    [
        interpolate_k(
            (0.0, 0.0),
            (0.5, 0.0),
            points_per_segment,
        ),
        interpolate_k(
            (0.5, 0.0),
            (0.5, 0.5),
            points_per_segment,
        ),
        interpolate_k(
            (0.5, 0.5),
            (0.0, 0.0),
            points_per_segment,
        ),
        np.array([[0.0, 0.0]]),
    ]
)

k_labels = ["Γ", "X", "M", "Γ"]

k_boundary_indices = [
    0,
    points_per_segment,
    2 * points_per_segment,
    3 * points_per_segment,
]


# ============================================================
# Construct one simulation
# ============================================================

def make_band_simulation(k_reduced, source_polarization):
    """
    Construct one unit-cell simulation.

    Parameters
    ----------
    k_reduced : array-like
        Reduced Bloch wavevector in units of 2*pi/a.

    source_polarization : str
        One of "Ex", "Ey", or "Ez".
    """

    kx_reduced, ky_reduced = k_reduced

    # Convert to physical Bloch wavevector.
    # Since a is in micrometers, kx and ky are in inverse micrometers.
    kx = 2 * np.pi * kx_reduced / a
    ky = 2 * np.pi * ky_reduced / a

    # Place the source away from the air hole.
    source = td.PointDipole(
        center=(0.35 * a, 0, 0),
        source_time=td.GaussianPulse(
            freq0=f0,
            fwidth=fwidth,
        ),
        polarization=source_polarization,
    )

    # Point time monitor in the dielectric region.
    monitor = td.FieldTimeMonitor(
        center=(-0.35 * a, 0, 0),
        size=(0, 0, 0),
        fields=list(recorded_fields),
        start=0,
        stop=run_time,
        interval=monitor_interval_steps,
        name="band_time",
    )

    # Bloch boundaries in x and y; PML in z.
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.bloch(
            bloch_vec=kx,
        ),
        y=td.Boundary.bloch(
            bloch_vec=ky,
        ),
        z=td.Boundary.pml(
            num_layers=num_pml_layers,
        ),
    )

    # Use the shortest wavelength in the scan for grid generation.
    grid_spec = td.GridSpec.auto(
        min_steps_per_wvl=min_steps_per_wavelength,
        wavelength=td.C_0 / fmax,
    )

    simulation = td.Simulation(
        center=(0, 0, 0),
        size=(a, a, sim_z_size),
        grid_spec=grid_spec,
        structures=structures,
        sources=[source],
        monitors=[monitor],
        run_time=run_time,
        shutoff=1e-7,
        boundary_spec=boundary_spec,
        medium=air,
    )

    return simulation


# ============================================================
# Plot the unit cell before running
# ============================================================

example_simulation = make_band_simulation(
    k_reduced=k_path[0],
    source_polarization=source_polarizations[0],
)

ax = example_simulation.plot(z=0)
ax.set_title("Photonic-crystal unit cell")
plt.tight_layout()
plt.show()


# ============================================================
# Helper: obtain one time-domain signal
# ============================================================

def get_time_signal(field_time_data, field_name):
    """
    Return the time coordinate and real-valued field signal.
    """

    field_data = getattr(
        field_time_data,
        field_name,
    ).squeeze()

    if "t" not in field_data.dims:
        raise ValueError(
            f"{field_name} data do not contain a time dimension. "
            f"Dimensions: {field_data.dims}"
        )

    # Put time first and remove any remaining singleton dimensions.
    field_data = field_data.transpose("t")

    t = np.asarray(
        field_data.coords["t"].values,
        dtype=float,
    )

    signal = np.asarray(
        field_data.values
    ).squeeze()

    # Tidy3D data may have a complex dtype.
    # The time-domain signal used by rfft must be real.
    signal = np.real(signal).astype(float)

    if signal.ndim != 1:
        raise ValueError(
            f"Expected a 1D time signal for {field_name}, "
            f"got shape {signal.shape}"
        )

    if len(signal) < 10:
        raise ValueError(
            f"Too few time samples for {field_name}: {len(signal)}"
        )

    return t, signal


# ============================================================
# Helper: diagnose and Fourier transform a time signal
# ============================================================

def compute_spectrum(t, signal):
    """
    Compute raw spectral power in the requested frequency range.

    Returns
    -------
    frequencies : ndarray
    spectral_power : ndarray
    growth_factor : float
    """

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    number_of_samples = len(signal)

    middle_start = number_of_samples // 3
    middle_end = 2 * number_of_samples // 3

    middle_signal = signal[
        middle_start:middle_end
    ]

    tail_signal = signal[
        middle_end:
    ]

    middle_level = (
        np.max(np.abs(middle_signal)) + 1e-30
    )

    tail_level = np.max(
        np.abs(tail_signal)
    )

    growth_factor = tail_level / middle_level

    # Treat strong late-time growth as numerical divergence.
    if growth_factor > 100:
        raise RuntimeError(
            f"Likely divergence: tail/middle growth = "
            f"{growth_factor:.3e}"
        )

    # Remove DC offset.
    signal = signal - np.mean(signal)

    # Hann window reduces spectral leakage.
    signal_windowed = (
        signal * np.hanning(len(signal))
    )

    dt = np.mean(np.diff(t))

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(
            f"Invalid time step dt={dt}"
        )

    # Real FFT.
    spectrum_amplitude = np.abs(
        np.fft.rfft(signal_windowed)
    )

    # Use power rather than amplitude for the heatmap.
    spectral_power = spectrum_amplitude**2

    frequencies_fft = np.fft.rfftfreq(
        len(signal_windowed),
        d=dt,
    )

    frequency_mask = (
        (frequencies_fft >= fmin)
        & (frequencies_fft <= fmax)
    )

    frequencies = frequencies_fft[
        frequency_mask
    ]

    spectral_power = spectral_power[
        frequency_mask
    ]

    if len(frequencies) == 0:
        raise ValueError(
            "No FFT frequencies fall inside the requested range."
        )

    if not np.all(np.isfinite(spectral_power)):
        raise ValueError(
            "Spectrum contains NaN or infinite values."
        )

    return frequencies, spectral_power, growth_factor


# ============================================================
# Data containers
# ============================================================

# One row per k-point.
# Each row will contain the combined raw spectral power from
# the selected source polarizations and field components.
spectrum_rows = [None] * len(k_path)

# Frequency axis used for the heatmap.
frequency_axis_reference = None

# Diagnostics
successful_k_indices = []
failed_k_indices = []
divergent_k_indices = []

# Optional extracted peaks
band_frequencies = []
band_k_indices = []


# ============================================================
# Run the simulations
# ============================================================

if not RUN_SIMULATIONS:
    raise RuntimeError(
        "RUN_SIMULATIONS is False. Set it to True to run the sweep."
    )


for k_index, k_reduced in enumerate(k_path):

    print()
    print(
        f"K-point {k_index + 1}/{len(k_path)}: "
        f"k=({k_reduced[0]:.4f}, {k_reduced[1]:.4f})"
    )

    combined_spectrum = None
    k_point_succeeded = False
    k_point_had_divergence = False

    for source_polarization in source_polarizations:

        print(
            f"  Source polarization: "
            f"{source_polarization}"
        )

        try:
            simulation = make_band_simulation(
                k_reduced=k_reduced,
                source_polarization=source_polarization,
            )

            task_name = (
                f"pc_band_k{k_index:03d}_"
                f"{source_polarization}"
            )

            band_data = web.run(
                simulation=simulation,
                task_name=task_name,
            )

            field_time_data = band_data["band_time"]

            source_spectrum = None
            source_frequency_axis = None

            # Combine spectra from Ex, Ey, and Ez.
            for field_name in recorded_fields:

                try:
                    t, signal = get_time_signal(
                        field_time_data,
                        field_name,
                    )

                    frequencies, spectral_power, growth_factor = (
                        compute_spectrum(t, signal)
                    )

                    print(
                        f"    {field_name}: "
                        f"max|E|={np.max(np.abs(signal)):.3e}, "
                        f"tail/middle={growth_factor:.3e}"
                    )

                    # Use the first valid frequency axis as reference.
                    if frequency_axis_reference is None:
                        frequency_axis_reference = frequencies

                    # Interpolate if necessary.
                    if not np.allclose(
                        frequencies,
                        frequency_axis_reference,
                        rtol=1e-8,
                        atol=1e-3,
                    ):
                        spectral_power = np.interp(
                            frequency_axis_reference,
                            frequencies,
                            spectral_power,
                        )

                    if source_spectrum is None:
                        source_spectrum = spectral_power.copy()
                    else:
                        source_spectrum += spectral_power

                except RuntimeError as error:

                    # A divergent field component should not contaminate
                    # the complete k-point.
                    print(
                        f"    WARNING: {field_name} rejected: "
                        f"{error}"
                    )

                    k_point_had_divergence = True
                    continue

                except Exception as error:

                    print(
                        f"    WARNING: could not process "
                        f"{field_name}: {error}"
                    )

                    continue

            # Add this source polarization to the k-point spectrum.
            if source_spectrum is not None:

                if combined_spectrum is None:
                    combined_spectrum = source_spectrum.copy()
                else:
                    combined_spectrum += source_spectrum

                k_point_succeeded = True

        except Exception as error:

            print(
                f"  FAILED source polarization "
                f"{source_polarization}: {error}"
            )

            if "divergence" in str(error).lower():
                k_point_had_divergence = True

    # Store the result for this k-point.
    if k_point_succeeded:

        spectrum_rows[k_index] = combined_spectrum
        successful_k_indices.append(k_index)

    else:

        spectrum_rows[k_index] = None
        failed_k_indices.append(k_index)

    if k_point_had_divergence:
        divergent_k_indices.append(k_index)


# ============================================================
# Check sweep results
# ============================================================

if frequency_axis_reference is None:
    raise RuntimeError(
        "No simulations completed successfully. "
        "Check the Tidy3D errors above."
    )

print()
print("Sweep summary")
print("-------------")
print(
    f"Successful k-points : "
    f"{len(successful_k_indices)}"
)
print(
    f"Failed k-points     : "
    f"{len(failed_k_indices)}"
)
print(
    f"Possibly divergent  : "
    f"{len(divergent_k_indices)}"
)

if failed_k_indices:
    print(
        "Failed indices      : ",
        failed_k_indices,
    )

if divergent_k_indices:
    print(
        "Divergent indices   : ",
        divergent_k_indices,
    )


# ============================================================
# Build the raw spectrum heatmap
# ============================================================

number_of_frequency_points = len(
    frequency_axis_reference
)

heatmap_rows = []

for row in spectrum_rows:

    if row is None:

        heatmap_rows.append(
            np.full(
                number_of_frequency_points,
                np.nan,
            )
        )

    else:

        heatmap_rows.append(row)


spectra_heatmap = np.asarray(
    heatmap_rows,
    dtype=float,
)

# Mask invalid or nonpositive values for LogNorm.
spectra_masked = np.ma.masked_where(
    (~np.isfinite(spectra_heatmap))
    | (spectra_heatmap <= 0),
    spectra_heatmap,
)

positive_values = spectra_heatmap[
    np.isfinite(spectra_heatmap)
    & (spectra_heatmap > 0)
]

if len(positive_values) == 0:
    raise RuntimeError(
        "The heatmap contains no positive spectral values."
    )

# Robust logarithmic color limits.
vmax = np.percentile(
    positive_values,
    99.5,
)

vmin = np.percentile(
    positive_values,
    5.0,
)

# Keep vmin strictly positive and below vmax.
vmin = max(
    vmin,
    vmax * 1e-8,
)

if vmin >= vmax:
    vmin = vmax * 1e-6


# Missing simulations appear white.
cmap = plt.get_cmap("magma").copy()
cmap.set_bad(color="white")


# ============================================================
# Plot the spectrum heatmap
# ============================================================

plt.figure(figsize=(11, 5.5))

plt.imshow(
    spectra_masked.T,
    origin="lower",
    aspect="auto",
    extent=[
        0,
        len(k_path) - 1,
        frequency_axis_reference[0] / 1e12,
        frequency_axis_reference[-1] / 1e12,
    ],
    cmap=cmap,
    norm=LogNorm(
        vmin=vmin,
        vmax=vmax,
    ),
)

plt.colorbar(
    label="Spectral power, logarithmic scale"
)

for boundary in k_boundary_indices:

    plt.axvline(
        boundary,
        color="white",
        linestyle="--",
        linewidth=0.9,
        alpha=0.9,
    )

plt.xticks(
    k_boundary_indices,
    k_labels,
)

plt.xlabel("Wavevector path")
plt.ylabel("Frequency (THz)")
plt.title(
    "Photonic-crystal slab spectrum versus wavevector"
)

plt.tight_layout()

plt.savefig(
    "photonic_crystal_band_heatmap_raw_log.png",
    dpi=200,
)

plt.show()


# ============================================================
# Optional strict peak extraction
# ============================================================

if EXTRACT_PEAKS:

    for k_index, row in enumerate(spectrum_rows):

        if row is None:
            continue

        # Smooth the logarithmic spectrum.
        log_spectrum = np.log10(
            np.maximum(row, np.finfo(float).tiny)
        )

        log_spectrum_smooth = gaussian_filter1d(
            log_spectrum,
            sigma=1.0,
        )

        # Peaks must be meaningfully above the local background.
        peaks, properties = find_peaks(
            log_spectrum_smooth,
            prominence=0.20,
            distance=8,
        )

        peak_frequencies = (
            frequency_axis_reference[peaks]
        )

        band_frequencies.extend(
            peak_frequencies
        )

        band_k_indices.extend(
            [k_index] * len(peak_frequencies)
        )

    if len(band_frequencies) > 0:

        band_frequencies = np.asarray(
            band_frequencies
        )

        band_k_indices = np.asarray(
            band_k_indices
        )

        plt.figure(figsize=(11, 5.5))

        plt.scatter(
            band_k_indices,
            band_frequencies / 1e12,
            s=10,
            color="navy",
        )

        for boundary in k_boundary_indices:

            plt.axvline(
                boundary,
                color="gray",
                linestyle="--",
                linewidth=0.8,
            )

        plt.xticks(
            k_boundary_indices,
            k_labels,
        )

        plt.xlabel("Wavevector path")
        plt.ylabel("Frequency (THz)")
        plt.title(
            "Strictly extracted candidate photonic bands"
        )
        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            "photonic_crystal_band_peaks.png",
            dpi=200,
        )

        plt.show()

    else:

        print(
            "No peaks passed the strict extraction criteria."
        )


# ============================================================
# Save numerical results
# ============================================================

# Replace missing rows with NaNs before saving.
np.savez(
    "photonic_crystal_band_spectra.npz",
    k_path=k_path,
    frequencies=frequency_axis_reference,
    spectra=spectra_heatmap,
    successful_k_indices=np.asarray(
        successful_k_indices,
        dtype=int,
    ),
    failed_k_indices=np.asarray(
        failed_k_indices,
        dtype=int,
    ),
    divergent_k_indices=np.asarray(
        divergent_k_indices,
        dtype=int,
    ),
)

print()
print(
    "Saved:"
)
print(
    "  photonic_crystal_band_heatmap_raw_log.png"
)
print(
    "  photonic_crystal_band_spectra.npz"
)

if EXTRACT_PEAKS:
    print(
        "  photonic_crystal_band_peaks.png"
    )