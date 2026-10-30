import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks


# ============================================================
# Load previously saved data
# ============================================================

filename = "photonic_crystal_band_spectra.npz"

data = np.load(filename)

k_path = data["k_path"]
frequencies = data["frequencies"]
spectra = data["spectra"]

# Frequency in THz for plotting
frequency_thz = frequencies / 1e12

number_of_k_points, number_of_frequency_points = spectra.shape

print("Loaded data")
print("-----------")
print(f"Number of k-points       : {number_of_k_points}")
print(f"Number of frequency bins: {number_of_frequency_points}")
print(
    f"Frequency range          : "
    f"{frequency_thz[0]:.2f} - {frequency_thz[-1]:.2f} THz"
)


# ============================================================
# Determine valid k-points
# ============================================================

valid_k = (
    np.any(np.isfinite(spectra), axis=1)
    & np.any(spectra > 0, axis=1)
)

valid_indices = np.where(valid_k)[0]

if len(valid_indices) == 0:
    raise RuntimeError(
        "No valid spectra were found in the NPZ file."
    )

print(f"Valid k-points           : {len(valid_indices)}")
print(
    f"Invalid k-points         : "
    f"{number_of_k_points - len(valid_indices)}"
)


# ============================================================
# Fill failed rows only for smoothing
# ============================================================
#
# Failed rows are not used as actual band data. They are filled
# temporarily so that the Gaussian smoothing operation does not
# propagate NaNs through the whole array.
# ============================================================

spectra_work = spectra.astype(float).copy()

# Use the median spectrum of successful k-points as a placeholder
# for failed rows.
median_spectrum = np.nanmedian(
    spectra[valid_k],
    axis=0,
)

for k_index in range(number_of_k_points):

    if not valid_k[k_index]:

        spectra_work[k_index, :] = median_spectrum


# Replace any remaining invalid or nonpositive values
# with a very small positive number.
positive_values = spectra_work[
    np.isfinite(spectra_work)
    & (spectra_work > 0)
]

if len(positive_values) == 0:
    raise RuntimeError(
        "The spectra contain no positive finite values."
    )

smallest_positive = np.min(positive_values)

spectra_work[
    ~np.isfinite(spectra_work)
    | (spectra_work <= 0)
] = smallest_positive


# ============================================================
# Convert to logarithmic spectral power
# ============================================================

log_spectra = np.log10(spectra_work)


# ============================================================
# Normalize each k-point for ridge detection
# ============================================================
#
# This normalization is used only to find bands.
# It should not be interpreted as physical intensity.
#
# It compensates for the fact that a point source couples
# differently to different modes at different k-points.
# ============================================================

ridge_map = np.zeros_like(log_spectra)

for k_index in range(number_of_k_points):

    row = log_spectra[k_index]

    lower = np.percentile(row, 5)
    upper = np.percentile(row, 99)

    scale = upper - lower

    if scale <= 1e-12:
        ridge_map[k_index, :] = 0.0
    else:
        ridge_map[k_index, :] = (
            row - lower
        ) / scale

ridge_map = np.clip(
    ridge_map,
    0.0,
    1.0,
)


# Invalid k-points should not contribute detected bands.
ridge_map[~valid_k, :] = 0.0


# ============================================================
# Smooth the response map
# ============================================================
#
# sigma=(k smoothing, frequency smoothing)
#
# A little smoothing in the k direction turns isolated
# spectral spots into continuous ridges without rerunning
# simulations.
# ============================================================

ridge_smooth = gaussian_filter(
    ridge_map,
    sigma=(1.2, 1.5),
)


# ============================================================
# Detect candidate peaks at every k-point
# ============================================================

frequency_step_thz = np.median(
    np.diff(frequency_thz)
)

# Require peaks to be separated by at least approximately 1 THz.
minimum_peak_distance = max(
    3,
    int(round(1.0 / frequency_step_thz)),
)

candidate_frequencies = []
candidate_scores = []

for k_index in range(number_of_k_points):

    if not valid_k[k_index]:

        candidate_frequencies.append([])
        candidate_scores.append([])
        continue

    row = ridge_smooth[k_index]

    peaks, properties = find_peaks(
        row,
        height=0.48,
        prominence=0.08,
        distance=minimum_peak_distance,
    )

    candidate_frequencies.append(
        frequency_thz[peaks]
    )

    candidate_scores.append(
        row[peaks]
    )

print()
print("Peak-detection parameters")
print("-------------------------")
print(f"Frequency step          : {frequency_step_thz:.4f} THz")
print(
    f"Minimum peak separation: "
    f"{minimum_peak_distance} bins"
)

total_candidates = sum(
    len(values)
    for values in candidate_frequencies
)

print(f"Candidate peaks found   : {total_candidates}")


# ============================================================
# Track continuous bands
# ============================================================

# A band should move continuously between neighboring k-points.
# This tracker connects peaks only if their frequency difference
# is physically/plausibly small.
#
# These are plotting parameters, not simulation parameters.

maximum_frequency_jump_thz = 7.0
maximum_missing_k_points = 2

tracks = []


def start_track(k_index, candidate_index):
    """
    Create a new candidate band track.
    """

    frequency = candidate_frequencies[
        k_index
    ][candidate_index]

    score = candidate_scores[
        k_index
    ][candidate_index]

    return {
        "points": [
            (k_index, frequency, score)
        ],
        "last_k": k_index,
        "last_frequency": frequency,
        "previous_frequency": None,
        "missing": 0,
    }


def predicted_frequency(track):
    """
    Predict the next frequency using the previous slope.
    """

    points = track["points"]

    if len(points) < 2:
        return track["last_frequency"]

    last_frequency = points[-1][1]
    previous_frequency = points[-2][1]

    slope = last_frequency - previous_frequency

    return last_frequency + slope


# Start tracks using the first valid k-point.
first_valid_k = valid_indices[0]

active_tracks = []

for candidate_index in range(
    len(candidate_frequencies[first_valid_k])
):
    active_tracks.append(
        start_track(
            first_valid_k,
            candidate_index,
        )
    )


# Process subsequent k-points.
for k_index in range(first_valid_k + 1, number_of_k_points):

    if not valid_k[k_index]:
        continue

    frequencies_here = candidate_frequencies[k_index]
    scores_here = candidate_scores[k_index]

    used_candidates = set()

    # Sort active tracks by their predicted frequency.
    # This makes matching more stable near crossings.
    active_tracks_sorted = sorted(
        active_tracks,
        key=predicted_frequency,
    )

    for track in active_tracks_sorted:

        prediction = predicted_frequency(track)

        if len(frequencies_here) == 0:

            track["missing"] += 1
            continue

        available_indices = [
            candidate_index
            for candidate_index in range(
                len(frequencies_here)
            )
            if candidate_index not in used_candidates
        ]

        if len(available_indices) == 0:

            track["missing"] += 1
            continue

        differences = np.array(
            [
                abs(
                    frequencies_here[candidate_index]
                    - prediction
                )
                for candidate_index in available_indices
            ]
        )

        nearest_position = np.argmin(differences)
        nearest_candidate = available_indices[
            nearest_position
        ]
        nearest_difference = differences[
            nearest_position
        ]

        # Allow a somewhat larger jump after a missed k-point.
        allowed_jump = (
            maximum_frequency_jump_thz
            * (track["missing"] + 1)
        )

        if nearest_difference <= allowed_jump:

            new_frequency = frequencies_here[
                nearest_candidate
            ]

            new_score = scores_here[
                nearest_candidate
            ]

            track["previous_frequency"] = (
                track["last_frequency"]
            )

            track["last_frequency"] = (
                new_frequency
            )

            track["last_k"] = k_index

            track["points"].append(
                (
                    k_index,
                    new_frequency,
                    new_score,
                )
            )

            track["missing"] = 0

            used_candidates.add(
                nearest_candidate
            )

        else:

            track["missing"] += 1

    # Remove tracks that have been missing too long.
    active_tracks = [
        track
        for track in active_tracks
        if track["missing"]
        <= maximum_missing_k_points
    ]

    # Start new tracks for unused candidates.
    for candidate_index in range(
        len(frequencies_here)
    ):

        if candidate_index not in used_candidates:

            active_tracks.append(
                start_track(
                    k_index,
                    candidate_index,
                )
            )

# `active_tracks` contains tracks that survive until the end.
# During the algorithm, some tracks may have been removed.
#
# To retain all tracks, this second pass is not required for the
# usual case. We use the active tracks plus any sufficiently long
# tracks collected during processing below.
#
# For this implementation, the final active tracks are enough
# for long continuous bands, but isolated bands may have ended
# earlier. Therefore, reconstruct tracks with a simpler full pass.


# ============================================================
# Full greedy track reconstruction
# ============================================================

all_tracks = []

# Start a track from every candidate.
for start_k in valid_indices:

    for start_candidate in range(
        len(candidate_frequencies[start_k])
    ):

        track_points = [
            (
                start_k,
                candidate_frequencies[start_k][
                    start_candidate
                ],
                candidate_scores[start_k][
                    start_candidate
                ],
            )
        ]

        last_frequency = track_points[-1][1]
        previous_frequency = None
        missing = 0

        for k_index in range(
            start_k + 1,
            number_of_k_points,
        ):

            if not valid_k[k_index]:
                missing += 1

                if missing > maximum_missing_k_points:
                    break

                continue

            frequencies_here = candidate_frequencies[
                k_index
            ]

            scores_here = candidate_scores[
                k_index
            ]

            if len(frequencies_here) == 0:
                missing += 1

                if missing > maximum_missing_k_points:
                    break

                continue

            if previous_frequency is None:

                prediction = last_frequency

            else:

                prediction = (
                    last_frequency
                    + (
                        last_frequency
                        - previous_frequency
                    )
                )

            differences = np.abs(
                frequencies_here - prediction
            )

            nearest_candidate = np.argmin(
                differences
            )

            nearest_difference = differences[
                nearest_candidate
            ]

            allowed_jump = (
                maximum_frequency_jump_thz
                * (missing + 1)
            )

            if nearest_difference <= allowed_jump:

                new_frequency = frequencies_here[
                    nearest_candidate
                ]

                new_score = scores_here[
                    nearest_candidate
                ]

                previous_frequency = last_frequency
                last_frequency = new_frequency
                missing = 0

                track_points.append(
                    (
                        k_index,
                        new_frequency,
                        new_score,
                    )
                )

            else:

                missing += 1

                if missing > maximum_missing_k_points:
                    break

        all_tracks.append(track_points)


# ============================================================
# Filter duplicate and unreliable tracks
# ============================================================

clean_tracks = []

for track in all_tracks:

    if len(track) < 8:
        continue

    k_values = np.array(
        [
            point[0]
            for point in track
        ]
    )

    f_values = np.array(
        [
            point[1]
            for point in track
        ]
    )

    scores = np.array(
        [
            point[2]
            for point in track
        ]
    )

    k_span = k_values[-1] - k_values[0]
    coverage = len(track) / (k_span + 1)

    average_score = np.mean(scores)

    # Reject tracks that are too short or too sparse.
    if k_span < 8:
        continue

    if coverage < 0.35:
        continue

    if average_score < 0.45:
        continue

    clean_tracks.append(track)


# Remove near-duplicate tracks.
# Two tracks are considered duplicates if they overlap in k and
# remain close in frequency.
unique_tracks = []

for track in sorted(
    clean_tracks,
    key=lambda item: len(item),
    reverse=True,
):

    k_values = np.array(
        [
            point[0]
            for point in track
        ]
    )

    f_values = np.array(
        [
            point[1]
            for point in track
        ]
    )

    is_duplicate = False

    for existing_track in unique_tracks:

        existing_k = np.array(
            [
                point[0]
                for point in existing_track
            ]
        )

        existing_f = np.array(
            [
                point[1]
                for point in existing_track
            ]
        )

        common_k = np.intersect1d(
            k_values,
            existing_k,
        )

        if len(common_k) < 5:
            continue

        current_interp = np.interp(
            common_k,
            k_values,
            f_values,
        )

        existing_interp = np.interp(
            common_k,
            existing_k,
            existing_f,
        )

        mean_difference = np.mean(
            np.abs(
                current_interp
                - existing_interp
            )
        )

        if mean_difference < 1.5:
            is_duplicate = True
            break

    if not is_duplicate:
        unique_tracks.append(track)


clean_tracks = unique_tracks

print()
print("Band-tracking result")
print("--------------------")
print(f"Tracks retained         : {len(clean_tracks)}")


# ============================================================
# Plot cleaned band diagram
# ============================================================

plt.figure(figsize=(11, 6))

colors = plt.cm.turbo(
    np.linspace(
        0.05,
        0.95,
        max(len(clean_tracks), 1),
    )
)

for track_index, track in enumerate(clean_tracks):

    k_values = np.array(
        [
            point[0]
            for point in track
        ]
    )

    f_values = np.array(
        [
            point[1]
            for point in track
        ]
    )

    scores = np.array(
        [
            point[2]
            for point in track
        ]
    )

    plt.plot(
        k_values,
        f_values,
        "-",
        color=colors[track_index],
        linewidth=1.4,
        alpha=0.85,
    )

    plt.scatter(
        k_values,
        f_values,
        s=8,
        color=colors[track_index],
        alpha=0.65,
    )


# High-symmetry-point locations.
# This assumes the same path construction used in the
# original simulation script.
points_per_segment = (
    number_of_k_points - 1
) // 3

k_boundary_indices = [
    0,
    points_per_segment,
    2 * points_per_segment,
    3 * points_per_segment,
]

for boundary in k_boundary_indices:

    plt.axvline(
        boundary,
        color="gray",
        linestyle="--",
        linewidth=0.8,
    )

plt.xticks(
    k_boundary_indices,
    ["Γ", "X", "M", "Γ"],
)

plt.xlabel("Wavevector path")
plt.ylabel("Frequency (THz)")
plt.title(
    "Extracted photonic-crystal band structure"
)

plt.ylim(
    frequency_thz[0],
    frequency_thz[-1],
)

plt.grid(
    alpha=0.25,
)

plt.tight_layout()

plt.savefig(
    "photonic_crystal_clean_band_structure.png",
    dpi=250,
)

plt.show()


# ============================================================
# Plot ridge-detection map for debugging
# ============================================================

plt.figure(figsize=(11, 5))

plt.imshow(
    ridge_smooth.T,
    origin="lower",
    aspect="auto",
    extent=[
        0,
        number_of_k_points - 1,
        frequency_thz[0],
        frequency_thz[-1],
    ],
    cmap="magma",
    vmin=0,
    vmax=1,
)

plt.colorbar(
    label="Ridge strength used for tracking"
)

for boundary in k_boundary_indices:

    plt.axvline(
        boundary,
        color="white",
        linestyle="--",
        linewidth=0.8,
    )

plt.xticks(
    k_boundary_indices,
    ["Γ", "X", "M", "Γ"],
)

plt.xlabel("Wavevector path")
plt.ylabel("Frequency (THz)")
plt.title(
    "Smoothed ridge map used for band extraction"
)

plt.tight_layout()

plt.savefig(
    "photonic_crystal_ridge_map.png",
    dpi=250,
)

plt.show()