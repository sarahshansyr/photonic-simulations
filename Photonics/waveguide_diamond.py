import numpy as np
import matplotlib.pyplot as plt

import tidy3d as td
from tidy3d import web
import gdstk


lda0 = 1.55  # central wavelength
ldas = np.linspace(1.5, 1.6, 101)  # wavelength range of interest
freq0 = td.C_0 / lda0  # central frequency
freqs = td.C_0 / ldas  # frequency range of interest
fwidth = 0.5 * (np.max(freqs) - np.min(freqs))  # frequency width of the source

# Define diamond using a constant permittivity
n_diamond = 2.41
diamond = td.Medium(permittivity=n_diamond**2)

sio2 = td.material_library["SiO2"]["Horiba"]
air = td.Medium(permittivity=1.0)

w = 0.5  # width of the waveguide
l = 12
h_si = 0.22  # thickness of the silicon layer
inf_eff = 1e2  # effective infinity

# simulation domain size
Lx = l + 4 * lda0
Ly = 2 * w + lda0
Lz = 9 * h_si

def straight_waveguide(x0, y0, z0, x1, y1, wg_width, wg_thickness, medium, sidewall_angle=0):
    """
    This function defines a straight strip waveguide and returns the tidy3d structure of it.

    Parameters
    ----------
    x0: x coordinate of the waveguide starting position (um)
    y0: y coordinate of the waveguide starting position (um)
    z0: z coordinate of the waveguide starting position (um)
    x1: x coordinate of the waveguide end position (um)
    y1: y coordinate of the waveguide end position (um)
    wg_width: width of the waveguide (um)
    wg_thickness: thickness of the waveguide (um)
    medium: medium of the waveguide
    sidewall_angle: side wall angle of the waveguide (rad)
    """

    cell = gdstk.Cell("waveguide")  # define a gds cell

    path = gdstk.RobustPath((x0, y0), wg_width, layer=1, datatype=0)  # define a path
    path.segment((x1, y1))

    cell.add(path)  # add path to the cell

    # define geometry from the gds cell
    wg_geo = td.PolySlab.from_gds(
        cell,
        gds_layer=1,
        axis=2,
        slab_bounds=(z0 - wg_thickness / 2, z0 + wg_thickness / 2),
        sidewall_angle=sidewall_angle,
    )

    # define tidy3d structure of the bend
    wg = td.Structure(geometry=wg_geo[0], medium=medium)

    return wg

waveguide = straight_waveguide(
    x0=-inf_eff,
    y0=0,
    z0=0,
    x1=inf_eff,
    y1=0,
    wg_width=w,
    wg_thickness=h_si,
    medium=diamond,
    sidewall_angle=0,
)

n_si = 3.47
# mode spec for the source
mode_spec_source = td.ModeSpec(num_modes=1, target_neff=n_si, filter_pol="te")
# mode spec for the through port
mode_spec_through = mode_spec_source

# add a mode source as excitation
mode_source = td.ModeSource(
    center=(-l/2 - lda0 / 4, 0, 0),
    size=(0, 6 * w, 6 * h_si),
    source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    direction="+",
    mode_spec=mode_spec_source,
    mode_index=0,

)

# add a mode monitor behind the source to measure reflected power
mode_monitor_reflection = td.ModeMonitor(
    center=(-l/2 - lda0 / 2, 0, 0),
    size=mode_source.size,
    freqs=freqs,
    mode_spec=mode_spec_through,
    name="reflection",
)

# add a mode monitor to measure transmission at the through port
mode_monitor_through = td.ModeMonitor(
    center=(l/2 + lda0 / 4, 0, 0),
    size=mode_source.size,
    freqs=freqs,
    mode_spec=mode_spec_through,
    name="through",
)

# add a field monitor to visualize the field distribution
field_monitor = td.FieldMonitor(
    center=(0, 0, 0), size=(td.inf, td.inf, 0), freqs=[freq0], name="field"
)

run_time = 2e-12  # simulation run time

sim = td.Simulation(
    center=(0, Ly / 4, 0),
    size=(Lx, Ly, Lz),
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=25, wavelength=lda0),
    structures=[waveguide],
    sources=[mode_source],
    monitors=[
        mode_monitor_reflection,
        mode_monitor_through,
        field_monitor,
    ],
    run_time=run_time,
    boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    medium=sio2,
    symmetry=(0, 0, 1),
)

ax = sim.plot(y=0.01)
plt.show()

# run simulation
waveguide_simulation = web.run(
    simulation=sim,
    task_name="waveguideDiamondSiO2",
    path="data/simulation_waveguideTDiamondSiO2.hdf5",
)