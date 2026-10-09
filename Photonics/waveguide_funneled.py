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

def tapered_waveguide(x0, y0, x1, y1, wg_width0, wg_width1,
                      wg_thickness, medium, sidewall_angle=0):
    cell = gdstk.Cell("taper")
    path = gdstk.RobustPath((x0, y0), wg_width0, layer=1, datatype=0)
    path.segment((x1, y1), width=wg_width1)  # linear width ramp along the segment
    cell.add(path)
    wg_geo = td.PolySlab.from_gds(
        cell,
        gds_layer=1,
        axis=2,
        slab_bounds=(0 - wg_thickness / 2, 0 + wg_thickness / 2),
        sidewall_angle=sidewall_angle,
    )
    return td.Structure(geometry=wg_geo[0], medium=medium)

taper_length = 8  # µm

# straight input lead (0.5 µm wide)
wg_in = tapered_waveguide(x0=-inf_eff, y0=0, x1=-taper_length/2, y1=0,
                          wg_width0=w, wg_width1=w,            # constant width
                          wg_thickness=h_si, medium=diamond)
# the actual taper: 0.5 -> 0.75 over 8 µm
taper = tapered_waveguide(x0=-taper_length/2, y0=0, x1=taper_length/2, y1=0,
                          wg_width0=w, wg_width1=2*w,
                          wg_thickness=h_si, medium=diamond)
# straight output lead (0.75 µm wide)
wg_out = tapered_waveguide(x0=taper_length/2, y0=0, x1=inf_eff, y1=0,
                           wg_width0=2*w, wg_width1=2*w,   # constant width
                           wg_thickness=h_si, medium=diamond)

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
    center=(0, 0, 0), size=(td.inf, td.inf, 0), freqs=[freq0], name="field0"
)

run_time = 2e-12  # simulation run time

sim = td.Simulation(
    center=(0, Ly / 4, 0),
    size=(Lx, Ly, Lz),
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=25, wavelength=lda0),
    structures=[wg_in, taper, wg_out],
    sources=[mode_source],
    monitors=[
        mode_monitor_reflection,
        mode_monitor_through,
        field_monitor
    ],
    run_time=run_time,
    boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    medium=sio2,
    symmetry=(0, 0, 1),
)

ax = sim.plot(z=0)
plt.show()

# run simulation
waveguide_simulation = web.run(
    simulation=sim,
    task_name="waveguideFunneledDiamondSiO2",
    path="data/simulation_waveguideFunneledDiamondSiO2.hdf5",
)



