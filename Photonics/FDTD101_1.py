# imports
import matplotlib.pylab as plt
import tidy3d as td
import tidy3d.web as web

# simulation parameters
side_length = 9.0
grid_size = 50e-3

# apply a PML in all directions
boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML())

# spectrum and resolution parameters
lambda0 = 1.0
freq0 = td.C_0 / lambda0
fwidth = freq0 / 50
run_time = 200 / freq0

# define dipole source at origin pointing in z
dipole_source = td.PointDipole(
    center=(0, 0, 0),
    source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    polarization='Ez',
    name='dipole',
)

# define monitor to measure fields in xz plane at central frequency
monitor = td.FieldMonitor(
    center=(0,0,0),
    size=(td.inf, 0, td.inf),
    freqs=[freq0],
    name='freq_domain',
)

# define the boxes
epsilon_box = 3
center_offset_box = 1.6
size_box = 1.8
medium=td.Medium(permittivity=epsilon_box,name='medium')

box_top = td.Structure(
    geometry=td.Box(
        center=(0, 0, center_offset_box),
        size=(size_box, size_box, size_box),
    ),
    medium=medium,
    name='box top',
)

bot_bot_r = td.Structure(
    geometry=td.Box(
        center=(+center_offset_box, 0, -center_offset_box),
        size=(size_box, size_box, size_box),
    ),
    medium=medium,
    name='box bottom right',
)

bot_bot_l = td.Structure(
    geometry=td.Box(
        center=(-center_offset_box, 0, -center_offset_box),
        size=(size_box, size_box, size_box),
    ),
    medium=medium,
    name='box bottom left',
)

# define simulation
# make a new simulation with the boxes added
simulation_boxes = td.Simulation(
    size=(side_length, side_length, side_length),
    grid_spec=td.GridSpec.uniform(dl=grid_size),
    structures=[box_top, bot_bot_r, bot_bot_l],
    sources=[dipole_source],
    monitors=[monitor],
    run_time=run_time,
    boundary_spec=boundary_spec,
)

# visualize the simulation first, before running
ax = simulation_boxes.plot(y=0.01)
plt.show()

# run simulation
sim_data_boxes = web.run(simulation_boxes, task_name='lecture01_several_box', path='data/data_several_box.hdf5')


# check that frequency domain fields look good
ax = sim_data_boxes.plot_field('freq_domain', 'Ez', y=0, f=freq0)
plt.show()