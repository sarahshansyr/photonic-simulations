import tidy3d as td
import numpy as np
import tidy3d.plugins.design as tdd


# import need be changed in some cases

def build_sim(src_mode: float) -> td.Simulation:
    wl_c = 1.55
    n_Si = 3.47
    a = 0.42
    wg_w = 0.4
    n_holes = 11
    n_rows = 7
    freq_c = td.C_0 / wl_c
    t = 0.75 * a
    r = 0.3 * a
    w = 0.73 * a
    phc_l = n_holes * a
    Ly = 2 * n_rows * a + wl_c
    Lz = t + 1.1 * wl_c
    Lx = 1.5 * phc_l
    Vacuum = td.Medium(
        name='Vacuum',
    )

    mode = td.ModeSource(
        name='mode',
        center=[-Lx / 2 + wl_c / 2, 0, 0],
        size=[0, 4 * wg_w, 5 * t],
        source_time=td.GaussianPulse(freq0=193809772230563.88, fwidth=17505398782115.469, ),
        direction='+',
        mode_spec=td.ModeSpec(num_modes=1, target_neff=3.47,
                              sort_spec={'filter_reference': 0, 'filter_order': 'over', 'sort_key': 'n_eff',
                                         'track_freq': 'central', 'keep_modes': 'all'}, group_index_step=True, ),
        mode_index=src_mode
    )

    field_xy = td.FieldMonitor(
        name='field_xy',
        size=[td.inf, td.inf, 0],
        freqs=td.C_0 / 1.55,
    )

    modal_flux = td.ModeMonitor(
        name='modal_flux',
        center=[Lx / 2 - wl_c / 2, 0, 0],
        size=[0, 4 * wg_w, 5 * t],
        freqs=td.C_0 / np.linspace(1.48, 1.62, 141),
        mode_spec=td.ModeSpec(num_modes=2, target_neff=3.47,
                              sort_spec={'filter_reference': 0, 'filter_order': 'over', 'sort_key': 'n_eff',
                                         'track_freq': 'central', 'keep_modes': 'all'}, group_index_step=True, )
    )

    reflected_flux = td.FluxMonitor(
        name='reflected_flux',
        center=[-Lx / 2 + 0.1, 0, 0],
        size=[0, Ly, Lz],
        freqs=td.C_0 / np.linspace(1.48, 1.62, 141),
    )

    mat_Si = td.Medium(
        name='mat_Si',
        permittivity=12.0409,
    )
    phc_slab = td.Structure(
        geometry=td.Box(size=[4.62, 5.398826999999999, 0.315]),
        name='phc_slab',
        medium=mat_Si
    )

    waveguide = td.Structure(
        geometry=td.Box(size=[td.inf, wg_w, t]),
        name='waveguide',
        medium=mat_Si
    )

    # This function is defined within script object
    # This example demonstrates the modeling of a compact TM-pass
    # polarization filter based on photonic crystal waveguide.
    # The photonic crystal is an air-bridged silicon slab with periodic
    # air holes arranged in a triangular lattice. It is possible to achieve
    # a TM-pass but TE-block device within a frequency range by
    # utilizing bandgap engineering and index guiding mechanism.
    # The design parameters adopted from Chandra Prakash and
    # Mrinal Sen , “Optimization of silicon-photonic crystal (PhC)
    # waveguide for a compact and high extinction ratio TM-pass
    # polarization filter”, Journal of Applied Physics 127, 023101 (2020)
    # are optimized for the telecom frequency to have
    # a ~-0.5 dB TM transmission and ~-40 dB TE transmission.

    from types import SimpleNamespace
    # gui medium used in script might need some changes yourself manually
    param = SimpleNamespace(
        **{"wl_c": 1.55, "freq_c": 193414489032258.06, "n_Si": 3.47, "a": 0.42, "t": 0.315, "r": 0.126, "w": 0.3066,
           "wg_w": 0.4, "n_holes": 11, "n_rows": 7, "phc_l": 4.62, "Lx": 6.93, "Ly": 7.43, "Lz": 2.0200000000000005,
           "src_mode": 1, "c": 299792458000000, "pi": 3.1415927, "inf": "Infinity", "medium": {}})
    param.medium = SimpleNamespace(**{"medium": {}}['medium'], Vacuum=Vacuum, mat_Si=mat_Si)

    # This is the @entrypoint script object function
    def make_phc_holes_0(param):  # hole
        # Creates a PhC waveguide structure based on the parameters:
        # a = lattice constant;
        # r = photonic crystal radius;
        # t = photonic crystal slab thickness;
        # w = photonic crystal waveguide width;
        # n_rows = number of PhC rows in each mirror;
        # n_holes = the number of PhC holes in each row.

        # Definition of PhC holes medium: Air
        air = param.medium.Vacuum

        # Now, we iterate over the number of rows and add the air hole structures
        # for each column on both PhC mirrors at the left and the right
        # side of the missing row.
        holes = []
        for i in range(param.n_rows):
            if i % 2 == 0:
                # Shift the positions of the holes in the propagation direction by a/2 on even `i` values.
                shift = param.a / 2
                N = param.n_holes
            else:
                # Adds one more hole in the column on odd `i` values.
                shift = 0
                N = param.n_holes + 1

            # Iterate over the column and add the air hole structures.
            for j in range(N):
                # Add a hole at a position based on the i, j values.
                holes.append(
                    td.Cylinder(
                        center=(
                            (j - (param.n_holes) / 2) * param.a + shift,
                            (param.w / 2 + param.r) + (i) * np.sqrt(3) * param.a / 2,
                            0,
                        ),
                        radius=param.r,
                        length=param.t,
                    )
                )
                # Add a hole at the opposite PhC mirror, at a position based on the -i, j values.
                holes.append(
                    td.Cylinder(
                        center=(
                            (j - (param.n_holes) / 2) * param.a + shift,
                            -((param.w / 2 + param.r) + (i) * np.sqrt(3) * param.a / 2),
                            0,
                        ),
                        radius=param.r,
                        length=param.t,
                    )
                )

                hole_geo = td.Structure(geometry=td.GeometryGroup(geometries=holes), medium=air)
        return hole_geo

    make_phc_holes_0_result = make_phc_holes_0(param)
    make_phc_holes_0_result = make_phc_holes_0_result if isinstance(make_phc_holes_0_result, list) else [
        make_phc_holes_0_result]

    def getObjectInArray(className, items):
        return list(filter(lambda item: isinstance(item, className), items))

    base_sim = td.Simulation(
        size=[6.93, 7.43, 2.02],
        grid_spec=td.GridSpec(grid_x=td.UniformGrid(dl=0.021), grid_y=td.UniformGrid(dl=0.01819),
                              grid_z=td.AutoGrid(min_steps_per_wvl=20), wavelength=1.5500000000000003),
        run_time=2e-11,
        medium=Vacuum,
        sources=[mode, *getObjectInArray(td.Source, [*make_phc_holes_0_result])],
        monitors=[field_xy, modal_flux, reflected_flux, *getObjectInArray(td.Monitor, [*make_phc_holes_0_result])],
        structures=[phc_slab, waveguide, *getObjectInArray(td.Structure, make_phc_holes_0_result)],
    )
    return base_sim


design_space = tdd.DesignSpace(
    name="sweep_0",
    method=tdd.MethodGrid(),
    parameters=[
        tdd.ParameterFloat(name="src_mode", span=(0, 1), num_points=2),
    ],
)


def collect_data(sim_data: td.SimulationData) -> td.SimulationData:
    # Pass-through post-processing: keep the full SimulationData (every monitor) for each case.
    return sim_data


# `build_sim` is the pre-processing function (assembles each Simulation) and
# `collect_data` runs after the cloud Batch completes. Supplying both lets the
# design plugin actually run the simulations in parallel and collect results.
result = design_space.run(fn=build_sim, fn_post=collect_data)
# result.values[i] -> full SimulationData (all monitors) for case i
# result.coords[i] -> the parameter tuple for case i