
""" MOOSAC: RANSAC + Multi-objective optimization for robust fitting of surface to points.
"""
from .sampling_grid import Grid
from .moo_indiv import MOOFitness, MOOIndiv, MOOTools
from .moo_pop import MOOPop

__all__ = [
    "Grid", "MOOTools", "MOOPop",
    "surface_area", "robust_fitting"
]


def surface_area(zyx, guide, len_grid):
    """ Calculate surface area of fitted spline surface.

    The grid length affects the area, especially when the length is small compared to membrane thickness.
    The length can be set to 4*membrane thickness where the area changes appear small.

    Args:
        zyx (np.ndarray): 3d points, with shape=(npts,3). Each point is [zi,yi,xi].
        guide (np.ndarray): 3d guideline points sorted in each slice, with shape=(npts_guide,3). Each point is [zi,yi,xi].
        len_grid (float): The length of grids in u(z) and v(xy) directions (set both to the same value).

    Returns:
        area (float): The area of fitted surface, in units of pixels^2.
        surf_fit (splipy.surface.Surface): Fitted surface.
    """
    # construct grid and mootools
    grid = Grid(zyx, guide, shrink_sidegrid=1, nz_eachu=1)
    grid.set_ngrids_by_len(len_grids=(len_grid, len_grid))
    grid.generate_grid()
    mtools = MOOTools().init_from_grid(grid, 1)

    # sample points, fit
    indiv = mtools.gen_middle(pin_side=True)
    pts_net = mtools.get_coords_net(indiv)
    surf_fit = mtools.surf_meta.interpolate(pts_net)

    # calc area
    area = surf_fit.area()
    return area, surf_fit


def robust_fitting(
        zyx, guide,
        len_grids=(50, 100), shrink_sidegrid=0.25,
        fitness_rthresh=1,
        pop_size=4, tol=(0.005, 10), max_iter=200,
        func_map=map
    ):
    """ Robust fitting of a surface to points.

    Steps: generate Grid, MOOTools, MOOPop - evolve - fit the best indivdual.

    Settings:
    The sampling grid:
        len_grids: (len_uz,len_vxy) in units of pixels.
        shrink_sidegrid: A smaller ratio facilitates the coverage of the side in sampling.
    Fitness calculation:
        fitness_rthresh: Can be set to the thickness of the membrane.
    Evolution:
        pop_size: Population size.
        tol, max_iter: Termination criteria.
    Multiprocessing:
        func_map: pass multiprocessing-map from __main__
            pool = multiprocessing.Pool()
            func_map = pool.map
            pool.close()


    Args:
        zyx (np.ndarray): 3d points, with shape=(npts,3). Each point is [zi,yi,xi].
        guide (np.ndarray): 3d guideline points sorted in each slice, with shape=(npts_guide,3). Each point is [zi,yi,xi].
        len_grids (2-tuple of float): The length of grids in u(z),v(xy) directions, (len_uz,len_vxy) in units of pixels.
        shrink_sidegrid (float): Grids close to the sides in xy are shrinked to this ratio.
        fitness_rthresh (float): Distance threshold for fitness calculation.
        pop_size (int): Population size.
        tol (2-tuple): (tol_value, n_back). Terminate if change_ratio < tol_value within the last n_back steps.
        max_iter (int): The max number of generations.
        func_map (Callable): Map function.
    
    Returns:
        zyx_fit (np.ndarray): Points on the fitted surface, with shape=(npts,3).
        mpop_state (dict): Dict of MOOPop attributes.
            {mootools_config,pop_size,pop_list,log_front_list,log_indicator}.
            mpop = etsynseg.moosac.MOOPop().init_from_state(mpop_state)
    """
    # setup grid
    grid = Grid(
        zyx, guide, shrink_sidegrid=shrink_sidegrid, nz_eachu=1
    ).set_ngrids_by_len(
        len_grids=len_grids, ngrids_min=(3, 3)
    ).generate_grid()
    
    # setup mootools and population
    mtools = MOOTools().init_from_grid(grid, fitness_rthresh=fitness_rthresh)
    mpop = MOOPop().init_from_mootools(mtools, pop_size=pop_size)

    # init, register map, evolve, clean map
    mpop.init_pop()
    mpop.register_map(func_map=func_map)
    mpop.evolve(tol=tol, max_iter=max_iter)
    mpop.register_map()

    # get the best individual and fit surface
    zyx_fit = mpop.fit_surface_best(indiv=mpop.log_front[-1][0])

    # get mpop state
    mpop_state = mpop.save_state()
    return zyx_fit, mpop_state

