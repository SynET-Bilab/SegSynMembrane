""" Multi-objective optimization: individuals.
"""

import numpy as np
import deap, deap.base, deap.tools

from etsynseg import pcdutils, bspline
from .grid import Grid


class MOOFitness(deap.base.Fitness):
    """ DEAP fitness for individuals.
    """
    # (coverage penalty, non-membrane penalty)
    weights = (-1, -1)


class MOOIndiv(list):
    """ DEAP individuals.

    Each instance contains sampling points on the grid, [[pt_u0v0, pt_u0v1,...],[pt_u1v0,...],...].
    """
    def __init__(self, iterable=()):
        super().__init__(iterable)
        self.fitness = MOOFitness()


class MOOTools:
    """ Tools for individuals.

    Examples:
        # setup
        mootools = MOOTools(zyx, n_vxy, n_uz, nz_eachu, degree, fitness_rthresh)
        # generate individuals
        indiv = mootools.indiv_random()
        # evolution
        mootools.mutate(indiv)
        indiv.fitness.values = mootools.evaluate(indiv)
        # save/load
        config = mootools.get_config()
        mootools_new = MOOTools(config=config)
        sample_list, fitness = mootools.to_list_fitness(indiv)
        indiv = mootools.from_list_fitness(sample_list, fitness)
        # surface fut
        zyx_fit, surf_fit = mootools.fit_surface_eval(indiv, u_evals, v_evals)
    
    Methods:
        # io
        get_config, to_list_fitness, from_list_fitness
        # indiv generation
        indiv_random, indiv_uniform, indiv_middle
        # operations
        mutate
        # get coordinates
        get_coord_net, flatten_net
        # evaluate
        fit_surface_eval, calc_fitness, evaluate
    """
    def __init__(self, zyx=None, n_uz=3, n_vxy=3, nz_eachu=1, shrink_sidegrid=1, fitness_rthresh=1, config=None):
        """ Initialization.

        Args:
            zyx (np.ndarray): Points with shape=(npts,3).
            n_vxy, n_uz (int): The number of sampling grids in v(xy) and u(z) directions.
            nz_eachu (int): The number of z-direction slices contained in each grid.
            shrink_sidegrid (float): Grids close to the sides in xy are shrinked to this ratio.
                A smaller ratio facilitates the coverage of sampling.
            fitness_rthresh (float): Distance threshold for fitness evaluation. Points outside this threshold contribute constant loss.
            config (dict): Config from self.get_config().
        """
        # read config
        # if config is provided as args
        if zyx is not None:
            self.zyx = zyx
            self.n_vxy = n_vxy
            self.n_uz = n_uz
            self.nz_eachu = nz_eachu
            self.shrink_sidegrid = shrink_sidegrid
            self.fitness_rthresh = fitness_rthresh
        # if config is provided as a dict
        elif config is not None:
            self.zyx = config["zyx"]
            self.n_vxy = config["n_vxy"]
            self.n_uz = config["n_uz"]
            self.nz_eachu = config["nz_eachu"]
            self.shrink_sidegrid = config["shrink_sidegrid"]
            self.fitness_rthresh = config["fitness_rthresh"]
        else:
            raise ValueError("Should provide either zyx or config.")

        # image shape, grid
        self.grid = Grid(
            self.zyx,
            n_vxy=self.n_vxy, n_uz=self.n_uz,
            nz_eachu=self.nz_eachu,
            shrink_sidegrid=self.shrink_sidegrid
        )
        # update nv, nu from grid: where there are additional checks
        self.n_vxy = self.grid.n_vxy
        self.n_uz = self.grid.n_uz

        # fitness
        # bspline
        self.surf_meta = bspline.Surface(degree=2)
        # pointcloud
        self.pcd = pcdutils.points2pointcloud(
            pcdutils.points_deduplicate(self.zyx)
        )
    
    def get_config(self):
        """ Get config as dict. Convenient for dumping and loading.

        Returns:
            config (dict): {zyx,shape,n_vxy,n_uz,nz_eachu}.
        """
        config = dict(
            zyx=self.zyx,
            n_vxy=self.n_vxy,
            n_uz=self.n_uz,
            nz_eachu=self.nz_eachu,
            shrink_sidegrid=self.shrink_sidegrid,
            fitness_rthresh=self.fitness_rthresh
        )
        return config

    def to_list_fitness(self, indiv):
        """ Convert individual to sample list and fitness. For easier dumping and loading.

        Args:
            indiv (MOOIndiv): Individual.

        Returns:
            sample_list (list): List of sampling point indexes in each grid.
            fitness (float): Fitness of the individual.
        """
        sample_list = list(indiv)
        fitness = indiv.fitness.values
        return sample_list, fitness

    def from_list_fitness(self, sample_list, fitness=None):
        """ Generate individual from sample list and fitness.

        Args:
            sample_list (list): List of sampling point indexes in each grid.
            fitness (float): Fitness.

        Returns:
            indiv (MOOIndiv): Individual.
        """
        indiv = MOOIndiv(sample_list)
        if fitness is not None:
            indiv.fitness.values = fitness
        return indiv

    def indiv_random(self):
        """ Generate individual with random sampling in each grid.

        Returns:
            indiv (MOOIndiv): Individual.
        """
        indiv = MOOIndiv()
        for iu in range(self.n_uz):
            indiv_u = []
            for iv in range(self.n_vxy):
                indiv_uv = np.random.randint(self.grid.uv_size[(iu, iv)])
                indiv_u.append(indiv_uv)
            indiv.append(indiv_u)
        return indiv

    def indiv_uniform(self, index=0):
        """ Generate individual with uniform index in each grid.

        Args:
            index (int): The index to select. Will clip according to grid size.

        Returns:
            indiv (MOOIndiv): Individual.
        """
        indiv = MOOIndiv()
        for iu in range(self.n_uz):
            indiv_u = []
            for iv in range(self.n_vxy):
                indiv_uv = np.clip(index, 0, self.grid.uv_size[(iu, iv)]-1)
                indiv_u.append(indiv_uv)
            indiv.append(indiv_u)
        return indiv
    
    def indiv_middle(self):
        """ Generate individual with the middle point in each grid.

        Returns:
            indiv (MOOIndiv): Individual.
        """
        indiv = MOOIndiv()
        for iu in range(self.n_uz):
            indiv_u = []
            for iv in range(self.n_vxy):
                size_uv = self.grid.uv_size[(iu, iv)]
                index = int((size_uv-1)/2)
                indiv_uv = np.clip(index, 0, size_uv-1)
                indiv_u.append(indiv_uv)
            indiv.append(indiv_u)
        return indiv

    def mutate(self, indiv):
        """ Mutate individual in-place. Randomly resample one of the grids.

        Args:
            indiv (MOOIndiv): Individual.
        Returns: None
        """
        # select one grid to mutate
        iu = np.random.randint(0, self.n_uz)
        iv = np.random.randint(0, self.n_vxy)
        # randomly select one sample from the rest
        indiv[iu][iv] = np.random.randint(self.grid.uv_size[(iu, iv)])

    def get_coord_net(self, indiv):
        """ Get coordinates from individual in a net-shaped form.

        Args:
            indiv (MOOIndiv): Individual.

        Returns:
            zyx_net (np.ndarray): Sample points arranged in a net shape, with shape=(n_uz,n_vxy,3).
        """
        zyx_net = np.zeros((self.n_uz, self.n_vxy, 3))
        for iu in range(self.n_uz):
            for iv in range(self.n_vxy):
                lb_i = indiv[iu][iv]
                zyx_net[iu][iv] = self.grid.uv_zyx[(iu, iv)][lb_i]
        return zyx_net
    
    def flatten_net(self, net):
        """ Flatten net-shaped sampling points.
        
        Args:
            net (np.ndarray): Points with shape=(n_uz, n_vxy, 3).
        Returns: flat
            flat: shape=(n_uz*n_vxy, 3)
        """
        return net.reshape((-1, 3))
  
    def fit_surface_eval(self, indiv, u_eval=None, v_eval=None):
        """ Fit surface from individual, evaluate at net.

        Args:
            indiv (MOOIndiv): Individual.
            u_eval, v_eval (np.ndarray): 1d arrays of u(z) and v(xy) to evaluate at, which range from [0,1]. Defaults to the max length of wireframes.

        Returns:
            zyx_fit (np.ndarray): Evaluated points on the fitted surface, with shape=(npts,3).
            surf_fit (splipy.surface.Surface): Fitted surface.
        """
        # fit
        sample_net = self.get_coord_net(indiv)
        surf_fit = self.surf_meta.interpolate(sample_net)

        # set eval points
        # default: max wireframe length
        if u_eval is None:
            nu_eval = int(np.max(pcdutils.wireframe_length(sample_net, axis=0)))
            u_eval = np.linspace(0, 1, nu_eval)
        if v_eval is None:
            nv_eval = int(np.max(pcdutils.wireframe_length(sample_net, axis=1)))
            v_eval = np.linspace(0, 1, nv_eval)

        # convert fitted surface to binary image
        # evaluate at dense points
        zyx_fit = self.flatten_net(surf_fit(u_eval, v_eval))
        return zyx_fit, surf_fit

    def calc_fitness(self, zyx_fit):
        """ Calculate fitness.

        Args:
            zyx_fit (np.ndarray): Evaluated points on the fitted surface, with shape=(npts,3).
        
        Returns:
            fitness (float): Fitness of the individual.
        """
        # deduplicate to reduce the number of points (in v/xy-direction)
        zyx_fit = pcdutils.points_deduplicate(zyx_fit)
        pcd_fit = pcdutils.points2pointcloud(zyx_fit)
        
        # coverage of zyx by fit
        dist = np.asarray(self.pcd.compute_point_cloud_distance(pcd_fit))
        fitness_coverage = np.sum(np.clip(dist, 0, self.fitness_rthresh)**2)

        # extra pixels of fit compared with zyx
        dist_fit = np.asarray(pcd_fit.compute_point_cloud_distance(self.pcd))
        fitness_extra = np.sum(dist_fit>self.fitness_rthresh)

        # moo fitness
        fitness = (fitness_coverage, fitness_extra)
        return fitness

    def evaluate(self, indiv, u_eval=None, v_eval=None):
        """ Evaluate fitness of individual. Fit surface then calculate fitness.

        Args:
            indiv (MOOIndiv): Individual.
            u_eval, v_eval (np.ndarray): 1d arrays of u(z) and v(xy) to evaluate at, which range from [0,1]. Defaults to the max length of wireframes.

        Returns:
            fitness (float): Fitness of the individual.
        """
        zyx_fit, _ = self.fit_surface_eval(indiv, u_eval=u_eval, v_eval=v_eval)
        fitness = self.calc_fitness(zyx_fit)
        return fitness

