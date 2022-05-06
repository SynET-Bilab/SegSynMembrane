""" single objective optimization: individuals
"""

import numpy as np
import skimage
import deap, deap.base, deap.tools

from etsynseg import utils, bspline
from etsynseg.evomsac import Grid


class MOOFitness(deap.base.Fitness):
    """ DEAP fitness for individuals
    """
    # (coverage penalty, non-membrane penalty)
    weights = (-1, -1)


class MOOIndiv(list):
    """ DEAP individuals
    Instance:
        sampling points on the image, [[pt_u0v0, pt_u0v1,...],[pt_u1v0,...],...]
    """
    def __init__(self, iterable=()):
        super().__init__(iterable)
        self.fitness = MOOFitness()


class MOOTools:
    """ info and tools for individuals
    Usage:
        # setup
        mootools = MOOTools(zyx, n_vxy, n_uz, nz_eachu, degree, fitness_rthresh)
        indiv = mootools.indiv_random()
        # evolution
        mootools.mutate(indiv)
        indiv.fitness.values = mootools.evaluate(indiv)
        # utils
        fit = mootools.fit_surface_interp(zyx)
        fit = mootools.fit_surface_approx(zyx, nctrl_uv=(4,3))
        # save/load
        config = mootools.get_config()
        mootools_new = MOOTools(config=config)
        sample_list, fitness = mootools.to_list_fitness(indiv)
        indiv = mootools.from_list_fitness(sample_list, fitness)
    """

    def __init__(self, zyx=None, n_uz=3, n_vxy=3, nz_eachu=1, shrink_sidegrid=1, fitness_rthresh=1, config=None):
        """ init, setups
        :param zyx: points
        :param n_vxy, n_uz: number of sampling grids in v(xy) and u(z) directions
        :param nz_eachu: number of z-direction slices contained in each grid
        :param shrink_sidegrid: grids close to the side in xy are shrinked to this ratio
        :param fitness_rthresh: distance threshold for fitness evaluation, r_outliers >= fitness_rthresh
        :param config: results from self.get_config()
        """
        # read config
        # if provided as args
        if zyx is not None:
            self.zyx = zyx
            self.n_vxy = n_vxy
            self.n_uz = n_uz
            self.nz_eachu = nz_eachu
            self.shrink_sidegrid = shrink_sidegrid
            self.fitness_rthresh = fitness_rthresh
        # if provided as a dict
        elif config is not None:
            self.zyx = config["zyx"]
            self.n_vxy = config["n_vxy"]
            self.n_uz = config["n_uz"]
            self.nz_eachu = config["nz_eachu"]
            self.shrink_sidegrid = config["shrink_sidegrid"]
            self.fitness_rthresh = config["fitness_rthresh"]
        else:
            raise ValueError("Should provide either B or config")

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
        self.surf_meta = bspline.Surface(
            nu=self.n_uz, nv=self.n_vxy, degree=2
        )
        # pointcloud
        self.pcd = utils.points_to_pointcloud(self.zyx)
    
    def get_config(self):
        """ save config to dict, convenient for dump and load
        :return: config={zyx,shape,n_vxy,n_uz,nz_eachu}
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

    def indiv_random(self):
        """ generate individual with random sampling in each grid
        :return: individual
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
        """ generate individual with uniform index in each grid
        :return: individual
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
        """ generate individual with middle index in each grid
        :return: individual
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
    
    def from_list_fitness(self, sample_list, fitness=None):
        """ generate individual from sample list, fitness
        :param sample_list: list of sampling points on the grid
        :param fitness: scalar fitness
        :return: indiv
        """
        indiv = MOOIndiv(sample_list)
        if fitness is not None:
            indiv.fitness.values = fitness
        return indiv
    
    def to_list_fitness(self, indiv):
        """ convert individual to sample list, fitness
        :param indiv: individual
        :return: sample_list, fitness
        """
        sample_list = list(indiv)
        fitness = indiv.fitness.values
        return sample_list, fitness

    def mutate(self, indiv):
        """ mutate individual in-place, by randomly resample one of the grids
        :return: None
        """
        # select one grid to mutate
        iu = np.random.randint(0, self.n_uz)
        iv = np.random.randint(0, self.n_vxy)
        # randomly select one sample from the rest
        indiv[iu][iv] = np.random.randint(self.grid.uv_size[(iu, iv)])

    def get_coord_net(self, indiv):
        """ get coordinates (net-shaped) from individual
        :param indiv: individual
        :return: zyx_net
            zyx_net: shape=(n_uz, n_vxy, 3)
        """
        zyx_net = np.zeros((self.n_uz, self.n_vxy, 3))
        for iu in range(self.n_uz):
            for iv in range(self.n_vxy):
                lb_i = indiv[iu][iv]
                zyx_net[iu][iv] = self.grid.uv_zyx[(iu, iv)][lb_i]
        return zyx_net
    
    def flatten_net(self, net):
        """ reshape data from net to flat
        :param net: shape=(n_uz, n_vxy, 3)
        :return: flat
            flat: shape=(n_uz*n_vxy, 3)
        """
        return net.reshape((-1, 3))
  
    def fit_surface_eval(self, indiv, u_eval=None, v_eval=None):
        """ fit surface from individual, evaluate at net
        :param indiv: individual
        :param u_eval, v_eval: 1d arrays, u(z) and v(xy) to evaluate at
        :return: zyx_fit, surf_fit
            zyx_fit: evaluated points on the fitted surface
            surf_fit: splipy surface
        """
        # fit
        sample_net = self.get_coord_net(indiv)
        surf_fit = self.surf_meta.interpolate(sample_net)

        # set eval points
        # default: max wireframe length
        if u_eval is None:
            nu_eval = int(np.max(utils.wireframe_lengths(sample_net, axis=0)))
            u_eval = np.linspace(0, 1, nu_eval)
        if v_eval is None:
            nv_eval = int(np.max(utils.wireframe_lengths(sample_net, axis=1)))
            v_eval = np.linspace(0, 1, nv_eval)

        # convert fitted surface to binary image
        # evaluate at dense points
        zyx_fit = self.flatten_net(surf_fit(u_eval, v_eval))
        return zyx_fit, surf_fit

    def calc_fitness(self, zyx_fit):
        """ calculate fitness
        :param zyx_fit: evaluated points on the fitted surface
        """
        pcd_fit = utils.points_to_pointcloud(zyx_fit)
        
        # coverage of zyx by fit
        dist = np.asarray(self.pcd.compute_point_cloud_distance(pcd_fit))
        fitness_coverage = np.sum(np.clip(dist, 0, self.fitness_rthresh)**2)

        # extra pixels of fit compared with zyx
        dist_fit = np.asarray(pcd_fit.compute_point_cloud_distance(self.pcd))
        fitness_extra = np.sum(dist_fit>self.fitness_rthresh)

        # moo fitness
        fitness = (fitness_coverage, fitness_extra)
        return fitness

    def evaluate(self, indiv):
        """ evaluate fitness of individual
        :param indiv: individual
        :return: fitness
        """
        zyx_fit, _ = self.fit_surface_eval(indiv)
        fitness = self.calc_fitness(zyx_fit)
        return fitness

