#!/usr/bin/env python
""" EvoMSAC
"""

import numpy as np
import sklearn.decomposition
import skimage
import sparse
import deap, deap.base, deap.tools

from synseg.utils import mask_to_coord, coord_to_mask
from synseg import bspline

#=========================
# Grid
#=========================

class Grid:
    """ Assigning points into grids
    Usage:
        grid = Grid(B, n_vxy, n_uz, nz_eachu)
        grid.uv_size, grid.uv_zyx
    Attributes:
        uv_size: uv_size[iu][iv] is the number of elements in the grid
        uv_zyx: uv_zyx[iu][iv] is the list of [z,y,x] of points in the grid
    """
    def __init__(self, B, n_vxy, n_uz, nz_eachu):
        """ init and generate grids
        :param B: binary image, shape=(nz,ny,nx)
        :param n_vxy, n_uz: number of sampling grids in v(xy) and u(z) directions
        :param nz_eachu: number of z-direction slices contained in each grid
        """
        # record of inputs
        self.B = B
        self.nz = B.shape[0]
        self.n_vxy = n_vxy
        self.n_uz = n_uz
        self.nz_eachu = nz_eachu

        # info of each bin[iu]: indexes of z, coordinates
        self.ubin_iz = self.get_ubin_iz()
        self.ubin_zyx = self.get_ubin_zyx()

        # info of each grid[iu][iv]: size, coordinates
        self.uv_size, self.uv_zyx = self.get_grid_zyx()

    def get_ubin_iz(self):
        """ for each bin in u(z)-direction, get z-indexes
        :return: ubin_iz
            ubin_iz: shape=(n_uz, nz_eachu), ubin_iz[iu]=(z1,z2,z3,...)
        """
        # interval between u-bins
        # used np.array_split to deal with uneven intervals
        interval_size = (self.nz - self.n_uz*self.nz_eachu)
        interval_arr = np.array_split(np.ones(interval_size), self.n_uz-1)

        # z-indexes for each u-bin
        ubin_iz = []
        iz = 0
        for i in range(self.n_uz-1):
            # take samples
            ubin_iz.append(tuple(range(iz, iz+self.nz_eachu)))
            # skip interval
            iz += self.nz_eachu + len(interval_arr[i])
        # take the last bin
        ubin_iz.append(tuple(range(self.nz-self.nz_eachu, self.nz)))

        ubin_iz = tuple(ubin_iz)
        return ubin_iz

    def get_ubin_zyx(self):
        """ for each bin and slice in u(z)-direction, get coordinates sorted by PC
        :return: ubin_zyx
            ubin_zyx: shape=(n_uz, nz_eachu, n_points, 3), ubin_zyx[iu][i]=((z1,y1,x1),...)
        """
        # pca of all points
        zyx_all = mask_to_coord(self.B)
        pca = sklearn.decomposition.PCA()
        pca.fit(zyx_all[:, 1:])

        # collect zyx in each u-bin
        ubin_zyx = []
        for iu in range(self.n_uz):
            zyx_iu = []
            # sort zyx for each iz
            for iz in self.ubin_iz[iu]:
                # project yx to pc
                yx_iz = mask_to_coord(self.B[iz])
                pc1 = pca.transform(yx_iz)[:, 0]
                # sort by pc1, get zyx
                idx_sorted = np.argsort(pc1)
                zyx_iz = tuple(
                    tuple([iz, *yx_iz[i]]) for i in idx_sorted
                )
                zyx_iu.append(zyx_iz)
            ubin_zyx.append(tuple(zyx_iu))
        ubin_zyx = tuple(ubin_zyx)
        return ubin_zyx

    def get_grid_zyx(self):
        """ for each uv-grid, get size and coordinates
        :return: uv_size, uv_zyx
            uv_size: uv_size[(iu,iv)]=n_uv, number of points in the grid
            uv_zyx: uv_zyx[(iu,iv)]=((z1,y1,x1),...), coordinates in the grid
        """
        # initialize
        uv_zyx = {
            (iu, iv): []
            for iu in range(self.n_uz)
            for iv in range(self.n_vxy)
        }

        # populate each grid
        for iu in range(self.n_uz):
            # distribute points in each z-slice into grids in v-direction
            for zyx_iz in self.ubin_zyx[iu]:
                idx_iz = np.arange(len(zyx_iz))
                # use array split to deal with uneven splits
                idx_split = np.array_split(idx_iz, self.n_vxy)
                for iv in range(self.n_vxy):
                    zyx_iz_iv = [zyx_iz[i] for i in idx_split[iv]]
                    uv_zyx[(iu, iv)].extend(zyx_iz_iv)

        # convert to dict of tuples
        uv_zyx = {k: tuple(v) for k, v in uv_zyx.items()}
        # get size
        uv_size = {k: len(v) for k, v in uv_zyx.items()}
        return uv_size, uv_zyx


#=========================
# Individual
#=========================

class EAFitness(deap.base.Fitness):
    """ DEAP fitness for individuals
    """
    # weights < 0 for minimization problem
    weights = (-1, )


class EAIndiv(list):
    """ DEAP individuals
    Instance:
        sampling points on the image, [[pt_u0v0, pt_u0v1,...],[pt_u1v0,...],...]
    """
    def __init__(self, iterable=()):
        super().__init__(iterable)
        self.fitness = EAFitness()


class IndivMeta:
    """ info and tools for individuals
    Usage:
        # setup
        imeta = IndivMeta(B, n_vxy, n_uz, nz_eachu, degree, r_thresh)
        indiv = imeta.generate()
        # evolution
        imeta.mutate(indiv)
        indiv.fitness.values = imeta.evaluate(indiv)
        # utils
        fit = imeta.fit_surface_interp(zyx)
        fit = imeta.fit_surface_approx(zyx, nctrl_uv=(4,3))
        # save/load
        config = imeta.get_config()
        imeta_new = IndivMeta(config=config)
    """
    def __init__(self, B=None, config=None, n_vxy=4, n_uz=3, nz_eachu=3, r_thresh=2):
        """ init, setups
        :param B: binary image
        :param n_vxy, n_uz: number of sampling grids in v(xy) and u(z) directions
        :param nz_eachu: number of z-direction slices contained in each grid
        :param degree: degree for NURBS fitting
        :param r_thresh: distance threshold for fitness evaluation, r_outliers >= r_thresh+1
        """
        # read config
        # if provided as args
        if B is not None:
            self.B = B
            self.n_vxy = n_vxy
            self.n_uz = n_uz
            self.nz_eachu = nz_eachu
            self.r_thresh = int(r_thresh)
        # if provided as a dict
        elif config is not None:
            self.B = coord_to_mask(config["zyx"], config["shape"])
            self.n_vxy = config["n_vxy"]
            self.n_uz = config["n_uz"]
            self.nz_eachu = config["nz_eachu"]
            self.r_thresh = int(config["r_thresh"])
        else:
            raise ValueError("Should provide either B or config")

        # image shape, grid
        self.shape = self.B.shape
        self.nz = self.B.shape[0]
        self.grid = Grid(self.B, n_vxy=self.n_vxy,
            n_uz=self.n_uz, nz_eachu=self.nz_eachu)

        # fitness
        # bspline
        self.surf_meta = bspline.Surface(
            uv_size=(self.n_uz, self.n_vxy), degree=2
        )
        # no. points in the image
        npts_iz = [np.sum(Biz > 0) for Biz in self.B]
        self.npts_B = np.sum(npts_iz)
        # evalpts in xy: set to npts_iz or diagonal length
        neval_xy = int(min(np.max(npts_iz),
            np.sqrt(self.shape[1]**2+self.shape[2]**2)
        ))
        # default evalpts
        self.u_eval_default = np.linspace(0, 1, self.nz)
        self.v_eval_default = np.linspace(0, 1, neval_xy)
        # image dilated at r's: in sparse format
        self.dilate_Br = {0: sparse.COO(self.B)}
        for r in range(1, self.r_thresh+1):
            dilate_Br_r = skimage.morphology.binary_dilation(
                self.B, skimage.morphology.ball(r)
            ).astype(int)
            self.dilate_Br[r] = sparse.COO(dilate_Br_r)
    
    def get_config(self):
        """ save config to dict, convenient for dump and load
        :return: config={zyx,shape,n_vxy,n_uz,nz_eachu}
        """
        config = dict(
            zyx=mask_to_coord(self.B),
            shape=self.shape,
            n_vxy=self.n_vxy,
            n_uz=self.n_uz,
            nz_eachu=self.nz_eachu,
            r_thresh=self.r_thresh
        )
        return config

    def random(self):
        """ generate individual, by random sampling in each grid
        :return: individual
        """
        indiv = EAIndiv()
        for iu in range(self.n_uz):
            indiv_u = []
            for iv in range(self.n_vxy):
                indiv_uv = np.random.randint(self.grid.uv_size[(iu, iv)])
                indiv_u.append(indiv_uv)
            indiv.append(indiv_u)
        return indiv
    
    def from_list_fitness(self, sample_list, fitness=None):
        """ generate individual from sample list, fitness
        :param sample_list: list of sampling points on the grid
        :param fitness: scalar fitness
        :return: indiv
        """
        indiv = EAIndiv(sample_list)
        if fitness is not None:
            indiv.fitness.values = (fitness,)
        return indiv
    
    def to_list_fitness(self, indiv):
        """ convert individual to sample list, fitness
        :param indiv: individual
        :return: sample_list, fitness
        """
        sample_list = list(indiv)
        fitness = indiv.fitness.values[0]
        return sample_list, fitness

    def mutate(self, indiv):
        """ mutate individual in-place, by randomly resample one of the grids
        :return: None
        """
        # select one grid to mutate
        iu = np.random.randint(0, self.n_uz)
        iv = np.random.randint(0, self.n_vxy)
        # randomly select one sample from the rest
        indiv_uv_new = np.random.randint(self.grid.uv_size[(iu, iv)])
        indiv[iu][iv] = indiv_uv_new

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
        :return: Bfit, fit
            Bfit: rough voxelization of fitted surface
            fit: splipy surface
        """
        # reading args
        u_eval = u_eval if (u_eval is not None) else self.u_eval_default
        v_eval = v_eval if (v_eval is not None) else self.v_eval_default

        # nurbs fit
        sample_net = self.get_coord_net(indiv)
        fit = self.surf_meta.interpolate(sample_net)

        # convert fitted surface to binary image
        # evaluate at dense points
        pts_fit = self.flatten_net(fit(u_eval, v_eval))
        Bfit = coord_to_mask(pts_fit, self.shape)
        return Bfit, fit

    def calc_fitness(self, Bfit):
        """ calculate fitness
        :param Bfit: binary image generated from fitted surface
        """
        # iterate over layers of r's
        fitness = 0
        Bfit_sparse = sparse.COO(Bfit)
        n_accum_prev = np.sum(self.dilate_Br[0] * Bfit_sparse)  # no. overlaps accumulated
        for r in range(1, self.r_thresh+1):
            n_accum_curr = np.sum(self.dilate_Br[r] * Bfit_sparse)
            n_r = n_accum_curr - n_accum_prev  # no. overlaps at r
            fitness += n_r * r**2
            n_accum_prev = n_accum_curr
        # counting points >= r_thresh
        n_rest = self.npts_B - n_accum_prev
        fitness += n_rest * (self.r_thresh+1)**2
        return fitness

    def evaluate(self, indiv):
        """ evaluate fitness of individual
        :param indiv: individual
        :return: (fitness,)
        """
        Bfit, _ = self.fit_surface_eval(indiv)
        fitness = self.calc_fitness(Bfit)
        return (fitness,)

