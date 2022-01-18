#!/usr/bin/env python
""" EvoMSAC
"""

import sys, pickle, functools
import multiprocessing
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.decomposition
import skimage
import deap, deap.base, deap.tools
import geomdl

from synseg.plot import imshow3d
from synseg.utils import mask_to_coord, coord_to_mask

#=========================
# auxiliary functions
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
    def __init__(self, B, n_vxy=4, n_uz=3, nz_eachu=1):
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


class Voxelize:
    """ Pixel-precision voxelization of a parameteric surface (from NURBS fitting)
    Usage:
        vox = Voxelize(surface)
        vox.run([0, 1], [0, 1])
        vox.pts
    Attributes:
        pts: coordinates of surface points, [[z1,y1,x1],[z2,y2,x2],...]
        uv: parameters of surface points, [[u1,v1],[u2,v2],...]
    """
    def __init__(self, surface):
        """ init
        """
        self.surface = surface
        self.pts = []
        self.uv = []
        self.evaluate.cache_clear()

    @functools.lru_cache
    def evaluate(self, u, v):
        """ cached evaluation of surface at location (u, v)
        :param u, v: parameteric positions
        :return: coord
            coord: [z,y,x] at (u,v)
        """
        return self.surface.evaluate_single((u, v))

    def run(self, bound_u, bound_v):
        """ recursive rasterization surface in a uv-square
        :param bound_u, bound_v: bounds in u,v directions, e.g. [u_min, u_max]
        :return: None
        :action:
            if corners are not connected: run for finer regions
            if corners are connected: append midpoint to self.pts
        """
        # evaluate corners, calculate differences
        fuv = [[self.evaluate(u, v) for v in bound_v] for u in bound_u]
        diff_v = np.abs(np.diff(fuv, axis=1)).max() > 1
        diff_u = np.abs(np.diff(fuv, axis=0)).max() > 1

        # calculate midpoints
        mid_u = np.mean(bound_u)
        mid_v = np.mean(bound_v)

        # if corners in both u,v directions have gaps
        # divide into 4 sub-regions and run for each
        if diff_v and diff_u:
            self.run([bound_u[0], mid_u], [bound_v[0], mid_v])
            self.run([bound_u[0], mid_u], [mid_v, bound_v[1]])
            self.run([mid_u, bound_u[1]], [bound_v[0], mid_v])
            self.run([mid_u, bound_u[1]], [mid_v, bound_v[1]])
        # if corners in v directions have gaps
        # divide into 2 sub-regions and run for each
        elif diff_v:
            self.run(bound_u, [bound_v[0], mid_v])
            self.run(bound_u, [mid_v, bound_v[1]])
        # if corners in u directions have gaps
        # divide into 2 sub-regions and run for each
        elif diff_u:
            self.run([bound_u[0], mid_u], bound_v)
            self.run([mid_u, bound_u[1]], bound_v)
        # if no corners have gaps
        # append midpoint and its value to results
        else:
            f_mid = np.mean(fuv, axis=(0, 1))
            self.pts.append(f_mid.tolist())
            self.uv.append([mid_u, mid_v])
            return


#=========================
# evolution individual
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
    def __init__(self):
        super().__init__()
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
    """
    def __init__(self, B, n_vxy=4, n_uz=3, nz_eachu=3, degree=2, r_thresh=3):
        """ init, setups
        :param B: binary image
        :param n_vxy, n_uz: number of sampling grids in v(xy) and u(z) directions
        :param nz_eachu: number of z-direction slices contained in each grid
        :param degree: degree for NURBS fitting
        :param r_thresh: distance threshold for fitness evaluation
        """
        # basic info
        self.B = B
        self.shape = B.shape
        self.nz = B.shape[0]

        # sampling and grid
        self.n_vxy = n_vxy
        self.n_uz = n_uz
        self.nz_eachu = nz_eachu
        self.grid = Grid(B, n_vxy=n_vxy, n_uz=n_uz, nz_eachu=nz_eachu)

        # fitness
        self.degree = degree
        npts_xy = [np.sum(Biz > 0) for Biz in B]
        self.npts = np.sum(npts_xy)
        self.vu_evallist = np.transpose(np.meshgrid(
            np.linspace(0, 1, self.nz),
            np.linspace(0, 1, np.max(npts_xy))
        )).reshape((-1, 2))
        self.r_thresh = int(r_thresh)
        self.ball = {r: skimage.morphology.ball(r)
            for r in range(1, self.r_thresh+1)}

    def generate(self):
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

    def get_coord(self, indiv):
        """ get coordinates from individual
        :param indiv: individual
        :return: zyx
            zyx: shape=(n_vxy*n_uz, 3), [[z,y,x]_u0v0,[z,y,x]_u0v1,...]
        """
        zyx = []
        for iu in range(self.n_uz):
            for iv in range(self.n_vxy):
                lb_i = indiv[iu][iv]
                zyx_i = self.grid.uv_zyx[(iu, iv)][lb_i]
                zyx.append(zyx_i)
        return zyx

    def fit_surface_interp(self, zyx):
        """ fitting surface using NURBS interpolation
        :param zyx: sampling points, shape=(n_vxy*n_uz, 3), order: [u0v0,u0v1,...]
        :return: fit
            fit: geomdl surface object
        """
        fit = geomdl.fitting.interpolate_surface(
            zyx,
            degree_u=self.degree, degree_v=self.degree,
            size_u=self.n_uz, size_v=self.n_vxy,
            centripetal=True
        )
        return fit
  
    def fit_surface_approx(self, zyx, nctrl_uv=(None, None)):
        """ fitting surface using NURBS approximation
        :param zyx: sampling points, shape=(n_vxy*n_uz, 3), order: [u0v0,u0v1,...]
        :param nctrl_uv: (ctrlpts_size_u,ctrlpts_size_v)
        :return: fit
            fit: geomdl surface object
        """
        # assign ctrlpts_size if not given
        nctrl_u = nctrl_uv[0] if (nctrl_uv[0] is not None) else (self.n_uz-1)
        nctrl_v = nctrl_uv[1] if (nctrl_uv[1] is not None) else (self.n_vxy-1)
        # fitting
        fit = geomdl.fitting.approximate_surface(
            zyx,
            degree_u=self.degree, degree_v=self.degree,
            size_u=self.n_uz, size_v=self.n_vxy,
            centripetal=True,
            ctrlpts_size_u=nctrl_u, ctrlpts_size_v=nctrl_v
        )
        return fit

    def get_surface(self, indiv):
        """ fit surface from individual, a wrapper of several functions
        :param indiv: individual
        :return: zyx, Bfit
            zyx: coord of individual
            Bfit: rough voxelization of fitted surface
        """
        # nurbs fit
        zyx = self.get_coord(indiv)
        fit = self.fit_surface_interp(zyx)

        # convert fitted surface to binary image
        # evaluate at dense points
        zyx_surf = fit.evaluate_list(self.vu_evallist)
        Bfit = coord_to_mask(zyx_surf, self.shape)
        return zyx, Bfit

    def calc_fitness(self, Bfit):
        """ calculate fitness
        :param Bfit: binary image generated from fitted surface
        """
        # iterate over layers of r's
        fitness = 0
        n_accum = np.sum(self.B * Bfit)  # no. overlaps accumulated
        for r in range(1, self.r_thresh):
            Bfit_r = skimage.morphology.binary_dilation(Bfit, self.ball[r])
            n_r = np.sum(self.B * Bfit_r) - n_accum  # no. overlaps at r
            fitness += n_r * r**2
            n_accum += n_r
        # counting points >= r_thresh
        n_rest = self.npts - n_accum
        fitness += n_rest * self.r_thresh**2
        return fitness

    def evaluate(self, indiv):
        """ evaluate fitness of individual
        :param indiv: individual
        :return: (fitness,)
        """
        _, Bfit = self.get_surface(indiv)
        fitness = self.calc_fitness(Bfit)
        return (fitness,)


#=========================
# evolution population
#=========================

class EAPop:
    """ evolving populations
    Usage:
        # evolve
        imeta = IndivMeta(B, n_vxy, n_uz, nz_eachu, degree, r_thresh)
        eap = EAPop(imeta)
        eap.init_pop(n_pop)
        eap.evolve(n_gen, p_cx, p_mut, dump_step, dump_name)
        # dump
        eap.dump_state(dump_name)
        # load
        eap = EAPop()
        eap.load_state(dump_name)
        # stats and plot
        df_stats = eap.get_log_stats()
        eap.plot_log_stats(xlim, save)
        # surface and plot
        B_arr = get_surfaces([indiv1, indiv2])
        imshow3d(imeta.B, B_arr)
    """
    def __init__(self, indiv_meta=None):
        """ init
        :param indiv_meta: IndivMeta(); if given, init; if None, init later
        """
        # variables
        # tools
        self.imeta = None
        self.toolbox = None
        self.stats = None
        # population
        self.n_pop = None
        self.pop = None
        # log
        self.log_stats = None
        self.log_best = None

        # init deap if given indiv_meta
        if indiv_meta is not None:
            self._init_toolbox(indiv_meta)

    def _init_toolbox(self, indiv_meta):
        """ initialize tools for evolution algorithm
        :param indiv_meta: IndivMeta()
        :return: None
        :action: assign variables imeta, toolbox, stats
        """
        # setup meta
        self.imeta = indiv_meta
        self.toolbox = deap.base.Toolbox()

        # population
        self.toolbox.register('population', deap.tools.initRepeat,
            list, self.imeta.generate)
        
        # operations
        self.toolbox.register("evaluate", self.imeta.evaluate)
        self.toolbox.register("select", deap.tools.selTournament, tournsize=2)
        self.toolbox.register("mate", deap.tools.cxTwoPoint)
        self.toolbox.register("mutate", self.imeta.mutate)

        # stats
        self.stats = deap.tools.Statistics(
            key=lambda indiv: indiv.fitness.values)
        self.stats.register('mean', np.mean)
        self.stats.register('best', np.min)
        self.stats.register('std', np.std)
    
    def register_map(self, func_map):
        """ for applying multiprocessing.Pool().map from __main__
        """
        self.toolbox.register("map", func_map)

    def init_pop(self, n_pop):
        """ initialize population, logbook, evaluate
        :param n_pop: size of population
        """
        # generation population
        self.n_pop = n_pop
        self.pop = self.toolbox.population(n_pop)
        # evaluate, sort, log stats
        self.log_stats = deap.tools.Logbook()
        n_evals = self.evaluate_pop(self.pop)
        self.pop = deap.tools.selBest(self.pop, self.n_pop, fit_attr='fitness')
        self.log_stats.record(n_evals=n_evals, **self.stats.compile(self.pop))
        # log best individual
        self.log_best = [self.toolbox.clone(self.pop[0])]

    def evaluate_pop(self, pop):
        """ evaluate population
        :param pop: list of individuals
        :return: n_evals
            n_evals: number of evaluations
        :action: assigned fitness to individuals
        """
        # find individuals that are not evaluated
        pop_invalid = [indiv for indiv in pop if not indiv.fitness.valid]
        # evaluate
        fit_invalid = self.toolbox.map(self.toolbox.evaluate, pop_invalid)
        for indiv, fit in zip(pop_invalid, fit_invalid):
            indiv.fitness.values = fit
        # number of evaluations
        n_evals = len(pop_invalid)
        return n_evals

    def evolve_one(self, p_cx, p_mut):
        """ evolve one generation
        :param p_cx: probability of crossover
        :param p_mut: probability of mutation
        :return: None
        :action: update self.pop, self.log_stats
        """
        # select for variation, copy, shuffle
        offspring = self.toolbox.select(self.pop, self.n_pop)
        offspring = [self.toolbox.clone(i) for i in offspring]
        np.random.shuffle(offspring)
        
        # crossover
        cx_pairs = list(zip(offspring[::2], offspring[1::2]))
        cx_idxes = np.nonzero(np.random.rand(len(cx_pairs)) < p_cx)[0]
        for i in cx_idxes:
            child1, child2 = cx_pairs[i]
            self.toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
        
        # mutation
        mut_idxes = np.nonzero(np.random.rand(len(offspring)) < p_mut)[0]
        for i in mut_idxes:
            mutant = offspring[i]
            self.toolbox.mutate(mutant)
            del mutant.fitness.values
        
        # update fitness
        n_evals = self.evaluate_pop(offspring)
        
        # select next generation, log
        self.pop = deap.tools.selBest(self.pop+offspring, self.n_pop, fit_attr='fitness')
        self.log_stats.record(n_evals=n_evals, **self.stats.compile(self.pop))
        self.log_best.append(self.toolbox.clone(self.pop[0]))

    def evolve(self, n_gen, p_cx=0.5, p_mut=0.5,
            dump_step=None, dump_name=None
        ):
        """ evolve n generations
        :param n_gen: number of generations
        :param p_cx, p_mut: probability of crossover, mutation
        :param dump_step, dump_name: dump into pkl file (dump_name) at intervals (dump_step)
        :return: None
        :action: update self.pop, self.log_stats
        """
        for i in range(1, n_gen+1):
            # evolve
            self.evolve_one(p_cx=p_cx, p_mut=p_mut)
            # dump
            if dump_step is not None:
                # at intervals or at the last step
                if (n_gen%dump_step == 0) or (i == n_gen-1):
                    self.dump_state(dump_name)
        
    def dump_state(self, dump_name):
        """ dump population state ={imeta,pop,log_stats}
        :param dump_name: name of pickle file to dump
        :return: None
        """
        state = dict(
            indiv_meta=self.imeta,
            pop=self.pop,
            log_stats=self.log_stats,
            log_best=self.log_best
        )
        with open(dump_name, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, dump_name):
        """ load population state ={imeta,pop,log_stats}
        :param dump_name: name of dumped pickle file
        :return: None
        :action: init with imeta, assign self.pop, self.log_stats
        """
        with open(dump_name, "rb") as pkl:
            state = pickle.load(pkl)

        self._init_toolbox(state["indiv_meta"])
        self.pop = state["pop"]
        self.log_stats = state["log_stats"]
        self.log_best = state["log_best"]

    def get_log_stats(self):
        """ get log_stats in DataFrame
        :return: DataFrame(log_stats)
        """
        return pd.DataFrame(self.log_stats)

    def plot_log_stats(self, xlim=(None, None), save=None):
        """ plot log_stats
        :param xlim: xlim for plotting
        :param save: name of figure file to save
        :return: fig, ax
        """
        # convert to dataframe
        df = self.get_log_stats()
        # plot
        fig, ax = plt.subplots()
        ax.plot(df["best"], label="best")
        ax.plot(df["mean"], label="mean")
        ax.legend()
        ax.set(xlabel="generation", ylabel="fitness")
        ax.set(xlim=xlim)
        # save
        if save is not None:
            fig.savefig(save)
        return fig, ax
    
    def get_surfaces(self, indiv_arr):
        """ get surfaces from array of individuals
        :param indiv_arr: array of individuals
        :return: B_arr
            B_arr: [B_sample_0, B_surf_0, B_sample_1, B_surf_1, ...]
        """
        B_arr = []
        for indiv in indiv_arr:
            zyx, B_surf = self.imeta.get_surface(indiv)
            B_sample = coord_to_mask(zyx, self.imeta.shape)
            B_arr.extend([B_sample, B_surf])
        return B_arr


if __name__ == "__main__":
    # setup
    # read args
    args = sys.argv[1:]
    dump_name = args[0]
    n_gen = int(args[1])
    
    # setup
    eap = EAPop()
    eap.load_state(dump_name)

    # run parallel
    pool = multiprocessing.Pool()
    eap.register_map(pool.map)
    eap.init_pop(n_pop=len(eap.pop))
    eap.evolve(n_gen, dump_interval=5, dump_name=dump_name)
    pool.close()
