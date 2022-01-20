#!/usr/bin/env python
""" EvoMSAC
"""

import pickle, functools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.decomposition
import skimage
import deap, deap.base, deap.tools
import geomdl

from synseg.plot import imshow3d
from synseg.utils import mask_to_coord, coord_to_mask

__all__ = [
    "Grid", "EAFitness", "EAIndiv", "IndivMeta", "EAPop",
    "Voxelize"
]

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
    def __init__(self, B=None, config=None, n_vxy=4, n_uz=3, nz_eachu=3, r_thresh=3):
        """ init, setups
        :param B: binary image
        :param n_vxy, n_uz: number of sampling grids in v(xy) and u(z) directions
        :param nz_eachu: number of z-direction slices contained in each grid
        :param degree: degree for NURBS fitting
        :param r_thresh: distance threshold for fitness evaluation
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
        npts_iz = [np.sum(Biz > 0) for Biz in self.B]
        self.npts = np.sum(npts_iz)
        # evalpoints in xy: set to npts_iz or diagonal length
        neval_xy = int(min(np.max(npts_iz),
            np.sqrt(self.shape[1]**2+self.shape[2]**2)
        ))
        self.uv_evallist = np.transpose(np.meshgrid(
            np.linspace(0, 1, self.nz),
            np.linspace(0, 1, neval_xy)
        )).reshape((-1, 2))
        self.ball = {r: skimage.morphology.ball(r)
            for r in range(1, self.r_thresh+1)}
    
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
    
    def from_points(self, points, fitness=None):
        """ generate individual from points, fitness
        :param points: sampling points on the grid
        :param fitness: scalar fitness
        :return: indiv
        """
        indiv = EAIndiv(points)
        if fitness is not None:
            indiv.fitness.values = (fitness,)
        return indiv
    
    def to_points(self, indiv):
        """ convert individual to points, fitness
        :param indiv: individual
        :return: points, fitness
        """
        points = list(indiv)
        fitness = indiv.fitness.values[0]
        return points, fitness

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

    def fit_surface_interp(self, zyx, degree=2):
        """ fitting surface using NURBS interpolation
        :param zyx: sampling points, shape=(n_vxy*n_uz, 3), order: [u0v0,u0v1,...]
        :return: fit
            fit: geomdl surface object
        """
        fit = geomdl.fitting.interpolate_surface(
            zyx,
            degree_u=degree, degree_v=degree,
            size_u=self.n_uz, size_v=self.n_vxy,
            centripetal=True
        )
        return fit
  
    def fit_surface_approx(self, zyx, nctrl_uv=(None, None), degree=2):
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
            degree_u=degree, degree_v=degree,
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
        zyx_surf = fit.evaluate_list(self.uv_evallist)
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
        eap.evolve(n_gen, p_cx, p_mut, dump_step, state_pkl)
        # dump
        eap.dump_state(state_pkl)
        # load
        eap = EAPop(state_pkl=state_pkl)
        # stats and plot
        df_stats = eap.get_log_stats()
        eap.plot_log_stats(xlim, save)
        # surface and plot
        B_arr = get_surfaces([indiv1, indiv2])
        imshow3d(imeta.B, B_arr)
    """
    def __init__(self, imeta=None, state_pkl=None, n_pop=2):
        """ init
        :param imeta: IndivMeta(); if given, init; if None, init later
        """
        # attributes
        self.imeta = None
        self.toolbox = None
        self.stats = None
        self.pop = None
        self.log_stats = None
        self.log_best = None

        # read config
        # if provided imeta
        if imeta is not None:
            self.init_from_imeta(imeta)
            self.n_pop = n_pop
        # if provided state pickle file
        elif state_pkl is not None:
            with open(state_pkl, "rb") as pkl:
                state = pickle.load(pkl)
            imeta = IndivMeta(config=state["imeta_config"])
            self.init_from_imeta(imeta)
            self.n_pop = state["n_pop"]
            if state["pop_points"] is not None:
                self.pop = [self.imeta.from_points(p) for p in state["pop_points"]]
            if state["log_stats"] is not None:
                self.log_stats = state["log_stats"]
            if state["log_best_points"] is not None:
                self.log_best = [self.imeta.from_points(*p) for p in state["log_best_points"]]
        else:
            raise ValueError("Should provide either imeta or state")


    def init_from_imeta(self, imeta):
        """ initialize tools for evolution algorithm
        :param imeta: IndivMeta()
        :return: None
        :action: assign variables imeta, toolbox, stats
        """
        # setup meta
        self.imeta = imeta
        self.toolbox = deap.base.Toolbox()

        # population
        self.toolbox.register('population', deap.tools.initRepeat,
            list, self.imeta.random)
        
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
    
    def dump_state(self, state_pkl):
        """ dump population state ={imeta,pop,log_stats}
        :param state_pkl: name of pickle file to dump
        :return: None
        """
        # convert EAIndiv to points
        pop_points = [self.imeta.to_points(i) for i in self.pop] if (self.pop is not None) else None
        log_best_points = [self.imeta.to_points(i) for i in self.log_best] if (self.log_best is not None) else None
        # collect state
        state = dict(
            imeta_config=self.imeta.get_config(),
            n_pop=self.n_pop,
            pop_points=pop_points,
            log_stats=self.log_stats,
            log_best_points=log_best_points
        )
        with open(state_pkl, "wb") as pkl:
            pickle.dump(state, pkl)

    def register_map(self, func_map):
        """ for applying multiprocessing.Pool().map from __main__
        """
        self.toolbox.register("map", func_map)

    def init_pop(self):
        """ initialize population, logbook, evaluate
        :param n_pop: size of population
        """
        # generation population
        self.pop = self.toolbox.population(self.n_pop)
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

    def evolve_one(self, action):
        """ evolve one generation, perform crossover or mutation
        :param action: select an action, 0=crossover, 1=mutation
        :return: None
        :action: update self.pop, self.log_stats
        """
        # select for variation, copy, shuffle
        offspring = self.toolbox.select(self.pop, self.n_pop)
        offspring = [self.toolbox.clone(i) for i in offspring]
        np.random.shuffle(offspring)
        
        # crossover
        if action == 0:
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values        
        # mutation
        elif action == 1:
            for mutant in offspring:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
        else:
            raise ValueError("action should be 0(crossover) or 1(mutation)")
        
        # update fitness
        n_evals = self.evaluate_pop(offspring)
        
        # select next generation, log
        self.pop = deap.tools.selBest(self.pop+offspring, self.n_pop, fit_attr='fitness')
        self.log_stats.record(n_evals=n_evals, **self.stats.compile(self.pop))
        self.log_best.append(self.toolbox.clone(self.pop[0]))

    def evolve(self, n_gen, dump_step=None, state_pkl=None):
        """ evolve n generations
        :param n_gen: number of generations
        :param dump_step, state_pkl: dump into pkl file (state_pkl) at intervals (dump_step)
        :return: None
        :action: update self.pop, self.log_stats
        """
        for i in range(1, n_gen+1):
            print(i)
            # evolve
            self.evolve_one(action=i%2)
            # dump
            if (dump_step is not None) and (state_pkl is not None):
                # at intervals or at the last step
                if (n_gen%dump_step == 0) or (i == n_gen-1):
                    self.dump_state(state_pkl)

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
