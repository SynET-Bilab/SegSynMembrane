# """ single objective optimization: individuals
# """

# import numpy as np
# import skimage
# import sparse
# import deap, deap.base, deap.tools

# from TomoSynSegAE.utils import mask_to_coord, coord_to_mask
# from TomoSynSegAE import bspline
# from TomoSynSegAE.evomsac import Grid


# class MOOFitness(deap.base.Fitness):
#     """ DEAP fitness for individuals
#     """
#     # (overlap penalty, non-membrane penalty)
#     weights = (-1, -1)


# class MOOIndiv(list):
#     """ DEAP individuals
#     Instance:
#         sampling points on the image, [[pt_u0v0, pt_u0v1,...],[pt_u1v0,...],...]
#     """
#     def __init__(self, iterable=()):
#         super().__init__(iterable)
#         self.fitness = MOOFitness()


# class MOOTools:
#     """ info and tools for individuals
#     Usage:
#         # setup
#         mootools = MOOTools(B, n_vxy, n_uz, nz_eachu, degree, r_thresh)
#         indiv = mootools.random()
#         # evolution
#         mootools.mutate(indiv)
#         indiv.fitness.values = mootools.evaluate(indiv)
#         # utils
#         fit = mootools.fit_surface_interp(zyx)
#         fit = mootools.fit_surface_approx(zyx, nctrl_uv=(4,3))
#         # save/load
#         config = mootools.get_config()
#         mootools_new = MOOTools(config=config)
#         sample_list, fitness = mootools.to_list_fitness(indiv)
#         indiv = mootools.from_list_fitness(sample_list, fitness)
#     """

#     def __init__(self, B=None, Bref=None, n_uz=3, n_vxy=4, nz_eachu=1, r_thresh=1, config=None):
#         """ init, setups
#         :param B: binary image for sampling
#         :param Bref: reference binary image for calculating fitness, default=B
#         :param n_vxy, n_uz: number of sampling grids in v(xy) and u(z) directions
#         :param nz_eachu: number of z-direction slices contained in each grid
#         :param r_thresh: distance threshold for fitness evaluation, r_outliers >= r_thresh+1
#         :param config: results from self.get_config()
#         """
#         # read config
#         # if provided as args
#         if B is not None:
#             self.B = B
#             self.Bref = Bref if (Bref is not None) else B
#             self.n_vxy = n_vxy
#             self.n_uz = n_uz
#             self.nz_eachu = nz_eachu
#             self.r_thresh = int(r_thresh)
#         # if provided as a dict
#         elif config is not None:
#             self.B = coord_to_mask(config["zyx"], config["shape"])
#             if "zyx_ref" not in config:
#                 config["zyx_ref"] = config["zyx"]
#             self.Bref = coord_to_mask(config["zyx_ref"], config["shape"])
#             self.n_vxy = config["n_vxy"]
#             self.n_uz = config["n_uz"]
#             self.nz_eachu = config["nz_eachu"]
#             self.r_thresh = int(config["r_thresh"])
#         else:
#             raise ValueError("Should provide either B or config")

#         # image shape, grid
#         self.shape = self.B.shape
#         self.nz = self.B.shape[0]
#         self.grid = Grid(self.B, n_vxy=self.n_vxy, n_uz=self.n_uz,
#             nz_eachu=self.nz_eachu)
#         # update nv, nu from grid: where there are additional checks
#         self.n_vxy = self.grid.n_vxy
#         self.n_uz = self.grid.n_uz

#         # fitness
#         # bspline
#         self.surf_meta = bspline.Surface(
#             uv_size=(self.n_uz, self.n_vxy), degree=2
#         )
#         # no. points in the reference image
#         npts_iz = np.sum(self.Bref, axis=(1, 2))
#         self.npts_Bref = np.sum(npts_iz)
#         # default evalpts
#         self.u_eval_default = np.linspace(0, 1, self.nz)
#         self.v_eval_default = np.linspace(0, 1, np.max(npts_iz))
#         # reference image dilated at r's: in sparse format
#         self.dilated_Bref = {0: sparse.COO(self.Bref)}
#         for r in range(1, self.r_thresh+1):
#             dilated_Bref_r = skimage.morphology.binary_dilation(
#                 self.B, skimage.morphology.ball(r)
#             ).astype(int)
#             self.dilated_Bref[r] = sparse.COO(dilated_Bref_r)
    
#     def get_config(self):
#         """ save config to dict, convenient for dump and load
#         :return: config={zyx,shape,n_vxy,n_uz,nz_eachu}
#         """
#         config = dict(
#             zyx=mask_to_coord(self.B),
#             zyx_ref=mask_to_coord(self.Bref),
#             shape=self.shape,
#             n_vxy=self.n_vxy,
#             n_uz=self.n_uz,
#             nz_eachu=self.nz_eachu,
#             r_thresh=self.r_thresh
#         )
#         return config

#     def random(self):
#         """ generate individual, by random sampling in each grid
#         :return: individual
#         """
#         indiv = MOOIndiv()
#         for iu in range(self.n_uz):
#             indiv_u = []
#             for iv in range(self.n_vxy):
#                 indiv_uv = np.random.randint(self.grid.uv_size[(iu, iv)])
#                 indiv_u.append(indiv_uv)
#             indiv.append(indiv_u)
#         return indiv

#     def uniform(self, index=0):
#         """ generate individual, by uniform index in each grid
#         :return: individual
#         """
#         indiv = MOOIndiv()
#         for iu in range(self.n_uz):
#             indiv_u = []
#             for iv in range(self.n_vxy):
#                 indiv_uv = np.clip(index, 0, self.grid.uv_size[(iu, iv)])
#                 indiv_u.append(indiv_uv)
#             indiv.append(indiv_u)
#         return indiv
    
#     def from_list_fitness(self, sample_list, fitness=None):
#         """ generate individual from sample list, fitness
#         :param sample_list: list of sampling points on the grid
#         :param fitness: scalar fitness
#         :return: indiv
#         """
#         indiv = MOOIndiv(sample_list)
#         if fitness is not None:
#             indiv.fitness.values = fitness
#         return indiv
    
#     def to_list_fitness(self, indiv):
#         """ convert individual to sample list, fitness
#         :param indiv: individual
#         :return: sample_list, fitness
#         """
#         sample_list = list(indiv)
#         fitness = indiv.fitness.values
#         return sample_list, fitness

#     def mutate(self, indiv):
#         """ mutate individual in-place, by randomly resample one of the grids
#         :return: None
#         """
#         # select one grid to mutate
#         iu = np.random.randint(0, self.n_uz)
#         iv = np.random.randint(0, self.n_vxy)
#         # randomly select one sample from the rest
#         indiv[iu][iv] = np.random.randint(self.grid.uv_size[(iu, iv)])

#     def get_coord_net(self, indiv):
#         """ get coordinates (net-shaped) from individual
#         :param indiv: individual
#         :return: zyx_net
#             zyx_net: shape=(n_uz, n_vxy, 3)
#         """
#         zyx_net = np.zeros((self.n_uz, self.n_vxy, 3))
#         for iu in range(self.n_uz):
#             for iv in range(self.n_vxy):
#                 lb_i = indiv[iu][iv]
#                 zyx_net[iu][iv] = self.grid.uv_zyx[(iu, iv)][lb_i]
#         return zyx_net
    
#     def flatten_net(self, net):
#         """ reshape data from net to flat
#         :param net: shape=(n_uz, n_vxy, 3)
#         :return: flat
#             flat: shape=(n_uz*n_vxy, 3)
#         """
#         return net.reshape((-1, 3))
  
#     def fit_surface_eval(self, indiv, u_eval=None, v_eval=None):
#         """ fit surface from individual, evaluate at net
#         :param indiv: individual
#         :param u_eval, v_eval: 1d arrays, u(z) and v(xy) to evaluate at
#         :return: Bfit, fit
#             Bfit: rough voxelization of fitted surface
#             fit: splipy surface
#         """
#         # reading args
#         u_eval = u_eval if (u_eval is not None) else self.u_eval_default
#         v_eval = v_eval if (v_eval is not None) else self.v_eval_default

#         # nurbs fit
#         sample_net = self.get_coord_net(indiv)
#         fit = self.surf_meta.interpolate(sample_net)

#         # convert fitted surface to binary image
#         # evaluate at dense points
#         pts_fit = self.flatten_net(fit(u_eval, v_eval))
#         Bfit = coord_to_mask(pts_fit, self.shape)
#         return Bfit, fit

#     def calc_fitness(self, Bfit):
#         """ calculate fitness
#         :param Bfit: binary image generated from fitted surface
#         """
#         # overlaps between fit and membrane
#         # iterate over layers of r's
#         fitness_overlap = 0
#         Bfit_sparse = sparse.COO(Bfit)
#         n_accum_prev = np.sum(self.dilated_Bref[0] * Bfit_sparse)  # no. overlaps accumulated
#         for r in range(1, self.r_thresh+1):
#             n_accum_curr = np.sum(self.dilated_Bref[r] * Bfit_sparse)
#             n_r = n_accum_curr - n_accum_prev  # no. overlaps at r
#             fitness_overlap += n_r * r**2
#             n_accum_prev = n_accum_curr
#         # counting points >= r_thresh
#         n_rest = self.npts_Bref - n_accum_prev
#         fitness_overlap += n_rest * (self.r_thresh+1)**2

#         # extra pixels of fit compared with membrane
#         fitness_extra = np.sum(Bfit_sparse) - n_accum_prev

#         # moo fitness
#         fitness = (fitness_overlap, fitness_extra)
#         return fitness

#     def evaluate(self, indiv):
#         """ evaluate fitness of individual
#         :param indiv: individual
#         :return: (fitness,)
#         """
#         Bfit, _ = self.fit_surface_eval(indiv)
#         fitness = self.calc_fitness(Bfit)
#         return fitness

