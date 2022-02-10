""" workflow
"""

import numpy as np
from synseg import io, utils, plot, trace
from synseg import hessian, dtvoting, nonmaxsup
from synseg import division, evomsac, matching

# __all__ = [
#     "MemDetect"
# ]

class Workflow:
    """ workflow for segmentation
    attribute formats:
        binary images: save coordinates (utils.mask_to_coord, self.coord_to_binary)
        sparse images (O): save as sparse (utils.sparsify3d, utils.densify3d)
        MOOPop: save state (MOOPop().dump_state, MOOPop(state=state))
    
    workflow:
        # setup
        tssa = synseg.workflow.Workflow()
        tssa.read_tomo(tomo_file, model_file, obj_bound, obj_ref, voxel_size_nm=None)
        # steps
        tssa.detect(factor_tv=1, factor_supp=5, qfilter=0.25)
        tssa.divide(size_ratio_thresh=0.5)
        tssa.evomsac(grid_z_nm=50, grid_xy_nm=150)
        tssa.match(factor_tv=1)
        tssa.surf_normal()
        tssa.surf_fit(grid_z_nm=10, grid_xy_nm=10)
        # io
        tssa.save_steps(filename)
        tssa.output_results(filenames=dict(tomo, match, plot, surf_normal, surf_fit), nslice=5)
    """
    def __init__(self):
        # record of parameters
        self.steps = dict(
            tomo=dict(
                finished=False,
                # parameters
                tomo_file=None,
                model_file=None,
                obj_bound=None,
                obj_ref=None,
                d_mem_nm=None,
                d_cleft_nm=None,
                # results
                I=None,
                shape=None,
                voxel_size_nm=None,
                clip_range=None,
                mask_bound=None,
                zyx_ref=None,
                d_mem_vx=None,
                d_cleft_vx=None,
            ),
            detect=dict(
                finished=False,
                # parameters
                factor_tv=None,
                factor_supp=None,
                qfilter=None,
                # results
                zyx=None,
                Oz=None
            ),
            divide=dict(
                finished=False,
                # parameters
                size_ratio_thresh=None,
                # results
                zyx1=None,
                zyx2=None,
            ),
            evomsac=dict(
                finished=False,
                # parameters
                grid_z_nm=None,
                grid_xy_nm=None,
                pop_size=None,
                max_iter=None,
                tol=None,
                # results
                mpop1z=None,
                mpop2z=None
            ),
            match=dict(
                finished=False,
                # parameters
                factor_tv=None,
                # results
                zyx1=None,
                zyx2=None,
            ),
            surf_normal=dict(
                finished=False,
                # results
                normal1=None,
                normal2=None,
            ),
            surf_fit=dict(
                finished=False,
                # parameters
                grid_z_nm=None,
                grid_xy_nm=None,
                # results
                zyx1=None,
                zyx2=None,
            )
        )

    #=========================
    # io
    #=========================

    def save_steps(self, filename):
        """ save steps to npz
        :param filename: filename to save to
        """
        np.savez_compressed(filename, **self.steps)
    
    def load_steps(self, filename):
        """ load steps from npz
        :param filename: filename to load from
        """
        steps = np.load(filename, allow_pickle=True)
        self.steps = {key: steps[key].item() for key in steps.keys()}
    
    def output_results(self, filenames, nslice=5):
        """ output results
        :param filenames: dict(tomo(mrc), match(mod), plot(png), surf_normal(npz), surf_fit(mod))
        """
        if ("tomo" in filenames) and self.check_steps(["tomo"]):
            io.write_mrc(
                data=self.steps["tomo"]["I"],
                mrcname=filenames["tomo"],
                voxel_size=self.steps["tomo"]["voxel_size_nm"]*10
            )
        
        if ("match" in filenames) and self.check_steps(["match"]):
            io.write_model(
                zyx_arr=[self.steps["match"][f"zyx{i}"] for i in (1, 2)],
                model_file=filenames["match"]
            )
        
        if ("plot" in filenames) and self.check_steps(["tomo", "match"]):
            fig, _ = self.plot_slices(
                I=-self.steps["tomo"]["I"],  # negated
                zyxs=tuple(self.steps["match"][f"zyx{i}"] for i in (1, 2)),
                nslice=nslice
            )
            fig.savefig(filenames["plot"], dpi=200)
        
        if ("surf_normal" in filenames) and self.check_steps(["match", "surf_normal"]):
            # coordinates: xyz(i) + xyz_shift = xyz in the original tomo
            np.savez(
                filenames["surf_normal"],
                xyz_shift=np.array([
                    self.steps["tomo"]["clip_range"][i][0]
                    for i in ['x', 'y', 'z']]),
                xyz1=utils.reverse_coord(self.steps["match"]["zyx1"]),
                xyz2=utils.reverse_coord(self.steps["match"]["zyx2"]),
                normal1=self.steps["surf_normal"]["normal1"],
                normal2=self.steps["surf_normal"]["normal2"],
            )
        
        if ("surf_fit" in filenames) and self.check_steps(["surf_fit"]):
            io.write_model(
                zyx_arr=[self.steps["surf_fit"][f"zyx{i}"] for i in (1, 2)],
                model_file=filenames["surf_fit"]
            )
    
    def plot_slices(self, I, zyxs, nslice):
        """ plot sampled slices of image
        :param I: 3d image
        :param zyxs: array of zyx to overlay on the image
        :param nslice: number of slices to show
        :return: fig, axes
        """
        izs = np.linspace(0, I.shape[0]-1, nslice, dtype=int)
        im_dict = {
            f"z={iz}": {
                "I": I[iz],
                "yxs": tuple(zyx_i[zyx_i[:, 0] == iz][:, 1:]
                    for zyx_i in zyxs)
            }
            for iz in izs
        }
        fig, axes = plot.imoverlay(im_dict)
        return fig, axes



    #=========================
    # auxiliary
    #=========================
    
    def check_steps(self, steps_prev, raise_error=False):
        """ raise error if any prerequisite steps is not finished
        :param steps_prev: array of names of prerequisite steps
        :param raise_error: if raise error when prerequisites are not met
        """
        satisfied = True
        for step in steps_prev:
            if not self.steps[step]["finished"]:
                if raise_error:
                    raise RuntimeError(f"unsatisfied prerequisite step: {step}")
                satisfied = False
        return satisfied

    def coord_to_binary(self, zyx):
        """ convert zyx to binary image
        :param zyx: coordinates
        """
        self.check_steps(["tomo"], raise_error=True)
        shape = self.steps["tomo"]["shape"]
        B = utils.coord_to_mask(zyx, shape=shape)
        return B
        
    def set_ngrids(self, B, voxel_size_nm, grid_z_nm, grid_xy_nm):
        """ setting number of grids for evomsac and fitting
        :param B: binary image
        :param voxel_size_nm: voxel spacing in nm
        :param grid_z_nm, grid_xy_nm: grid spacing in z, xy
        :return: n_uz, n_vxy
        """
        # auxiliary
        def set_one(span_vx, grid_nm, n_min):
            span_nm = span_vx * voxel_size_nm
            n_grid = int(np.round(span_nm/grid_nm)+1)
            return max(n_min, n_grid)

        # grids in z
        nz = B.shape[0]
        n_uz = set_one(nz, grid_z_nm, 3)

        # grids in xy
        dydx = utils.spans_xy(B)
        l_xy = np.median(np.linalg.norm(dydx, axis=1))
        n_vxy = set_one(l_xy, grid_xy_nm, 3)
        
        return n_uz, n_vxy

    #=========================
    # read tomo
    #=========================
    
    def read_tomo(self, tomo_file, model_file,
        obj_bound=1, obj_ref=2, voxel_size_nm=None,
        d_mem_nm=5, d_cleft_nm=20
        ):
        """ load and clip tomo and model
        :param tomo_file, model_file: filename of tomo, model
        :param obj_bound, obj_ref: obj label for boundary and presynapse, begins with 1
        :param voxel_size_nm: manually set; if None then read from tomo_file
        :action: assign steps["tomo"]: I, voxel_size_nm, mask_bound, zyx_ref, d_mem_vx, d_cleft_vx
        """
        # read model
        model = io.read_model(model_file)
        
        # set the range of clipping
        # use np.floor/ceil -> int to ensure integers
        model_bound = model[model["object"] == obj_bound]
        clip_range = {
            i: (int(np.floor(model_bound[i].min())),
                int(np.ceil(model_bound[i].max())))
            for i in ["x", "y", "z"]
        }

        # read tomo, clip to bound
        I, voxel_size_A = io.read_clip_tomo(
            tomo_file, clip_range
        )
        I = np.asarray(I)

        # voxel_size in nm
        if voxel_size_nm is None:
            # assumed voxel_size_A is the same for x,y,z
            voxel_size_nm = voxel_size_A.tolist()[0] / 10

        # shift model coordinates
        for i in ["x", "y", "z"]:
            model[i] -= clip_range[i][0]

        # generate mask for bound (after clipping)
        mask_bound = io.model_to_mask(
            model=model[model["object"] == obj_bound],
            yx_shape=I[0].shape
        )

        # get coordinates of presynaptic label
        series_ref = model[model["object"] == obj_ref].iloc[0]
        zyx_ref = np.array(
            [series_ref[i] for i in ["z", "y", "x"]]
        )

        # save parameters and results
        self.steps["tomo"].update(dict(
            finished=True,
            # parameters
            tomo_file=tomo_file,
            model_file=model_file,
            obj_bound=obj_bound,
            obj_ref=obj_ref,
            d_mem_nm=d_mem_nm,
            d_cleft_nm=d_cleft_nm,
            # results
            I=I,
            shape=I.shape,
            voxel_size_nm=voxel_size_nm,
            clip_range=clip_range,
            mask_bound=mask_bound,
            zyx_ref=zyx_ref,
            d_mem_vx=d_mem_nm/voxel_size_nm,
            d_cleft_vx=d_cleft_nm/voxel_size_nm,
        ))

    #=========================
    # detect
    #=========================
    
    def detect(self, factor_tv=1, factor_supp=5, qfilter=0.25):
        """ detect membrane features
        :param factor_tv: sigma for tv = factor_tv*d_cleft_vx
        :param factor_supp: sigma for normal suppression = factor_supp*d_cleft_vx
        :param qfilter: quantile to filter out by Stv and Ssupp
        :action: assign steps["detect"]: B, O
        """
        # load from self
        self.check_steps(["tomo"], raise_error=True)
        I = self.steps["tomo"]["I"]
        mask_bound = self.steps["tomo"]["mask_bound"]
        d_mem_vx = self.steps["tomo"]["d_mem_vx"]
        d_cleft_vx = self.steps["tomo"]["d_cleft_vx"]

        # negate, hessian
        Ineg = utils.negate_image(I)
        S, O = hessian.features3d(Ineg, sigma=d_mem_vx)
        
        # tv, nms
        Stv, Otv = dtvoting.stick3d(
            S*mask_bound,
            O*mask_bound,
            sigma=d_cleft_vx*factor_tv
        )
        Btv = mask_bound*nonmaxsup.nms3d(Stv, Otv)
        
        # refine: orientation, nms, orientation changes
        _, Oref = hessian.features3d(Btv, sigma=d_mem_vx)
        Bref = nonmaxsup.nms3d(Stv*Btv, Oref*Btv)
        
        # normal suppression
        Bsupp, Ssupp = dtvoting.suppress_by_orient(
            Bref, Oref*Bref,
            sigma=d_cleft_vx*factor_supp,
            dO_threshold=np.pi/4
        )

        # filter in xy: small Stv, small Ssup
        Bfilt_xy = utils.filter_connected_xy(
            Bsupp, [Stv, Ssupp],
            connectivity=2, stats="median",
            qfilter=qfilter, min_size=1
        )

        # filter in 3d: z
        # z spans a large fraction
        dzfilter = int(I.shape[0]*0.75)
        Bfilt_dz = utils.filter_connected_dz(
            Bfilt_xy, dzfilter=dzfilter, connectivity=3)

        # settle detected
        Bdetect = Bfilt_dz
        Odetect = Oref*Bdetect

        # save parameters and results
        self.steps["detect"].update(dict(
            finished=True,
            # parameters
            factor_tv=factor_tv,
            factor_supp=factor_supp,
            qfilter=qfilter,
            # results
            zyx=utils.mask_to_coord(Bdetect),
            Oz=utils.sparsify3d(Odetect)
        ))
    
    #=========================
    # divide
    #=========================
    
    def divide(self, size_ratio_thresh=0.5):
        """ divide detected image into pre-post candidates
        :param size_ratio_thresh: if size2/size1<size_ratio_thresh, consider that membranes are connected
        :action: assign steps["divide"]: B1, B2
        """
        # load from self
        self.check_steps(["tomo", "detect"], raise_error=True)
        d_mem_vx = self.steps["tomo"]["d_mem_vx"]
        d_cleft_vx = self.steps["tomo"]["d_cleft_vx"]
        zyx_ref = self.steps["tomo"]["zyx_ref"]
        B = self.coord_to_binary(self.steps["detect"]["zyx"])
        O = utils.densify3d(self.steps["detect"]["Oz"])

        # extract two largest components
        comp_arr = list(utils.extract_connected(
            B, n_keep=2, connectivity=3
        ))
        size1_raw, comp1_raw = comp_arr[0]
        size2_raw, comp2_raw = comp_arr[1]

        # if two membranes seem connected, divide
        size_ratio = size2_raw / size1_raw
        if size_ratio > size_ratio_thresh:
            comp1 = comp1_raw
            comp2 = comp2_raw
        else:
            comps_div = division.divide_connected(
                comp1_raw, O*comp1_raw,
                seg_max_size=max(1, int(d_cleft_vx)),
                seg_neigh_thresh=max(1, int(d_mem_vx)),
                n_clusters=2
            )
            comp1, comp2 = comps_div[:2]
        
        # compare components' distance to ref
        iz_ref = int(zyx_ref[0])
        yx1 = utils.mask_to_coord(comp1[iz_ref])
        yx2 = utils.mask_to_coord(comp2[iz_ref])
        dist1 = np.sum((yx1 - zyx_ref[1:])**2, axis=1).min()
        dist2 = np.sum((yx2 - zyx_ref[1:])**2, axis=1).min()

        # identify pre and post membranes
        if dist1 < dist2:
            Bdiv1 = comp1
            Bdiv2 = comp2
        else:
            Bdiv1 = comp2
            Bdiv2 = comp1

        # save parameters and results
        self.steps["divide"].update(dict(
            finished=True,
            # parameters
            size_ratio_thresh=size_ratio_thresh,
            # results
            zyx1=utils.mask_to_coord(Bdiv1),
            zyx2=utils.mask_to_coord(Bdiv2),
        ))
    
    #=========================
    # evomsac
    #=========================

    def evomsac_one(self, B, voxel_size_nm,
            grid_z_nm, grid_xy_nm,
            pop_size, max_iter, tol
        ):
        """ evomsac for one divided part
        :param B: 3d binary image
        :param voxel_size_nm: voxel spacing in nm
        :param grid_z_nm, grid_xy_nm: grid spacing in z, xy
        :param pop_size: size of population
        :param tol: (tol_value, n_back), terminate if change ratio < tol_value within last n_back steps
        :param max_iter: max number of generations
        """
        # setup grid and mootools
        n_uz, n_vxy = self.set_ngrids(B, voxel_size_nm, grid_z_nm, grid_xy_nm)
        mtools = evomsac.MOOTools(
            B, n_vxy=n_vxy, n_uz=n_uz, nz_eachu=1, r_thresh=1
        )

        # setup pop, evolve
        mpop = evomsac.MOOPop(mtools, pop_size=pop_size)
        mpop.init_pop()
        mpop.evolve(max_iter=max_iter, tol=tol)

        return mpop

    def evomsac(self, grid_z_nm=50, grid_xy_nm=150,
            pop_size=40, max_iter=500, tol=(0.01, 10)
        ):
        """ evomsac for both divided parts
        :param grid_z_nm, grid_xy_nm: grid spacing in z, xy
        :param pop_size: size of population
        :param tol: (tol_value, n_back), terminate if change ratio < tol_value within last n_back steps
        :param max_iter: max number of generations
        :action: assign steps["evomsac"]: mpop1, mpop2
        """
        # load from self
        self.check_steps(["tomo", "divide"], raise_error=True)
        voxel_size_nm = self.steps["tomo"]["voxel_size_nm"]
        Bdiv1 = self.coord_to_binary(self.steps["divide"]["zyx1"])
        Bdiv2 = self.coord_to_binary(self.steps["divide"]["zyx2"])

        # do for each divided part
        params = dict(
            grid_z_nm=grid_z_nm,
            grid_xy_nm=grid_xy_nm,
            pop_size=pop_size,
            max_iter=max_iter,
            tol=tol
        )
        mpop1 = self.evomsac_one(Bdiv1, voxel_size_nm=voxel_size_nm, **params)
        mpop2 = self.evomsac_one(Bdiv2, voxel_size_nm=voxel_size_nm, **params)

        # save parameters and results
        self.steps["evomsac"].update(params)
        self.steps["evomsac"].update(dict(
            finished=True,
            # results
            mpop1z=mpop1.dump_state(),
            mpop2z=mpop2.dump_state()
        ))

    #=========================
    # matching
    #=========================

    def match_one(self, B, O, mpop, d_cleft_vx, d_mem_vx, factor_tv):
        """ match for one divided part
        :param B, O: 3d binary image, orientation
        :param mpop: MOOPop from evomsac_one
        :param d_cleft_vx, d_mem_vx: lengthscales in voxel
        :param factor_tv: sigma for tv = factor_tv*d_cleft_vx
        :return: Bsmooth, zyx_sorted
            Bsmooth: resulting binary image from matching
            zyx_sorted: coordinates sorted by tracing
        """
        # fit surface
        indiv = mpop.select_by_hypervolume(mpop.log_front[-1])
        pts_net = mpop.mootools.get_coord_net(indiv)
        nu_eval = np.max(utils.wireframe_lengths(pts_net, axis=0))
        nv_eval = np.max(utils.wireframe_lengths(pts_net, axis=1))
        Bfit, _ = mpop.mootools.fit_surface_eval(
            indiv,
            u_eval=np.linspace(0, 1, 2*int(nu_eval)),
            v_eval=np.linspace(0, 1, 2*int(nv_eval))
        )

        # tv
        Stv, Otv = dtvoting.stick3d(B, O,
            sigma=d_cleft_vx*factor_tv
        )
        Btv = nonmaxsup.nms3d(Stv, Otv)

        # matching
        Bmatch = matching.match_spatial_orient(
            Btv, Otv*Btv, Bfit,
            sigma_gauss=d_mem_vx,
            sigma_tv=d_mem_vx
        )

        # smoothing
        _, Osmooth = hessian.features3d(Bmatch, d_mem_vx)
        Bsmooth = nonmaxsup.nms3d(Bmatch, Osmooth*Bmatch)
        Bsmooth = next(iter(utils.extract_connected(Bsmooth)))[1]
        Osmooth = Osmooth*Bsmooth

        # ordering
        zyx_sorted = trace.Trace(Bsmooth, Osmooth).sort_coord()
        
        return Bsmooth, zyx_sorted

    def match(self, factor_tv=1):
        """ match for both divided parts
        :param factor_tv: sigma for tv = factor_tv*d_cleft_vx
        :action: assign steps["match"]: zyx1,  zyx2
        """
        # load from self
        self.check_steps(["tomo", "detect", "divide", "evomsac"], raise_error=True)
        O = utils.densify3d(self.steps["detect"]["Oz"])
        Bdiv1 = self.coord_to_binary(self.steps["divide"]["zyx1"])
        Bdiv2 = self.coord_to_binary(self.steps["divide"]["zyx2"])
        mpop1 = evomsac.MOOPop(state=self.steps["evomsac"]["mpop1z"])
        mpop2 = evomsac.MOOPop(state=self.steps["evomsac"]["mpop2z"])

        # match
        params = dict(
            d_mem_vx=self.steps["tomo"]["d_mem_vx"],
            d_cleft_vx=self.steps["tomo"]["d_cleft_vx"],
            factor_tv=factor_tv
        )
        _, zyx1 = self.match_one(Bdiv1, O*Bdiv1, mpop1, **params)
        _, zyx2 = self.match_one(Bdiv2, O*Bdiv2, mpop2, **params)

        # save parameters and results
        self.steps["match"].update(dict(
            finished=True,
            # parameters
            factor_tv=factor_tv,
            # results
            zyx1=zyx1,
            zyx2=zyx2,
        ))

    #=========================
    # surface normal
    #=========================

    def surf_normal_one(self, zyx, zyx_ref, shape, d_mem_vx):
        """ calculate surface normal for one divided part
        :param zyx: coordinates
        :param zyx_ref: reference zyx
        :param shape: (nz,ny,nx) of 3d image
        :param d_mem_vx: membrane thickness in voxel
        :return: normal
            normal: each=(nx,ny,nz)
        """
        B = utils.coord_to_mask(zyx, shape)
        pos = tuple(zyx.T)
        _, normal = hessian.surface_normal(
            B, sigma=d_mem_vx, zyx_ref=zyx_ref, pos=pos
        )
        return normal
    
    def surf_normal(self):
        """ calculate surface normal for both divided parts
        :action: assign steps["surf_normal"]: normal1,  normal2
        """
        # load from self
        self.check_steps(["tomo", "match"], raise_error=True)
        params = dict(
            shape=self.steps["tomo"]["shape"],
            zyx_ref=self.steps["tomo"]["zyx_ref"],
            d_mem_vx=self.steps["tomo"]["d_mem_vx"],
        )
        zyx1 = self.steps["match"]["zyx1"]
        zyx2 = self.steps["match"]["zyx2"]

        # calc normal
        normal1 = self.surf_normal_one(zyx1, **params)
        normal2 = self.surf_normal_one(zyx2, **params)

        # save parameters and results
        self.steps["surf_normal"].update(dict(
            finished=True,
            # results
            normal1=normal1,
            normal2=normal2,
        ))

    #=========================
    # fitting
    #=========================
    
    def surf_fit_one(self, B, voxel_size_nm, d_mem_vx, grid_z_nm=10, grid_xy_nm=10):
        """ surface fitting for one divided part
        :param B: 3d binary image
        :param voxel_size_nm: voxel spacing in nm
        :param grid_z_nm, grid_xy_nm: grid spacing in z, xy
        :return: Bfit, zyx_sorted
        """
        # setup grids, mtools
        n_uz, n_vxy = self.set_ngrids(B, voxel_size_nm, grid_z_nm, grid_xy_nm)
        mtools = evomsac.MOOTools(
            B, n_uz=n_uz, n_vxy=n_vxy, nz_eachu=1, r_thresh=1
        )

        # generate indiv
        grid_sizes = list(mtools.grid.uv_size.values())
        index = int((np.median(grid_sizes)-1)/2)
        indiv = mtools.uniform(index=index)
        
        # fit
        pts_net = mtools.get_coord_net(indiv)
        nu_eval = np.max(utils.wireframe_lengths(pts_net, axis=0))
        nv_eval = np.max(utils.wireframe_lengths(pts_net, axis=1))
        Bfit, _ = mtools.fit_surface_eval(
            indiv,
            u_eval=np.linspace(0, 1, 5*int(nu_eval)),
            v_eval=np.linspace(0, 1, 5*int(nv_eval))
        )

        # ordering
        _, Ofit = hessian.features3d(Bfit, d_mem_vx)
        zyx_sorted = trace.Trace(Bfit, Ofit).sort_coord()

        return Bfit, zyx_sorted

    def surf_fit(self, grid_z_nm=10, grid_xy_nm=10):
        """ surface fitting for both divided parts
        :param grid_z_nm, grid_xy_nm: grid spacing in z, xy
        :action: assign steps["surf_fit"]: zyx1,  zyx2
        """
        # load from self
        self.check_steps(["tomo", "match"], raise_error=True)
        zyx1 = self.steps["match"]["zyx1"]
        zyx2 = self.steps["match"]["zyx2"]

        # fit
        params = dict(
            voxel_size_nm=self.steps["tomo"]["voxel_size_nm"],
            d_mem_vx=self.steps["tomo"]["d_mem_vx"],
            grid_z_nm=grid_z_nm,
            grid_xy_nm=grid_xy_nm
        )
        B1 = self.coord_to_binary(zyx1)
        B2 = self.coord_to_binary(zyx2)
        _, zyx_fit1 = self.surf_fit_one(B1, **params)
        _, zyx_fit2 = self.surf_fit_one(B2, **params)

        # save parameters and results
        self.steps["surf_fit"].update(dict(
            finished=True,
            # parameters
            grid_z_nm=grid_z_nm,
            grid_xy_nm=grid_xy_nm,
            # results
            zyx1=zyx_fit1,
            zyx2=zyx_fit2,
        ))

