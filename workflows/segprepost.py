""" workflow
"""

import time
import numpy as np
import mrcfile
from etsynseg import io, utils, plot
from etsynseg import division
from etsynseg.workflows import SegBase, SegSteps


class SegPrePost(SegBase):
    """ workflow for segmentation
    attribute formats:
        binary images: save coordinates (utils.mask_to_coord, utils.coord_to_mask)
        sparse images (O): save as sparse (utils.sparsify3d, utils.densify3d)
        MOOPop: save state (MOOPop().dump_state, MOOPop(state=state))
    
    workflow:
        see segprepost_script.py
    """
    def __init__(self):
        self.steps = dict(
            tomo=dict(
                finished=False,
                timing=None,
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
                model=None,
                clip_range=None,
                zyx_shift=None,
                zyx_bound=None,
                contour_bound=None,
                contour_len_bound=None,
                zyx_ref=None,
                d_mem=None,
                d_cleft=None,
            ),
            detect=dict(
                finished=False,
                timing=None,
                # parameters: input
                factor_tv=None,
                factor_supp=None,
                xyfilter=None,
                zfilter=None,
                # parameters: inferred
                sigma_tv=None,
                sigma_supp=None,
                dzfilter=None,
                zyx_supp=None,
                # results
                zyx=None,
                Oz=None
            ),
            divide=dict(
                finished=False,
                timing=None,
                # parameters: input
                size_ratio_thresh=None,
                zfilter=None,
                # parameters: inferred
                dzfilter=None,
                # results
                zyx1=None,
                zyx2=None,
            ),
            evomsac=dict(
                finished=False,
                timing=None,
                # parameters: input
                grid_z_nm=None,
                grid_xy_nm=None,
                pop_size=None,
                max_iter=None,
                tol=None,
                # results
                zyx1=None,
                zyx2=None
            ),
            match=dict(
                finished=False,
                timing=None,
                # parameters: input
                factor_tv=None,
                factor_extend=None,
                # results
                zyx1=None,
                zyx2=None,
            ),
            surf_fit=dict(
                finished=False,
                timing=None,
                # parameters: input
                grid_z_nm=None,
                grid_xy_nm=None,
                # results
                zyx1=None,
                zyx2=None,
            ),
            surf_normal=dict(
                finished=False,
                timing=None,
                # results
                nxyz_seg1=None,
                nxyz_seg2=None,
                nxyz_fit1=None,
                nxyz_fit2=None,
            ),
        )

    #=========================
    # io
    #=========================
    
    def output_tomo(self, filename):
        """ output clipped tomo
        :param filename: filename(.mrc) for saving
        """
        self.check_steps(["tomo"], raise_error=True)
        io.write_mrc(
            data=self.steps["tomo"]["I"],
            mrcname=filename,
            voxel_size=self.steps["tomo"]["voxel_size_nm"]*10
        )

    def output_model(self, step, filename, clipped=False):
        """ output results to a model file
        :param step: step name, one of {match, surf_fit}
        :param filename: filename(.mod) for saving
        :param clipped: if coordinates are for the clipped data
        """
        self.check_steps(["tomo", step], raise_error=True)
        
        # collect zyx's
        contour_bound = self.steps["tomo"]["contour_bound"]
        zyx_segs = [self.steps[step][f"zyx{i}"] for i in (1, 2)]
        zyx_arr = [contour_bound] + zyx_segs

        # shift
        if not clipped:
            zyx_shift = self.steps["tomo"]["zyx_shift"]
            zyx_arr = [zyx_i+zyx_shift for zyx_i in zyx_arr]
        
        # write model
        io.write_model(zyx_arr=zyx_arr, model_file=filename)
    
    def output_normal(self, filename):
        """ output points and normal
        :param filename: filename(.npz) for saving
        """
        self.check_steps(["tomo", "match", "surf_normal", "surf_fit"],
            raise_error=True)
        steps = self.steps
        np.savez(
            filename,
            xyz_shift=steps["tomo"]["zyx_shift"][::-1],
            # segs
            xyz_seg1=utils.reverse_coord(steps["match"]["zyx1"]),
            xyz_seg2=utils.reverse_coord(steps["match"]["zyx2"]),
            normal_seg1=steps["surf_normal"]["nxyz_seg1"],
            normal_seg2=steps["surf_normal"]["nxyz_seg2"],
            # fits
            xyz_fit1=utils.reverse_coord(steps["surf_fit"]["zyx1"]),
            xyz_fit2=utils.reverse_coord(steps["surf_fit"]["zyx2"]),
            normal_fit1=steps["surf_normal"]["nxyz_fit1"],
            normal_fit2=steps["surf_normal"]["nxyz_fit2"],
        )

    def output_figure(self, step, filename, clipped=True, nslice=5, dpi=200):
        """ output results to a figure
        :param step: step name, one of {match, surf_fit}
        :param filename: filename(.png) for saving
        :param clipped: if coordinates are for the clipped data
        :param nslice: no. of slices to plot
        :param dpi: dpi for saving
        """
        self.check_steps(["tomo", step], raise_error=True)

        # collect zyx's
        contour_bound = self.steps["tomo"]["contour_bound"]
        zyx_segs = [self.steps[step][f"zyx{i}"] for i in (1, 2)]
        zyx_arr = [contour_bound] + zyx_segs

        # shift
        if not clipped:
            zyx_shift = self.steps["tomo"]["zyx_shift"]
            zyx_arr = [zyx_i+zyx_shift for zyx_i in zyx_arr]
            with mrcfile.mmap(self.steps["tomo"]["tomo_file"], permissive=True) as mrc:
                I = mrc.data
        else:
            I = self.steps["tomo"]["I"]
        
        fig, _ = self.plot_slices(I=I, zyxs=zyx_arr, nslice=nslice)
        fig.savefig(filename, dpi=dpi)
    

    #=========================
    # plotting
    #=========================
    
    def plot_slices(self, I, zyxs, nslice):
        """ plot sampled slices of image
        :param I: 3d image
        :param zyxs: array of zyx to overlay on the image
        :param nslice: number of slices to show
        :return: fig, axes
        """
        iz_min = np.min([np.min(zyx_i[:, 0]) for zyx_i in zyxs])
        iz_max = np.max([np.max(zyx_i[:, 0]) for zyx_i in zyxs])
        izs = np.linspace(iz_min, iz_max, nslice, dtype=int)
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
    
    def imshow3d_steps(self, vec_width=0.25):
        """ imshow important intermediate results
        :param vec_width: width for plotting normal vectors
        """
        # setup
        self.check_steps(
            ["tomo", "detect", "divide", "evomsac", "match", "surf_normal"],
            raise_error=True
        )
        steps = self.steps

        # image
        I = steps["tomo"]["I"]
        name_I = "clipped image"

        # results from steps
        Is_overlay = [
            self.coord_to_mask(zyx) for zyx in [
                steps["detect"]["zyx_supp"],
                steps["detect"]["zyx"],
                steps["divide"]["zyx1"],
                steps["divide"]["zyx2"],
                steps["evomsac"]["zyx1"],
                steps["evomsac"]["zyx2"],
                steps["surf_fit"]["zyx1"],
                steps["surf_fit"]["zyx2"],
            ]
        ]
        name_Is = [
            "detect(not filtered)", "detect(filtered)",
            "divide(pre)", "divide(post)",
            "evomsac(pre)", "evomsac(post)",
            "surf_fit(pre)", "surf_fit(post)",
        ]
        cmap_Is = [
            "red", "magenta",
            "green", "green",
            "cyan", "cyan",
            "yellow", "yellow",
        ]
        cmap_vecs = ["yellow", "yellow"]
        visible_Is = [True]*2 + [False]*6
        visible_vecs = True

        # normals
        vecs_zyx = [steps["surf_fit"][f"zyx{i}"] for i in (1, 2)]
        vecs_dir = [
            utils.reverse_coord(steps["surf_normal"][f"nxyz_fit{i}"])
            for i in (1, 2)
        ]
        name_vecs = ["normal(pre)", "normal(post)"]

        # imshow
        plot.imshow3d(
            I, Is_overlay,
            vecs_zyx=vecs_zyx, vecs_dir=vecs_dir, vec_width=vec_width,
            name_I=name_I, name_Is=name_Is, name_vecs=name_vecs,
            cmap_Is=cmap_Is, cmap_vecs=cmap_vecs,
            visible_Is=visible_Is, visible_vecs=visible_vecs
        )


    #=========================
    # utils
    #=========================
    
    def coord_to_mask(self, coord):
        """ coord to mask, use default shape
        """
        self.check_steps(["tomo"], raise_error=True)
        shape = self.steps["tomo"]["shape"]
        return utils.coord_to_mask(coord, shape)

    #=========================
    # read tomo
    #=========================
    
    def read_tomo(self, tomo_file, model_file,
            voxel_size_nm=None, d_mem_nm=5, d_cleft_nm=20,
            obj_bound=1, obj_ref=2
        ):
        """ load and clip tomo and model
        :param tomo_file, model_file: filename of tomo, model
        :param obj_bound, obj_ref: obj label for boundary and presynapse, begins with 1
        :param voxel_size_nm: manually set; if None then read from tomo_file
        :action: assign steps["tomo"]: I, voxel_size_nm, zyx_shift, zyx_bound, contour_bound, zyx_ref, d_mem, d_cleft
        """
        time_start = time.process_time()

        # read tomo and model, clip
        results = SegSteps.read_tomo(
            tomo_file, model_file,
            voxel_size_nm=voxel_size_nm, d_mem_nm=d_mem_nm,
            obj_bound=obj_bound
        )
        
        # get coordinates of presynaptic label
        model = results["model"]
        series_ref = model[model["object"] == obj_ref].iloc[0]
        zyx_ref = np.array(
            [series_ref[i] for i in ["z", "y", "x"]]
        )

        # save parameters and results
        self.steps["tomo"].update(dict(
            finished=True,
            # parameters
            obj_ref=obj_ref,
            d_cleft_nm=d_cleft_nm,
            # results
            zyx_ref=zyx_ref,
            d_cleft=d_cleft_nm/results["voxel_size_nm"],
        ))
        self.steps["tomo"].update(results)
        self.steps["tomo"]["timing"] = time.process_time()-time_start

    #=========================
    # detect
    #=========================
    
    def set_dzfilter(self, zfilter, nz):
        """ set dzfilter. see self.detect.
        :return: dzfilter
        """
        # set dzfilter
        if zfilter <= 0:  # as offset
            dzfilter = int(nz + zfilter)
        elif zfilter < 1:  # as fraction
            dzfilter = int(nz * zfilter)
        else:  # as direct value
            dzfilter = int(zfilter)
        return dzfilter

    def detect(self, factor_tv=5, factor_supp=0.25, xyfilter=2.5, zfilter=-1):
        """ detect membrane features
        :param factor_tv: sigma for tv = factor_tv*d_mem
        :param factor_supp: sigma for normal suppression = factor_supp*mean(contour_len_bound)
        :param xyfilter: for each xy plane, filter out pixels with Ssupp below quantile threshold; the threshold = 1-xyfilter*fraction_mems. see SegSteps().detect()
        :param zfilter: a component will be filtered out if its z-span < dzfilter;
            dzfilter = {nz+zfilter if zfilter<=0, nz*zfilter if 0<zfilter<1}
        :action: assign steps["detect"]: B, O
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo"], raise_error=True)
        I = self.steps["tomo"]["I"]
        mask_bound = self.coord_to_mask(self.steps["tomo"]["zyx_bound"])
        d_mem = self.steps["tomo"]["d_mem"]
        
        # sets sigma_supp, dzfilter
        sigma_supp = factor_supp * np.mean(self.steps["tomo"]["contour_len_bound"])
        dzfilter = self.set_dzfilter(zfilter, nz=I.shape[0])

        # detect
        results = SegSteps.detect(
            I, mask_bound,
            contour_len_bound=self.steps["tomo"]["contour_len_bound"],
            sigma_hessian=d_mem,
            sigma_tv=d_mem*factor_tv,
            sigma_supp=sigma_supp,
            dO_threshold=np.pi/4,
            xyfilter=xyfilter,
            dzfilter=dzfilter
        )

        # save parameters and results
        self.steps["detect"].update(dict(
            finished=True,
            # parameters
            factor_tv=factor_tv,
            factor_supp=factor_supp,
            zfilter=zfilter
        ))
        self.steps["detect"].update(results)
        self.steps["detect"]["timing"] = time.process_time()-time_start
    
    #=========================
    # divide
    #=========================

    def divide(self, size_ratio_thresh=0.5, zfilter=-1):
        """ divide detected image into pre-post candidates
        :param size_ratio_thresh: divide the largest component if size2/size1<size_ratio_thresh
        :param zfilter: consider a component as candidate if its z-span >= dzfilter. see self.detect for relations between zfilter and dzfilter.
        :action: assign steps["divide"]: zyx1, zyx2
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo", "detect"], raise_error=True)
        shape = self.steps["tomo"]["shape"]
        d_mem = self.steps["tomo"]["d_mem"]
        d_cleft = self.steps["tomo"]["d_cleft"]
        zyx_ref = self.steps["tomo"]["zyx_ref"]
        B = self.coord_to_mask(self.steps["detect"]["zyx"])
        O = utils.densify3d(self.steps["detect"]["Oz"])

        # extract two largest components
        # could get only one component
        comps = [comp[1] for comp in utils.extract_connected(B, n_keep=2, connectivity=3)]

        # check if components need to be divided
        dzfilter = self.set_dzfilter(zfilter, nz=shape[0])
        def need_to_divide(comps):
            # only one component: True
            if len(comps) == 1:
                return True
            else:
                zyx1, zyx2 = [utils.mask_to_coord(comp) for comp in comps[:2]]
                # size of component-2 too small: True
                if len(zyx2)/len(zyx1) < size_ratio_thresh:
                    return True
                # z-span of component-2 too small: True
                # loophole here: z-span of component-1 too small?
                elif np.ptp(zyx2[:, 0]) < dzfilter:
                    return True
                else:
                    return False

        # if two membranes seem connected, divide
        while need_to_divide(comps):
            comps = division.divide_connected(
                comps[0], O*comps[0],
                seg_max_size=max(1, int(d_cleft)),
                seg_neigh_thresh=max(1, int(d_mem)),
                n_clusters=2
            )

        # set two components
        comp1, comp2 = comps[:2]
        
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
            # parameters: input
            size_ratio_thresh=size_ratio_thresh,
            zfilter=zfilter,
            # parameters: inferred
            dzfilter=dzfilter,
            # results
            zyx1=utils.mask_to_coord(Bdiv1),
            zyx2=utils.mask_to_coord(Bdiv2),
        ))
        self.steps["divide"]["timing"] = time.process_time()-time_start
    
    #=========================
    # evomsac
    #=========================

    def evomsac(self, grid_z_nm=50, grid_xy_nm=150,
            pop_size=40, max_iter=500, tol=(0.01, 10), factor_eval=1
        ):
        """ evomsac for both divided parts
        :param grid_z_nm, grid_xy_nm: grid spacing in z, xy
        :param pop_size: size of population
        :param tol: (tol_value, n_back), terminate if change ratio < tol_value within last n_back steps
        :param max_iter: max number of generations
        :param factor_eval: factor for assigning evaluation points
        :action: assign steps["evomsac"]: mpop1, mpop2
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo", "divide"], raise_error=True)
        voxel_size_nm = self.steps["tomo"]["voxel_size_nm"]
        Bdiv1 = self.coord_to_mask(self.steps["divide"]["zyx1"])
        Bdiv2 = self.coord_to_mask(self.steps["divide"]["zyx2"])

        # do for each divided part
        params = dict(
            grid_z_nm=grid_z_nm,
            grid_xy_nm=grid_xy_nm,
            pop_size=pop_size,
            max_iter=max_iter,
            tol=tol,
            factor_eval=factor_eval
        )
        _, zyx1 = SegSteps.evomsac(Bdiv1, voxel_size_nm=voxel_size_nm, **params)
        _, zyx2 = SegSteps.evomsac(Bdiv2, voxel_size_nm=voxel_size_nm, **params)

        # save parameters and results
        self.steps["evomsac"].update(params)
        self.steps["evomsac"].update(dict(
            finished=True,
            # results
            zyx1=zyx1,
            zyx2=zyx2
        ))
        self.steps["evomsac"]["timing"] = time.process_time()-time_start

    #=========================
    # matching
    #=========================


    def match(self, factor_tv=1, factor_extend=5):
        """ match for both divided parts
        :param factor_tv: sigma for tv on detected = factor_tv*d_mem
        :param factor_extend: sigma for tv extension on evomsac surface = factor_extend*d_mem
        :action: assign steps["match"]: zyx1,  zyx2
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo", "detect", "divide", "evomsac"], raise_error=True)
        d_mem = self.steps["tomo"]["d_mem"]
        O = utils.densify3d(self.steps["detect"]["Oz"])
        Bdiv1 = self.coord_to_mask(self.steps["divide"]["zyx1"])
        Bdiv2 = self.coord_to_mask(self.steps["divide"]["zyx2"])
        Bsac1 = self.coord_to_mask(self.steps["evomsac"]["zyx1"])
        Bsac2 = self.coord_to_mask(self.steps["evomsac"]["zyx2"])

        # match
        params = dict(
            sigma_tv=d_mem*factor_tv,
            sigma_hessian=d_mem,
            sigma_extend=d_mem*factor_extend,
            mask_bound=self.coord_to_mask(self.steps["tomo"]["zyx_bound"])
        )
        _, zyx1 = SegSteps.match(Bdiv1, O*Bdiv1, Bsac1, **params)
        _, zyx2 = SegSteps.match(Bdiv2, O*Bdiv2, Bsac2, **params)

        # save parameters and results
        self.steps["match"].update(dict(
            finished=True,
            # parameters
            factor_tv=factor_tv,
            factor_extend=factor_extend,
            # results
            zyx1=zyx1,
            zyx2=zyx2,
        ))
        self.steps["match"]["timing"] = time.process_time()-time_start

    #=========================
    # fitting
    #=========================

    def surf_fit(self, grid_z_nm=10, grid_xy_nm=10, factor_eval=1):
        
        """ surface fitting for both divided parts
        :param grid_z_nm, grid_xy_nm: grid spacing in z, xy
        :action: assign steps["surf_fit"]: zyx1,  zyx2
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo", "match"], raise_error=True)
        zyx1 = self.steps["match"]["zyx1"]
        zyx2 = self.steps["match"]["zyx2"]
        mask_bound = self.coord_to_mask(self.steps["tomo"]["zyx_bound"])

        # fit
        params = dict(
            voxel_size_nm=self.steps["tomo"]["voxel_size_nm"],
            grid_z_nm=grid_z_nm,
            grid_xy_nm=grid_xy_nm,
            factor_eval=factor_eval
        )
        B1 = mask_bound*self.coord_to_mask(zyx1)
        B2 = mask_bound*self.coord_to_mask(zyx2)
        _, zyx_fit1 = SegSteps.surf_fit(B1, **params)
        _, zyx_fit2 = SegSteps.surf_fit(B2, **params)

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
        self.steps["surf_fit"]["timing"] = time.process_time()-time_start

    #=========================
    # surface normal
    #=========================

    def surf_normal(self):
        """ calculate surface normal(order is nx,ny,nz) for both divided parts
        :action: assign steps["surf_normal"]: nxyz_seg1/2,  nxyz_fit1/2
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo", "match", "surf_fit"], raise_error=True)
        params = dict(
            shape=self.steps["tomo"]["shape"],
            zyx_ref=self.steps["tomo"]["zyx_ref"],
            d_mem=self.steps["tomo"]["d_mem"],
        )

        # calc normal from segs
        nxyz_seg1 = SegSteps.surf_normal(self.steps["match"]["zyx1"], **params)
        nxyz_seg2 = -SegSteps.surf_normal(self.steps["match"]["zyx2"], **params)

        # calc normal from segs from fits
        nxyz_fit1 = SegSteps.surf_normal(self.steps["surf_fit"]["zyx1"], **params)
        nxyz_fit2 = -SegSteps.surf_normal(self.steps["surf_fit"]["zyx2"], **params)

        # save parameters and results
        self.steps["surf_normal"].update(dict(
            finished=True,
            # results
            nxyz_seg1=nxyz_seg1,
            nxyz_seg2=nxyz_seg2,
            nxyz_fit1=nxyz_fit1,
            nxyz_fit2=nxyz_fit2,
        ))
        self.steps["surf_normal"]["timing"] = time.process_time()-time_start
