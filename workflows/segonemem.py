""" workflow
"""

import time
import numpy as np
import mrcfile
from etsynseg import io, imgutils, plot
from etsynseg.workflows import SegBase, SegSteps
from utilities import utils


class SegOneMem(SegBase):
    """ workflow for segmentation
    attribute formats:
        binary images: save coordinates (utils.pixels2points, utils.points2pixels)
        sparse images (O): save as sparse (utils.sparsify3d, utils.densify3d)
        MOOPop: save state (MOOPop().dump_state, MOOPop(state=state))
    
    workflow:
        see run/segonemem_run.py
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
                # results
                I=None,
                shape=None,
                pixel_nm=None,
                model=None,
                clip_range=None,
                zyx_shift=None,
                zyx_bound=None,
                contour_bound=None,
                contour_len_bound=None,
                zyx_ref=None,
                d_mem=None,
            ),
            detect=dict(
                finished=False,
                timing=None,
                # parameters: input
                factor_tv=None,
                factor_supp=None,
                xyfilter=None,
                zfilter=None,
                # results
                zyx_nofilt=None,
                zyx=None,
                Oz=None
            ),
            evomsac=dict(
                finished=False,
                timing=None,
                # parameters: input
                grid_z_nm=None,
                grid_xy_nm=None,
                shrink_sidegrid=None,
                fitness_fthresh=None,
                pop_size=None,
                max_iter=None,
                tol=None,
                # results
                mpopz=None,
                zyx=None
            ),
            match=dict(
                finished=False,
                timing=None,
                # parameters: input
                factor_tv=None,
                factor_extend=None,
                # results
                zyx=None
            ),
            meshrefine=dict(
                finished=False,
                timing=None,
                # parameters: input
                factor_normal=None,
                factor_mesh=None,
                # results
                zyx=None,
                nxyz=None
            ),
        )

    #=========================
    # io
    #=========================
    
    def output_tomo(self, filename):
        """ output clipped tomo
            filename: filename(.mrc) for saving
        """
        self.check_steps(["tomo"], raise_error=True)
        io.write_tomo(
            tomo=self.steps["tomo"]["I"],
            tomo_file=filename,
            voxel_size=self.steps["tomo"]["pixel_nm"]*10
        )

    def output_model(self, step, filename, clipped=False):
        """ output results to a model file
            step: step name, one of {match, surf_fit}
            filename: filename(.mod) for saving
            clipped: if coordinates are for the clipped data
        """
        self.check_steps(["tomo", step], raise_error=True)
        
        # collect zyx's
        contour_bound = self.steps["tomo"]["contour_bound"]
        zyx_segs = [self.steps[step]["zyx"]]
        zyx_arr = [contour_bound] + zyx_segs

        # shift
        if not clipped:
            zyx_shift = self.steps["tomo"]["zyx_shift"]
            zyx_arr = [zyx_i+zyx_shift for zyx_i in zyx_arr]
        
        # write model
        io.write_model(zyx_arr=zyx_arr, model_file=filename)
    
    def output_seg(self, filename, clipped=False):
        """ output final seg, including points and normals
            filename: filename(.npz) for saving
            clipped: if coordinates are for the clipped data
        """
        self.check_steps(["tomo", "meshrefine"], raise_error=True)
        steps = self.steps

        if not clipped:
            zyx_shift = self.steps["tomo"]["zyx_shift"]
        else:
            zyx_shift = np.zeros(3)
        np.savez(
            filename,
            tomo_file=steps["tomo"]["tomo_file"],
            pixel_nm=steps["tomo"]["pixel_nm"],
            xyz=utils.reverse_coord(steps["meshrefine"]["zyx"]+zyx_shift),
            normal=steps["meshrefine"]["nxyz"]
        )

    def output_figure(self, step, filename, clipped=True, nslice=5, dpi=300):
        """ output results to a figure
            step: step name, one of {match, surf_fit}
            filename: filename(.png) for saving
            clipped: if coordinates are for the clipped data
            nslice: no. of slices to plot
            dpi: dpi for saving
        """
        self.check_steps(["tomo", step], raise_error=True)

        # collect zyx's
        contour_bound = self.steps["tomo"]["contour_bound"]
        zyx_segs = [self.steps[step]["zyx"]]
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
            I: 3d image
            zyxs: array of zyx to overlay on the image
            nslice: number of slices to show
        Returns: fig, axes
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
    
    def imshow3d_steps(self, vec_width=0.25, vec_length=2):
        """ imshow important intermediate results
            vec_width: width for plotting normal vectors
            vec_length: length for plotting normal vectors
        """
        # setup
        self.check_steps(["tomo"], raise_error=True)

        # image
        I = self.steps["tomo"]["I"]
        name_I = "clipped image"

        # results from steps
        Is_overlay = []
        Is_overlay.append(
            self.points2pixels(self.steps["tomo"]["zyx_bound"])
        )
        if self.check_steps(["detect"]):
            Is_overlay.extend([
                self.points2pixels(self.steps["detect"][f"zyx{i}"])
                for i in ("_nofilt", "")
            ])
        for step in ["evomsac", "match", "meshrefine"]:
            if self.check_steps([step]):
                Is_overlay.extend([
                    self.points2pixels(self.steps[step]["zyx"])
                ])
            else:
                break
                
        name_Is = [
            "mask",
            "detect(not filtered)", "detect(filtered)",
            "evomsac", "match", "meshrefine"
        ]
        cmap_Is = [
            "bop blue",
            "red", "bop orange",
            "green", "cyan", "yellow"
        ]
        cmap_vecs = ["yellow"]
        visible_Is = [False] + [True]*2 + [False]*4
        visible_vecs = True

        # normals
        if self.check_steps(["meshrefine"]):
            vecs_zyx = [
                self.steps["meshrefine"]["zyx"]
            ]
            vecs_dir = [
                vec_length*utils.reverse_coord(self.steps["meshrefine"]["nxyz"])
            ]
            name_vecs = ["normal"]
        else:
            vecs_zyx = ()
            vecs_dir = ()
            name_vecs = ()

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
    
    def points2pixels(self, coord):
        """ coord to mask, use default shape
        """
        self.check_steps(["tomo"], raise_error=True)
        shape = self.steps["tomo"]["shape"]
        return utils.points2pixels(coord, shape)

    #=========================
    # read tomo
    #=========================
    
    def read_tomo(self, tomo_file, model_file,
            pixel_nm=None, d_mem_nm=5,
            obj_bound=1, obj_ref=2
        ):
        """ load and clip tomo and model
            tomo_file, model_file: filename of tomo, model
            obj_bound, obj_ref: obj label for boundary and presynapse, begins with 1
            pixel_nm: manually set; if None then read from tomo_file
        :action: assign steps["tomo"]: I, pixel_nm, zyx_shift, zyx_bound, contour_bound, zyx_ref, d_mem, d_cleft
        """
        time_start = time.process_time()

        # check model file
        model = io.read_model(model_file)
        if obj_bound not in model["object"].values:
            raise ValueError(f"object bound (index={obj_bound}) not found in the model")

        # read tomo and model, clip
        results = SegSteps.read_tomo(
            tomo_file, model_file,
            pixel_nm=pixel_nm, d_mem_nm=d_mem_nm,
            obj_bound=obj_bound
        )
        
        # get coordinates of presynaptic label
        model = results["model"]
        if obj_ref in model["object"].values:
            series_ref = model[model["object"] == obj_ref].iloc[0]
            zyx_ref = np.array(
                [series_ref[i] for i in ["z", "y", "x"]]
            )
        # if no obj_ref, set to midpoint of the bound
        else:
            obj_ref = None,
            zyx_ref = model[model["object"]==obj_bound][["z", "y", "x"]].values
            zyx_ref = (np.max(zyx_ref, axis=0)+np.min(zyx_ref, axis=0))/2

        # save parameters and results
        self.steps["tomo"].update(dict(
            finished=True,
            # parameters
            obj_ref=obj_ref,
            # results
            zyx_ref=zyx_ref,
        ))
        self.steps["tomo"].update(results)
        self.steps["tomo"]["timing"] = time.process_time()-time_start

    #=========================
    # detect
    #=========================
    
    def set_dzfilter(self, zfilter, nz):
        """ set dzfilter. see self.detect.
        Returns: dzfilter
        """
        # set dzfilter
        if zfilter <= 0:  # as offset
            dzfilter = int(nz + zfilter)
        elif zfilter < 1:  # as fraction
            dzfilter = int(nz * zfilter)
        else:  # as direct value
            dzfilter = int(zfilter)
        return dzfilter

    def detect(self, factor_tv=5, factor_supp=0.25, xyfilter=3, zfilter=-1):
        """ detect membrane features
            factor_tv: sigma for tv = factor_tv*d_mem
            factor_supp: sigma for normal suppression = factor_supp*mean(contour_len_bound)
            xyfilter: for each xy plane, filter out pixels with Ssupp below quantile threshold; the threshold = 1-xyfilter*fraction_mems. see SegSteps().detect()
            zfilter: a component will be filtered out if its z-span < dzfilter;
            dzfilter = {nz+zfilter if zfilter<=0, nz*zfilter if 0<zfilter<1}
        :action: assign steps["detect"]: B, O
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo"], raise_error=True)
        I = self.steps["tomo"]["I"]
        mask_bound = self.points2pixels(self.steps["tomo"]["zyx_bound"])
        d_mem = self.steps["tomo"]["d_mem"]
        
        # sets sigma_supp, dzfilter
        sigma_supp = factor_supp * np.mean(self.steps["tomo"]["contour_len_bound"])
        dzfilter = self.set_dzfilter(zfilter, nz=I.shape[0])

        # detect
        zyx_nofilt, zyx_raw, Oz_raw = SegSteps.detect(
            I, mask_bound,
            contour_len_bound=self.steps["tomo"]["contour_len_bound"],
            sigma_hessian=d_mem,
            sigma_tv=d_mem*factor_tv,
            sigma_supp=sigma_supp,
            dO_thresh=np.pi/4,
            xyfilter=xyfilter,
            dzfilter=dzfilter
        )

        # extract connected
        B_raw = self.points2pixels(zyx_raw)
        B_c = next(imgutils.connected_components(B_raw, n_keep=1, connectivity=3))[1]
        zyx = utils.pixels2points(B_c)
        Oz = imgutils.sparsify3d(utils.densify3d(Oz_raw)*B_c)

        # save parameters and results
        self.steps["detect"].update(dict(
            finished=True,
            # parameters
            factor_tv=factor_tv,
            factor_supp=factor_supp,
            xyfilter=xyfilter,
            zfilter=zfilter,
            # results
            zyx_nofilt=zyx_nofilt,
            zyx=zyx,
            Oz=Oz
        ))
        self.steps["detect"]["timing"] = time.process_time()-time_start
     
    #=========================
    # evomsac
    #=========================

    def evomsac(self, grid_z_nm=50, grid_xy_nm=150,
            shrink_sidegrid=0.2, fitness_fthresh=1,
            pop_size=40, max_iter=200, tol=(0.01, 10), factor_eval=1
        ):
        """ evomsac
            grid_z_nm, grid_xy_nm: grid spacing in z, xy
            pop_size: size of population
            tol: (tol_value, n_back), terminate if change ratio < tol_value within last n_back steps
            max_iter: max number of generations
            factor_eval: factor for assigning evaluation points
        :action: assign steps["evomsac"]
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo"], raise_error=True)
        d_mem = self.steps["tomo"]["d_mem"]
        pixel_nm = self.steps["tomo"]["pixel_nm"]

        # evomsac
        params = dict(
            grid_z_nm=grid_z_nm,
            grid_xy_nm=grid_xy_nm,
            shrink_sidegrid=shrink_sidegrid,
            pop_size=pop_size,
            max_iter=max_iter,
            tol=tol,
        )
        params_extend = dict(
            params,
            **dict(
                fitness_rthresh=fitness_fthresh*d_mem,
                factor_eval=factor_eval,
                pixel_nm=pixel_nm
            )
        )
        zyx, mpopz = SegSteps.evomsac(self.steps["detect"]["zyx"], **params_extend)

        # save parameters and results
        self.steps["evomsac"].update(params)
        self.steps["evomsac"].update(dict(
            finished=True,
            # parameters
            fitness_fthresh=fitness_fthresh,
            # results
            mpopz=mpopz,
            zyx=zyx,
        ))
        self.steps["evomsac"]["timing"] = time.process_time()-time_start


    #=========================
    # matching
    #=========================

    def match(self, factor_tv=0, factor_extend=1):
        """ match
            factor_tv: sigma for tv on detected = factor_tv*d_mem
            factor_extend: sigma for tv extension on evomsac surface = factor_extend*d_mem
        :action: assign steps["match"]: zyx1,  zyx2
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo", "detect", "evomsac"], raise_error=True)
        d_mem = self.steps["tomo"]["d_mem"]
        O = utils.densify3d(self.steps["detect"]["Oz"])
        Bdiv = self.points2pixels(self.steps["detect"]["zyx"])
        Bsac = self.points2pixels(self.steps["evomsac"]["zyx"])

        # match
        params = dict(
            sigma_tv=d_mem*factor_tv,
            sigma_hessian=d_mem,
            sigma_extend=d_mem*factor_extend,
            mask_bound=self.points2pixels(self.steps["tomo"]["zyx_bound"])
        )
        _, zyx = SegSteps.match(Bdiv, O*Bdiv, Bsac, **params)
        
        # save parameters and results
        self.steps["match"].update(dict(
            finished=True,
            # parameters
            factor_tv=factor_tv,
            factor_extend=factor_extend,
            # results
            zyx=zyx,
        ))
        self.steps["match"]["timing"] = time.process_time()-time_start


    #=========================
    # meshrefine
    #=========================

    def meshrefine(self, factor_normal=2, factor_mesh=2):
        """ surface fitting
            grid_z_nm, grid_xy_nm: grid spacing in z, xy
            factor_<normal,mesh>: sigma for normal,mesh,hull calculations = d_mem*factor
        :action: assign steps["surf_fit"]: zyx
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo", "match"], raise_error=True)
        d_mem = self.steps["tomo"]["d_mem"]
        zyx = self.steps["match"]["zyx"]

        # parameters
        # sigma_mesh: should be smaller than z_span
        z_span = np.min([np.ptp(zyx_i[:, 0]) for zyx_i in [zyx]])
        sigma_mesh = min(factor_mesh*d_mem, z_span/3)
        # other parameters
        params = dict(
            zyx_ref=self.steps["tomo"]["zyx_ref"],
            sigma_normal=factor_normal*d_mem,
            sigma_mesh=sigma_mesh,
            sigma_hull=d_mem,
            mask_bound=self.points2pixels(self.steps["tomo"]["zyx_bound"])
        )

        # normal directions: towards cleft
        zyxref, nxyz = SegSteps.meshrefine(zyx, **params)
        
        # save parameters and results
        self.steps["meshrefine"].update(dict(
            finished=True,
            # parameters
            factor_normal=factor_normal,
            factor_mesh=factor_mesh,
            # results
            zyx=zyxref,
            nxyz=nxyz,
        ))
        self.steps["meshrefine"]["timing"] = time.process_time()-time_start
