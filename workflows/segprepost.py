""" workflow
"""

import time
import numpy as np
import mrcfile
from etsynseg import io, utils, plot
from etsynseg import dividing
from etsynseg.workflows import SegBase, SegSteps


class SegPrePost(SegBase):
    """ workflow for segmentation
    attribute formats:
        binary images: save coordinates (utils.voxels_to_points, utils.points_to_voxels)
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
                # results
                zyx_nofilt=None,
                zyx=None,
                Oz=None
            ),
            divide=dict(
                finished=False,
                timing=None,
                # parameters: input
                ratio_comps=None,
                zfilter=None,
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
                shrink_sidegrid=None,
                fitness_fthresh=None,
                pop_size=None,
                max_iter=None,
                tol=None,
                # results
                mpopz1=None,
                mpopz2=None,
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
            meshrefine=dict(
                finished=False,
                timing=None,
                # parameters: input
                factor_normal=None,
                factor_mesh=None,
                # results
                zyx1=None,
                zyx2=None,
                nxyz1=None,
                nxyz2=None,
                dist1=None,
                dist2=None
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
    
    def output_seg(self, filename, clipped=False):
        """ output final seg, including points and normals
        :param filename: filename(.npz) for saving
        :param clipped: if coordinates are for the clipped data
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
            voxel_size_nm=steps["tomo"]["voxel_size_nm"],
            xyz1=utils.reverse_coord(steps["meshrefine"]["zyx1"]+zyx_shift),
            xyz2=utils.reverse_coord(steps["meshrefine"]["zyx2"]+zyx_shift),
            normal1=steps["meshrefine"]["nxyz1"],
            normal2=steps["meshrefine"]["nxyz2"],
            dist1=steps["meshrefine"]["dist1"],
            dist2=steps["meshrefine"]["dist2"]
        )

    def output_figure(self, step, filename, clipped=True, nslice=5, dpi=300):
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
    
    def imshow3d_steps(self, vec_width=0.25, vec_length=2):
        """ imshow important intermediate results
        :param vec_width: width for plotting normal vectors
        :param vec_length: length for plotting normal vectors
        """
        # setup
        self.check_steps(["tomo"], raise_error=True)

        # image
        I = self.steps["tomo"]["I"]
        name_I = "clipped image"

        # results from steps
        Is_overlay = []
        if self.check_steps(["detect"]):
            Is_overlay.extend([
                self.points_to_voxels(self.steps["detect"][f"zyx{i}"])
                for i in ("_nofilt", "")
            ])
        for step in ["divide", "evomsac", "match", "meshrefine"]:
            if self.check_steps([step]):
                Is_overlay.extend([
                    self.points_to_voxels(self.steps[step][f"zyx{i}"])
                    for i in (1, 2)
                ])
            else:
                break
                
        name_Is = [
            "detect(not filtered)", "detect(filtered)",
            "divide(pre)", "divide(post)",
            "evomsac(pre)", "evomsac(post)",
            "match(pre)", "match(post)",
            "meshrefine(pre)", "meshrefine(post)",
        ]
        cmap_Is = [
            "red", "bop orange",
            "magenta", "magenta",
            "green", "green",
            "cyan", "cyan",
            "yellow", "yellow",
        ]
        cmap_vecs = ["yellow", "yellow"]
        visible_Is = [True]*2 + [False]*8
        visible_vecs = True

        # normals
        if self.check_steps(["meshrefine"]):
            vecs_zyx = [
                self.steps["meshrefine"][f"zyx{i}"]
                for i in (1, 2)
            ]
            vecs_dir = [
                vec_length*utils.reverse_coord(self.steps["meshrefine"][f"nxyz{i}"])
                for i in (1, 2)
            ]
            name_vecs = ["normal(pre)", "normal(post)"]
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
    
    def points_to_voxels(self, coord):
        """ coord to mask, use default shape
        """
        self.check_steps(["tomo"], raise_error=True)
        shape = self.steps["tomo"]["shape"]
        return utils.points_to_voxels(coord, shape)

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

        # check model file
        model = io.read_model(model_file)
        if obj_bound not in model["object"].values:
            raise ValueError(f"object bound (index={obj_bound}) not found in the model")
        if obj_ref not in model["object"].values:
            raise ValueError(f"object presynaptic reference (index={obj_ref}) not found in the model")

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
        mask_bound = self.points_to_voxels(self.steps["tomo"]["zyx_bound"])
        d_mem = self.steps["tomo"]["d_mem"]
        
        # sets sigma_supp, dzfilter
        sigma_supp = factor_supp * np.mean(self.steps["tomo"]["contour_len_bound"])
        dzfilter = self.set_dzfilter(zfilter, nz=I.shape[0])

        # detect
        zyx_nofilt, zyx, Oz = SegSteps.detect(
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
            xyfilter=xyfilter,
            zfilter=zfilter,
            # results
            zyx_nofilt=zyx_nofilt,
            zyx=zyx,
            Oz=Oz
        ))
        self.steps["detect"]["timing"] = time.process_time()-time_start
    
    #=========================
    # divide
    #=========================

    def divide(self, ratio_comps=0.5, zfilter=-1):
        """ divide detected image into pre-post candidates
        :param ratio_comps: divide the largest component if size2/size1<ratio_comps
        :param zfilter: consider a component as candidate if its z-span >= dzfilter. see self.detect for relations between zfilter and dzfilter.
        :action: assign steps["divide"]: zyx1, zyx2
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo", "detect"], raise_error=True)
        d_mem = self.steps["tomo"]["d_mem"]
        d_cleft = self.steps["tomo"]["d_cleft"]
        zyx_ref = self.steps["tomo"]["zyx_ref"]
        zyx = self.steps["detect"]["zyx"]

        # extract two largest components
        zyx_comps = dividing.divide_to_two(
            zyx,
            group_rthresh=d_mem,
            group_size=int(d_cleft),
            ratio_comps=ratio_comps,
            max_iter=10,
            zfilter=zfilter
        )
        zyx_comp1, zyx_comp2 = zyx_comps[:2]
        
        # compare components' distance to ref
        dist1 = np.sum((zyx_comp1 - zyx_ref)**2, axis=1).min()
        dist2 = np.sum((zyx_comp2 - zyx_ref)**2, axis=1).min()

        # identify pre and post membranes
        if dist1 < dist2:
            zyx1 = zyx_comp1
            zyx2 = zyx_comp2
        else:
            zyx1 = zyx_comp2
            zyx2 = zyx_comp1

        # save parameters and results
        self.steps["divide"].update(dict(
            finished=True,
            # parameters: input
            ratio_comps=ratio_comps,
            zfilter=zfilter,
            # results
            zyx1=zyx1,
            zyx2=zyx2,
        ))
        self.steps["divide"]["timing"] = time.process_time()-time_start
    
    #=========================
    # evomsac
    #=========================

    def evomsac(self, grid_z_nm=50, grid_xy_nm=150,
            shrink_sidegrid=0.2, fitness_fthresh=1,
            pop_size=40, max_iter=200, tol=(0.01, 10), factor_eval=1
        ):
        """ evomsac for both divided parts
        :param grid_z_nm, grid_xy_nm: grid spacing in z, xy
        :param pop_size: size of population
        :param tol: (tol_value, n_back), terminate if change ratio < tol_value within last n_back steps
        :param max_iter: max number of generations
        :param factor_eval: factor for assigning evaluation points
        :action: assign steps["evomsac"]
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo", "divide"], raise_error=True)
        d_mem = self.steps["tomo"]["d_mem"]
        voxel_size_nm = self.steps["tomo"]["voxel_size_nm"]

        # do for each divided part
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
                voxel_size_nm=voxel_size_nm
            )
        )
        zyx1, mpopz1 = SegSteps.evomsac(self.steps["divide"]["zyx1"], **params_extend)
        zyx2, mpopz2 = SegSteps.evomsac(self.steps["divide"]["zyx2"], **params_extend)

        # save parameters and results
        self.steps["evomsac"].update(params)
        self.steps["evomsac"].update(dict(
            finished=True,
            # parameters
            fitness_fthresh=fitness_fthresh,
            # results
            mpopz1=mpopz1,
            mpopz2=mpopz2,
            zyx1=zyx1,
            zyx2=zyx2
        ))
        self.steps["evomsac"]["timing"] = time.process_time()-time_start


    #=========================
    # matching
    #=========================

    def match(self, factor_tv=0, factor_extend=1):
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
        Bdiv1 = self.points_to_voxels(self.steps["divide"]["zyx1"])
        Bdiv2 = self.points_to_voxels(self.steps["divide"]["zyx2"])
        Bsac1 = self.points_to_voxels(self.steps["evomsac"]["zyx1"])
        Bsac2 = self.points_to_voxels(self.steps["evomsac"]["zyx2"])

        # match
        params = dict(
            sigma_tv=d_mem*factor_tv,
            sigma_hessian=d_mem,
            sigma_extend=d_mem*factor_extend,
            mask_bound=self.points_to_voxels(self.steps["tomo"]["zyx_bound"])
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
    # meshrefine
    #=========================

    def meshrefine(self, factor_normal=2, factor_mesh=2):
        """ surface fitting for both divided parts
        :param grid_z_nm, grid_xy_nm: grid spacing in z, xy
        :param factor_<normal,mesh>: sigma for normal,mesh,hull calculations = d_mem*factor
        :action: assign steps["surf_fit"]: zyx1,  zyx2
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo", "match"], raise_error=True)
        d_mem = self.steps["tomo"]["d_mem"]
        zyx1 = self.steps["match"]["zyx1"]
        zyx2 = self.steps["match"]["zyx2"]

        # parameters
        # sigma_mesh: should be smaller than z_span
        z_span = np.min([np.ptp(zyx_i[:, 0]) for zyx_i in (zyx1, zyx2)])
        sigma_mesh = min(factor_mesh*d_mem, z_span/3)
        # other parameters
        params = dict(
            zyx_ref=self.steps["tomo"]["zyx_ref"],
            sigma_normal=factor_normal*d_mem,
            sigma_mesh=sigma_mesh,
            sigma_hull=d_mem,
            mask_bound=self.points_to_voxels(self.steps["tomo"]["zyx_bound"])
        )

        # normal directions: towards cleft
        zyxref1, nxyz1 = SegSteps.meshrefine(zyx1, **params)
        zyxref2, nxyz2 = SegSteps.meshrefine(zyx2, **params)
        nxyz2 = -nxyz2

        # distance to the other membrane
        pcdref1 = utils.points_to_pointcloud(zyxref1)
        pcdref2 = utils.points_to_pointcloud(zyxref2)
        dist1 = np.asarray(pcdref1.compute_point_cloud_distance(pcdref2))
        dist2 = np.asarray(pcdref2.compute_point_cloud_distance(pcdref1))
        
        # save parameters and results
        self.steps["meshrefine"].update(dict(
            finished=True,
            # parameters
            factor_normal=factor_normal,
            factor_mesh=factor_mesh,
            # results
            zyx1=zyxref1,
            zyx2=zyxref2,
            nxyz1=nxyz1,
            nxyz2=nxyz2,
            dist1=dist1,
            dist2=dist2
        ))
        self.steps["meshrefine"]["timing"] = time.process_time()-time_start
