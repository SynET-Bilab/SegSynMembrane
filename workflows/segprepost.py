""" workflow
"""

import time
import numpy as np
from etsynseg import io, utils, plot
from etsynseg import division, evomsac
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
                clip_range=None,
                mask_bound=None,
                zyx_ref=None,
                d_mem=None,
                d_cleft=None,
            ),
            detect=dict(
                finished=False,
                timing=None,
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
                timing=None,
                # parameters
                size_ratio_thresh=None,
                # results
                zyx1=None,
                zyx2=None,
            ),
            evomsac=dict(
                finished=False,
                timing=None,
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
                timing=None,
                # parameters
                factor_tv=None,
                # results
                zyx1=None,
                zyx2=None,
            ),
            surf_normal=dict(
                finished=False,
                timing=None,
                # results
                normal1=None,
                normal2=None,
            ),
            surf_fit=dict(
                finished=False,
                timing=None,
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
   
    def output_results(self, filenames, plot_nslice=5, plot_dpi=200):
        """ output results
        :param filenames: dict(tomo(mrc), match(mod), plot(png), surf_normal(npz), surf_fit(mod))
        :param plot_nslice, plot_dpi: plot nslice slices, save at dpt=plot_dpi
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
                I=utils.negate_image(self.steps["tomo"]["I"]),  # negated
                zyxs=tuple(self.steps["match"][f"zyx{i}"] for i in (1, 2)),
                nslice=plot_nslice
            )
            fig.savefig(filenames["plot"], dpi=plot_dpi)
        
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
        :action: assign steps["tomo"]: I, voxel_size_nm, mask_bound, zyx_ref, d_mem, d_cleft
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
        self.steps["tomo"].update(results)
        self.steps["tomo"].update(dict(
            finished=True,
            # parameters
            obj_ref=obj_ref,
            d_cleft_nm=d_cleft_nm,
            # results
            zyx_ref=zyx_ref,
            d_cleft=d_cleft_nm/results["voxel_size_nm"],
        ))
        self.steps["tomo"]["timing"] = time.process_time()-time_start

    #=========================
    # detect
    #=========================
    
    def detect(self, factor_tv=1, factor_supp=5, qfilter=0.25):
        """ detect membrane features
        :param factor_tv: sigma for tv = factor_tv*d_cleft
        :param factor_supp: sigma for normal suppression = factor_supp*d_cleft
        :param qfilter: quantile to filter out by Stv and Ssupp
        :action: assign steps["detect"]: B, O
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo"], raise_error=True)
        I = self.steps["tomo"]["I"]
        mask_bound = self.steps["tomo"]["mask_bound"]
        d_mem = self.steps["tomo"]["d_mem"]
        d_cleft = self.steps["tomo"]["d_cleft"]
        
        # detect: update qfilter, zyx, Oz
        results = SegSteps.detect(
            I, mask_bound,
            sigma_hessian=d_mem,
            sigma_tv=d_cleft*factor_tv,
            sigma_supp=d_cleft*factor_supp,
            qfilter=qfilter,
            dzfilter=int(I.shape[0]*0.75)
        )

        # save parameters and results
        self.steps["detect"].update(results)
        self.steps["detect"].update(dict(
            finished=True,
            # parameters
            factor_tv=factor_tv,
            factor_supp=factor_supp,
        ))
        self.steps["detect"]["timing"] = time.process_time()-time_start
    
    #=========================
    # divide
    #=========================
    
    def divide(self, size_ratio_thresh=0.5):
        """ divide detected image into pre-post candidates
        :param size_ratio_thresh: if size2/size1<size_ratio_thresh, consider that membranes are connected
        :action: assign steps["divide"]: B1, B2
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo", "detect"], raise_error=True)
        d_mem = self.steps["tomo"]["d_mem"]
        shape = self.steps["tomo"]["shape"]
        d_cleft = self.steps["tomo"]["d_cleft"]
        zyx_ref = self.steps["tomo"]["zyx_ref"]
        B = utils.coord_to_mask(self.steps["detect"]["zyx"], shape)
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
                seg_max_size=max(1, int(d_cleft)),
                seg_neigh_thresh=max(1, int(d_mem)),
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
        self.steps["divide"]["timing"] = time.process_time()-time_start
    
    #=========================
    # evomsac
    #=========================


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
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo", "divide"], raise_error=True)
        voxel_size_nm = self.steps["tomo"]["voxel_size_nm"]
        shape = self.steps["tomo"]["shape"]
        Bdiv1 = utils.coord_to_mask(self.steps["divide"]["zyx1"], shape)
        Bdiv2 = utils.coord_to_mask(self.steps["divide"]["zyx2"], shape)

        # do for each divided part
        params = dict(
            grid_z_nm=grid_z_nm,
            grid_xy_nm=grid_xy_nm,
            pop_size=pop_size,
            max_iter=max_iter,
            tol=tol
        )
        mpop1 = SegSteps.evomsac(Bdiv1, voxel_size_nm=voxel_size_nm, **params)
        mpop2 = SegSteps.evomsac(Bdiv2, voxel_size_nm=voxel_size_nm, **params)

        # save parameters and results
        self.steps["evomsac"].update(params)
        self.steps["evomsac"].update(dict(
            finished=True,
            # results
            mpop1z=mpop1.dump_state(),
            mpop2z=mpop2.dump_state()
        ))
        self.steps["evomsac"]["timing"] = time.process_time()-time_start

    #=========================
    # matching
    #=========================


    def match(self, factor_tv=1):
        """ match for both divided parts
        :param factor_tv: sigma for tv = factor_tv*d_cleft
        :action: assign steps["match"]: zyx1,  zyx2
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo", "detect", "divide", "evomsac"], raise_error=True)
        shape = self.steps["tomo"]["shape"]
        O = utils.densify3d(self.steps["detect"]["Oz"])
        Bdiv1 = utils.coord_to_mask(self.steps["divide"]["zyx1"], shape)
        Bdiv2 = utils.coord_to_mask(self.steps["divide"]["zyx2"], shape)
        mpop1 = evomsac.MOOPop(state=self.steps["evomsac"]["mpop1z"])
        mpop2 = evomsac.MOOPop(state=self.steps["evomsac"]["mpop2z"])

        # match
        params = dict(
            sigma_tv=self.steps["tomo"]["d_cleft"]*factor_tv,
            sigma_hessian=self.steps["tomo"]["d_mem"],
            sigma_dilate=self.steps["tomo"]["d_mem"]
        )
        _, zyx1 = SegSteps.match(Bdiv1, O*Bdiv1, mpop1, **params)
        _, zyx2 = SegSteps.match(Bdiv2, O*Bdiv2, mpop2, **params)

        # save parameters and results
        self.steps["match"].update(dict(
            finished=True,
            # parameters
            factor_tv=factor_tv,
            # results
            zyx1=zyx1,
            zyx2=zyx2,
        ))
        self.steps["match"]["timing"] = time.process_time()-time_start

    #=========================
    # surface normal
    #=========================
    
    def surf_normal(self):
        """ calculate surface normal for both divided parts
        :action: assign steps["surf_normal"]: normal1,  normal2
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo", "match"], raise_error=True)
        params = dict(
            shape=self.steps["tomo"]["shape"],
            zyx_ref=self.steps["tomo"]["zyx_ref"],
            d_mem=self.steps["tomo"]["d_mem"],
        )
        zyx1 = self.steps["match"]["zyx1"]
        zyx2 = self.steps["match"]["zyx2"]

        # calc normal
        normal1 = SegSteps.surf_normal(zyx1, **params)
        normal2 = SegSteps.surf_normal(zyx2, **params)

        # save parameters and results
        self.steps["surf_normal"].update(dict(
            finished=True,
            # results
            normal1=normal1,
            normal2=normal2,
        ))
        self.steps["surf_normal"]["timing"] = time.process_time()-time_start

    #=========================
    # fitting
    #=========================

    def surf_fit(self, grid_z_nm=10, grid_xy_nm=10):
        """ surface fitting for both divided parts
        :param grid_z_nm, grid_xy_nm: grid spacing in z, xy
        :action: assign steps["surf_fit"]: zyx1,  zyx2
        """
        time_start = time.process_time()

        # load from self
        self.check_steps(["tomo", "match"], raise_error=True)
        shape = self.steps["tomo"]["shape"]
        zyx1 = self.steps["match"]["zyx1"]
        zyx2 = self.steps["match"]["zyx2"]

        # fit
        params = dict(
            voxel_size_nm=self.steps["tomo"]["voxel_size_nm"],
            grid_z_nm=grid_z_nm,
            grid_xy_nm=grid_xy_nm
        )
        B1 = utils.coord_to_mask(zyx1, shape)
        B2 = utils.coord_to_mask(zyx2, shape)
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
