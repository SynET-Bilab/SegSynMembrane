""" 
"""

import time
import numpy as np
import etsynseg

__all__ = [
    "Timer", "SegBase"
]

class Timer:
    """ A timer class.

    Examples:
        timer = Timer()
        dt = timer.click()
    """
    def __init__(self):
        """ Init and record current time.
        """
        self.t_last = time.perf_counter()

    def click(self):
        """ Record current time and calc time difference.
        """
        t_curr = time.perf_counter()
        del_t = t_curr - self.t_last
        self.t_last = t_curr
        del_t = f"{del_t:.1f}s"
        return del_t

class SegBase:
    """ Base class for segmentation.
    """
    def __init__(self):
        """ Init. Example attributes.
        """
        # logging
        self.timer = None
        self.logger = None
        # map
        self.func_map = None
        
        # args
        self.args = dict(
            mode=None, inputs=None, outputs=None,
            pixel_nm=None, extend_nm=None, d_mem_nm=None, neigh_thresh_nm=None,
            detect_tv_nm=None, detect_filt=None, detect_supp=None,
            moosac_lengrids=None, moosac_shrinkside=None, moosac_popsize=None, moosac_maxiter=None
        )
        # intermediate steps: in clipped coordinates
        self.steps = dict(
            tomod=dict(
                I=None, shape=None, pixel_nm=None,
                model=None, clip_low=None,
                bound=None, bound_plus=None, bound_minus=None,
                guide=None, normal_ref=None
            ),
            detect=dict(zyx_nofilt=None, zyx=None),
            components=dict(zyx1=None),
            moosac=dict(mpopz1=None, zyx1=None),
            match=dict(zyx1=None),
            meshrefine=dict(zyx1=None)
        )
        # results: in original coordinates
        self.results=dict(
            xyz1=None, nxyz1=None, area1_nm2=None
        )

    #=========================
    # io
    #=========================
    
    def load_state(self, state_file):
        """ Load info from state file.

        Args:
            state_file (str): Filename of the state file.
        """
        state = np.load(state_file, allow_pickle=True)
        self.args = state["args"].item()
        self.steps = state["steps"].item()
        self.results = state["results"].item()
        return self

    def save_state(self, state_file):
        """ Save data to state file.

        Args:
            state_file (str): Filename of the state file.
        """
        np.savez_compressed(
            state_file,
            args=self.args,
            steps=self.steps,
            results=self.results
        )   

    #=========================
    # 
    #=========================
        

    #=========================
    # steps
    #=========================
    
    def load_tomod(self):
        """ Load tomo and model.
        
        Prerequisites: args are read.
        Effects: updates self.steps["tomod"].
        """
        # log
        self.timer.click()

        # read tomo, model
        args = self.args
        tomod = etsynseg.modutil.read_tomo_model(
            tomo_file=args["tomo_file"],
            model_file=args["model_file"],
            extend_nm=args["extend_nm"],
            pixel_nm=args["pixel_nm"]
        )

        # update parameters
        tomod["d_mem"] = args["d_mem_nm"] / tomod["pixel_nm"]
        # neigh thresh >= 1
        tomod["neigh_thresh"] = max(
            1, args["neigh_thresh_nm"]/tomod["pixel_nm"])

        # save
        self.steps["tomod"].update(tomod)

        # log
        self.save_state(self.args["outputs_state"])
        self.logger.info(f"""loaded data: {self.timer.click()}""")

    def detect(self):
        """ Detect membrane-candidates from the image.

        Prerequisites: tomod.
        Effects: updates self.steps["detect"].
        """
        # log
        self.timer.click()

        # setup
        args = self.args
        tomod = self.steps["tomod"]
        pixel_nm = tomod["pixel_nm"]

        # detect mem-like structures
        B, _, B_nofilt = etsynseg.detecting.detect_memlike(
            tomod["I"],
            guide=tomod["guide"],
            bound=tomod["bound"],
            sigma_gauss=tomod["d_mem"],
            sigma_tv=args["detect_tv_nm"]/pixel_nm,
            factor_filt=args["detect_filt"],
            factor_supp=args["detect_supp"],
            return_nofilt=True
        )

        # save results
        self.steps["detect"]["zyx_nofilt"] = etsynseg.pcdutil.pixels2points(
            B_nofilt)
        self.steps["detect"]["zyx"] = etsynseg.pcdutil.pixels2points(B)

        # log
        self.save_state(self.args["outputs_state"])
        self.logger.info(f"""finished detecting: {self.timer.click()}""")

    def fit_refine(self, label):
        """ Fit, match, refine a surface.

        Prerequisites: components are extracted.
        Effects: updates self.steps[field], field="moosac","match","meshrefine"

        Args:
            label (int): Name label for the component. 
        """
        # log
        self.timer.click()

        # setup
        args = self.args
        tomod = self.steps["tomod"]
        pixel_nm = tomod["pixel_nm"]
        guide = tomod["guide"]
        zyx = self.steps["components"][f"zyx{label}"]

        # moosac fitting
        len_grids = tuple(l/pixel_nm for l in args["moosac_lengrids"])
        zyx_fit, mpop_state = etsynseg.moosac.robust_fitting(
            zyx, guide,
            len_grids=len_grids,
            shrink_sidegrid=args["moosac_shrinkside"],
            fitness_rthresh=tomod["neigh_thresh"],
            pop_size=args["moosac_popsize"],
            tol=(0.005, 10),
            max_iter=args["moosac_maxiter"],
            func_map=self.func_map
        )
        # save results
        self.steps["moosac"][f"zyx{label}"] = zyx_fit
        self.steps["moosac"][f"mpopz{label}"] = mpop_state
        # log
        self.logger.info(
            f"""finished moosac ({label}): {self.timer.click()}""")
        self.save_state(self.args["outputs_state"])

        # matching
        zyx_match = etsynseg.matching.match_candidate_to_ref(
            zyx, zyx_fit, guide, r_thresh=tomod["neigh_thresh"]
        )
        # save results
        self.steps["match"][f"zyx{label}"] = zyx_match
        # log
        self.save_state(self.args["outputs_state"])
        self.logger.info(
            f"""finished matching ({label}): {self.timer.click()}""")

        # meshrefine
        zyx_refine = etsynseg.meshrefine.refine_surface(
            zyx_match,
            sigma_normal=tomod["neigh_thresh"]*2,
            sigma_mesh=tomod["neigh_thresh"]*2,
            sigma_hull=tomod["d_mem"],
            target_spacing=1,
            bound=tomod["bound"]
        )
        # sort
        zyx_refine = etsynseg.pcdutil.sort_pts_by_guide_3d(zyx_refine, guide)
        # save results
        self.steps["meshrefine"][f"zyx{label}"] = zyx_refine
        # log
        self.save_state(self.args["outputs_state"])
        self.logger.info(
            f"""finished meshrefine ({label}): {self.timer.click()}""")
