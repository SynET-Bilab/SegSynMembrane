#!/usr/bin/env python

import argparse
import multiprocessing
import pathlib
import logging
import numpy as np
import etsynseg
from etsynseg.bin.segbase import SegBase, Timer

class SegPrePost(SegBase):
    def __init__(self, func_map=map):
        """ Initialization
        """
        # logging
        self.timer = Timer()
        self.logger = logging.getLogger("segprepost")
        
        # func
        self.func_map = func_map

        # data
        self.args = dict(
            mode=None, inputs=None, outputs=None,
            pixel_nm=None, extend_nm=None, d_mem_nm=None, neigh_thresh_nm=None,
            detect_tv_nm=None, detect_filt=None, detect_supp=None,
            components_min=None,
            moosac_lengrids=None, moosac_shrinkside=None, moosac_popsize=None, moosac_maxiter=None
        )
        self.steps = dict(
            tomod=dict(
                I=None, shape=None, pixel_nm=None,
                model=None, clip_low=None,
                bound=None, bound_plus=None, bound_minus=None,
                guide=None, normal_ref=None
            ),
            detect=dict(
                zyx_nofilt=None, zyx=None
            ),
            components=dict(
                zyx1=None, zyx2=None
            ),
            moosac=dict(
                mpopz1=None, mpopz2=None,
                zyx1=None, zyx2=None
            ),
            match=dict(
                zyx1=None, zyx2=None,
            ),
            meshrefine=dict(
                zyx1=None, zyx2=None,
            ),
        )
        self.results=dict(
            zyx1=None, zyx2=None,
            nzyx1=None, nzyx2=None,
            dist1=None, dist2=None
        )

    def build_parser(self):
        """ Build parser for segmentation.

        Returns:
            parser (argparse.ArgumentParser): 
        """
        # parser
        parser = argparse.ArgumentParser(
            prog="segprepost.py",
            description="Semi-automatic membrane segmentation. Inputs: tomo and boundary model. Outputs: record of steps (-steps.npz), segmentation result (-seg.npz), model (-seg.mod), quickview image (-seg.png).",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        # mode
        parser.add_argument("mode", type=str, choices=["run", "runfine"], help="Segmentation mode.")
        # input/output
        parser.add_argument("inputs", type=str, nargs='+',help="Input files. Tomo file, model file for new segmentation. Seg file for continuation.")
        parser.add_argument("-o", "--outputs", type=str, default=None, help="Basename for output files. Defaults to the basename of model file.")
        # info
        parser.add_argument("-px", "--pixel_nm", type=float, default=None, help="Pixel size in nm. If not set, then read from the header of tomo.")
        parser.add_argument("--extend_nm", type=float, default=30, help="The distance (in nm) that the bounding region extends from guiding lines.")
        parser.add_argument("--d_mem_nm", type=float, default=5, help="Membrane thickness in nm.")
        parser.add_argument("--neigh_thresh_nm", type=float, default=5, help="Distance threshold (in nm) for neighboring points in graph construction.")
        # detect
        parser.add_argument("--detect_tv_nm", type=float, default=20, help="Step 'detect': sigma for tensor voting (tv) in nm. Can be set to cleft width.")
        parser.add_argument("--detect_filt", type=float, default=3, help="Step 'detect': keep the strongest (detect_filt * size of guiding surface) pixels during filtering.")
        parser.add_argument("--detect_supp", type=float, default=0.5, help="Step 'detect': sigma for normal suppression = (detect_supp * length of guiding line).")
        # components
        parser.add_argument("--components_min", type=float, default=0.75, help="Step 'components': min size of component = (components_min * size of guiding surface).")
        # moosac
        parser.add_argument("--moosac_lengrids", type=float, nargs=2, default=[50, 150], help="Step 'moosac': length of sampling grids in z- and xy-axes.")
        parser.add_argument("--moosac_shrinkside", type=float, default=0.25, help="Step 'moosac': grids on the side in xy are shrinked to this value.")
        parser.add_argument("--moosac_popsize", type=int, default=40, help="Step 'moosac': population size for evolution.")
        parser.add_argument("--moosac_maxiter", type=int, default=150, help="Step 'moosac': max number of iterations.")
        return parser

    def load_args(self, args):
        """ Load args into self.args.

        Args:
            args (dict or argparse.Namespace): Args as a dict, or from parser.parse_args.
        """
        # conversion
        if type(args) is not dict:
            args = vars(args)
        
        # processing
        # modes reading tomo and model files
        if args["mode"] in ["run", "runfine"]:
            # amend tomo, model
            args["tomo_file"] = args["inputs"][0]
            if len(args["inputs"]) == 2:
                args["model_file"] = args["inputs"][1]
            else:
                args["model_file"] = str(pathlib.Path(
                    args["tomo_file"]).with_suffix(".mod"))
            # amend outputs
            if args["outputs"] is None:
                args["outputs"] = pathlib.Path(args["model_file"]).stem + "-seg"
            # save
            self.args.update(args)
        # modes reading state file
        else:
            state_file = args["inputs"][0]
            self.load_state(state_file)
        # add state filename
        self.args["outputs_state"] = self.args["outputs"]+".npz"

        # logging
        log_handler = logging.FileHandler(self.args["outputs"]+".log")
        log_formatter = logging.Formatter(
            "%(asctime)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        log_handler.setFormatter(log_formatter)
        self.logger.addHandler(log_handler)
        self.logger.setLevel("INFO")
        self.logger.info("----segprepost----")
        self.logger.info(f"read args")

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
        tomod["neigh_thresh"] = max(1, args["neigh_thresh_nm"]/tomod["pixel_nm"])
        
        # save
        self.steps["tomod"].update(tomod)

        # log
        self.logger.info(f"""loaded data: {self.timer.click()}""")
        self.save_state(self.args["outputs_state"])

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
        self.steps["detect"]["zyx_nofilt"] = etsynseg.pcdutil.pixels2points(B_nofilt)
        self.steps["detect"]["zyx"] = etsynseg.pcdutil.pixels2points(B)

        # log
        self.logger.info(f"""finished detecting: {self.timer.click()}""")
        self.save_state(self.args["outputs_state"])

    def components_auto(self):
        """ Extract two components automatically.

        Prerequisites: membranes are detected.
        Effects: updates self.steps["components"].
        """
        # log
        self.timer.click()

        # setup
        args = self.args
        tomod = self.steps["tomod"]
        
        # extract by division
        zyx = self.steps["detect"]["zyx"]
        min_size = len(tomod["guide"])*args["components_min"]
        zyx1, zyx2 = etsynseg.components.extract_components_two(
            zyx,
            r_thresh=tomod["neigh_thresh"],
            orients=None, sigma_dO=np.pi/4,
            min_size=min_size
        )

        # sort by distance to ref point
        zyx1, zyx2 = etsynseg.pcdutil.sort_pcds_by_ref(
            [zyx1, zyx2],
            pt_ref=tomod["normal_ref"]
        )
        
        # save results
        self.steps["components"]["zyx1"] = zyx1
        self.steps["components"]["zyx2"] = zyx2

        # log
        self.logger.info(f"""extracted components: {self.timer.click()}""")
        self.save_state(self.args["outputs_state"])

    def components_by_mask(self):
        """ Extract two components using masks.
        
        Prerequisites: membranes are detected.
        Effects: updates self.steps["components"].
        """
        # log
        self.timer.click()

        # setup
        tomod = self.steps["tomod"]

        # extract by masking
        zyx = self.steps["detect"]["zyx"]
        zyx1, zyx2 = etsynseg.components.extract_components_regions(
            zyx,
            region_arr=[tomod["bound_plus"], tomod["bound_minus"]],
            r_thresh=tomod["neigh_thresh"]
        )

        # save results
        self.steps["components"]["zyx1"] = zyx1
        self.steps["components"]["zyx2"] = zyx2

        # log
        self.logger.info(f"""extracted components: {self.timer.click()}""")
        self.save_state(self.args["outputs_state"])

    def fit_refine(self, label):
        """ Fit, match, refine a surface.

        Prerequisites: components are extracted.
        Effects: updates self.steps[field], field="moosac","match","meshrefine"

        Args:
            label (int): 1 for presynapse, 2 for postsynapse. 
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
        self.logger.info(f"""finished moosac ({label}): {self.timer.click()}""")
        self.save_state(self.args["outputs_state"])

        # matching
        zyx_match = etsynseg.matching.match_candidate_to_ref(
            zyx, zyx_fit, guide, r_thresh=tomod["neigh_thresh"]
        )
        # save results
        self.steps["match"][f"zyx{label}"] = zyx_match
        # log
        self.logger.info(f"""finished matching ({label}): {self.timer.click()}""")
        self.save_state(self.args["outputs_state"])

        # meshrefine
        zyx_refine = etsynseg.meshrefine.refine_surface(
            zyx_match,
            sigma_normal=tomod["neigh_thresh"]*2,
            sigma_mesh=tomod["neigh_thresh"]*2,
            sigma_hull=tomod["d_mem"],
            target_spacing=1,
            bound=tomod["bound"]
        )
        # save results
        self.steps["meshrefine"][f"zyx{label}"] = zyx_refine
        # log
        self.logger.info(f"""finished meshrefine ({label}): {self.timer.click()}""")
        self.save_state(self.args["outputs_state"])

    def finalize(self):
        # setup
        tomod = self.steps["tomod"]
        meshrefine = self.steps["meshrefine"]

        # collect results
        results = {}
        for i in (1, 2):
            # points
            xyz_i = etsynseg.pcdutil.reverse_coord(
                tomod["clip_low"]+meshrefine[f"zyx{i}"]
            )
            results[f"xyz{i}"] = xyz_i
            # normals
            nxyz_i = etsynseg.pcdutil.normals_points(
                xyz_i,
                sigma=tomod["neigh_thresh"]*2,
                pt_ref=tomod["normal_ref"][::-1]
            )
            if i == 2:
                nxyz_i = -nxyz_i
            results[f"nxyz{i}"] = nxyz_i
        # distance
        results["dist1"], results["dist2"] = etsynseg.pcdutil.points_distance(
            meshrefine["zyx1"], meshrefine["zyx2"],
            return_2to1=True
        )

        # save
        self.results.update(results)
        self.save_state(self.args["outputs_state"])

    def workflow(self):
        # load tomod
        if self.args["mode"] in ["run", "runfine"]:
            self.load_tomod()

        # detecting
        self.detect()

        # extract components
        if self.args["mode"] in ["runfine"]:
            self.components_by_mask()
        else:
            self.components_auto()

        # fit refine
        self.fit_refine(1)
        self.fit_refine(2)

        # finalize
        self.finalize()


if __name__ == "__main__":
    # init
    pool = multiprocessing.Pool()
    seg = SegPrePost(func_map=pool.map)
    # parse args
    parser = seg.build_parser()
    args = parser.parse_args()
    # run
    seg.load_args(args)
    seg.workflow()
    # clean
    pool.close()

