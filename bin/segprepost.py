#!/usr/bin/env python

import argparse, pathlib, logging
import multiprocessing
import numpy as np
import etsynseg

class SegPrePost(etsynseg.segbase.SegBase):
    def __init__(self):
        """ Initialization
        """
        # init from SegBase
        super().__init__()
        
        # logging
        self.timer = etsynseg.segbase.Timer(format="string")
        self.logger = logging.getLogger("segprepost")
        
        # update fields
        self.labels = (1, 2)
        self.args.update(dict(components_min=None))
        self.steps["components"].update(dict(zyx2=None))
        self.steps["moosac"].update(dict(
            mpopz2=None, zyx2=None
        ))
        self.steps["match"].update(dict(zyx2=None))
        self.steps["meshrefine"].update(dict(zyx2=None))
        self.results.update(dict(
            xyz2=None, nxyz2=None, area2_nm2=None,
            dist1_nm=None, dist2_nm=None,
        ))

    def build_parser(self):
        """ Build parser for segmentation.

        Returns:
            parser (argparse.ArgumentParser): 
        """
        # parser
        parser = argparse.ArgumentParser(
            prog="segprepost.py",
            description="Semi-automatic membrane segmentation.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        # mode
        parser.add_argument("mode", type=str, choices=["run", "runfine", "rewrite", "showim", "showpcd"], help="Mode. run: normal segmentation; runfine: run with finely-drawn model which separates pre and post; rewrite: model and figures; showim: draw steps; showpcd: draw membranes as pointclouds.")
        # input/output
        parser.add_argument("inputs", type=str, nargs='+',help="Input files. Tomo and model files for modes in (run, runfine). State file for other modes.")
        parser.add_argument("-o", "--outputs", type=str, default=None, help="Basename for output files. Defaults to the basename of model file.")
        # info
        parser.add_argument("-px", "--pixel", type=float, default=None, help="Pixel size in nm. If not set, then read from the header of tomo.")
        parser.add_argument("--extend", type=float, default=30, help="The distance (in nm) that the bounding region extends from guiding lines.")
        parser.add_argument("--neigh_thresh", type=float, default=5, help="Distance threshold (in nm) for neighboring points in graph construction.")
        # detect
        parser.add_argument("--detect_smooth", type=float, default=5, help="Step 'detect': sigma for gaussian smoothin in nm. Can be set to membrane thickness.")
        parser.add_argument("--detect_tv", type=float, default=20, help="Step 'detect': sigma for tensor voting in nm. Can be set to cleft width.")
        parser.add_argument("--detect_filt", type=float, default=3, help="Step 'detect': keep the strongest (detect_filt * size of guiding surface) pixels during filtering.")
        parser.add_argument("--detect_supp", type=float, default=0.5, help="Step 'detect': sigma for normal suppression = (detect_supp * length of guiding line).")
        # components
        parser.add_argument("--components_min", type=float, default=0.75, help="Step 'components': min size of component = (components_min * size of guiding surface).")
        # moosac
        parser.add_argument("--moosac_lengrids", type=float, nargs=2, default=[50, 150], help="Step 'moosac': length of sampling grids in z- and xy-axes.")
        parser.add_argument("--moosac_shrinkside", type=float, default=0.25, help="Step 'moosac': grids on the side in xy are shrinked to this value.")
        parser.add_argument("--moosac_popsize", type=int, default=40, help="Step 'moosac': population size for evolution.")
        parser.add_argument("--moosac_tol", type=float, default=0.005, help="Step 'moosac': terminate if fitness change < tol in all last 10 steps.")
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
        elif args["mode"] in ["rewrite", "showim", "showpcd"]:
            state_file = args["inputs"][0]
            self.load_state(state_file)
        else:
            raise ValueError(f"""Unrecognized mode: {args["mode"]}""")
            
        # state filename
        self.args["outputs_state"] = self.args["outputs"]+".npz"

        # setup logging
        log_handler = logging.FileHandler(self.args["outputs"]+".log")
        log_formatter = logging.Formatter(
            "%(asctime)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        log_handler.setFormatter(log_formatter)
        self.logger.addHandler(log_handler)
        self.logger.setLevel("INFO")
        
        # log for modes that run segmentation
        if args["mode"] in ["run", "runfine"]:
            self.logger.info("----segprepost----")
            self.logger.info(f"read args")
            # save state, backup for the first time
            self.save_state(self.args["outputs_state"], backup=True)
        # print for modes that just show
        else:
            print(f"----segprepost----")
            print(f"""mode: {args["mode"]}""")

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
        self.save_state(self.args["outputs_state"])
        self.logger.info(f"""extracted components: {self.timer.click()}""")

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
        # pre in normal minus, post in normal plus
        zyx1, zyx2 = etsynseg.components.extract_components_regions(
            zyx,
            region_arr=[tomod["bound_minus"], tomod["bound_plus"]],
            r_thresh=tomod["neigh_thresh"]
        )

        # save results
        self.steps["components"]["zyx1"] = zyx1
        self.steps["components"]["zyx2"] = zyx2

        # log
        self.save_state(self.args["outputs_state"])
        self.logger.info(f"""extracted components: {self.timer.click()}""")

    def final_results(self):
        """ Calculate final results. Save in self.results.
        """
        # log
        self.timer.click()

        # setup
        tomod = self.steps["tomod"]
        meshrefine = self.steps["meshrefine"]
        pixel_nm = tomod["pixel_nm"]
        zyx_low = tomod["clip_low"]

        # collect results
        results = {}
        for i in [1, 2]:
            # points
            zyx_i = meshrefine[f"zyx{i}"]
            results[f"xyz{i}"] = etsynseg.pcdutil.reverse_coord(zyx_low+zyx_i)

            # normals
            nzyx_i = etsynseg.pcdutil.normals_points(
                zyx_i,
                sigma=tomod["neigh_thresh"]*2,
                pt_ref=tomod["normal_ref"]
            )
            if i == 2:  # flip sign for postsynapse
                nzyx_i = -nzyx_i
            results[f"nxyz{i}"] = etsynseg.pcdutil.reverse_coord(nzyx_i)

            # surface area
            area_i, _ = etsynseg.moosac.surface_area(
                zyx_i, tomod["guide"],
                len_grid=tomod["neigh_thresh"]*4
            )
            results[f"area{i}_nm2"] = area_i * pixel_nm**2

        # distance
        dist1, dist2 = etsynseg.pcdutil.points_distance(
            meshrefine["zyx1"], meshrefine["zyx2"],
            return_2to1=True
        )
        results["dist1_nm"] = dist1 * pixel_nm
        results["dist2_nm"] = dist2 * pixel_nm

        # save
        self.results.update(results)
        self.save_state(self.args["outputs_state"])

        # log
        self.logger.info(f"finalized: {self.timer.click()}")

    def final_outputs(self):
        """ Final outputs.
        """
        self.output_model(self.args["outputs"]+".mod")
        self.output_slices(self.args["outputs"]+".png", nslice=5)

    def workflow(self):
        """ Segmentation workflow.
        """
        mode = self.args["mode"]

        # load tomod
        if mode in ["run"]:
            self.load_tomod(interp_degree=2)
        elif mode in ["runfine"]:
            self.load_tomod(interp_degree=1)

        # detecting
        self.detect()

        # extract components
        if mode in ["runfine"]:
            self.components_by_mask()
        else:
            self.components_auto()

        # fit refine
        self.fit_refine(1)
        self.fit_refine(2)

        # finalize
        self.final_results()
        self.final_outputs()
        self.logger.info(f"""segmentation finished: total {self.timer.total()}""")

if __name__ == "__main__":
    # init
    seg = SegPrePost()
    
    # parse and load args
    parser = seg.build_parser()
    args = vars(parser.parse_args())
    seg.load_args(args)
    
    # workflow
    # read mode from args, not seg.args
    mode = args["mode"]
    # run segmentation
    if mode in ["run", "runfine"]:
        pool = multiprocessing.Pool()
        seg.register_map(pool.map)
        seg.workflow()
        seg.register_map()
        pool.close()
    # recalculate and rewrite final results
    elif mode == "rewrite":
        seg.final_results()
        seg.final_outputs()
    # visualizations
    elif mode == "showim":
        seg.show_steps()
    elif mode == "showpcd":
        seg.show_pcds()
    
    