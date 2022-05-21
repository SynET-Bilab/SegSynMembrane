#!/usr/bin/env python

import argparse, pathlib, logging, textwrap
import multiprocessing
import numpy as np
import etsynseg

class SegOneMem(etsynseg.segbase.SegBase):
    def __init__(self):
        """ Initialization
        """
        # init from SegBase
        super().__init__()
        
        # logging
        self.timer = etsynseg.segbase.Timer(return_format="string")
        self.logger = logging.getLogger("segonemem")
        
        # update fields
        self.labels = (1,)

    def build_parser(self):
        """ Build parser for segmentation.

        Returns:
            parser (argparse.ArgumentParser): Parser with arguments.
        """
        # parser
        description = textwrap.dedent("""
        Semi-automatic membrane segmentation.
        
        Usage:
            (u1) segonemem.py mode tomo.mrc model.mod -o outputs [options]
                model: if not provided, then set as tomo.mod
                outputs: if not provided, then set as model-seg
            (u2) segonemem.py mode state.npz [options]
        
        Modes:
            run (u1): normal segmentation
            runfine (u1): run with finely-drawn model which separates pre and post
            rewrite (u2): model and figures
            showarg (u2): print args
            showim (u2): draw steps
            showpcd (u2): draw membranes as pointclouds
            showmoo (u2): plot moosac trajectory
        """)

        parser = argparse.ArgumentParser(
            prog="segonemem.py",
            description=description,
            formatter_class=etsynseg.segbase.HelpFormatterCustom
        )
        # mode
        parser.add_argument("mode", type=str, choices=["run", "runfine", "rewrite", "showarg", "showim", "showpcd"])
        # input/output
        parser.add_argument("inputs", type=str, nargs='+',help="Input files. Tomo and model files for modes in (run, runfine). State file for other modes.")
        parser.add_argument("-o", "--outputs", type=str, default=None, help="Basename for output files. Defaults to the basename of model file.")
        # info
        parser.add_argument("-px", "--pixel", type=float, default=None, help="Pixel size in nm. If not set, then read from the header of tomo.")
        parser.add_argument("--extend", type=float, default=20, help="The distance (in nm) that the bounding region extends from guiding lines.")
        parser.add_argument("--neigh_thresh", type=float, default=5, help="Distance threshold (in nm) for neighboring points in graph construction.")
        # detect
        parser.add_argument("--detect_smooth", type=float, default=5, help="Step 'detect': sigma for gaussian smoothin in nm. Can be set to membrane thickness.")
        parser.add_argument("--detect_tv", type=float, default=20, help="Step 'detect': sigma for tensor voting in nm. Can be set to cleft width.")
        parser.add_argument("--detect_filt", type=float, default=1.5, help="Step 'detect': keep the strongest (detect_filt * size of guiding surface) pixels during filtering.")
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
        mode = args["mode"]
        # modes reading tomo and model files
        if mode in ["run", "runfine"]:
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
        elif (mode in ["rewrite"]) or  mode.startswith("show"):
            state_file = args["inputs"][0]
            self.load_state(state_file)
        else:
            raise ValueError(f"""Unrecognized mode: {mode}""")
            
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

    def components_one(self):
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
        zyx1 = etsynseg.components.extract_components_one(
            zyx,
            r_thresh=tomod["neigh_thresh"],
            min_size=min_size
        )

        # save results
        self.steps["components"]["zyx1"] = zyx1

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
        for i in self.labels:
            # points
            zyx_i = meshrefine[f"zyx{i}"]
            results[f"xyz{i}"] = etsynseg.pcdutil.reverse_coord(zyx_low+zyx_i)

            # normals
            nzyx_i = etsynseg.pcdutil.normals_points(
                zyx_i,
                sigma=tomod["neigh_thresh"]*2,
                pt_ref=tomod["normal_ref"]
            )
            results[f"nxyz{i}"] = etsynseg.pcdutil.reverse_coord(nzyx_i)

            # surface area
            area_i, _ = etsynseg.moosac.surface_area(
                zyx_i, tomod["guide"],
                len_grid=tomod["neigh_thresh"]*4
            )
            results[f"area{i}_nm2"] = area_i * pixel_nm**2

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
            self.load_tomod(interp_degree=2, raise_noref=False)
        elif mode in ["runfine"]:
            self.load_tomod(interp_degree=1, raise_noref=False)

        # detecting
        self.detect()

        # extract components
        self.components_one()

        # fit refine
        self.fit_refine(1)

        # finalize
        self.final_results()
        self.final_outputs()
        self.logger.info(f"""segmentation finished: total {self.timer.total()}""")

if __name__ == "__main__":
    # init
    seg = SegOneMem()
    
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
    elif mode == "showarg":
        seg.show_args()
    elif mode == "showim":
        seg.show_steps()
    elif mode == "showpcd":
        seg.show_pcds()
    