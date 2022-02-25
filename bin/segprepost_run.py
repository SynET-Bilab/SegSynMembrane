#!/usr/bin/env python

import argparse
import pathlib
import logging
import napari
from etsynseg.workflows import SegPrePost


def build_parser():
    """ build parser for segmentation.
    :return: parser
    """
    # parser
    parser = argparse.ArgumentParser(
        prog="segprepost_run.py",
        description="Segmentation of synaptic membranes. Outputs: record of steps (-steps.npz), segmentation model (-segs.mod), its quickview image (-segs.png), smoothed fittings (-fits.mod), and surface normal (-normal.npz).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # tomo, model
    parser.add_argument("input_files", type=str, nargs='+',
        help="str, or str str. Case 1 for new segmentation: filename of tomo (.mrc) and filename of model (.mod, optional, if not provided then set to tomo name with suffix replaced by .mod). Case 2 for continuing segmentation: filename of steps (-steps.npz), start from steps where parameters are changed.")
    
    # output
    parser.add_argument("-o", "--output_base", type=str, default=None,
        help="str. Basename for output files. If not set, then use --output_prefix + basename of tomo_file."
    )
    parser.add_argument("-op", "--output_prefix", type=str, default="mem-",
        help="str. See --output_base."
    )

    # diagnosis
    parser.add_argument("-d", "--diagnose", default=False, action="store_true",
        help="flag. Used when the input file is -steps.npz. Show diagnostic plots and arguments."
    )

    # tomo continued
    parser.add_argument("--model_objs", type=int, nargs=2, default=[1, 2],
        help="int int (1-based). Object id's of the boundary and the reference point in the model file.")
    parser.add_argument("-vx", "--voxel_size", type=float, default=None,
        help="float (in nm). Voxel size in nm. If not set, then read from tomo file's header.")
    parser.add_argument("--lengths", type=float, default=[5, 20],
        help="float float (in nm). Membrane thickness and cleft width.")
    
    # detect
    parser.add_argument("--detect_tv", type=float, default=5,
        help="float. Step 'detect': sigma for tensor voting = membrane thickness * detect_tv. The larger the smoother and more connected.")
    parser.add_argument("--detect_xyfilter", type=float, default=2.5,
        help="float (>1). Step 'detect': after tensor voting and normal suppression, filter out voxels with Ssupp below quantile threshold, the threshold = 1-xyfilter*fraction_mems.Smaller values filter out more voxels.")
    
    # evomsac
    parser.add_argument("--evomsac_grids", type=float, nargs=2, default=[50, 150],
        help="float float (in nm). Step 'evomsac': spacings in z- and xy-axes of the sampling grids. Fine grids may be prone to noise; too-coarse grids may miss too much of the membrane.")

    # match
    parser.add_argument("--match_extend", type=float, default=2,
        help="float. Step 'match': factor for extension (by tensor voting, TV) of surface from evomsac. The sigma value for TV = match_extend * membrane thickness (set in --lengths). Larger value means more detected segments would be matched with evomsac surface.")

    # fit
    parser.add_argument("--meshrefine_factors", type=float, nargs=2, default=[2, 2],
        help="float float. Step 'meshrefine': lengthscales (=factor*membrane thickness) for normal calculation and mesh reconstruction. Larger factors remove more noises.")
    return parser

class IfRunStep:
    """ check if to run a step
    """
    def __init__(self):
        self.run = False
    def check(self, seg_step, param_cmp):
        # if prev steps was newly run, run
        if self.run:
            return True
        # if step is not finished, run
        elif not seg_step["finished"]:
            self.run = True
            return True
        else:
            # if any param is changed, run
            for k, v in param_cmp.items():
                if seg_step[k] != v:
                    self.run = True
                    return True
            # step has finished and no param change, no run
            return False

def backup_file(filename):
    """ if file exists, rename to filename~
    """
    p = pathlib.Path(filename)
    if p.is_file():
        p.rename(filename+"~")

def run_seg(args):
    """ run segmentation
    :param args: args from parser.parse_args()
    """
    #=========================
    # setups
    #=========================
    
    # init
    seg = SegPrePost()
    ifrun = IfRunStep()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )
    logging.info("starting etsynseg")

    # inputs
    # case: tomo (and model)
    if not args.input_files[0].endswith(".npz"):
        # set tomo, infer model
        if len(args.input_files) == 1:
            tomo_file = args.input_files[0]
            model_file = str(pathlib.Path(tomo_file).with_suffix(".mod"))
        # set tomo and model
        else:
            tomo_file = args.input_files[0]
            model_file = args.input_files[1]
        
        # check files
        if not pathlib.Path(tomo_file).is_file():
            print(f"tomo file not found: {tomo_file}")
            return
        if not pathlib.Path(model_file).is_file():
            print(f"model file not found: {model_file}")
            return
        
        # read tomo and model
        logging.info("reading tomo and model: %s %s", tomo_file, model_file)
        seg.read_tomo(
            tomo_file, model_file,
            obj_bound=args.model_objs[0],
            obj_ref=args.model_objs[1],
            voxel_size_nm=args.voxel_size,
            d_mem_nm=args.lengths[0],
            d_cleft_nm=args.lengths[1]
        )

    # case: steps npz
    else:
        step_file = args.input_files[0]
        logging.info("reading steps: %s", step_file)
        seg.load_steps(step_file)
        tomo_file = seg.steps["tomo"]["tomo_file"]
        
        # show diagnostics and quit
        if args.diagnose:
            logging.info("showing diagnostics")
            log_args = {
                "tomo": ["voxel_size_nm"],
                "detect": ["factor_tv", "xyfilter"],
                "evomsac": ["grid_xy_nm", "grid_z_nm"],
                "match": ["factor_extend"],
                "meshrefine": ["factor_normal", "factor_mesh"]
            }
            log_str = "arguments:\n  " + "\n  ".join([
                f"{k}: " + ",".join([f"{vi}={seg.steps[k][vi]}" for vi in v])
                for k, v in log_args.items()
            ])
            logging.info(log_str)
            seg.imshow3d_steps()
            napari.run()
            return

    # setup namings
    if args.output_base is not None:
        name = args.output_base
    else:
        name = args.output_prefix + pathlib.Path(tomo_file).stem

    #=========================
    # run steps, save
    #=========================
    
    filenames = {
        "steps": f"{name}-steps.npz",
        "meshrefine_mod": f"{name}-seg.mod",
        "meshrefine_fig": f"{name}-seg.png",
        "meshrefine_pts": f"{name}-seg.npz",
    }
    
    # initial save
    backup_file(filenames["steps"])
    seg.save_steps(filenames["steps"])

    # detect
    if ifrun.check(seg.steps["detect"],
        {"factor_tv": args.detect_tv,
        "xyfilter": args.detect_xyfilter}
    ):
        logging.info("starting detect")
        seg.detect(
            factor_tv=args.detect_tv,
            factor_supp=0.25,
            xyfilter=args.detect_xyfilter,
            zfilter=-1,
        )
        # save
        seg.save_steps(filenames["steps"])

    # divide
    if ifrun.check(seg.steps["divide"], {}):
        logging.info("starting divide")
        seg.divide(size_ratio_thresh=0.5, zfilter=-1)
        # save
        seg.save_steps(filenames["steps"])

    # evomsac
    if ifrun.check(seg.steps["evomsac"],
        {"grid_z_nm": args.evomsac_grids[0],
        "grid_xy_nm": args.evomsac_grids[1]}
    ):
        logging.info("starting evomsac")
        seg.evomsac(
            grid_z_nm=args.evomsac_grids[0],
            grid_xy_nm=args.evomsac_grids[1]
        )
        # save
        seg.save_steps(filenames["steps"])

    # match
    if ifrun.check(seg.steps["match"],
        {"factor_extend": args.match_extend}
    ):
        logging.info("starting match")
        seg.match(factor_tv=1, factor_extend=args.match_extend)
        # save
        seg.save_steps(filenames["steps"])

    # meshrefine
    if ifrun.check(seg.steps["meshrefine"],
        {"factor_normal": args.meshrefine_factors[0],
        "factor_mesh": args.meshrefine_factors[1]}
    ):
        logging.info("starting meshrefine")
        seg.meshrefine(
            factor_normal=args.meshrefine_factors[0],
            factor_mesh=args.meshrefine_factors[1]
        )
        # save
        seg.save_steps(filenames["steps"])
        seg.output_model("meshrefine", filenames["meshrefine_mod"], clipped=False)
        backup_file(filenames["meshrefine_fig"])
        seg.output_figure(
            "meshrefine", filenames["meshrefine_fig"], nslice=5, dpi=300, clipped=True)
        seg.output_points(filenames["meshrefine_pts"])
    
    logging.info("ending etsynseg")

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_seg(args)
