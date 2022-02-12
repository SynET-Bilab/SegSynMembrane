#!/usr/bin/env python

import argparse
import pathlib
from etsynseg.workflows import SegPrePost


def build_parser():
    """ build parser for segmentation.
    :return: parser
    """
    # parser
    parser = argparse.ArgumentParser(
        prog="segprepost_script.py",
        description="Segmentation of synaptic membranes. Outputs: record of steps (-steps.npz), clipped tomo (-clip.mrc), segmentation model (-segs.mod), its quickview image (-segs.png), surface normal (-normal.npz), and smoothed fittings (-fits.mod).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # tomo
    parser.add_argument("input_files", type=str, nargs='+',
        help="str, or str str. Filename of tomo (.mrc) and filename of model (.mod, optional, if not provided then set to tomo name with suffix replaced by .mod).")
    
    # output
    parser.add_argument("-o", "--output_base", type=str, default=None,
        help="str. Basename for output files. If not set, then use --output_prefix + basename of tomo_file."
    )
    parser.add_argument("-op", "--output_prefix", type=str, default="mem-",
        help="str. See --output_base."
    )

    # tomo continued
    parser.add_argument("--model_objs", type=int, nargs=2, default=[1, 2],
        help="int int (1-based). Object id's of the boundary and the reference point in the model file.")
    parser.add_argument("-vx", "--voxel_size", type=float, default=None,
        help="float (in nm). Voxel size in nm. If not set, then read from tomo file's header.")
    parser.add_argument("--lengths", type=float, default=[5, 20],
        help="float float (in nm). Membrane thickness and cleft width.")
    
    # detect
    parser.add_argument("--detect_thresh", type=float, default=0.25,
        help="float (between 0 and 1). Step 'detect': after tensor voting and normal suppression, there is a simple saliency-based filtering. Segments with saliency below this quantile threshold will be filtered out. Higher values could potentially filter out more noises.")
    
    # divide
    parser.add_argument("--divide_thresh", type=float, default=0.5,
        help="float (between 0 and 1). Step 'divide': after step 'detect', the largest two connected components are extracted; if the ratio of their sizes is below this threshold, consider that the largest component contains both membranes, so that it has to be divided")
    
    # evomsac
    parser.add_argument("--evomsac_grids", type=float, nargs=2, default=[50, 150],
        help="float float (in nm). Step 'evomsac': spacings in z- and xy-axes of the sampling grids. Fine grids may be prone to noise; too-coarse grids may miss too much of the membrane.")

    # match
    parser.add_argument("--match_extend", type=float, default=1,
        help="float. Step 'match': factor for extension (by tensor voting, TV) of surface from evomsac. The sigma value for TV = match_extend * membrane thickness (set in --lengths). Larger value means more detected segments would be matched with evomsac surface.")

    # fit
    parser.add_argument("--fit_grids", type=float, nargs=2, default=[10, 10],
        help="float float (in nm). Step 'surf_fit': spacings in z- and xy-axes of the sampling grids. Set to a scale that could be enough to capture membranes' uneveness.")
    return parser

def run_seg(args):
    """ run segmentation
    :param args: args from parser.parse_args()
    """
    # input files
    if len(args.input_files) == 1:
        tomo_file = args.input_files[0]
        model_file = str(pathlib.Path(tomo_file).with_suffix(".mod"))
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

    # output namings
    if args.output_base is not None:
        name = args.output_base
    else:
        name = args.output_prefix + pathlib.Path(tomo_file).stem
    name_steps = f"{name}-steps.npz"
    
    # setups
    seg = SegPrePost()
    
    seg.read_tomo(
        tomo_file, model_file,
        obj_bound=args.model_objs[0],
        obj_ref=args.model_objs[1],
        voxel_size_nm=args.voxel_size,
        d_mem_nm=args.lengths[0],
        d_cleft_nm=args.lengths[1]
    )
    seg.save_steps(name_steps)

    # run steps
    seg.detect(
        factor_tv=1, factor_supp=5,
        qfilter=args.detect_thresh
    )
    seg.save_steps(name_steps)

    seg.divide(size_ratio_thresh=args.divide_thresh)
    seg.save_steps(name_steps)

    seg.evomsac(
        grid_z_nm=args.evomsac_grids[0],
        grid_xy_nm=args.evomsac_grids[1]
    )
    seg.save_steps(name_steps)

    seg.match(factor_smooth=1, factor_extend=args.match_extend)
    seg.surf_normal()
    seg.surf_fit(
        grid_z_nm=args.fit_grids[0],
        grid_xy_nm=args.fit_grids[1]
    )
    seg.save_steps(name_steps)

    # output results
    filenames = dict(
        tomo=f"{name}-clip.mrc",
        match=f"{name}-segs.mod",
        match_shift=f"{name}-segs-shift.mod",
        plot_shift=f"{name}-segs-shift.png",
        surf_normal=f"{name}-normal.npz",
        surf_fit=f"{name}-fits.mod",
        surf_fit_shift=f"{name}-fits-shift.mod"
    )
    seg.output_results(filenames, plot_nslice=5, plot_dpi=500)

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_seg(args)
