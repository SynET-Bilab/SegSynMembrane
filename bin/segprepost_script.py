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
    parser.add_argument("tomo_file", type=str,
        help="str. Filename of tomo (.mrc)")
    parser.add_argument("--model_file", type=str, default=None,
        help="str. Filename of imod model (.mod), which contains a boundary of the synaptic region and a reference point in the presynapse. If not provided, defaults to the name of tomo_file with suffix replaced by .mod.")
    
    # output
    parser.add_argument("--output_base", type=str, default=None,
        help="str. Basename for output files. If not set, then use --output_prefix + basename of tomo_file."
    )
    parser.add_argument("--output_prefix", type=str, default="mem-",
        help="str. See --output_base."
    )

    # tomo continued
    parser.add_argument("--model_objs", type=int, nargs=2, default=[1, 2],
        help="int int (1-based). Object id's of the boundary and the reference point in the model file.")
    parser.add_argument("--voxel_size", type=float, default=None,
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

    # fit
    parser.add_argument("--fit_grids", type=float, nargs=2, default=[10, 10],
        help="float float (in nm). Step 'surf_fit': spacings in z- and xy-axes of the sampling grids. Set to a scale that could be enough to capture membranes' uneveness.")
    return parser

def run_seg(args):
    """ run segmentation
    :param args: args from parser.parse_args()
    """
    # default namings
    if args.model_file is not None:
        model_file = args.model_file
    else:
        model_file = str(pathlib.Path(args.tomo_file).with_suffix(".mod"))

    if args.output_base is not None:
        name = args.output_base
    else:
        name = args.output_prefix + pathlib.Path(args.tomo_file).stem
    name_steps = f"{name}-steps.npz"
    
    # setups
    seg = SegPrePost()
    
    seg.read_tomo(
        args.tomo_file, model_file,
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

    seg.match(factor_tv=1)
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
        plot=f"{name}-segs.png",
        surf_normal=f"{name}-normal.npz",
        surf_fit=f"{name}-fits.mod"
    )
    seg.output_results(filenames, plot_nslice=5, plot_dpi=200)

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_seg(args)
