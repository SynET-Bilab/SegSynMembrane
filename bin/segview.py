#!/usr/bin/env python

import argparse
import textwrap
import etsynseg


def build_argparser():
    """ Build argument parser for inputs.

    Returns:
        parser (argparse.ArgumentParser): The parser.
    """
    # parser
    description = textwrap.dedent("""
        View membrane segmentation results.
        
        Usage:
            segview.py mode state.npz -t tomo.mrc
        
        Modes:
            args: print args
            steps: view steps
            3d: view membranes as 3d pointclouds
            moosac: plot moosac trajectory
        """)
    parser = argparse.ArgumentParser(
        prog="segview.py",
        description=description,
        formatter_class=etsynseg.miscutil.HelpFormatterCustom
    )
    parser.add_argument("mode", type=str, choices=[
        "args", "steps", "3d", "moosac"
    ])
    parser.add_argument("input", type=str, help="Input state file.")
    parser.add_argument("-t", "--tomo_file", type=str, default=None, help="Tomo file. Defaults to the one in seg_file.")
    return parser


if __name__ == '__main__':
    # get args
    parser = build_argparser()
    args = vars(parser.parse_args())

    # init seg
    # read prog from state file
    seg = etsynseg.segbase.SegBase()
    seg.load_state(args["input"])

    # view
    mode = args["mode"]
    if mode == "args":
        seg.show_args()
    elif mode == "steps":
        seg.reload_tomo(args["tomo_file"])
        seg.show_steps()
    elif mode == "3d":
        seg.show_segpcds()
    elif mode == "moosac":
        seg.show_moosac()
    