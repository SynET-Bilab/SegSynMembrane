#!/usr/bin/env python

import argparse
import textwrap
import numpy as np
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
            segview.py mode state.npz
        
        Modes:
            args: print args
            steps: view steps
            3d: view membranes as 3d pointclouds
            moosac: plot moosac trajectory
        """)
    parser = argparse.ArgumentParser(
        prog="segview.py",
        description=description,
        formatter_class=etsynseg.segbase.HelpFormatterCustom
    )
    parser.add_argument("mode", type=str, choices=[
        "args", "steps", "3d", "moosac"
    ])
    parser.add_argument("input", type=str, help="Input state file.")
    return parser


if __name__ == '__main__':
    # get args
    parser = build_argparser()
    args = vars(parser.parse_args())

    # init seg according to program
    state_file = args["input"]
    prog = np.load(state_file, allow_pickle=True)["prog"].item()
    if prog == "segprepost":
        seg = etsynseg.segprepost.SegPrePost()
    elif prog == "segonemem":
        seg = etsynseg.segonemem.SegOneMem()
    else:
        raise ValueError(f"Unrecognized prog in {state_file}.")
    seg.load_state(state_file)

    # view
    mode = args["mode"]
    if mode == "args":
        seg.show_args()
    elif mode == "steps":
        seg.show_steps()
    elif mode == "3d":
        seg.show_segpcds()
    elif mode == "moosac":
        seg.show_moosac()
    