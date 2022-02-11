#!/usr/bin/env python

import argparse
from etsynseg.workflows import SegPrePost


def build_parser():
    """ build parser for segmentation.
    :return: parser
    """
    # parser
    parser = argparse.ArgumentParser(
        prog="segprepost_check.py",
        description="Checking the status of the segmentation process."
    )
    parser.add_argument("steps_file", type=str,
        help="str. Filename of steps (.npz)")
    return parser

def run_check(args):
    """ run checking
    :param args: args from parser.parse_args()
    """
    # load status
    seg = SegPrePost()
    seg.load_steps(args.steps_file)
    status = seg.view_status()

    # print
    for k, v in status.items():
        timing = v['timing'] if v['finished'] else 0
        message = f"{k:<15s} finished={v['finished']}, timing={timing:.1f}s"
        print(message)

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_check(args)
