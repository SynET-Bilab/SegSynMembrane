#!/usr/bin/env python

import argparse
import os
import pathlib
import numpy as np
import pandas as pd
import starfile

def normal_to_euler(normal):
    """ convert normal to euler angles
    :param normal: (nx,ny,nz)
    :return: euler
        euler: ()
    """
    nx, ny, nz = normal
    theta = np.mod(-np.rad2deg(np.arccos(nz)), 360)  # np outputs [0,pi]
    phi = np.mod(-np.rad2deg(np.arctan2(ny, nx)), 360)  # np outputs [-pi,pi]
    euler = (180., theta, phi)
    return euler

def downsample(xyz, normal, step_xy, step_z, factor_offset=0.5):
    """ downsample points and normals
    :param xyz, normal: array of points and normals, shape=(npts,3)
    :param step_xy, step_z: int, step in xy and z
    :param factor_offset: index of the first sample = int(factor_offset*step)
    :return: xyz_sample, normal_sample
    """
    xyz_sample = []
    normal_sample = []

    # sample along z
    z_sample = np.unique(xyz[:, 2])[int(factor_offset*step_z)::step_z]
    for z in z_sample:
        # sample in xy planes
        mask = (xyz[:, 2]==z)
        xyz_sample.append(xyz[mask][int(factor_offset*step_xy)::step_xy])
        normal_sample.append(normal[mask][int(factor_offset*step_xy)::step_xy])

    xyz_sample = np.concatenate(xyz_sample, axis=0)
    normal_sample = np.concatenate(normal_sample, axis=0)
    return xyz_sample, normal_sample

def get_tomo_path(seg_file, tomo_file=None):
    """ get path to tomo file from seg-npz
    :param seg_file: path to seg.npz
    :param tomo_file: tomo_file in seg.npz
        which is relative path to tomo w.r.t seg
    :return: tomo_rel
        tomo_rel: path to tomo relative to pwd
    """
    if tomo_file is None:
        tomo_file = np.load(seg_file, allow_pickle=True)["tomo_file"].item()

    # get absolute path to tomo
    # combine path/to/seg with path/tomo/relative/to/seg
    # use os.path.abspath to not expand symlink (pathlib.Path().resolve does)
    seg_dir = pathlib.Path(seg_file).parent
    tomo_abs = pathlib.Path(os.path.abspath(seg_dir/tomo_file))
    # get relative path to tomo w.r.t. current dir
    pwd = pathlib.Path(os.path.abspath("."))
    tomo_rel = str(tomo_abs.relative_to(pwd))
    return tomo_rel

def star_optics(cs, voltage, apix, box_size):
    """ generate star file sections: optics
    :param cs: spherical aberration
    :param voltage: voltage
    :param apix: pixel in angstrom
    :param box_size: output box size in pixel
    :return: df_optics
    """
    df_optics = pd.DataFrame(
        data=[[1, "opticsGroup1", cs, voltage, apix, apix, box_size, 3]],
        columns=(
            "rlnOpticsGroup", "rlnOpticsGroupName",
            "rlnSphericalAberration", "rlnVoltage",
            "rlnMicrographPixelSize", "rlnImagePixelSize",
            "rlnImageSize", "rlnImageDimensionality"
        ))
    return df_optics

def star_samples(seg_files, key_xyz, key_normal, step_xy=1, step_z=1, factor_offset=0.5):
    """ generate star file sections: micrographs, particles
    :param seg_files: list of seg.npz
    :param key_xyz, key_normal: keys in seg.npz for xyz and normal
    :param step_xy, step_z: int, step in xy and z
    :param factor_offset: index of the first sample = int(factor_offset*step)
    :return: df_micrographs, df_particles, apix
    """
    col_micrographs = ("rlnMicrographName", "rlnOpticsGroup")
    col_particles = (
        "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ",
        "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi",
        "rlnOriginX", "rlnOriginY", "rlnOriginZ",
        "rlnMicrographName", "rlnOpticsGroup"
    )

    data_micrographs = []
    data_particles = []
    voxel_sizes_nm = []

    for seg_file in seg_files:
        if not pathlib.Path(seg_file).is_file():
            continue

        # load seg, get tomo
        seg = np.load(seg_file, allow_pickle=True)
        tomo_file = get_tomo_path(seg_file, seg["tomo_file"].item())
        data_micrographs.append([tomo_file, 1])
        voxel_sizes_nm.append(seg["voxel_size_nm"].item())

        # get particles
        xyz, normal = downsample(
            seg[key_xyz], seg[key_normal],
            step_xy, step_z, factor_offset
        )
        for xyz_i, normal_i in zip(xyz, normal):
            euler_i = normal_to_euler(normal_i)
            data_i = [*xyz_i, *euler_i, 0., 0., 0., tomo_file, 1]
            data_particles.append(data_i)
    
    df_micrographs = pd.DataFrame(data=data_micrographs, columns=col_micrographs)
    df_particles = pd.DataFrame(data=data_particles, columns=col_particles)
    apix = np.mean(voxel_sizes_nm) * 10
    return df_micrographs, df_particles, apix
    
def build_parser():
    """ build parser for sampling.
    :return: parser
    """
    parser = argparse.ArgumentParser(
        prog="segmemsample.py",
        description="Sampling on segmented membranes. Inputs: segmentation results (-seg.npz). Outputs: ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # inputs
    parser.add_argument("seg_files", type=str, nargs='+',
        help="list of str. List of segmentation results (-seg.npz)")

    # outputs
    parser.add_argument("-ot", "--output_tomos", type=str, default="mem-tomos.star",
        help="str. Filename for starfile of tomos."
    )
    parser.add_argument("-op", "--output_particles", type=str, default="mem-particles.star",
        help="str. Filename for starfile of particles."
    )

    # sampling
    parser.add_argument("--keys_seg", type=str, nargs=2, default=("xyz2", "normal2"),
        help="str str. Keys in seg_files for xyz and normal."
    )
    parser.add_argument("-s", "--step_xy_z", type=int, nargs='+', default=(8, 8),
        help="int or int int. Downsampling stepsize in xy and z. If given only one number, then set both steps to this number."
    )
    parser.add_argument("-b", "--box_size", type=int, default=64,
        help="int. Box size in voxels in starfile."
    )

    # optics
    parser.add_argument("--cs", type=float, default=2.7,
        help="float. Spherical aberration."
    )
    parser.add_argument("--voltage", type=float, default=300,
        help="float. Voltage in kV."
    )

    return parser

def memsample(args):
    """ sample on membrane and output to star files
    """
    # generate starfile sections
    step_xy = args.step_xy_z[0]
    step_z = args.step_xy_z[1] if len(args.step_xy_z)>1 else step_xy
    df_micrographs, df_particles, apix = star_samples(
        seg_files=args.seg_files,
        key_xyz=args.keys_seg[0], key_normal=args.keys_seg[1],
        step_xy=step_xy, step_z=step_z,
        factor_offset=0.5
    )
    df_optics = star_optics(
        cs=args.cs, voltage=args.voltage,
        apix=apix, box_size=args.box_size
    )

    # write starfiles
    starfile.write(
        {"optics":df_optics, "micrographs": df_micrographs},
        args.output_tomos, overwrite=True
    )
    starfile.write(
        {"optics":df_optics, "particles": df_particles},
        args.output_particles, overwrite=True
    )


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    memsample(args)
