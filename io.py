""" input-output
"""

import tempfile
import subprocess
import numpy as np
import pandas as pd
import skimage
import tslearn.metrics
import mrcfile
from etsynseg import utils

__all__ = [
    # tomo io
    "read_tomo", "read_tomo_clip", "write_tomo",
    # model io
    "read_model", "write_model",
    # model interpolation
    "interpolate_contours", "contours_to_mask_closed", "contours_to_mask_open", "model_to_mask"
]


#=========================
# tomo
#=========================

def read_tomo(tomo_file, negate=False):
    """ read tomo file (mrc)
    :param tomo_file: mrc tomo file
    :param negate: if True, return negated image
        convenient for loading original black foreground tomo
    :return: data, voxel_size
        voxel_size: in A, in mrcfile format (tuple)
    """
    with mrcfile.open(tomo_file, permissive=True) as mrc:
        tomo = mrc.data
        voxel_size = mrc.voxel_size
    if negate:
        tomo = utils.negate_image(tomo)
    return tomo, voxel_size

def read_tomo_clip(tomo_file, clip_range=None):
    """ clip mrc according to the range of model
    :param tomo_file: filename of tomo
    :param clip_range: {x: (min,max), y:..., z:...}
    :return: tomo, voxel_size
        tomo: clipped
        voxel_size: read with mrcfile, in A
    """
    # set clip_range
    if clip_range is None:
        sub = slice(None, None)
    else:
        sub = tuple(
            slice(clip_range[i][0], clip_range[i][1]+1)
            for i in ["z", "y", "x"]
        )

    # read tomo and clip
    with mrcfile.mmap(tomo_file, permissive=True) as mrc:
        tomo = mrc.data[sub]
        voxel_size = mrc.voxel_size

    return tomo, voxel_size

def write_tomo(tomo, tomo_file, voxel_size=None, dtype=None):
    """ write 3d data to mrc
    :param voxel_size: in A, (x,y,z), or a number, or None
    :param dtype: datatype, if None use data.dtype
        mrc-compatible: int8/16, uint8/16, float32, complex64
    """
    # ensure data is numpy array
    tomo = np.asarray(tomo)

    # set mrc-compatible dtype (see mrc doc)
    # https://mrcfile.readthedocs.io/en/latest/usage_guide.html#data-types
    if dtype is None:
        dtype = tomo.dtype
    if np.issubdtype(dtype, np.signedinteger):
        if dtype not in [np.int8, np.int16]:
            dtype = np.int16
    elif np.issubdtype(dtype, np.unsignedinteger):
        if dtype not in [np.uint8, np.uint16]:
            dtype = np.uint16
    elif np.issubdtype(dtype, np.floating):
        if dtype not in [np.float32]:
            dtype = np.float32
    elif np.issubdtype(dtype, np.complexfloating):
        if dtype not in [np.complex64]:
            dtype = np.complex64

    # write new mrc
    # note: first set data, then set voxel_size
    tomo = tomo.astype(dtype)
    with mrcfile.new(tomo_file, overwrite=True) as mrc:
        mrc.set_data(tomo)
        if voxel_size is not None:
            mrc.voxel_size = voxel_size


#=========================
# imod model
#=========================

def read_model(model_file):
    """ load xyz from model file
    :return: DataFrame[object,contour,x,y,z] with correct dtypes
        dtypes: z is also int
        range: object,contour are 1-based; x,y,z are 0-based
    """
    # model2point
    temp_file = tempfile.NamedTemporaryFile(suffix=".point")
    point_file = temp_file.name
    cmd = f"model2point -ob {model_file} {point_file} >/dev/null"
    subprocess.run(cmd, shell=True, check=True)

    # load point
    point = np.loadtxt(point_file)

    # convert to dataframe
    columns = ["object", "contour", "x", "y", "z"]
    dtypes = [int, int, float, float, int]
    point_struct = np.array(
        list(map(tuple, point)), # [(ob1,c1,x1,y1,z1), ...]
        dtype=list(zip(columns, dtypes)) # [("object", int), ...]
    )
    model = pd.DataFrame.from_records(point_struct)
    
    # close tempfile
    temp_file.close()
    return model

def write_model(zyx_arr, model_file, dist_thresh=5):
    """ write points to model
    one object for each zyx in array, one open contour for each consecutive line in each z
    :param zyx_arr: array of zyx=[[z1,y1,x1],...]
    :param model_file: filename of model
    :param dist_thresh: if the distance between adjacent points > this threshold, break contour
    :return: data
        data: object, contour, x, y, z
    """
    # make data from zyx's
    # data format: object, contour, x, y, z
    data_arr = []
    for i_obj, zyx_obj in enumerate(zyx_arr):
        z_obj = zyx_obj[:, 0]
        ct_prev = 0
        for z in np.unique(z_obj):
            xy_z = utils.reverse_coord(zyx_obj[z_obj == z][:, 1:])
            n_z = len(xy_z)

            # break array by distance
            dist_z = np.linalg.norm(np.diff(xy_z, axis=0), axis=1)
            ct_z = np.zeros(n_z, dtype=int)
            ct_z[np.nonzero(dist_z > dist_thresh)[0] + 1] = 1  # jumps
            ct_z = np.cumsum(ct_z)  # cumulate
            ct_z += ct_prev + 1  # add prev
            ct_prev = ct_z[-1]

            ones = np.ones((n_z, 1), dtype=int)
            data_ct = np.concatenate(
                [(i_obj+1)*ones, ct_z.reshape((-1, 1)), xy_z, z*ones],
                axis=1
            )
            data_arr.append(data_ct)
    data = np.concatenate(data_arr)

    # save as point file, convert to model
    temp_file = tempfile.NamedTemporaryFile(suffix=".point")
    point_file = temp_file.name
    # format: int for object, contour; float for x, y, z
    np.savetxt(point_file, data, fmt=(['%d']*2 + ['%.2f']*3))
    # point2model: -op for open contour
    cmd = f"point2model -op {point_file} {model_file} >/dev/null"
    subprocess.run(cmd, shell=True, check=True)
    
    return data


#=========================
# model interpolation
#=========================

def interpolate_two_open(zyx1, zyx2):
    """ interpolate between two z's for open contours
    finds dtw with lower score for the original and the reverse points
    :param zyx1, zyx2: contours' points on the two z's
    :return: path, score
        path: [(i11,i12),(i21,i22),...], correspondence between two contours
    """
    # dtw for both the original sequence and its reverse
    path, score = tslearn.metrics.dtw_path(zyx1, zyx2)
    path_rev, score_rev = tslearn.metrics.dtw_path(zyx1, zyx2[::-1])
    # select one with lower score
    if score_rev < score:
        # get path with original indexes
        # point index: ix in rev = i_rev[ix] in original
        i_rev = np.arange(len(zyx2))[::-1]
        path = [(p[0], i_rev[p[1]]) for p in path_rev]
        score = score_rev
    return path, score

def interpolate_two_closed(zyx1, zyx2):
    """ interpolate between two z's for closed contours
    finds dtw with the lowest score amoing all cycles
    :param zyx1, zyx2: contours' points on the two z's
    :return: path, score
        path: [(i11,i12),(i21,i22),...], correspondence between two contours
    """
    # calc dtw for each possible roll
    n2 = len(zyx2)
    scores = []
    paths_roll = []
    for i in range(n2):
        path_roll_i, score_i = interpolate_two_open(
            zyx1, np.roll(zyx2, i, axis=0)
        )
        scores.append(score_i)
        paths_roll.append(path_roll_i)

    # select the best roll
    i_best = np.argmin(scores)

    # get path with original indexes
    # point index: ix in roll = i_roll[ix] in original
    i_roll = np.roll(np.arange(n2), i_best)
    path = [(p[0], i_roll[p[1]]) for p in paths_roll[i_best]]
    return path, scores[i_best]


def interpolate_contours(zyx, closed=True):
    """ interpolate contours through z, using dynamic time warping (dtw)
    :param zyx: points, ordered in each z
    :param closed: if the contour is treated as closed
    :return: zyx_interp
    """
    zyx = np.round(zyx).astype(int)
    z_given = sorted(np.unique(zyx[:, 0]))
    z_arr = zyx[:, 0]
    zyx_interp = []

    for z1, z2 in zip(z_given[:-1], z_given[1:]):
        # get correspondence between two given contours
        zyx1 = zyx[z_arr == z1]
        zyx2 = zyx[z_arr == z2]
        if not closed:
            path, _ = interpolate_two_open(zyx1, zyx2)
        else:
            path, _ = interpolate_two_closed(zyx1, zyx2)

        # linear interpolation on intermediate z's
        zyx_interp.append(zyx1)
        for zi in range(z1+1, z2):
            zyxi = np.array([
                ((z2-zi)*zyx1[p[0]] + (zi-z1)*zyx2[p[1]])/(z2-z1)
                for p in path
            ])
            zyx_interp.append(zyxi)
        if z2 == z_given[-1]:
            zyx_interp.append(zyx2)

    # round to int
    zyx_interp = np.round(
        np.concatenate(zyx_interp, axis=0)
    ).astype(int)

    return zyx_interp

def contours_to_mask_closed(zyx, shape):
    """ convert closed contours to mask (0,1's)
    :param zyx: points
    :param shape: shape of mask, (nz,ny,nx)
    :return: mask
    """
    # use polygon
    zyx = np.round(zyx).astype(int)
    mask = np.zeros(shape, dtype=int)
    for z in sorted(np.unique(zyx[:, 0])):
        if (z < 0) or (z > shape[0]-1):
            continue
        yx = zyx[zyx[:, 0] == z][:, 1:]
        mask[z] = skimage.draw.polygon2mask(
            shape[1:], yx
        )
    return mask

def contours_to_mask_open(zyx, shape):
    """ convert closed contours to mask (0,1's)
    :param zyx: points
    :param shape: shape of mask, (nz,ny,nx)
    :return: mask
    """
    # use lines
    zyx = np.round(zyx).astype(int)
    mask = np.zeros(shape, dtype=int)
    for z in sorted(np.unique(zyx[:, 0])):
        if (z < 0) or (z > shape[0]-1):
            continue
        # connect lines piecewise
        yx = zyx[zyx[:, 0] == z][:, 1:]
        pos_y = []
        pos_x = []
        for yx0, yx1 in zip(yx[:-1], yx[1:]):
            pos_yi, pos_xi = skimage.draw.line(*yx0, *yx1)
            pos_y.extend(list(pos_yi))
            pos_x.extend(list(pos_xi))
        pos_y = tuple(np.clip(pos_y, 0, shape[1]-1))
        pos_x = tuple(np.clip(pos_x, 0, shape[2]-1))
        mask[z, pos_y, pos_x] = 1
    return mask

def model_to_mask(zyx_mod, shape, closed=True, amend=True):
    """ convert model to mask, dtw-interpolate for missing z's
    :param zyx_mod: model points, ordered in each z
    :param shape: shape of mask, (nz,ny,nx)
    :param closed: if the contour is treated as closed
    :param amend: if amend, i.e. set model intersections to 1
    :return: mask
    """
    zyx_mod = np.round(zyx_mod).astype(int)

    # interpolate
    zyx_interp = interpolate_contours(zyx_mod, closed=closed)

    # convert to mask
    if closed:
        mask = contours_to_mask_closed(zyx_interp, shape)
    else:
        mask = contours_to_mask_open(zyx_interp, shape)
    
    # amend
    if amend:
        z_mod = sorted(np.unique(zyx_mod[:, 0]))
        for z1, z2 in zip(z_mod[:-1], z_mod[1:]):
            z_intersect = (mask[z1]*mask[z2]).astype(bool)
            mask[z1+1:z2, z_intersect] = 1
    
    return mask
