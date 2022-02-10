""" input-output
"""

import tempfile
import subprocess
import numpy as np
import pandas as pd
import skimage
import mrcfile
from etsynseg import utils

__all__ = [
    # mrc
    "read_mrc", "write_mrc",
    # model
    "read_model", "model_to_mask", "write_model",
    # processing
    "read_clip_tomo",
]


#=========================
# mrc
#=========================

def read_mrc(mrcname, negate=False):
    """ read 3d data from mrc
    :param negate: if True, return negated image
        convenient for loading original black foreground tomo
    :return: data, voxel_size
        voxel_size: in A
    """
    with mrcfile.open(mrcname, permissive=True) as mrc:
        data = mrc.data
        voxel_size = mrc.voxel_size
    if negate:
        data = utils.negate_image(data)
    return data, voxel_size

def write_mrc(data, mrcname, voxel_size=None, dtype=None):
    """ write 3d data to mrc
    :param voxel_size: in A, (x,y,z), or a number, or None
    :param dtype: datatype, if None use data.dtype
        mrc-compatible: int8/16, uint8/16, float32, complex64
    """
    # ensure data is numpy array
    data = np.asarray(data)

    # set mrc-compatible dtype (see mrc doc)
    # https://mrcfile.readthedocs.io/en/latest/usage_guide.html#data-types
    if dtype is None:
        dtype = data.dtype
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
    data = data.astype(dtype)
    with mrcfile.new(mrcname, overwrite=True) as mrc:
        mrc.set_data(data)
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

def model_to_mask(model, yx_shape):
    """ convert model to mask, interpolate at missing z's
    :param model: DataFrame, result of read_model() or clip_tomo()
    :param yx_shape: shape in yx-dims
    """
    # prepare array to be filled
    #   shape in z: max value + 1
    #   dtype: set to float first, to allow for interpolation
    mask = np.zeros((model["z"].max()+1, *yx_shape))

    # setup mask at given slices
    z_given = sorted(model["z"].unique())
    for z in z_given:
        mask_yx = model[model["z"] == z][["y", "x"]].values
        mask[z] = skimage.draw.polygon2mask(
            yx_shape, mask_yx
        ).astype(float)

    # interpolate at z's between slices
    z_pairs = zip(z_given[:-1], z_given[1:])
    for z_low, z_high in z_pairs:
        for z in range(z_low+1, z_high):
            mask[z] = (mask[z_low]*(z_high-z)
                       + mask[z_high]*(z-z_low)
                       ) / (z_high-z_low)

    # round to int
    mask = np.round(mask).astype(int)
    return mask

def write_model(zyx_arr, model_file):
    """ write points to model
    one object for each zyx in array
    one open contour for each z
    :param zyx_arr: array of zyx=[[z1,y1,x1],...]
    :param model_file: filename of model
    :return: data
        data: object, contour, x, y, z
    """
    # make data from zyx's
    # data format: object, contour, x, y, z
    data_arr = []
    for i_obj, zyx_obj in enumerate(zyx_arr):
        obj = i_obj + 1
        z_obj = zyx_obj[:, 0]
        for i_ct, z_ct in enumerate(np.unique(z_obj)):
            ct = i_ct + 1
            xyz_ct = utils.reverse_coord(zyx_obj[z_obj==z_ct])
            ones = np.ones((len(xyz_ct), 1), dtype=np.int_)
            data_ct = np.concatenate(
                [obj*ones, ct*ones, xyz_ct],
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
# processing tomo, model
#=========================

def read_clip_tomo(tomo_file, clip_range=None):
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
