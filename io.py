#!/usr/bin/env python
""" utils: utilities
"""

import tempfile
import subprocess
import numpy as np
import pandas as pd
import skimage
import mrcfile
from synseg.utils import negate_image

__all__ = [
    # read/write
    "read_mrc", "write_mrc", "read_model",
    # processing
    "read_clip_tomo", "model_to_mask"
]


#=========================
# mrc
#=========================

def read_mrc(mrcname, negate=False):
    """ read 3d data from mrc
    :param negate: if True, return negated image
        convenient for loading original black foreground tomo
    :return: data, voxel_size
    """
    with mrcfile.open(mrcname, permissive=True) as mrc:
        data = mrc.data
        voxel_size = mrc.voxel_size
    if negate:
        data = negate_image(data)
    return data, voxel_size

def write_mrc(data, mrcname, voxel_size=None, dtype=None):
    """ write 3d data to mrc
    :param voxel_size: (x,y,z), or None for auto
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
    subprocess.call(cmd, shell=True)

    # load point
    point = np.loadtxt(point_file)

    # convert to dataframe
    columns = ["object", "contour", "x", "y", "z"]
    dtypes = [int, int, float, float, int]
    point_struct = np.array(
        list(map(tuple, point)), # [(ob1,c1,x1,y1,z1), ...]
        dtype=list(zip(columns, dtypes)) # [("object", int), ...]
    )
    df = pd.DataFrame.from_records(point_struct)
    
    # close tempfile
    temp_file.close()
    return df


#=========================
# processing tomo, model
#=========================

def read_clip_tomo(mrc_file, model_file):
    """ clip mrc according to the range of model
    :param mrc_file, model_file: filename of mrc, model
    :return: data, model, voxel_size, clip_range
        data, model are clipped
        voxel_size: read with mrcfile
        clip_range: {x: (min,max), y:..., z:...}
    """
    # read model
    model = read_model(model_file)

    # set the range of clipping
    # use np.floor/ceil -> int to ensure integers
    clip_range = {
        i: (
            int(np.floor(model[i].min())),
            int(np.ceil(model[i].max()))
        )
        for i in ["x", "y", "z"]
    }

    # read mrc data within clip_range
    sub = tuple(
        slice(clip_range[i][0], clip_range[i][1]+1)
        for i in ["z", "y", "x"]
    )
    with mrcfile.mmap(mrc_file, permissive=True) as mrc:
        data = mrc.data[sub]
        voxel_size = mrc.voxel_size

    # clip model to clip_range
    for i in ["x", "y", "z"]:
        model[i] -= clip_range[i][0]

    return data, model, voxel_size, clip_range

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
    z_given = model["z"].unique()
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
