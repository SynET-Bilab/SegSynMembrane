#!/usr/bin/env python
""" utils: utilities
"""

import tempfile
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mrcfile

__all__ = [
    # basics
    "zscore", "negate",
    # mrc
    "read_mrc", "write_mrc",
    # model
    "read_model",
]


#=========================
# basic processing
#=========================

def zscore(I):
    """ return zscore of I """
    z = (I-np.mean(I))/np.std(I)
    return z

def negate(I):
    """ switch between white and dark foreground, zscore->negate
    """
    std = np.std(I)
    if std > 0:
        return -(I-np.mean(I))/std
    else:
        return np.zeros_like(I)


#=========================
# mrc
#=========================

def read_mrc(mrcname, return_negated=False):
    """ read 3d data from mrc
    :param return_negated: if True, return negated image
        convenient for loading original black foreground tomo
    :return: data, voxel_size
    """
    with mrcfile.open(mrcname, permissive=True) as mrc:
        data = mrc.data
        voxel_size = mrc.voxel_size
    if return_negated:
        data = negate(data)
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
# plotting
#=========================

def setup_subplots(n, shape, figsize1):
    """ setup subplots
    :param n: number of subplots, can be different from shape
    :param shape: (nrows, ncols), either can be None
    :param figsize1: size of one subplot
    :return: fig, axes
    """
    # setup shape
    if shape is None:
        shape = (1, n)
    elif (shape[0] is None) and (shape[1] is None):
        shape = (1, n)
    elif shape[0] is None:  # (None, ncols)
        shape = (int(np.ceil(n/shape[1])), shape[1])
    elif shape[1] is None:  # (nrows, None)
        shape = (shape[0], int(np.ceil(n/shape[0])))

    # setup figure
    figsize = (
        figsize1[0]*shape[1],  # size_x * ncols
        figsize1[1]*shape[0]  # size_y * nrows
    )
    fig, axes = plt.subplots(
        nrows=shape[0], ncols=shape[1],
        sharex=True, sharey=True,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False  # always return axes as 2d array
    )
    return fig, axes

def imshow(
        I_arr, shape=None, style="custom",
        vrange=None, qrange=(0, 1),
        cmap="gray", colorbar=True, colorbar_shrink=0.6,
        title_arr=None, suptitle=None,
        supxlabel="x/pixel", supylabel="y/pixel",
        figsize1=(3.5, 3.5), save=None
    ):
    """ show multiple images
    :param I_arr: a 1d list of images
    :param shape: (nrows, ncols), will auto set if either is None
    :param style: custom, gray, orient
    :param vrange, qrange: range of value(v) or quantile(q)
    :param cmap, colorbar, colorbar_shrink: set colors
    :param title_arr, suptitle, supxlabel, supylabel: set labels
    :param figsize1: size of one subplot
    :param save: name to save fig
    :return: fig, axes
    """
    # setup styles
    # grayscale image: adjust qrange
    if style == "gray":
        cmap = "gray"
        vrange = None
        qrange = (0.02, 0.98)
    # orientation: circular cmap, convert rad to deg
    elif style == "orient":
        cmap = "hsv"
        vrange = (0, 180)
        I_arr = [np.mod(I, np.pi)/np.pi*180 for I in I_arr]
    elif style == "custom":
        pass
    else:
        raise ValueError("style options: custom, gray, orient")

    # setup subplots
    fig, axes = setup_subplots(len(I_arr), shape, figsize1)
    shape = axes.shape

    # plot on each ax
    for idx1d, I in enumerate(I_arr):
        # get 2d index
        idx2d = np.unravel_index(idx1d, shape)

        # setup vrange
        if vrange is not None:
            vmin, vmax = vrange
        else:
            vmin = np.quantile(I, qrange[0])
            vmax = np.quantile(I, qrange[1])
        
        # plot
        axes[idx2d].set_aspect(1)
        im = axes[idx2d].imshow(
            I, vmin=vmin, vmax=vmax,
            cmap=cmap, origin="lower"
        )

        # setup colorbar, title
        if colorbar:
            fig.colorbar(im, ax=axes[idx2d], shrink=colorbar_shrink)
        if title_arr is not None:
            axes[idx2d].set_title(title_arr[idx1d])
        
    # setup fig title
    fig.suptitle(suptitle)
    fig.supxlabel(supxlabel)
    fig.supylabel(supylabel)

    # save fig
    if save is not None:
        fig.savefig(save)

    return fig, axes


def scatter(
        xy_arr,
        labels_arr=None,
        shape=None,
        marker_size=0.1,
        cmap="viridis", colorbar=True, colorbar_shrink=0.6,
        title_arr=None, suptitle=None,
        supxlabel="x/pixel", supylabel="y/pixel",
        figsize1=(3.5, 3.5), save=None
    ):
    """ show multiple scatters
    :param xy_arr: 1d list of 2d array [x, y]
    :param labels_arr: 1d list of labels for each xy
    :param shape: (nrows, ncols), will auto set if either is None
    :param cmap, colorbar, colorbar_shrink: set colors
    :param title_arr, suptitle, supxlabel, supylabel: set labels
    :param figsize1: size of one subplot
    :param save: name to save fig
    :return: fig, axes
    """
    # regularize labels_arr
    if labels_arr is None:
        labels_arr = [None]*len(xy_arr)

    # setup subplots
    fig, axes = setup_subplots(len(xy_arr), shape, figsize1)
    shape = axes.shape

    # plot on each ax
    for idx1d, (xy, labels) in enumerate(zip(xy_arr, labels_arr)):
        # get 2d index
        idx2d = np.unravel_index(idx1d, shape)

        # plot
        axes[idx2d].set_aspect(1)
        im = axes[idx2d].scatter(
            xy[:, 0], xy[:, 1],
            s=marker_size,
            c=labels, cmap=cmap
        )

        # setup colorbar, title
        if colorbar:
            fig.colorbar(im, ax=axes[idx2d], shrink=colorbar_shrink)
        if title_arr is not None:
            axes[idx2d].set_title(title_arr[idx1d])

    # setup fig title
    fig.suptitle(suptitle)
    fig.supxlabel(supxlabel)
    fig.supylabel(supylabel)

    # save fig
    if save is not None:
        fig.savefig(save)

    return fig, axes
