#!/usr/bin/env python
""" utils: utilities
"""

import numpy as np
import skimage.filters
import matplotlib.pyplot as plt
import mrcfile

__all__ = [
    # basics
    "zscore", "negate", "gaussian",
    # coordinates
    "mask_to_coord", "coord_to_mask", "reverse_coord",
    # mrc
    "read_mrc", "write_mrc",
    # plotting
    "imshow"
]


#=========================
# basic image processing
#=========================

def zscore(I):
    """ return zscore of I """
    z = (I-np.mean(I))/np.std(I)
    return z

def negate(I):
    """ switch foreground between white and dark
    zscore then negate
    """
    std = np.std(I)
    if std > 0:
        return -(I-np.mean(I))/std
    else:
        return np.zeros_like(I)

def gaussian(I, sigma):
    """ gaussian smoothing, a wrapper of skimage.filters.gaussian
    :param sigma: if sigma=0, return I
    """
    if sigma == 0:
        return I
    else:
        return skimage.filters.gaussian(I, sigma, mode="nearest")


#=========================
# coordinates tools
#=========================

def mask_to_coord(mask):
    """ convert mask[y,x] to coordinates (y,x) of points>0
    :return: coord, shape=(npts, mask.ndim)
    """
    coord = np.argwhere(mask)
    return coord

def coord_to_mask(coord, shape):
    """ convert coordinates (y,x) to mask[y,x] with 1's on points
    :return: mask
    """
    mask = np.zeros(shape, dtype=int)
    index = tuple(
        coord[:, i].astype(int) for i in range(coord.shape[1])
    )
    mask[index] = 1
    return mask

def reverse_coord(coord):
    """ convert (y,x) to (x,y)
    :return: reversed coord
    """
    index_rev = np.arange(coord.shape[1])[::-1]
    return coord[:, index_rev]


#=========================
# mrc
#=========================

def read_mrc(mrcname):
    """ read 3d data from mrc """
    #TODO: read headers
    with mrcfile.open(mrcname, permissive=True) as mrc:
        data = mrc.data
    return data

def write_mrc(data, mrcname, dtype=np.float32):
    """ write 3d data to mrc """
    #TODO: set header
    with mrcfile.new(mrcname, overwrite=True) as mrc:
        mrc.set_data(data.astype(dtype))


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
