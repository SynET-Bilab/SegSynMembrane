#!/usr/bin/env python
""" utils: common utilities
"""
import numpy as np
import skimage.filters

__all__ = [
    # basics
    "zscore_image", "minmax_image", "negate_image", "gaussian",
    # orientation
    "rotate_orient",
    # coordinates
    "mask_to_coord", "coord_to_mask", "reverse_coord"
]

#=========================
# basic processing
#=========================

def zscore_image(I):
    """ zscore image I """
    z = (I-np.mean(I))/np.std(I)
    return z

def minmax_image(I, qrange=(0, 1), vrange=(0, 1)):
    """ minmax-scale of image I
    :param qrange: clip I by quantile range
    :param vrange: target range of values
    """
    # clip I by quantiles, set by qrange
    I_min = np.quantile(I, qrange[0])
    I_max = np.quantile(I, qrange[1])
    I_clip = np.clip(I, I_min, I_max)

    # scale to
    I_scaled = vrange[0] + (I_clip-I_min)/(I_max-I_min)*(vrange[1]-vrange[0])
    return I_scaled

def negate_image(I):
    """ switch between white and dark foreground, zscore->negate
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

def rotate_orient(O):
    """ rotate orientation by pi/2, then mod pi
    :return: mod(O+pi/2, pi)
    """
    return np.mod(O+np.pi/2, np.pi)

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
    mask = np.zeros(shape, dtype=np.int_)
    index = tuple(
        coord[:, i].astype(np.int_)
        for i in range(coord.shape[1])
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
# deprecated
#=========================

# def draw_line(yx0, yx1):
#     """ wraps skimage.draw.line
#     :param yx0, yx1: [y0,x0], [y1,x1]
#     :return: line_yx=[[y0,x0],...,[yi,xi],...,[y1,x1]]
#     """
#     line_rc = skimage.draw.line(
#         yx0[0], yx0[1],
#         yx1[0], yx1[1],
#     )
#     line_yx = np.transpose(line_rc)
#     return line_yx
