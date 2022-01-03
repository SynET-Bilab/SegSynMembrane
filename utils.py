#!/usr/bin/env python
""" utils: common utilities
"""
import numpy as np
import pandas as pd
import skimage.filters

__all__ = [
    # basics
    "zscore_image", "minmax_image", "negate_image", "gaussian",
    # orientation
    "rotate_orient", "absdiff_orient",
    # coordinates
    "mask_to_coord", "coord_to_mask", "reverse_coord",
    # segments
    "stats_per_label", "filter_connected_xy", "filter_connected_dz"
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

#=========================
# orientation tools
#=========================

def rotate_orient(O):
    """ rotate orientation by pi/2, then mod pi
    :return: mod(O+pi/2, pi)
    """
    return np.mod(O+np.pi/2, np.pi)

def absdiff_orient(O1, O2):
    """ abs diff between O1, O2, converted to (0, pi/2)
    :param O1, O2: orientations, in (-pi/2, pi/2)
    :return: dO
    """
    dO = np.abs(O1-O2)
    dO = np.where(dO<=np.pi/2, dO, np.pi-dO)
    return dO


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
# segment tools
#=========================

def stats_per_label(L, V_arr, name_arr=None, stats="mean",
    qfilter=0.25, min_size=1):
    """ calc statistics for each label in the image
    :param L: image with integer labels on pixels
    :param V_arr: image array with values on pixels
    :param name_arr: names of values, default=["value0","value1",...]
    :param stats: statistics to apply on values, e.g. "mean", "median"
    :param qfilter: filter out labels if any of their stats < qfilter quantile
    :param min_size: min size of segments
    :return: df_stats
        df_stats: columns=["label","count","value0",...], row=each label
    """
    # positions of nonzero pixels, match with values
    pos = np.nonzero(L)
    if name_arr is None:
        name_arr = [f"value{v}" for v in range(len(V_arr))]
    df_px = pd.DataFrame(
        data=np.transpose([L[pos], *[V[pos] for V in V_arr]]),
        columns=["label", *name_arr]
    ).astype({"label": int})

    # count
    df_count = df_px.value_counts("label").to_frame("count")
    df_count = df_count[df_count["count"]>=min_size]
    df_count = df_count.reset_index()

    # group by labels, then stat
    df_stats = df_px.groupby("label").agg(stats)

    # filter
    threshold = df_stats.quantile(qfilter)
    df_stats = df_stats[df_stats>=threshold].dropna(axis=0)
    df_stats = df_stats.reset_index()

    # merge count into stats
    df_stats = pd.merge(df_count, df_stats, on="label", how="inner")
    return df_stats

def filter_connected_xy(nms, V_arr,
    connectivity=2, stats="median",
    qfilter=0.25, min_size=1):
    """ label by connectivity for each xy-slice, filter out small values
    :param nms: nms image
    :param V_arr: array of valued-images
    :param connectivity: used for determining connected segments, 1 or 2
    :param stats: statistics to apply on values
    :param qfilter: filter out labels if any of their stats < qfilter quantile
    :param min_size: min size of segments
    :return: nms_filt
        nms_filt: filtered nms image
    """
    nms_filt = np.zeros_like(nms)
    for i in range(nms_filt.shape[0]):
        # label by connectivity
        Li = skimage.measure.label(nms[i], connectivity=connectivity)

        # stats
        df_stats = stats_per_label(
            Li, [V[i] for V in V_arr],
            stats=stats, qfilter=qfilter, min_size=min_size
        )

        # filter out labels from image
        nms_filt[i] = nms[i]*np.isin(Li, df_stats["label"])
    return nms_filt

def filter_connected_dz(nms, dzfilter=1, connectivity=2):
    """ label by connectivity in 3d, filter out dz<dzfilter segments
    :param nms: nms image
    :param connectivity: used for determining connected segments, 1 or 2
    :param dzfilter: threshold of z-range
    :return: nms_filt
        nms_filt: filtered nms image
    """
    # label
    L = skimage.measure.label(nms, connectivity=connectivity)
    # z-value of each pixel
    nz = L.shape[0]
    Z = np.ones(L.shape)*np.arange(nz).reshape((-1,1,1))
    # z-range for each label
    df = stats_per_label(L, [Z], name_arr=["z"],
        stats=(lambda x: np.max(x)-np.min(x)),
        qfilter=0, min_size=dzfilter
    )
    # filter
    mask = np.isin(L, df["label"][df["z"] >= dzfilter])
    nms_filt = nms * mask
    return nms_filt


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
