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
    "extract_connected", "stats_per_label", "filter_connected_xy", "filter_connected_dz",
    # grid helpers
    "spans_xy", "wireframe_lengths",
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
    coord = np.asarray(coord)
    mask = np.zeros(shape, dtype=np.int_)
    ndim = coord.shape[1]
    index = tuple(
        np.clip(
            np.round(coord[:, i]).astype(np.int_),
            0, shape[i]-1
        ) for i in range(ndim)
    )
    mask[index] = 1
    return mask

def reverse_coord(coord):
    """ convert (y,x) to (x,y)
    :return: reversed coord
    """
    coord = np.asarray(coord)
    index_rev = np.arange(coord.shape[1])[::-1]
    return coord[:, index_rev]


#=========================
# segment tools
#=========================

def extract_connected(B, n_keep=None, connectivity=2):
    """ extract n_keep largest connected components
    :param B: binary image
    :param n_keep: number of components to keep
    :param connectivity: sense of neighboring, 1(-|) or 2(-|\/)
    :return: yield (count, B_i)
    """
    # label
    L = skimage.measure.label(B, connectivity=connectivity)
    # count
    df = (pd.Series(L[L > 0])
          .value_counts(sort=True, ascending=False)
          .to_frame("count").reset_index()
          )
    # yield
    for item in df.iloc[:n_keep].itertuples():
        B_i = B * (L == item.index)
        yield (item.count, B_i)


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

def filter_connected_xy(B, V_arr, connectivity=2, stats="median",qfilter=0.25, min_size=1):
    """ label by connectivity for each xy-slice, filter out small values
    :param B: binary image
    :param V_arr: array of valued-images
    :param connectivity: used for determining connected segments, 1 or 2
    :param stats: statistics to apply on values
    :param qfilter: filter out labels if any of their stats < qfilter quantile
    :param min_size: min size of segments
    :return: B_filt
        B_filt: filtered binary image
    """
    B_filt = np.zeros_like(B)
    for i in range(B_filt.shape[0]):
        # label by connectivity
        Li = skimage.measure.label(B[i], connectivity=connectivity)

        # stats
        df_stats = stats_per_label(
            Li, [V[i] for V in V_arr],
            stats=stats, qfilter=qfilter, min_size=min_size
        )

        # filter out labels from image
        B_filt[i] = B[i]*np.isin(Li, df_stats["label"])
    return B_filt

def filter_connected_dz(B, dzfilter=1, connectivity=2):
    """ label by connectivity in 3d, filter out dz<dzfilter segments
    :param B: binary image
    :param connectivity: used for determining connected segments, 1 or 2
    :param dzfilter: threshold of z-range
    :return: B_filt
        B_filt: filtered binary image
    """
    # label
    L = skimage.measure.label(B, connectivity=connectivity)
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
    B_filt = B * mask
    return B_filt


#=========================
# grid helpers
#=========================

def spans_xy(B):
    """ calculate span in xy for each z
    :param B: image, shape=(nz, ny, nx)
    :return: dydx
        dydx: 2d np.ndarray, [[dy1, dx1],...]
    """
    nz = B.shape[0]
    dydx = np.zeros((nz, 2))
    for iz in range(nz):
        yx = mask_to_coord(B[iz])
        dydx[iz] = np.ptp(yx, axis=0)
    return dydx

def wireframe_lengths(pts_net, axis):
    """ calculate lengths of wireframe along one axis
    :param pts_net: shape=(nu,nv,3)
    :param axis: u - 0, v - 1
    :return: wires
        wires: 1d np.ndarray, [length1, length2, ...]
    """
    # A, B - axes
    # [dz,dy,dx] along A for each B
    diff_zyx = np.diff(pts_net, axis=axis)
    # len of wire segments along A for each B
    segments = np.linalg.norm(diff_zyx, axis=-1)
    # len of wire along A for each B
    wires = np.sum(segments, axis=axis)
    return wires
