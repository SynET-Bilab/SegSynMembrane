""" utils: common utilities
"""
import numpy as np
import pandas as pd
import scipy as sp
import skimage
import open3d as o3d

__all__ = [

    # segments
    "stats_per_label", "filter_connected_xy", "filter_connected_dz",

]

#=========================
# orientation tools
#=========================






#=========================
# segment tools
#=========================




def stats_per_label(L, V_arr, name_arr=None, stats="mean",
    qfilter=0.25, min_size=1):
    """ calc statistics for each label in the image
        L: image with integer labels on pixels
        V_arr: image array with values on pixels
        name_arr: names of values, default=["value0","value1",...]
        stats: statistics to apply on values, e.g. "mean", "median"
        qfilter: filter out labels if any of their stats < qfilter quantile
        min_size: min size of segments
    Returns: df_stats
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
        B: binary image
        V_arr: array of valued-images
        connectivity: used for determining connected segments, 1 or 2
        stats: statistics to apply on values
        qfilter: filter out labels if any of their stats < qfilter quantile
        min_size: min size of segments
    Returns: B_filt
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
        B: binary image
        connectivity: used for determining connected segments, 1 or 2
        dzfilter: minimal value of z-range
    Returns: B_filt
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

# def spans_xy(B):
#     """ calculate span in xy for each z
#         B: image, shape=(nz, ny, nx)
#     Returns: dydx
#         dydx: 2d np.ndarray, [[dy1, dx1],...]
#     """
#     nz = B.shape[0]
#     dydx = np.zeros((nz, 2))
#     for iz in range(nz):
#         yx = pixels2points(B[iz])
#         dydx[iz] = np.ptp(yx, axis=0)
#     return dydx

