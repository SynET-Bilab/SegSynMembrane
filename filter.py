#!/usr/bin/env python
""" filter
"""

import numpy as np
import pandas as pd
import skimage

__all__ = [
    # statistics
    "stats_per_label", "filter_connected"
]

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

def filter_connected(nms, V_arr, connectivity=2, stats="mean",
    qfilter=0.25, min_size=1):
    """ label by connectivity, filter out small values
    :param nms: nms image
    :param S: Stv image
    :param connectivity: used for determining connected segments, 1 or 2
    :param stats: statistics to apply on values
    :param qfilter: filter out labels if any of their stats < qfilter quantile
    :param min_size: min size of segments
    :return: F
        F: filtered nms image
    """
    F = np.zeros_like(nms)
    for i in range(F.shape[0]):
        # label by connectivity
        Li = skimage.measure.label(nms[i], connectivity=connectivity)

        # stats
        df_stats = stats_per_label(
            Li, [V[i] for V in V_arr],
            stats=stats, qfilter=qfilter, min_size=min_size
        )

        # filter out labels from image
        F[i] = nms[i]*np.isin(Li, df_stats["label"])
    return F
