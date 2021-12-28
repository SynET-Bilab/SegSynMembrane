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

def stats_per_label(L, V, stats="mean"):
    """ calc statistics for each label in the image
    :param L: image with integer labels on pixels
    :param V: image with values on pixels
    :param stats: statistics to apply on values, e.g. "mean", "median"
    :return: df_label
        df_label: columns=["label", "count", "stats"], row=each label
    """
    # positions of nonzero pixels, match with values
    pos = np.nonzero(L)
    df_px = pd.DataFrame(
        data=np.transpose([L[pos], V[pos]]),
        columns=["label", "value"]
    )
    # group by labels
    df_label = (df_px.groupby("label")["value"]
        .agg(["count", stats])
        .rename(columns={stats: "stats"})
        .sort_values("stats", ascending=False).reset_index()
        .astype({f: int for f in ["label", "count"]})
    )
    return df_label

def filter_connected(nms, S, connectivity=2, stats="mean", frac=0.75):
    """ label by connectivity, filter out small values
    :param nms: nms image
    :param S: Stv image
    :param connectivity: used for determining connected segments, 1 or 2
    :param stats: statistics to apply on values
    :param frac: fraction to keep
    :return: F
        F: filtered nms image
    """
    F = np.zeros_like(nms)
    for i in range(F.shape[0]):
        # label by connectivity
        Li = skimage.measure.label(nms[i], connectivity=connectivity)
        # stats
        df_label = stats_per_label(Li, S[i], stats=stats)
        # keep first frac labels
        n_thresh = int(len(df_label)*frac)
        F[i] = nms[i]*np.isin(Li, df_label["label"][:n_thresh])
    return F
