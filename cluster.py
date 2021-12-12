#!/usr/bin/env python
""" cluster: for clustering
"""

import functools
import numpy as np
import pandas as pd
import sklearn
import sklearn.cluster
import sklearn.metrics
from synseg.utils import mask_to_coord, coord_to_mask, reverse_coord

__all__ = [
    # coordinates
    "prep_xyo",
    # distance metrics
    "dist_xy", "dist_o", "dist_xyo",
    # clustering
    "cluster2d", "cluster3d",
    # labeling
    "labels_to_2dimage", "labels_reindex3d",
]


#=========================
# distance metrics
#=========================

def prep_xyo(I, O):
    """ prepare xyo from >0 points in image I and orientation O
    :param I: image, pixels with value>0 are samples
    :param O: orientation, in rad, in (-pi/2, pi/2)
    :return: [x, y, o], shape=(n points, 3)
    """
    # set mask
    mask = I > 0
    # get xy array
    yx = mask_to_coord(mask)
    xy = reverse_coord(yx)
    # get o array
    o = O[mask]
    # concat to xyo
    xyo = np.concatenate((xy, o[:, np.newaxis]), axis=1)
    return xyo

def dist_xy(xyo1, xyo2):
    """ Euclidean distance in xy
    :param xyo: [x, y, o]
    """
    dxy = np.linalg.norm(xyo2[:2]-xyo1[:2])
    return dxy

def dist_o(xyo1, xyo2):
    """ difference in orientation (rad), mod to (0, pi/2)
    :param xyo: [x, y, o]
    """
    do = np.abs(xyo2[2]-xyo1[2])
    do_mod = do if do < np.pi/2 else np.pi-do
    return do_mod

def dist_xyo(xyo1, xyo2, scale_xy=3, scale_o=10):
    """ distance between xyo's
    :param xyo: [x, y, o]
    :param scale_xy: scale in xy
    :param scale_o: scale in orientation (deg)
    :return: d = dist_xy()/scale_xy + dist_o()/pi*180/scale_o
    """
    dxy = dist_xy(xyo1, xyo2)
    do = dist_o(xyo1, xyo2)
    d = dxy/scale_xy + do/np.pi*180/scale_o
    return d


#=========================
# clustering
#=========================

def cluster2d(I, O,
        scale_xy=3, scale_o=10,
        eps=2, min_samples=4,
        core_only=True, remove_noise=True, min_cluster_size=20
    ):
    """ cluster on 2d slice using custom metric and DBSCAN
    :param I, O: shape=(ny,nx)
    :param scale_xy, scale_o: scales, for setting metrics
    :param eps, min_samples: parameters of DBSCAN
    :param core_only: use only core points
        note this filter is applied before min_cluster_size and noise
    :param min_cluster_size, remove_noise: filter on clusters
    :return: xyo[npts, 3], labels[npts]
    """
    # note: xyo, labels gets changed when filtered
    # setups: xyo, metric
    xyo = prep_xyo(I, O)
    dist = functools.partial(
        dist_xyo, scale_xy=scale_xy, scale_o=scale_o
    )

    # DBSCAN
    clust = sklearn.cluster.DBSCAN(
        eps=eps, min_samples=min_samples,
        metric=dist
    )
    clust.fit(xyo)
    labels = np.copy(clust.labels_)

    # filter in core samples
    # use core to reduce possibility of unwanted overlaps
    if core_only:
        mask_core = np.isin(
            np.arange(len(labels)), clust.core_sample_indices_
        )
        xyo = xyo[mask_core]
        labels = labels[mask_core]
    
    # filter out noise
    if remove_noise:
        mask_noise = (labels!=-1)
        xyo = xyo[mask_noise]
        labels = labels[mask_noise]

    # filter out clusters with size < min
    labels_df = (pd.Series(labels)
        .value_counts(ascending=False)  # count by number of each label
        .to_frame("count").reset_index()  # frame colums=[index, count]
        .reset_index()  # frame columns=[level_0, index, count]
        .rename(columns={"level_0": "label_size"})
    )
    labels_df = labels_df[labels_df["count"]>=min_cluster_size]
    mask_cluster = np.isin(labels, labels_df["index"])
    xyo = xyo[mask_cluster]
    labels = labels[mask_cluster]

    # reindex based on cluster size
    labels_map = dict(labels_df[["index", "label_size"]].values)
    labels = np.array([labels_map[l] for l in labels])

    return xyo, labels

def cluster3d(I, O,
        scale_xy=3, scale_o=10,
        eps=2, min_samples=4,
        core_only=True, remove_noise=True, min_cluster_size=20
    ):
    """ cluster on 3d using custom metric and DBSCAN
    :param I, O: shape=(nz,ny,nx)
    :param scale_xy, scale_o: scales, for setting metrics
    :param eps, min_samples: parameters of DBSCAN
    :param core_only: use only core points
        note this filter is applied before min_cluster_size and noise
    :param min_cluster_size, remove_noise: filter on clusters
    :return: xyo_stack, labels_stack
        xyo_stack: array of xyo[npts, 3]
        labels_stack: array of labels[npts]
    """
    nz = I.shape[0]
    xyo_stack = []
    labels_stack = []

    for i in range(nz):
        xyo, labels = cluster2d(
            I[i], O[i],
            scale_xy=scale_xy, scale_o=scale_o,
            eps=eps, min_samples=min_samples,
            core_only=core_only, remove_noise=remove_noise, min_cluster_size=min_cluster_size
        )
        xyo_stack.append(xyo)
        labels_stack.append(labels)
    
    return xyo_stack, labels_stack


#=========================
# convert xy,labels to image
#=========================

def labels_to_2dimage(xyo, labels, yx_shape):
    """ convert results from dbscan to 2d image
    :param xyo: array of [x_i, y_i, o_i]
    :param labels: assumed to range from 0 to n, consecutively
    :param yx_shape: (ny, nx) for the image
    :return: labels2d
        labels2d: image[ny,nx] with labels on corresponding pixels; labeling indexes are shifted by 1, to 1->n+1
    """
    labels2d = np.zeros(yx_shape, dtype=np.int64)
    yx = reverse_coord(xyo[:, :2])
    for i in np.unique(labels):
        yx_i = yx[labels == i]
        labels2d_i = (i+1)*coord_to_mask(yx_i, yx_shape)
        labels2d += labels2d_i
    return labels2d

def labels_reindex3d(labels3d):
    """ reindex stack of labels2d, making labels in each slice distint
    :param labels3d: stack of labels2d, clusters in each slice are indexed from 1->n
    :return: labels3d
        labels3d: reindexed all clusters from 1 consecutively, no overlap between slices
    """
    nz = len(labels3d)
    # max index of previous slice
    prev = 0
    # reindex slice by slice
    for iz in range(nz):
        # max of current slice before reindexing
        label_iz_max = labels3d[iz].max()
        # select real clusters with label>0
        mask = labels3d[iz] > 0
        # shift cluster 
        labels3d[iz][mask] += prev
        # update prev
        prev += label_iz_max
    return labels3d
