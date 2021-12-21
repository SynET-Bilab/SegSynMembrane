#!/usr/bin/env python
""" cluster: for clustering
"""

import numpy as np
import scipy as sp
import pandas as pd
import skimage
import sklearn.cluster
import numba

__all__ = [
    # neighbors search
    "neighbor_shift", "neighbor_distance",
    # clustering
    "cluster2d", "cluster3d"
]

#=========================
# neighbors search
#=========================

def neighbor_shift(r):
    """ find neighbors' shifts for pixels <= radius r
    :param r: max radius of neighbors
    :return: dxdy_arr=[[dx1,dy1],[dx2,dy2],...]
    """
    # set reference values of r
    rmax = np.ceil(r).astype(np.int_)+1
    r2 = r**2

    # add to array if (-dx,-dy) have not been added
    dxdy_arr = []
    for dy in range(0, rmax):
        for dx in range(rmax, -rmax, -1):
            dr2 = dx**2+dy**2
            if (dr2 > 0) and (dr2 <= r2):
                if (-dx, -dy) not in dxdy_arr:
                    dxdy_arr.append((dx, dy))
    dxdy_arr = np.array(dxdy_arr)
    return dxdy_arr

def neighbor_distance_one(dxdy, O, L, pos):
    """ calculate distance for each point (y,x) with (y+dy,x+dx)
    :param dxdy: [dx,dy], relative shift of the neighbor
    :param O: orientation of points, [ny, nx]
    :param L: labeling of points, 1-based, [ny, nx]
    :param pos: (pos_ys,pos_xs), corresponding position in image for each index
    :return: idx_pre, idx_post, dO
        1d arrays, distance(idx_pre[i], idx_post[i]) = dO[i]
    """
    # shift L by (-dy, -dx)
    # skimage.transform says SimilarityTransform uses fast routine
    shift = skimage.transform.SimilarityTransform(
        translation=-dxdy
    )
    L_shifted = skimage.transform.warp(
        L, shift.inverse,
        mode="constant", cval=0, clip=True,
        order=0, preserve_range=True
    )

    # index of pre-post pairs
    mask = L_shifted[pos] > 0  # where there is overlap
    idx_pre = L[pos][mask] - 1  # idx for original image
    idx_post = L_shifted[pos][mask] - 1  # index for shifted image

    # positions of pre-post pairs
    pos_pre = tuple(p[idx_pre] for p in pos)
    pos_post = tuple(p[idx_post] for p in pos)

    # # distance - spatial
    # ds = np.sqrt(dx**2+dy**2) * np.ones(len(idx_pre))

    # distance - orientation
    dO = np.abs(O[pos_pre]-O[pos_post])
    dO = np.where(dO < np.pi/2, dO, np.pi-dO)  # wrap around pi
    return idx_pre, idx_post, dO

def neighbor_distance(S, O, r):
    """ calculate distance for each point with neighbors <= radius r
    notations
    index: 1d mapping of points, 0-based
    label: 1d mapping of points, 1-based, index+1
    pos: 2d positions of points, 0-based
    
    :param S: image, [ny, nx], points are identified as S>0
    :param O: orientation, [ny, nx]
    :param r: max radius of neighbors
    :return: mat_dO, pos
        mat_dO: csr_matrix for orientational distance between neighbors
        pos: (pos_ys,pos_xs), corresponding position for each index of mat_dO
    """
    # find points from image
    pos = np.nonzero(S)  # pos=(pos_ys, pos_xs)
    npts = len(pos[0])

    # labeling points
    L = np.zeros(S.shape, dtype=np.int_)
    L[pos] = np.arange(1, npts+1)

    # all shifts with radius r (deduplicated by symmetry)
    dxdy_arr = neighbor_shift(r)

    # calculate orientational distances for each shift
    dO_arr = []
    idx_pre_arr = []
    idx_post_arr = []
    for dxdy in dxdy_arr:
        # distance from pre=(y,x) to post=(y+dy,x+dx)
        idx_pre, idx_post, dO = neighbor_distance_one(dxdy, O, L, pos)
        idx_pre_arr.append(idx_pre)
        idx_post_arr.append(idx_post)
        dO_arr.append(dO)
        # distance from post=(y,x) to pre=(y+dy,x+dx)
        idx_pre_arr.append(idx_post)
        idx_post_arr.append(idx_pre)
        dO_arr.append(dO)
    idx_pre_cat = np.concatenate(idx_pre_arr)
    idx_post_cat = np.concatenate(idx_post_arr)
    dO_cat = np.concatenate(dO_arr)

    # distance array to csr matrix
    mat_dO = sp.sparse.csr_matrix(
        (dO_cat, (idx_pre_cat, idx_post_cat)),
        shape=(npts, npts)
    )
    return mat_dO, pos


#=========================
# DBSCAN clustering
#=========================

def cluster2d(
        S, O,
        eps_r=1.5, eps_O=np.deg2rad(3), min_samples=2,
        core_only=False, min_cluster_size=5
    ):
    """ cluster on 2d slice based on orientational distance and DBSCAN
    filters: keep only core samples; remove noise; sort by cluster size; remove small clusters; label clusters according to size

    :param S, O: shape=(ny,nx)
    :param eps_r: only search neighbors with radius <= eps_r
    :param eps_O, min_samples: DBSCAN parameters
    :param core_only: whether to only keep core points
    :param min_cluster_size: keep clusters with size >= min_cluster_size
    :return: labels2d[ny,nx]
        labels2d: 1-based for points
    """
    # clustering

    # precalculate orientational distance matrix mat_dO
    # only consider neighbors with spatial distance <= eps_r
    mat_dO, pos_pts = neighbor_distance(S, O, r=eps_r)

    # DBSCAN clustering using precomputed distance
    # min_samples: includes self
    # index_clust: index matches that of mat_dO
    clust = sklearn.cluster.DBSCAN(
        eps=eps_O, min_samples=min_samples,
        metric="precomputed"
    )
    clust.fit(mat_dO)
    index_clust = np.copy(clust.labels_)

    # filtering, sorting

    # keep only core points
    # use core to reduce possibility of unwanted overlaps
    if core_only:
        mask_core = np.isin(
            np.arange(len(index_clust)), clust.core_sample_indices_
        )
        pos_pts = tuple(p[mask_core] for p in pos_pts)
        index_clust = index_clust[mask_core]

    # filter out noise
    mask_noise = (index_clust != -1) # index of noise cluster is -1
    pos_pts = tuple(p[mask_noise] for p in pos_pts)
    index_clust = index_clust[mask_noise]

    # sort index by size, from large to small
    clust_df = (pd.Series(index_clust)
        .value_counts(ascending=False)  # count each label
        .to_frame("count").reset_index()  # columns=[index, count]
        .reset_index()  # columns=[level_0, index, count]
        .rename(columns={"level_0": "label"})
    )
    clust_df["label"] += 1  # +1 so that labels start from 1

    # filter out clusters with size < min
    clust_df = clust_df[clust_df["count"] >= min_cluster_size]
    mask_clust = np.isin(index_clust, clust_df["index"])
    pos_pts = tuple(p[mask_clust] for p in pos_pts)
    index_clust = index_clust[mask_clust]

    # map filtered index to labels (1-based)
    clust_map = dict(clust_df[["index", "label"]].values)
    label_clust = np.array([clust_map[l] for l in index_clust])

    # convert labels-array to labels-image
    labels2d = np.zeros(S.shape, dtype=np.int64)
    labels2d[pos_pts] = label_clust
    return labels2d


@numba.njit(parallel=True)
def cluster3d_loop(
        S, O,
        eps_r, eps_O, min_samples,
        core_only, min_cluster_size
    ):
    """ cluster on 2d stack based on orientational distance and DBSCAN
    :param S, O: shape=(nz,ny,nx)
    :param eps_r: only search neighbors with radius <= eps_r
    :param eps_O, min_samples: DBSCAN parameters
    :param core_only: whether to only keep core points
    :return: labels3d[nz,ny,nx]
        labels3d: 1-based for each slice, but labels between slices are not regulated
    """
    # setup
    nz = S.shape[0]
    labels3d = np.zeros(S.shape, dtype=np.int64)

    # loop over slices
    for i in numba.prange(nz):
        with numba.objmode(labels3d_i="intp[:,:]"):
            labels3d_i = cluster2d(
                S[i], O[i],
                eps_r=eps_r, eps_O=eps_O, min_samples=min_samples,
                core_only=core_only, min_cluster_size=min_cluster_size
            )
        labels3d[i] = labels3d_i
    return labels3d

def cluster3d_relabel(labels3d):
    """ relabel labels3d, making labels in each slice distint
    :param labels3d: labels[nz,ny,nx], range 1->n(z) for each slice
    :return: labels3d
        labels3d: reindexed all clusters from 1 consecutively, no duplicates between slices
    """
    # max index of previous slice
    prev = 0
    # relabel slice by slice
    for iz in range(labels3d.shape[0]):
        # max of current slice before relabeling
        label_iz_max = labels3d[iz].max()
        # select real clusters with label>0
        mask = labels3d[iz] > 0
        # shift cluster
        labels3d[iz][mask] += prev
        # update prev
        prev += label_iz_max
    return labels3d

def cluster3d(
        S, O,
        eps_r=1.5, eps_O=np.deg2rad(3), min_samples=2,
        core_only=False, min_cluster_size=5
    ):
    """ cluster on 2d stack based on orientational distance and DBSCAN
    :param S, O: shape=(nz,ny,nx)
    :param eps_r: only search neighbors with radius <= eps_r
    :param eps_O, min_samples: DBSCAN parameters
    :param core_only: whether to only keep core points
    :param min_cluster_size: keep clusters with size >= min_cluster_size
    :return: labels3d[nz,ny,nx]
        labels3d: 1-based for each slice, no duplicates between slices
    """
    labels3d = cluster3d_loop(
        S, O,
        eps_r=eps_r, eps_O=eps_O, min_samples=min_samples,
        core_only=core_only, min_cluster_size=min_cluster_size
    )
    labels3d = cluster3d_relabel(labels3d)
    return labels3d


# #=========================
# # distance metrics
# #=========================

# def prep_xyo(I, O):
#     """ prepare xyo from >0 points in image I and orientation O
#     :param I: image, pixels with value>0 are samples
#     :param O: orientation, in rad, in (-pi/2, pi/2)
#     :return: [x, y, o], shape=(n points, 3)
#     """
#     # set mask
#     mask = I > 0
#     # get xy array
#     yx = mask_to_coord(mask)
#     xy = reverse_coord(yx)
#     # get o array
#     o = O[mask]
#     # concat to xyo
#     xyo = np.concatenate((xy, o[:, np.newaxis]), axis=1)
#     return xyo


# def dist_xy(xyo1, xyo2):
#     """ Euclidean distance in xy
#     :param xyo: [x, y, o]
#     """
#     dxy = np.linalg.norm(xyo2[:2]-xyo1[:2])
#     return dxy


# def dist_o(xyo1, xyo2):
#     """ difference in orientation (rad), mod to (0, pi/2)
#     :param xyo: [x, y, o]
#     """
#     do = np.abs(xyo2[2]-xyo1[2])
#     do_mod = do if do < np.pi/2 else np.pi-do
#     return do_mod


# def dist_xyo(xyo1, xyo2, scale_xy=3, scale_o=10):
#     """ distance between xyo's
#     :param xyo: [x, y, o]
#     :param scale_xy: scale in xy
#     :param scale_o: scale in orientation (deg)
#     :return: d = dist_xy()/scale_xy + dist_o()/pi*180/scale_o
#     """
#     dxy = dist_xy(xyo1, xyo2)
#     do = dist_o(xyo1, xyo2)
#     d = dxy/scale_xy + do/np.pi*180/scale_o
#     return d

# #=========================
# # clustering
# #=========================

# def cluster2d(I, O,
#               scale_xy=3, scale_o=10,
#               eps=2, min_samples=4,
#               core_only=True, remove_noise=True, min_cluster_size=20
#               ):
#     """ cluster on 2d slice using custom metric and DBSCAN
#     :param I, O: shape=(ny,nx)
#     :param scale_xy, scale_o: scales, for setting metrics
#     :param eps, min_samples: parameters of DBSCAN
#     :param core_only: use only core points
#         note this filter is applied before min_cluster_size and noise
#     :param min_cluster_size, remove_noise: filter on clusters
#     :return: xyo[npts, 3], labels[npts]
#     """
#     # note: xyo, labels gets changed when filtered
#     # setups: xyo, metric
#     xyo = prep_xyo(I, O)
#     dist = functools.partial(
#         dist_xyo, scale_xy=scale_xy, scale_o=scale_o
#     )

#     # DBSCAN
#     clust = sklearn.cluster.DBSCAN(
#         eps=eps, min_samples=min_samples,
#         metric=dist
#     )
#     clust.fit(xyo)
#     labels = np.copy(clust.labels_)

#     # filter in core samples
#     # use core to reduce possibility of unwanted overlaps
#     if core_only:
#         mask_core = np.isin(
#             np.arange(len(labels)), clust.core_sample_indices_
#         )
#         xyo = xyo[mask_core]
#         labels = labels[mask_core]

#     # filter out noise
#     if remove_noise:
#         mask_noise = (labels != -1)
#         xyo = xyo[mask_noise]
#         labels = labels[mask_noise]

#     # filter out clusters with size < min
#     labels_df = (pd.Series(labels)
#                  # count by number of each label
#                  .value_counts(ascending=False)
#                  # frame colums=[index, count]
#                  .to_frame("count").reset_index()
#                  .reset_index()  # frame columns=[level_0, index, count]
#                  .rename(columns={"level_0": "label_size"})
#                  )
#     labels_df = labels_df[labels_df["count"] >= min_cluster_size]
#     mask_cluster = np.isin(labels, labels_df["index"])
#     xyo = xyo[mask_cluster]
#     labels = labels[mask_cluster]

#     # reindex based on cluster size
#     labels_map = dict(labels_df[["index", "label_size"]].values)
#     labels = np.array([labels_map[l] for l in labels])

#     return xyo, labels


# def cluster3d(I, O,
#               scale_xy=3, scale_o=10,
#               eps=2, min_samples=4,
#               core_only=True, remove_noise=True, min_cluster_size=20
#               ):
#     """ cluster on 3d using custom metric and DBSCAN
#     :param I, O: shape=(nz,ny,nx)
#     :param scale_xy, scale_o: scales, for setting metrics
#     :param eps, min_samples: parameters of DBSCAN
#     :param core_only: use only core points
#         note this filter is applied before min_cluster_size and noise
#     :param min_cluster_size, remove_noise: filter on clusters
#     :return: xyo_stack, labels_stack
#         xyo_stack: array of xyo[npts, 3]
#         labels_stack: array of labels[npts]
#     """
#     nz = I.shape[0]
#     xyo_stack = []
#     labels_stack = []

#     for i in range(nz):
#         xyo, labels = cluster2d(
#             I[i], O[i],
#             scale_xy=scale_xy, scale_o=scale_o,
#             eps=eps, min_samples=min_samples,
#             core_only=core_only, remove_noise=remove_noise, min_cluster_size=min_cluster_size
#         )
#         xyo_stack.append(xyo)
#         labels_stack.append(labels)

#     return xyo_stack, labels_stack


# #=========================
# # convert xy,labels to image
# #=========================

# def labels_to_2dimage(xyo, labels, yx_shape):
#     """ convert results from dbscan to 2d image
#     :param xyo: array of [x_i, y_i, o_i]
#     :param labels: assumed to range from 0 to n, consecutively
#     :param yx_shape: (ny, nx) for the image
#     :return: labels2d
#         labels2d: image[ny,nx] with labels on corresponding pixels; labeling indexes are shifted by 1, to 1->n+1
#     """
#     labels2d = np.zeros(yx_shape, dtype=np.int64)
#     yx = reverse_coord(xyo[:, :2])
#     for i in np.unique(labels):
#         yx_i = yx[labels == i]
#         labels2d_i = (i+1)*coord_to_mask(yx_i, yx_shape)
#         labels2d += labels2d_i
#     return labels2d


# def labels_reindex3d(labels3d):
#     """ reindex stack of labels2d, making labels in each slice distint
#     :param labels3d: stack of labels2d, clusters in each slice are indexed from 1->n
#     :return: labels3d
#         labels3d: reindexed all clusters from 1 consecutively, no overlap between slices
#     """
#     nz = len(labels3d)
#     # max index of previous slice
#     prev = 0
#     # reindex slice by slice
#     for iz in range(nz):
#         # max of current slice before reindexing
#         label_iz_max = labels3d[iz].max()
#         # select real clusters with label>0
#         mask = labels3d[iz] > 0
#         # shift cluster
#         labels3d[iz][mask] += prev
#         # update prev
#         prev += label_iz_max
#     return labels3d
