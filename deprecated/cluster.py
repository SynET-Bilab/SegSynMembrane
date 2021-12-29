#!/usr/bin/env python
""" cluster: for clustering
"""

import numpy as np
import scipy as sp
import pandas as pd
import skimage
import sklearn.cluster
import numba
import warnings

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

# def propose_epsO(S, O, sigma, eps_r=1.5, q=0.9):
#     """ a proposal of eps_O value
#     find seg with larges sum(stv), set eps_O as its q-quantile
#     :param S: image, [ny, nx], points are identified as S>0
#     :param O: orientation, [ny, nx]
#     :param sigma: sigma for stick tv
#     :param eps_r: max radius of neighbors
#     :param q: quantile for eps_O
#     :return: eps_O
#     """
#     # label connected components
#     L = skimage.measure.label(S, connectivity=2)
#     # strong stick tv
#     df_stv = stats_by_seg(L, O, sigma)
#     # get largest segment
#     idx_seg = df_stv["label"].values[0]
#     mask = (L==idx_seg)
#     # calculate O-distance
#     mat_O, _ = neighbor_distance(S*mask, O*mask, eps_r)
#     # return q-quantile
#     eps_O = np.quantile(mat_O.data, q)
#     return eps_O

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
    warnings.filterwarnings("ignore")
    labels3d = cluster3d_loop(
        S, O,
        eps_r=eps_r, eps_O=eps_O, min_samples=min_samples,
        core_only=core_only, min_cluster_size=min_cluster_size
    )
    labels3d = cluster3d_relabel(labels3d)
    return labels3d

