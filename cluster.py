#!/usr/bin/env python
""" cluster: for clustering
"""

import functools
import numpy as np
import pandas as pd
import sklearn
import sklearn.cluster
import sklearn.metrics
import igraph
import bidict

from synseg.utils import mask_to_coord, coord_to_mask, reverse_coord


__all__ = [
    # encoding
    "prep_xyo",
    # distance metrics
    "dist_xy", "dist_o", "dist_xyo",
    # clustering
    "cluster2d", "cluster3d",
    # stitching
    "gen_cid_bidict", "build_graph",
    # extraction
    "graph_components", "graph_to_image",
]


#=========================
# encoding
#=========================

def prep_xyo(I, O):
    """ prepare xyo from >0 points in image I and orientation O
    :param I: image, pixels with value>0 are samples
    :param O: orientation, in rad, in (-pi/2, pi/2)
    :return: [x, y, o], shape=(n points, 3)
    """
    # set mask
    mask = I>0
    # get xy array
    yx = mask_to_coord(mask)
    xy = reverse_coord(yx)
    # get o array
    o = O[mask]
    # concat to xyo
    xyo = np.concatenate((xy, o[:, np.newaxis]), axis=1)
    return xyo


#=========================
# distance metrics
#=========================

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
    :return: 
        xyo_stack = array of xyo[npts, 3]
        labels_stack = array of labels[npts]
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
# stitching clusters
#=========================

def gen_cid_bidict(df3d):
    """ generate cid bidict from df3d
    :param df3d: pd.DataFrame(columns=["c3d", "iz", "c2d"])
    :return: bidict(c3d:(iz, c2d))
    """
    cid_map_32 = bidict.bidict(
        list(zip(
            df3d["c3d"].values,
            zip(*df3d[["iz", "c2d"]].values.T)
        ))
    )
    return cid_map_32

def build_graph_vertices(labels_stack):
    """ build vertices from each cluster(label)
    :param labels_stack: [labels_z0, labels_z1, ...]
    :return:
        vertices: array of vertices, start from 0
        vs_attrs: {c3d, iz, c2d, weight}
        cid_map_32: bidict( c3d:(iz, c2d) )
    """
    # get c2ds and weights(no. of pixels) for each iz
    # df2d: columns=[iz, c2d, weight]
    df2d_arr = []
    for iz, labels in enumerate(labels_stack):
        df2d = (pd.Series(labels)
                .value_counts().to_frame("weight") # sort by weight
                .reset_index().rename(columns={"index": "c2d"})
                )
        df2d["iz"] = iz
        df2d_arr.append(df2d)

    # combine iz's, reindex
    # df3d: columns=[c3d, iz, c2d, weight]
    df3d = (pd.concat(df2d_arr)
            .reset_index(drop=True) # reindex from 0
            .reset_index().rename(columns={"index": "c3d"})
            )

    # extract index and attributes
    vertices = df3d["c3d"].to_list()
    vs_attrs = dict(
        c3d=df3d["c3d"].to_list(),
        iz=df3d["iz"].to_list(),
        c2d=df3d["c2d"].to_list(),
        weight=df3d["weight"].to_list(),
    )

    # generate bidict: {c3d: (iz, c2d)}
    cid_map_32 = gen_cid_bidict(df3d)
    return vertices, vs_attrs, cid_map_32

def build_graph_edges_adjacent(iz, xyo_stack, labels_stack, yx_shape, cid_map_32):
    """ stitch vertices in adjacent slices iz and iz-1
    strategy: convert xy to 2d mask, calculate overlaps
    :param iz: z index, should >=1
    :param xyo_stack, labels_stack: results of cluster3d()
    :param yx_shape: image boundary [ny,nx]
    :param cid_map_32: bidict(c3d=(iz, c2d))
    :return:
        edges: array of edges, (v1, v2)
        es_attrs: {weight}
    """
    # check iz >=1
    if iz == 0:
        raise ValueError("iz should >= 1")
    elif iz >= len(xyo_stack):
        raise ValueError("iz should < len(xyo_stack)")

    # convert yx_prev to mask
    # all clusters are combined to the same mask
    # cluster-i is assigned with value i+1 (i starts from 0)
    yx_prev = xyo_stack[iz-1][:, [1, 0]]
    labels_prev = np.asarray(labels_stack[iz-1])
    mask_prev = np.zeros(yx_shape, dtype=int)
    for i in np.unique(labels_prev):
        yx_prev_i = yx_prev[labels_prev == i]
        mask_prev_i = (i+1)*coord_to_mask(yx_prev_i, yx_shape)
        mask_prev += mask_prev_i

    # calculate overlap between curr and prev, get edges
    edges = []
    es_attrs = {"weight": []}
    yx_curr = xyo_stack[iz][:, [1, 0]]
    labels_curr = np.asarray(labels_stack[iz])
    # loop over current clusters
    for i in np.unique(labels_curr):
        # convert yx_curr to mask
        yx_curr_i = yx_curr[labels_curr == i]
        mask_curr_i = coord_to_mask(yx_curr_i, yx_shape)

        # calculate overlap
        # count number of each elements using pandas
        # value=j+1 indicates overlap with prev cluster-j
        overlap = mask_curr_i * mask_prev
        counts = pd.Series(overlap[overlap > 0]).value_counts()

        # generate edges
        c3d_i = cid_map_32.inv[(iz, i)]
        for j, weight in zip(counts.index-1, counts.values):
            c3d_j = cid_map_32.inv[(iz-1, j)]
            edges.append((c3d_i, c3d_j))
            es_attrs["weight"].append(weight)
            
    return edges, es_attrs

def build_graph_edges(xyo_stack, labels_stack, yx_shape, cid_map_32):
    """ stitch vertices through all slices in the stack
    :param xyo_stack, labels_stack: results of cluster3d()
    :param yx_shape: image boundary [ny,nx]
    :param cid_map_32: bidict(c3d=(iz, c2d))
    :return:
        edges: array of edges, (v1, v2)
        es_attrs: {weight}
    """
    edges = []
    es_attrs = dict()
    nz = len(xyo_stack)
    # stitch adjacent slices one by one
    for iz in range(1, nz):
        edges_i, es_attrs_i = build_graph_edges_adjacent(
            iz, xyo_stack, labels_stack, yx_shape, cid_map_32
        )
        edges.extend(edges_i)
        for key, value in es_attrs_i.items():
            # setdefault: can assign to [] if key is not present
            es_attrs.setdefault(key, []).extend(value)
    return edges, es_attrs

def build_graph(xyo_stack, labels_stack, yx_shape):
    """ build graph from clusters
    vertex: each cluster, weight = cluster size
    edge: clusters connected in adjacent z's, weight = no. overlaps
    :param xyo_stack, labels_stack: results of cluster3d()
    :param yx_shape: image boundary [ny,nx]
    :return: igraph.Graph(), cid_map_32
        vertices: 0 to n, attrs=[weight, c3d, iz, c2d]
        edges: attrs=[weight]
    """
    # get vertices and weights
    vertices, vs_attrs, cid_map_32 = build_graph_vertices(labels_stack)
    
    # get edges and weights
    edges, es_attrs = build_graph_edges(
        xyo_stack, labels_stack, yx_shape, cid_map_32)

    # build graph
    g = igraph.Graph()
    g.add_vertices(vertices)
    g.add_edges(edges)
    for key, value in vs_attrs.items():
        g.vs[key] = value
    for key, value in es_attrs.items():
        g.es[key] = value
    return g


#=========================
# analysis of stitched
#=========================

def graph_components(g):
    """ 
    :param: graph from build_graph()
    :return:
        df_comps: columns=[index,nv,ne,v_weight,e_weight,iz_min,iz_max,iz_span]
        g_comps: [vs of comp0, vs of comp1, ...]
    """
    # fields to collect
    columns = [
        "index", "nv", "ne",
        "v_weight", "e_weight",
        "iz_min", "iz_max", "iz_span"
    ]

    # collect info for each component
    g_comps = list(g.components())
    comp_arr = []
    for i, vi in enumerate(g_comps):
        gi = g.subgraph(vi)
        v_weight = np.sum(gi.vs["weight"])
        e_weight = np.sum(gi.es["weight"])
        iz_min = np.min(gi.vs["iz"])
        iz_max = np.max(gi.vs["iz"])
        comp_i = (
            i, gi.vcount(), gi.ecount(),
            v_weight, e_weight,
            iz_min, iz_max, iz_max-iz_min
        )
        comp_arr.append(comp_i)
    
    # make a DataFrame
    df_comps = pd.DataFrame(
        data=comp_arr,
        columns=columns
    )
    df_comps = (df_comps
        .sort_values("v_weight", ascending=False)
        .reset_index(drop=True)
    )
    return df_comps, g_comps

def subgraph_to_image(
        g, vertices,
        xyo_stack, labels_stack, zyx_shape
    ):
    """ convert subgraph of clusters to image
    :param g: graph of clusters
    :param vertices: vertex IDs in graph, e.g. g_comps[i]
    :param xyo_stack, labels_stack: results of cluster3d()
    :param zyx_shape: shape of image, [nz,ny,nx]
    :return: image I
    """
    # generate xyz for each cluster
    gi = g.subgraph(vertices)
    xyz = None
    for i in range(gi.vcount()):
        # get iz, c2d
        iz = gi.vs["iz"][i]
        c2d = gi.vs["c2d"][i]
        # select points from slice iz
        mask_c2d = np.asarray(labels_stack[iz]) == c2d
        xy_i = xyo_stack[iz][mask_c2d][:, :2]
        # combine xy with z
        xyz_i = np.concatenate((xy_i, np.ones((len(xy_i), 1))*iz), axis=1)
        # concaternate xyz's
        if xyz is None:
            xyz = xyz_i
        else:
            xyz = np.concatenate((xyz, xyz_i), axis=0)

    # convert xyz to image
    zyx = reverse_coord(xyz)
    I = coord_to_mask(zyx, zyx_shape)
    return I
