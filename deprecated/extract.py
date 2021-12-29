#!/usr/bin/env python
""" extract: stitch clusters and extract connected components
"""

import numpy as np
import pandas as pd
import igraph

__all__ = [
    # graph method
    "build_graph", "graph_components",
    # extraction
    "labels_to_image", "extract_connected"
]

#=========================
# stitch clusters
#=========================

def build_graph_vertices(labels3d):
    """ build vertices from cluster labels
    :param label3d: array[nz,ny,nx], clusters are indexed from 1
    :return: vs_labels, vs_attrs
        vs_labels: labels of clusters
        vs_attrs: {weight}
    """
    # use pandas to sort clusters by weight
    df3d = (pd.Series(labels3d[labels3d > 0])
            .value_counts().to_frame("weight")
        )

    # generate index and attributes
    vs_label = df3d.index.to_list()
    vs_attrs = dict(
        weight=df3d["weight"].to_list(),
    )
    return vs_label, vs_attrs

def build_graph_edges_adjacent(iz, labels3d, dict_label_vid):
    """ stitch vertices in adjacent slices iz and iz-1
    :param iz: z index, should >=1
    :param label3d: array[nz,ny,nx], clusters are indexed from 1
    :param dict_label_vid: {label: vid}
    :return: edges, es_attrs
        edges: array of edges, (vid1, vid2)
        es_attrs: {weight}
    """
    # setups
    edges = []
    es_attrs = {"weight": []}

    yx_shape = labels3d[0].shape
    labels2d_prev = labels3d[iz-1]
    labels2d_curr = labels3d[iz]

    # unique labels of iz, note to exclude label=0
    labels_uniq_curr = np.unique(labels2d_curr[labels2d_curr>0])

    # loop over current clusters
    for i in labels_uniq_curr:
        # calculate overlap using image-based method
        # value=j+1 indicates cluster-i overlaps with prev cluster-j
        # count each cluster using pandas
        mask_curr_i = np.zeros(yx_shape, dtype=np.int_)
        mask_curr_i[labels2d_curr == i] = 1
        overlap = mask_curr_i * labels2d_prev
        counts = pd.Series(overlap[overlap > 0]).value_counts()

        # generate edges
        # note to convert label to vertex id
        vid_i = dict_label_vid[i]
        edges_curr = [(vid_i, dict_label_vid[j]) for j in counts.index]
        edges.extend(edges_curr)
        es_attrs["weight"].extend(list(counts.values))

    return edges, es_attrs

def build_graph_edges(labels3d, dict_label_vid):
    """ stitch vertices through all slices in the stack
    :param label3d: array[nz,ny,nx], clusters are indexed from 1
    :param dict_label_vid: {label: vid}
    :return: edges, es_attrs
        edges: array of edges, (v1, v2)
        es_attrs: {weight}
    """
    edges = []
    es_attrs = dict()
    nz = labels3d.shape[0]
    # stitch adjacent slices one by one
    for iz in range(1, nz):
        edges_i, es_attrs_i = build_graph_edges_adjacent(
            iz, labels3d, dict_label_vid
        )
        edges.extend(edges_i)
        for key, value in es_attrs_i.items():
            # setdefault: can assign to [] if key is not present
            es_attrs.setdefault(key, []).extend(value)
    return edges, es_attrs

def build_graph(labels3d):
    """ build graph from clusters
    :param label3d: array[nz,ny,nx], clusters are indexed from 1
    :return: g=igraph.Graph()
        vertices: 0 to n, attrs={weight,label}
        edges: attrs={weight}
    """
    # build graph vertices
    # vs["name"]=vs_labels, vertexID=range(0, n)
    g = igraph.Graph()
    vs_label, vs_attrs = build_graph_vertices(labels3d)
    g.add_vertices(vs_label)
    for key, value in vs_attrs.items():
        g.vs[key] = value

    # build graph edges
    dict_label_vid = dict(
        g.get_vertex_dataframe()
        .reset_index()[["name", "vertex ID"]].values
    )
    edges, es_attrs = build_graph_edges(labels3d, dict_label_vid)
    g.add_edges(edges)
    for key, value in es_attrs.items():
        g.es[key] = value

    return g


#=========================
# analysis of stitched
#=========================

def graph_components(g):
    """ extract connected components for graph
    :param: graph from build_graph()
    :return: df_comps, g_comps
        df_comps: columns=[index,nv,ne,v_weight,e_weight]
        g_comps: [vs of comp0, vs of comp1, ...]
    """
    # fields to collect
    columns = [
        "index", "nv", "ne",
        "v_weight", "e_weight"
    ]

    # collect info for each component
    label_comps = []
    df_comps_row = []
    for i, vi in enumerate(g.components()):
        gi = g.subgraph(vi)
        # info of the component
        v_weight = np.sum(gi.vs["weight"])
        e_weight = np.sum(gi.es["weight"])
        comp_i = (
            i, gi.vcount(), gi.ecount(),
            v_weight, e_weight
        )
        df_comps_row.append(comp_i)
        # labels of the vertices
        label_comps.append(gi.vs["name"])

    # make a DataFrame
    df_comps = pd.DataFrame(
        data=df_comps_row,
        columns=columns
    )
    df_comps = (df_comps
        .sort_values("v_weight", ascending=False)
        .reset_index(drop=True)
    )

    return df_comps, label_comps

def labels_to_image(labels3d, labels_selected):
    """ convert selected labels to image (1 at labels, 0 otherwise)
    :param label3d: array[nz,ny,nx], clusters are indexed from 1
    :param labels_selected: array of labels to select from
    :return: I
    """
    mask = np.isin(labels3d, labels_selected)
    I = np.zeros(labels3d.shape, dtype=np.int_)
    I[mask] = 1
    return I

#=========================
# extract largest
#=========================

def extract_connected(labels3d, n_largest=2):
    """ extract connected clusters
    :param label3d: array[nz,ny,nx], clusters are indexed from 1
    :param n_largest: return n_largest connected components
    :return: seg_arr, df_comps
        seg_arr: array of segmented binary image of components
        df_comps: columns=[index,nv,ne,v_weight,e_weight]
    """
    # find connected
    g = build_graph(labels3d)
    df_comps, label_comps = graph_components(g)

    # convert to image
    seg_arr = []
    for i in range(n_largest):
        idx = df_comps["index"][i]
        L_i = labels_to_image(labels3d, label_comps[idx])
        seg_arr.append(L_i)
    return seg_arr, df_comps

