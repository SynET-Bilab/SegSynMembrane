#!/usr/bin/env python
""" extract: stitch clusters and extract connected components
"""

import numpy as np
import pandas as pd
import igraph

__all__ = [
    # stitching
    "build_graph",
    # extraction
    "graph_components", "labels_to_image"
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
    :return: L, df_comps
        L: image of components, labeled from 1
        df_comps: columns=[index,nv,ne,v_weight,e_weight]
    """
    # find connected
    g = build_graph(labels3d)
    df_comps, label_comps = graph_components(g)

    # convert to image
    L = np.zeros(labels3d.shape, dtype=np.int_)
    for i in range(n_largest):
        idx = df_comps["index"][i]
        L_i = labels_to_image(labels3d, label_comps[idx])
        L += (i+1)*L_i
    return L, df_comps

# def subgraph_to_image(
#         g, vertices,
#         xyo_stack, labels_stack, zyx_shape
#     ):
#     """ convert subgraph of clusters to image
#     :param g: graph of clusters
#     :param vertices: vertex IDs in graph, e.g. g_comps[i]
#     :param xyo_stack, labels_stack: results of cluster3d()
#     :param zyx_shape: shape of image, [nz,ny,nx]
#     :return: image I
#     """
#     # generate xyz for each cluster
#     gi = g.subgraph(vertices)
#     xyz = None
#     for i in range(gi.vcount()):
#         # get iz, c2d
#         iz = gi.vs["iz"][i]
#         c2d = gi.vs["c2d"][i]
#         # select points from slice iz
#         mask_c2d = np.asarray(labels_stack[iz]) == c2d
#         xy_i = xyo_stack[iz][mask_c2d][:, :2]
#         # combine xy with z
#         xyz_i = np.concatenate((xy_i, np.ones((len(xy_i), 1))*iz), axis=1)
#         # concaternate xyz's
#         if xyz is None:
#             xyz = xyz_i
#         else:
#             xyz = np.concatenate((xyz, xyz_i), axis=0)

    # # convert xyz to image
    # zyx = reverse_coord(xyz)
    # I = coord_to_mask(zyx, zyx_shape)
    # return I


# #=========================
# # stitch clusters
# #=========================

# def gen_cid_bidict(df3d):
#     """ generate cid bidict from df3d
#     :param df3d: pd.DataFrame(columns=["c3d", "iz", "c2d"])
#     :return: bidict(c3d:(iz, c2d))
#     """
#     cid_map_32 = bidict.bidict(
#         list(zip(
#             df3d["c3d"].values,
#             zip(*df3d[["iz", "c2d"]].values.T)
#         ))
#     )
#     return cid_map_32


# def build_graph_vertices(labels_stack):
#     """ build vertices from each cluster(label)
#     :param labels_stack: [labels_z0, labels_z1, ...]
#     :return:
#         vertices: array of vertices, start from 0
#         vs_attrs: {c3d, iz, c2d, weight}
#         cid_map_32: bidict( c3d:(iz, c2d) )
#     """
#     # get c2ds and weights(no. of pixels) for each iz
#     # df2d: columns=[iz, c2d, weight]
#     df2d_arr = []
#     for iz, labels in enumerate(labels_stack):
#         df2d = (pd.Series(labels)
#                 .value_counts().to_frame("weight")  # sort by weight
#                 .reset_index().rename(columns={"index": "c2d"})
#                 )
#         df2d["iz"] = iz
#         df2d_arr.append(df2d)

#     # combine iz's, reindex
#     # df3d: columns=[c3d, iz, c2d, weight]
#     df3d = (pd.concat(df2d_arr)
#             .reset_index(drop=True)  # reindex from 0
#             .reset_index().rename(columns={"index": "c3d"})
#             )

#     # extract index and attributes
#     vertices = df3d["c3d"].to_list()
#     vs_attrs = dict(
#         c3d=df3d["c3d"].to_list(),
#         iz=df3d["iz"].to_list(),
#         c2d=df3d["c2d"].to_list(),
#         weight=df3d["weight"].to_list(),
#     )

#     # generate bidict: {c3d: (iz, c2d)}
#     cid_map_32 = gen_cid_bidict(df3d)
#     return vertices, vs_attrs, cid_map_32


# def build_graph_edges_adjacent(iz, xyo_stack, labels_stack, yx_shape, cid_map_32):
#     """ stitch vertices in adjacent slices iz and iz-1
#     strategy: convert xy to 2d mask, calculate overlaps
#     :param iz: z index, should >=1
#     :param xyo_stack, labels_stack: results of cluster3d()
#     :param yx_shape: image boundary [ny,nx]
#     :param cid_map_32: bidict(c3d=(iz, c2d))
#     :return:
#         edges: array of edges, (v1, v2)
#         es_attrs: {weight}
#     """
#     # check iz >=1
#     if iz == 0:
#         raise ValueError("iz should >= 1")
#     elif iz >= len(xyo_stack):
#         raise ValueError("iz should < len(xyo_stack)")

#     # convert yx_prev to mask
#     # all clusters are combined to the same mask
#     # cluster-i is assigned with value i+1 (i starts from 0)
#     yx_prev = xyo_stack[iz-1][:, [1, 0]]
#     labels_prev = np.asarray(labels_stack[iz-1])
#     mask_prev = np.zeros(yx_shape, dtype=int)
#     for i in np.unique(labels_prev):
#         yx_prev_i = yx_prev[labels_prev == i]
#         mask_prev_i = (i+1)*coord_to_mask(yx_prev_i, yx_shape)
#         mask_prev += mask_prev_i

#     # calculate overlap between curr and prev, get edges
#     edges = []
#     es_attrs = {"weight": []}
#     yx_curr = xyo_stack[iz][:, [1, 0]]
#     labels_curr = np.asarray(labels_stack[iz])
#     # loop over current clusters
#     for i in np.unique(labels_curr):
#         # convert yx_curr to mask
#         yx_curr_i = yx_curr[labels_curr == i]
#         mask_curr_i = coord_to_mask(yx_curr_i, yx_shape)

#         # calculate overlap
#         # count number of each elements using pandas
#         # value=j+1 indicates overlap with prev cluster-j
#         overlap = mask_curr_i * mask_prev
#         counts = pd.Series(overlap[overlap > 0]).value_counts()

#         # generate edges
#         c3d_i = cid_map_32.inv[(iz, i)]
#         for j, weight in zip(counts.index-1, counts.values):
#             c3d_j = cid_map_32.inv[(iz-1, j)]
#             edges.append((c3d_i, c3d_j))
#             es_attrs["weight"].append(weight)

#     return edges, es_attrs


# def build_graph_edges(xyo_stack, labels_stack, yx_shape, cid_map_32):
#     """ stitch vertices through all slices in the stack
#     :param xyo_stack, labels_stack: results of cluster3d()
#     :param yx_shape: image boundary [ny,nx]
#     :param cid_map_32: bidict(c3d=(iz, c2d))
#     :return:
#         edges: array of edges, (v1, v2)
#         es_attrs: {weight}
#     """
#     edges = []
#     es_attrs = dict()
#     nz = len(xyo_stack)
#     # stitch adjacent slices one by one
#     for iz in range(1, nz):
#         edges_i, es_attrs_i = build_graph_edges_adjacent(
#             iz, xyo_stack, labels_stack, yx_shape, cid_map_32
#         )
#         edges.extend(edges_i)
#         for key, value in es_attrs_i.items():
#             # setdefault: can assign to [] if key is not present
#             es_attrs.setdefault(key, []).extend(value)
#     return edges, es_attrs


# def build_graph(xyo_stack, labels_stack, yx_shape):
#     """ build graph from clusters
#     vertex: each cluster, weight = cluster size
#     edge: clusters connected in adjacent z's, weight = no. overlaps
#     :param xyo_stack, labels_stack: results of cluster3d()
#     :param yx_shape: image boundary [ny,nx]
#     :return: g=igraph.Graph()
#         vertices: 0 to n, attrs=[weight, c3d, iz, c2d]
#         edges: attrs=[weight]
#     """
#     # get vertices and weights
#     vertices, vs_attrs, cid_map_32 = build_graph_vertices(labels_stack)

#     # get edges and weights
#     edges, es_attrs = build_graph_edges(
#         xyo_stack, labels_stack, yx_shape, cid_map_32)

#     # build graph
#     g = igraph.Graph()
#     g.add_vertices(vertices)
#     g.add_edges(edges)
#     for key, value in vs_attrs.items():
#         g.vs[key] = value
#     for key, value in es_attrs.items():
#         g.es[key] = value
#     return g
