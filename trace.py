#!/usr/bin/env python
""" trace
"""

import functools
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import geomdl.fitting
from synseg.dtvoting import stats_by_seg

__all__ = [
    # trace within segment
    "trace_within_seg", "trace_within_labels",
    # trace across segments
    "across_next_seg",
    # graph tools
    "pathL_to_LE", "pathLE_to_yx",
    "curve_fit_pathLE", "curve_fit_smooth",
    "graph_to_image",
    # build graph
    "MemGraph"
]


#=========================
# trace within segment
#=========================

def within_next_yxs(d_curr):
    """ find next yx candidates according to current direction
    :param d_curr: current direction
    :return: dydx_candidates
        dydx_candidates: three possible dydx's among 8 neighbors
    """
    # direction: keep in range (0, 2pi)
    d_curr = np.mod(d_curr, 2*np.pi)

    # categorize: into bins from
    bins = np.pi*np.arange(0, 16.1, 1)/8
    loc_bin = np.histogram(d_curr, bins=bins)[0].argmax()

    # dict from direction to dydx
    # key = index in histogram
    # value = dydx for d_curr, and neighbors as candidates
    map_dydx = {
        0: [(0, 1), (1, 1), (-1, 1)],  # (pi*0/8 ,pi*1/8)
        1: [(1, 1), (0, 1), (1, 0)],  # (pi*1/8 ,pi*2/8)
        2: [(1, 1), (1, 0), (0, 1)],  # (pi*2/8 ,pi*3/8)
        3: [(1, 0), (1, 1), (1, -1)],  # (pi*3/8 ,pi*4/8)
        4: [(1, 0), (1, -1), (1, 1)],  # (pi*4/8 ,pi*5/8)
        5: [(1, -1), (1, 0), (0, -1)],  # (pi*5/8 ,pi*6/8)
        6: [(1, -1), (0, -1), (1, 0)],  # (pi*6/8 ,pi*7/8)
        7: [(0, -1), (1, -1), (-1, -1)],  # (pi*7/8 ,pi*8/8)
        8: [(0, -1), (-1, -1), (1, -1)],  # (pi*8/8 ,pi*9/8)
        9: [(-1, -1), (0, -1), (-1, 0)],  # (pi*9/8 ,pi*10/8)
        10: [(-1, -1), (-1, 0), (0, -1)],  # (pi*10/8 ,pi*11/8)
        11: [(-1, 0), (-1, -1), (-1, 1)],  # (pi*11/8 ,pi*12/8)
        12: [(-1, 0), (-1, 1), (-1, -1)],  # (pi*12/8 ,pi*13/8)
        13: [(-1, 1), (-1, 0), (0, 1)],  # (pi*13/8 ,pi*14/8)
        14: [(-1, 1), (0, 1), (-1, 0)],  # (pi*14/8 ,pi*15/8)
        15: [(0, 1), (-1, 1), (1, 1)],  # (pi*15/8 ,pi*16/8)
    }

    # obtain candidates
    dydx_candidates = map_dydx[loc_bin]
    return dydx_candidates

def within_next_direction(d_curr, o_next):
    """ convert next orientation to direction that's closest to current direction
    :param d_curr: current direction
    :param o_next: next orientation
    :return: d_next
    """
    # diff if d_next=o_next
    diff1 = np.mod(o_next-d_curr, 2*np.pi)
    diff1 = diff1 if diff1 <= np.pi else diff1-2*np.pi
    # diff if d_next=o_next+pi
    diff2 = np.mod(o_next+np.pi-d_curr, 2*np.pi)
    diff2 = diff2 if diff2 <= np.pi else diff2-2*np.pi

    # pick one with smaller diff
    diff = diff1 if np.abs(diff1) <= np.abs(diff2) else diff2
    d_next = d_curr + diff
    return d_next

def trace_within_one(yx_curr, trace, map_yxd):
    """ trace segment from current (y,x) in one direction
    :param yx_curr: current (y,x)
    :param trace: list of (y,x)'s in the trace
    :param map_yxd: {(y,x): direction}
    :return: success
        success: True if no loop, otherwise False
        action: append yx_next (if exists) to trace
    """
    # update trace
    trace.append(yx_curr)

    # find next yx's
    d_curr = map_yxd[yx_curr]
    dydx_candidates = within_next_yxs(d_curr)

    # visit next yx's
    idx = 0
    dydx_tovisit = [dydx_candidates[idx]]
    for dydx in dydx_tovisit:
        yx_next = (yx_curr[0]+dydx[0], yx_curr[1]+dydx[1])
        # next pixel visited, found loop, return false
        if yx_next in trace:
            return False
        # next pixel is still in segment, trace from it
        elif yx_next in map_yxd:
            # convert next orientation to direction, update dict
            d_next = within_next_direction(d_curr, map_yxd[yx_next])
            map_yxd[yx_next] = d_next
            # trace next
            return trace_within_one(yx_next, trace, map_yxd)
        else:
            idx += 1
            # previous dydx candidate not found, to visit next
            if idx < 3:
                dydx_tovisit.append(dydx_candidates[idx])
            # no more candidates, return true
            else:
                return True

def trace_within_seg(nms, O):
    """ trace a 8-connected segment
    :param nms: shape=(ny,nx), nonmaxsup'ed image
    :param O: orientation
    :return: yx_trace, d_trace, success
        yx_trace: sequence of yx's, [(y1,x1),...]
        d_trace: directions from front to end, [d1,d2,...]
        success: True if no loop, otherwise False
    """
    # get position of pixels
    pos = np.nonzero(nms)

    # set starting point: as a midpoint
    idx_start = int(len(pos[0])/2)
    yx_start = tuple(pos[i][idx_start] for i in [0, 1])

    # trace direction of orientation
    map_yxd_plus = dict(zip(zip(*pos), O[pos]))
    yx_plus = []
    success_plus = trace_within_one(yx_start, yx_plus, map_yxd_plus)
    d_plus = [map_yxd_plus[yx] for yx in yx_plus]

    # trace direction of orientation+pi
    map_yxd_minus = dict(zip(zip(*pos), O[pos]))
    map_yxd_minus[yx_start] = O[yx_start]+np.pi
    yx_minus = []
    success_minus = trace_within_one(yx_start, yx_minus, map_yxd_minus)
    # reverse sequence, align direction with d_plus
    yx_minus_reverse = yx_minus[-1:0:-1]
    d_minus_reverse = [map_yxd_minus[yx]-np.pi for yx in yx_minus_reverse]

    # concatenate plus and minus directions
    yx_trace = yx_minus_reverse + yx_plus
    d_trace = d_minus_reverse + d_plus
    success = success_plus and success_minus
    return yx_trace, d_trace, success

def trace_within_labels(L, O, min_size):
    """ trace segments from labels
    :param L: 2d label-image
    :param O: 2d orientation-image
    :param min_size: min segment size after tracing
        this is also the number of last pixels for averaging
    :return: df_segs, traces
        df_segs:
            index=(label,end), looks redundant but is convenient
            columns=["label", "end", "y", "x", "d"]
            end: 1/-1, start/end of yx_trace
            y, x: average coord of the last min_size points
                reducing the probability that ends "nearly overlap"
            d: average outward direction of the last min_size points,
                d(-1)-(d(1)+pi) gives continuous change of directions
        traces: { yx: {label: yx_trace}, d: {label: d_trace} }
    """
    labels = np.unique(L[L > 0])
    df_columns = ["label", "end", "size", "y", "x", "d"]
    df_data = []
    traces = {"yx": {}, "d": {}}
    # loop over segments
    for l in labels:
        # trace segment
        Ll = L*(L == l)
        yx_trace, d_trace, success = trace_within_seg(Ll, O)
        # process non-cyclic segments
        if success:
            # record trace
            traces["yx"][l] = yx_trace
            traces["d"][l] = d_trace

            # end 1
            data_i1 = [
                l, 1, len(yx_trace),  # label, end, size
                *np.mean(yx_trace[:min_size], axis=0),  # y, x
                np.mean(d_trace[:min_size])-np.pi,  # d1, -pi to point outwards
            ]
            df_data.append(data_i1)
            # end -1
            data_i2 = [
                l, -1, len(yx_trace),  # label, end
                *np.mean(yx_trace[-min_size:], axis=0),  # y, x
                np.mean(d_trace[-min_size:])  # d
            ]
            df_data.append(data_i2)
    
    # make dataframe, filter by size
    df_segs = pd.DataFrame(data=df_data, columns=df_columns)
    df_segs = df_segs[df_segs["size"] >= min_size]
    # correct dtype, set index
    df_segs = df_segs.astype(
        {f: int for f in ["label", "end", "size", "y", "x"]}
    )
    df_segs = df_segs.set_index(zip(df_segs["label"], df_segs["end"]))
    return df_segs, traces


#=========================
# trace across segments
#=========================

def across_next_seg_1end(df_curr, df_next):
    """ find the end (1 or -1) of the next segment
    :param df_next: df_segs subset to end=1 or -1
    :param df_curr: df_segs subset to curr (label,end)
    :return: mask={pos,dir,angle,dist,combined}
        pos: position in plus direction
        dir: direction wrt connection < pi/2
        angle: direction wrt d_curr < pi/2
    """
    # quantities: from curr to next
    # connecting-line
    dxdy_next = df_next[["x", "y"]].values-df_curr[["x", "y"]].values
    # vector for d_curr
    dvec_curr = np.array([np.cos(df_curr["d"]), np.sin(df_curr["d"])])
    # vector for d_next: "-" for pointing from curr to next
    dvec_next = np.transpose([
        -np.cos(df_next["d"].values),
        -np.sin(df_next["d"].values),
    ])

    # masks
    mask = {}
    # position of next wrt curr direction: should > 0
    mask["pos"] = np.sum(dxdy_next*dvec_curr, axis=1) > 0
    # direction of next wrt connecting-line: should < pi/2
    mask["dir"] = np.sum(dxdy_next*dvec_next, axis=1) > 0
    # direction of next wrt curr direction: should < pi/2
    mask["angle"] = np.sum(dvec_curr*dvec_next, axis=1) > 0
    # combine masks
    mask["combined"] = mask["dir"] & mask["angle"]
    return mask

def across_next_seg(label, end, df_segs, search_dist_thresh):
    """ find next segment starting from current one
    :param label, end: segment's label and end
    :param df_segs: result of trace_within_labels()
    :param search_dist_thresh: discard points whose distance >= threshold
    :return: df_next[label,end,y,x,d,dist,dd_segs,dd_ends]
        dist: from curr end to next end
        dd_segs: directional change from curr seg to next
        dd_ends: directional change from next seg end to the other end
    """
    # subset df_segs to curr and next 1(start) and -1(end)
    df_curr = df_segs.loc[(label, end)]
    df_next_s = df_segs[df_segs["end"] == 1].sort_values("label")
    df_next_e = df_segs[df_segs["end"] == -1].sort_values("label")
    assert np.all(df_next_e["label"].values == df_next_s["label"].values)

    # get masks for two ends of next
    mask_s = across_next_seg_1end(df_curr, df_next_s)
    mask_e = across_next_seg_1end(df_curr, df_next_e)
    # both ends in plus direction
    mask_pos = mask_s["pos"] & mask_e["pos"]
    # only one end has a small direction change
    mask_dir = np.logical_xor(mask_s["dir"], mask_e["dir"])

    # combine all masks
    # &dir: the end with small dir change is retained
    df_next = pd.concat([
        df_next_s[mask_pos & mask_dir & mask_s["dir"] & mask_s["angle"]],
        df_next_e[mask_pos & mask_dir & mask_e["dir"] & mask_e["angle"]]
    ])

    # distance: calculate, thresholding, sort
    dxdy_next = df_next[["x", "y"]].values-df_curr[["x", "y"]].values
    df_next["dist"] = np.linalg.norm(dxdy_next, axis=1)
    df_next = df_next[
        df_next["dist"] < search_dist_thresh
    ].sort_values("dist")

    # change in directions from current segment to the next
    dd_segs = np.mod(
        df_next["d"].values+np.pi-df_curr["d"],
        2*np.pi
    )
    dd_segs = np.where(dd_segs <= np.pi, dd_segs, dd_segs-2*np.pi)
    df_next["dd_segs"] = dd_segs

    # change in directions from next segment's end-in to end-out
    n_next = len(df_next)
    d_end_s = df_segs.loc[zip(df_next["label"], [1]*n_next), "d"].values
    d_end_e = df_segs.loc[zip(df_next["label"], [-1]*n_next), "d"].values
    dd_ends = d_end_e - (d_end_s + np.pi)  # d-change from end -1 to 1
    dd_ends = dd_ends*df_next["end"].values  # consider actual end
    df_next["dd_ends"] = dd_ends

    return df_next


#=========================
# graph tools
#=========================

def pathL_to_LE(path_l, G):
    """ convert path of labels to path of (label,end_in)
    :param path_l: [label1,label2,...]
    :param G: graph, provides end_in info for each node(label)
    :return: path_le=((label1, end_in1),...)
    """
    path_le = tuple((i, G.nodes[i]["end_in"]) for i in path_l)
    return path_le

def pathLE_to_yx(path, traces_yx, endsegs=(False, False), n_last=1):
    """ convert path of (label,end_in) to yx
    :param path: ((label1, end_in1),...), at least two elements
    :param traces_yx: {label1: yx_trace1,...}
    :param endsegs: if include whole start/end segments
    :param n_last: start/end if not included in full, use n_last points
    :return: yx=[(y1,x1),...]
    """
    if len(path) < 2:
        raise ValueError("len(path) should >= 2")

    yx = []
    # starting segment
    # end: can be used as step
    label_s, end_s = path[0]
    yx_s = traces_yx[label_s][::end_s]
    yx_s = yx_s if endsegs[0] else yx_s[-n_last:]
    yx.extend(yx_s)

    # middle segments
    for i in range(1, len(path)-1):
        label_i, end_i = path[i]
        yx_i = traces_yx[label_i][::end_i]
        yx.extend(yx_i)

    # ending segment
    label_e, end_e = path[-1]
    yx_e = traces_yx[label_e][::end_e]
    yx_e = yx_e if endsegs[1] else yx_e[:n_last]
    yx.extend(yx_e)

    yx = np.asarray(yx)
    return yx

def curve_fit_pathLE(path, traces_yx, n_last=5):
    """ NURBS curve fitting of pathLE
    :param path: ((label1, end_in1),...), at least two elements
    :param traces_yx: {label1: yx_trace1,...}
    :param n_last: use n_last points for segments at ends
    :return: fit, yx
        fit: geomdl Curve
        yx: original data points
    """
    # convert path to yx
    yx = pathLE_to_yx(path, traces_yx,
        n_last=n_last, endsegs=(False, False))
    # ctrlpts: 1 for every 5 samples, 2 for each gap
    ctrlpts_size = int(np.round(len(yx)/n_last))+2*(len(path)-1)
    # nurbs fit
    fit = geomdl.fitting.approximate_curve(
        yx.tolist(), degree=3, ctrlpts_size=ctrlpts_size
    )
    return fit, yx

def curve_fit_smooth(fit):
    """ test if fitting is smooth
    :param fit: geomdl Curve from fitting
    :return: True/False
        True if all ctrl pts are in the same direction
    """
    dydx = np.diff(fit.ctrlpts, axis=0)
    dots = np.sum(dydx[1:]*dydx[:-1], axis=1)
    test = np.all(dots>=0)
    return test

def graph_to_image(L, G):
    """ convert graph to image
    :param L: 2d label image
    :param G: graph, nodes are labels
    :return: I
        I: binary image, 1 at positions corresponding to graph
    """
    I = np.zeros(L.shape, dtype=np.int_)
    mask = np.isin(L, G.nodes)
    I[mask] = 1
    return I


#=========================
# build membrane graph
#=========================

class MemGraph():
    """ class for building membrane graph recursively
    usage: mg=MemGraph(); Gs,traces_yx=mg.build_prepost(L,O,sigma)

    __init__: parameters
        search_dist_thresh: search next segment within this threshold
        path_dist_thresh: add new edge only if dist >= this threshold
        dd_thresh: total change in direction should < this threshold
    build_graph_one: parameters
        label, end: segment's label and end
        G: networkx.DiGraph()
    build_prepost: parameters
        L, O, sigma, min_size
    """
    def __init__(self,
            min_size=5,
            search_dist_thresh=50,
            path_dist_thresh=2,
            dd_thresh=np.pi/2
        ):
        """ init, save parameters """
        # params
        self.min_size = min_size
        self.search_dist_thresh = search_dist_thresh
        self.path_dist_thresh = path_dist_thresh
        self.dd_thresh = dd_thresh
        # variables to be used later
        self.df_segs = pd.DataFrame()
        self.traces_yx = {}
        self.G = nx.DiGraph()

    def set_traced_segs(self, df_segs, traces_yx):
        """ save result of trace_within_labels()
        """
        self.df_segs = df_segs
        self.traces_yx = traces_yx

    def reset_graph(self):
        """ reset directed graph, clear cache for fitting """
        self.G = nx.DiGraph()
        self.curve_fit_pathLE_cached.cache_clear()

    @functools.lru_cache
    def curve_fit_pathLE_cached(self, pathLE):
        fit, _ = curve_fit_pathLE(
            pathLE, self.traces_yx, n_last=self.min_size
        )
        return fit

    def graph_add_node(self, label, end_out, node_next):
        """ add node to graph if not existed
        :param label, end_out: for current node
        :param node_next: next node defined in build_graph_one()
        :return: add
            add: True/False for whether to add the node
        """
        # threshold on direction change
        if ((np.abs(node_next.d) >= self.dd_thresh)
            or (np.abs(node_next.d_out) >= self.dd_thresh)):
            return False

        # smoothness test: via NURBS control points
        pathLE_new = ((label, -end_out), (node_next.label, node_next.end))
        fit_new = self.curve_fit_pathLE_cached(pathLE_new)
        smooth = curve_fit_smooth(fit_new)
        if smooth:
            return True

        # otherwise
        else:
            return False

    def graph_add_edge(self, label, end_out, node_next):
        """ add edge to graph if no overlap
        :param label, end_out: for current node
        :param node_next: next node defined in build_graph_one()
        :return: add
            add: True/False for whether to add the edge
        """
        # fit new
        pathLE_new = ((label, -end_out), (node_next.label, node_next.end))
        fit_new = self.curve_fit_pathLE_cached(pathLE_new)

        # loop through existed paths
        pathL_exist_arr = nx.all_simple_paths(
            self.G, label, node_next.label)
        for pathL_exist in pathL_exist_arr:
            # fit exist
            pathLE_exist = tuple(pathL_to_LE(pathL_exist, self.G))
            fit_exist = self.curve_fit_pathLE_cached(pathLE_exist)

            # distance from new to exist
            dist_path = sp.spatial.distance.directed_hausdorff(
                fit_new.evalpts, fit_exist.evalpts
            )[0]
            # if two fits are close, do not add edge
            if dist_path < self.path_dist_thresh:
                return False

        # if new is not close to any existed, add edge
        return True

    def build_graph_one(self, label, end_out):
        """ build graph starting from (label,end_out)
        :param label, end: identity of segment end
        :param G: networkx.DiGraph()
            node attrs: {end_in, d_in, end_out, d_out}
                d_in, d_out: wrt root's direction of end_out
        """
        # if label not in G, create node
        if label not in self.G:
            self.G.add_node(
                label,
                end_in=-end_out, d_in=None,
                end_out=end_out, d_out=0.
            )

        # find next segs
        df_next = across_next_seg(
            label, end_out, self.df_segs, self.search_dist_thresh)
        # update direction
        df_next["d"] = self.G.nodes[label]["d_out"] + df_next["dd_segs"].values
        df_next["d_out"] = df_next["d"].values + df_next["dd_ends"].values

        # update graph and visit next node
        for node_next in df_next.itertuples():
            # add new node
            if node_next.label not in self.G:
                add_node = self.graph_add_node(label, end_out, node_next)
                if add_node:
                    self.G.add_node(
                        node_next.label,
                        end_in=node_next.end, d_in=node_next.d,
                        end_out=-node_next.end, d_out=node_next.d_out
                    )
            else:
                add_node = False

            # add new edge
            # if node existed before or newly added
            if (node_next.label in self.G) or add_node:
                add_edge = self.graph_add_edge(label, end_out, node_next)
                if add_edge:
                    self.G.add_edge(label, node_next.label)
            
            # if new node is added, visit next
            if add_node:
                self.build_graph_one(node_next.label, -node_next.end)
        return

    def build_prepost(self, L, O, sigma):
        """ find prepost membranes starting with 2d labels
        :param L, O: 2d label, orientation
        :param sigma: sigma for sticktv, e.g. 2*cleft
        :param min_size: param for trace_within_labels()
        :return: Gs, traces_yx
            Gs: array with 4 graphs for two membranes
        """
        # setup trace
        df_segs, traces = trace_within_labels(L, O, min_size=self.min_size)
        self.set_traced_segs(df_segs, traces["yx"])
        
        # # setup search dist: cleft+min_size
        # self.search_dist_thresh = sigma/2 + self.min_size*2

        # sort labels by tv
        labels_sorted = stats_by_seg(L, O, sigma, stats=np.sum)["label"].values
        
        # trace both ends of membrane 1, with largest stv
        Gs = []
        label1 = labels_sorted[0]
        for end in [1, -1]:
            self.reset_graph()
            self.build_graph_one(label1, end)
            Gs.append(self.G)

        # find segment with largest stv from remainings
        labels_used = np.concatenate([list(g.nodes) for g in Gs])
        labels_remained = labels_sorted[
            np.isin(labels_sorted, labels_used, invert=True)
        ]
        label2 = labels_remained[0]

        # trace both ends of membrane 2
        for end in [1, -1]:
            self.reset_graph()
            self.build_graph_one(label2, end)
            Gs.append(self.G)

        self.reset_graph()
        return Gs, traces["yx"]
