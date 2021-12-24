#!/usr/bin/env python
""" trace
"""

import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
from synseg.utils import draw_line
from synseg.dtvoting import stick2d

__all__ = [
    # trace within segment
    "trace_within_seg", "trace_within_labels",
    # trace across segments
    "across_next_seg",
    # graph methods
    "pathL_to_LE", "pathLE_to_yx", "pathLEs_distance",
    "order_segs_by_tv",
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

def trace_within_labels(L, O, n_avg):
    """ trace segments from labels
    :param L: 2d label-image
    :param O: 2d orientation-image
    :param n_avg: last n pixels for averaging
    :return: df_segs, traces
        df_segs:
            index=(label,end), looks redundant but is convenient
            columns=["label", "end", "y", "x", "d"]
            end: 1/-1, start/end of yx_trace
            y, x: average coord of the last n_avg points
                reducing the probability that ends "nearly overlap"
            d: average outward direction of the last n_avg points,
                d(-1)-(d(1)+pi) gives continuous change of directions
        traces: { yx: {label: yx_trace}, d: {label: d_trace} }
    """
    labels = np.unique(L[L > 0])
    df_columns = ["label", "end", "y", "x", "d"]
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
                l, 1,  # label, end
                *np.mean(yx_trace[:n_avg], axis=0),  # y, x
                np.mean(d_trace[:n_avg])-np.pi,  # d1, -pi to point outwards
            ]
            df_data.append(data_i1)
            # end -1
            data_i2 = [
                l, -1,  # label, end
                *np.mean(yx_trace[-n_avg:], axis=0),  # y, x
                np.mean(d_trace[-n_avg:])  # d
            ]
            df_data.append(data_i2)
    
    # make dataframe, correct dtype, set index
    df_segs = pd.DataFrame(data=df_data, columns=df_columns)
    df_segs = df_segs.astype({f: int for f in ["label", "end", "y", "x"]})
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
    :return: path_le=[(label1, end_in1),...]
    """
    path_le = [(i, G.nodes[i]["end_in"]) for i in path_l]
    return path_le

def pathLE_to_yx(path, traces_yx, endsegs=(False, False), n_last=1, fill_inter=False):
    """ convert path of (label,end_in) to yx
    :param path: [(label1, end_in1),...], at least two elements
    :param traces_yx: {label1: yx_trace1,...}
    :param endsegs: if include whole start/end segments
    :param n_last: start/end if not included in full, use n_last points
    :param fill_inter: if fills inter-segment gaps by line
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
        # fill inter if selected
        if fill_inter:
            inter_yx = list(draw_line(yx[-1], yx_i[0]))
            yx.extend(inter_yx)
        yx.extend(yx_i)

    # ending segment
    label_e, end_e = path[-1]
    yx_e = traces_yx[label_e][::end_e]
    yx_e = yx_e if endsegs[1] else yx_e[:n_last]
    if fill_inter:
        inter_yx = list(draw_line(yx[-1], yx_e[0]))
        yx.extend(inter_yx)
    yx.extend(yx_e)

    yx = np.asarray(yx)
    return yx

def pathLEs_distance(path1, path2, traces_yx):
    """ Hausdorff distance between two paths of (label,end_in)
    :param path1, path2: [(label11, end_in11),...], at least two elements
    :param traces_yx: {label1: yx_trace1,...}
    :return: dist(Hausdorff distance)
    """
    # convert paths to yxs
    kwargs = dict(traces_yx=traces_yx, fill_inter=True)
    yx1 = pathLE_to_yx(path1, **kwargs)
    yx2 = pathLE_to_yx(path2, **kwargs)

    # compute symmetric Hausdorff distance
    h12 = sp.spatial.distance.directed_hausdorff(yx1, yx2)[0]
    h21 = sp.spatial.distance.directed_hausdorff(yx2, yx1)[0]
    dist = max(h12, h21)
    return dist

#=========================
# build membrane graph
#=========================

def order_segs_by_tv(L, O, sigma, stats=np.median):
    """ apply tv, order by stats (e.g. median)
    :param L, O: 2d label, orientation
    :param sigma: sigma for sticktv, e.g. 2*cleft
    :param stats: function to calculate stats
    :return: df["label", "count", "S_stats"]
        df: sorted by S_stats, descending
    """
    # tv on binary image
    nms = (L>0).astype(np.int_)
    Stv, _ = stick2d(nms, O, sigma)

    # stat for each label
    columns = ["label", "count", "S_stats"]
    data = []
    for l in np.unique(L[L>0]):
        pos_l = np.nonzero(L==l)
        Stv_l = Stv[pos_l]
        data_l = [
            l, len(pos_l[0]), stats(Stv_l)
        ]
        data.append(data_l)

    # make dataframe, sort
    df = pd.DataFrame(data=data, columns=columns)
    df = (df.sort_values("S_stats", ascending=False)
        .reset_index(drop=True)
    )

    return df
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
        L, O, sigma, n_avg
    """
    def __init__(self,
            search_dist_thresh=50,
            path_dist_thresh=2,
            dd_thresh=np.pi/2
        ):
        """ init, save parameters """
        self.search_dist_thresh = search_dist_thresh
        self.path_dist_thresh = path_dist_thresh
        self.dd_thresh = dd_thresh

    def set_traced(self, df_segs, traces_yx):
        """ save result of trace_within_labels()
        """
        self.df_segs = df_segs
        self.traces_yx = traces_yx

    def build_prepost(self, L, O, sigma, n_avg=5):
        """ find prepost membranes starting with 2d labels
        :param L, O: 2d label, orientation
        :param sigma: sigma for sticktv, e.g. 2*cleft
        :param n_avg: param for trace_within_labels()
        :return: Gs, traces_yx
            Gs: array with 4 graphs for two membranes
        """
        # setup trace
        df_segs, traces = trace_within_labels(L, O, n_avg)
        self.set_traced(df_segs, traces["yx"])
        
        # setup search dist: cleft+n_avg
        self.search_dist_thresh = sigma/2 + n_avg*2

        # sort labels by tv
        labels_sorted = (order_segs_by_tv(L, O, sigma)["label"]
            .values.astype(np.int_)
        )
        
        # trace both ends of membrane 1, with largest stv
        Gs = []
        label1 = labels_sorted[0]
        for end in [1, -1]:
            Gi = nx.DiGraph()
            self.build_graph_one(label1, end, Gi)
            Gs.append(Gi)

        # find segment with largest stv from remainings
        labels_used = np.concatenate([list(g.nodes) for g in Gs])
        labels_remained = labels_sorted[
            np.isin(labels_sorted, labels_used, invert=True)
        ]
        label2 = labels_remained[0]

        # trace both ends of membrane 2
        for end in [1, -1]:
            Gi = nx.DiGraph()
            self.build_graph_one(label2, end, Gi)
            Gs.append(Gi)

        return Gs, traces["yx"]
        

    def build_graph_one(self, label, end, G):
        """ build graph starting from (label,end)
        :param label, end: identity of segment end
        :param G: networkx.DiGraph()
            node attrs: {end_in, d_in, end_out, d_out}
                d_in, d_out: wrt root's direction of end_out
        """
        # if label not in G, create node
        if label not in G:
            G.add_node(
                label,
                end_in=-end, d_in=None,
                end_out=end, d_out=0.
            )

        # find next segs
        df_next = across_next_seg(label, end, self.df_segs, self.search_dist_thresh)
        # update direction 
        df_next["d"] = G.nodes[label]["d_out"] + df_next["dd_segs"].values
        df_next["d_out"] = df_next["d"].values + df_next["dd_ends"].values

        for next in df_next.itertuples():
            # skip if total change in direction >= thresh
            if ((np.abs(next.d)>=self.dd_thresh)
                or (np.abs(next.d_out)>=self.dd_thresh)):
                continue

            # add new node (if not existed)
            # prepare for visiting the new node
            if next.label not in G:
                G.add_node(
                    next.label,
                    end_in=next.end, d_in=next.d,
                    end_out=-next.end, d_out=next.d_out
                )
                visit_next = True
            else:
                visit_next = False

            # add new edge
            # (if tentative new edge is not close to existing ones)
            # can still add edge if node existed
            add_new_edge = True
            pathLE_new = pathL_to_LE([label, next.label], G)
            for pathL_exist in nx.all_simple_paths(G, label, next.label):
                pathLE_exist = pathL_to_LE(pathL_exist, G)
                dist_path = pathLEs_distance(
                    pathLE_new, pathLE_exist, self.traces_yx
                )
                # if new is close to one existed, do not add new edge
                if dist_path < self.path_dist_thresh:
                    add_new_edge = False
                    break
            if add_new_edge:
                G.add_edge(label, next.label)

            # build graph from new (label,end)
            if visit_next:
                self.build_graph_one(next.label, -next.end, G)
        return
