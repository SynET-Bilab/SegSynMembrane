#!/usr/bin/env python
""" trace
"""

import numpy as np
import pandas as pd

__all__ = [
    # trace within segment
    "trace_within_seg", "segs_to_frame",
    # trace across segments
    "across_next_seg"
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

def segs_to_frame(L, O, n_avg):
    """ convert segments ends info to dataframe
    :param L: 2d label-image
    :param O: 2d orientation-image
    :param n_avg: last n pixels for averaging
    :return: df["label", "end", "y", "x", "d"]
        index: (label,end), looks redundant but is convenient
        end: -1 or 1
        y, x: average coord of the last n_avg points
            reducing the probability that ends "nearly overlap"
        d: average outward direction of the last n_avg points,
            d(1)-(d(-1)+pi) gives continuous change of directions
    """
    labels = np.unique(L[L > 0])
    columns = ["label", "end", "y", "x", "d"]
    data = []
    # loop over segments
    for l in labels:
        # trace segment
        Ll = L*(L == l)
        yx_trace, d_trace, success = trace_within_seg(Ll, O)
        # process non-cyclic segments
        if success:
            # end -1
            data_i1 = [
                l, -1,  # label, end
                *np.mean(yx_trace[:n_avg], axis=0),  # y, x
                np.mean(d_trace[:n_avg])-np.pi,  # d1, -pi to point outwards
            ]
            data.append(data_i1)
            # end 1
            data_i2 = [
                l, 1,  # label, end
                *np.mean(yx_trace[-n_avg:], axis=0),  # y, x
                np.mean(d_trace[-n_avg:])  # d
            ]
            data.append(data_i2)
    
    # make dataframe, correct dtype, set index
    df = pd.DataFrame(data=data, columns=columns)
    df = df.astype({f: int for f in ["label", "end", "y", "x"]})
    df = df.set_index(zip(df["label"], df["end"]))
    return df


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
    mask["combined"] = mask["dir"]&mask["angle"]
    return mask

def across_next_seg(label, end, df_segs, dist_cutoff):
    """ find next segment starting from current one
    :param label, end: segment's label and end
    :param df_segs: result of segs_to_frame()
    :param dist_cutoff: discard points >= cutoff
    :return: df_next[label,end,y,x,d,dist,dd_segs,dd_ends]
        dist: from curr end to next end
        dd_segs: directional change from curr seg to next
        dd_ends: directional change from next seg end to the other end
    """
    # subset df_segs to curr and next 1 and -1
    df_curr = df_segs.loc[(label, end)]
    df_next1 = df_segs[df_segs["end"] == 1].sort_values("label")
    df_next2 = df_segs[df_segs["end"] == -1].sort_values("label")
    assert np.all(df_next1["label"].values == df_next2["label"].values)

    # get masks for two ends of next
    mask1 = across_next_seg_1end(df_curr, df_next1)
    mask2 = across_next_seg_1end(df_curr, df_next2)
    # both ends in plus direction
    mask_pos = mask1["pos"] & mask2["pos"]
    # only one end has a small direction change
    mask_dir = np.logical_xor(mask1["dir"], mask2["dir"])

    # combine all masks
    # &dir: the end with small dir change is retained
    df_next = pd.concat([
        df_next1[mask_pos & mask_dir & mask1["dir"] & mask1["angle"]],
        df_next2[mask_pos & mask_dir & mask2["dir"] & mask2["angle"]]
    ])

    # distance: calculate, thresholding, sort
    dxdy_next = df_next[["x", "y"]].values-df_curr[["x", "y"]].values
    df_next["dist"] = np.linalg.norm(dxdy_next, axis=1)
    df_next = df_next[
        df_next["dist"] < dist_cutoff
    ].sort_values("dist")

    # change in directions from current segment to the next
    dd_segs = np.mod(
        df_next["d"].values+np.pi-df_curr["d"],
        2*np.pi
    )
    dd_segs = np.where(dd_segs<=np.pi, dd_segs, dd_segs-2*np.pi)
    df_next["dd_segs"] = dd_segs

    # change in directions from next segment's end-in to end-out
    n_next = len(df_next)
    d_end1 = df_segs.loc[zip(df_next["label"], [1]*n_next), "d"].values
    d_end2 = df_segs.loc[zip(df_next["label"], [-1]*n_next), "d"].values
    dd_ends = d_end1 - (d_end2 + np.pi)  # d-change from end -1 to 1
    dd_ends = dd_ends*(-df_next["end"].values)  # consider actual end
    df_next["dd_ends"] = dd_ends

    return df_next
