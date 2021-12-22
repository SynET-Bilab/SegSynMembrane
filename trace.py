#!/usr/bin/env python
""" trace
"""

import numpy as np

__all__ = [
    "trace_direction", "trace_segment"
]

def find_next_yxs(d_curr):
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

def find_next_direction(d_curr, o_next):
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

def trace_direction(yx_curr, trace, map_yxd):
    """ trace segment from current (y,x)
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
    dydx_candidates = find_next_yxs(d_curr)

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
            d_next = find_next_direction(d_curr, map_yxd[yx_next])
            map_yxd[yx_next] = d_next
            # trace next
            return trace_direction(yx_next, trace, map_yxd)
        else:
            idx += 1
            # previous dydx candidate not found, to visit next
            if idx < 3:
                dydx_tovisit.append(dydx_candidates[idx])
            # no more candidates, return true
            else:
                return True

def trace_segment(nms, O):
    """ trace a 8-connected segment
    :param nms: shape=(ny,nx), nonmaxsup'ed image
    :param O: orientation
    :return: yx_trace, d_trace, success
        yx_trace: yx's, [(y1,x1),...]
        d_trace: directions, [d1,d2,...]
        success: True if no loop, otherwise False
    """
    # get position of pixels
    pos = np.nonzero(nms)

    # set starting point
    idx_start = int(len(pos[0])/2)
    yx_start = tuple(pos[i][idx_start] for i in [0, 1])

    # trace direction of orientation
    map_yxd_plus = dict(zip(zip(*pos), O[pos]))
    yx_plus = []
    success_plus = trace_direction(yx_start, yx_plus, map_yxd_plus)
    d_plus = [map_yxd_plus[yx] for yx in yx_plus]

    # trace direction of orientation+pi
    map_yxd_minus = dict(zip(zip(*pos), O[pos]))
    map_yxd_minus[yx_start] = O[yx_start]+np.pi
    yx_minus = []
    success_minus = trace_direction(yx_start, yx_minus, map_yxd_minus)
    # reverse sequence, align direction with d_plus
    yx_minus_reverse = yx_minus[-1:0:-1]
    d_minus_reverse = [map_yxd_minus[yx]-np.pi for yx in yx_minus_reverse]

    # concatenate plus and minus directions
    yx_trace = yx_minus_reverse + yx_plus
    d_trace = d_minus_reverse + d_plus
    success = success_plus and success_minus
    return yx_trace, d_trace, success
