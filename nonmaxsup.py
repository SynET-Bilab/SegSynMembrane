#!/usr/bin/env python
""" nonmaxsup: non-maximum suppression
"""

from functools import reduce
import numpy as np
import numba

__all__ = [
    "nms2d", "nms3d"
]

#=========================
# find local maximum
#=========================

def slice_adjacent(x, y):
    """ generate slicing for adjacent pixels
    e.g. for image I and mask pts, if x="+", y="-"
        I[sub_I][pts[sub_pts]] are points on the right(x+)-below(y-)
    :param x, y: one of "+","-","0"
    :return: sub_I, sub_pts
    """
    s = {
        "0": {
            "pts": slice(None, None),
            "I": slice(None, None)
        },
        "-": {
            "pts": slice(1, None),
            "I": slice(None, -1)
        },
        "+": {
            "I": slice(1, None),
            "pts": slice(None, -1)
        }
    }
    sub_I = (s[y]["I"], s[x]["I"])
    sub_pts = (s[y]["pts"], s[x]["pts"])
    return sub_I, sub_pts

def compare_local(N_range, x, y, S, N, mask):
    """ compare local pixels
    :param N_range, x, y: direction to consider
        N_range: range of angles, (N_min, N_max)
        x, y: relative position of adjacent pixel, "+"/"-"/"0"
        e.g. N_range=(0, np.pi/8), x="+", y="0"
    :param S: values of saliency, shape=(ny, nx)
    :param N: values of normal direction, in (0, pi)
    :param mask: additional mask, bool, shape=S.shape
    :return: local_max, local_supp
        local_max: bool, shape=S.shape, true if the pixel is local max
        local_supp: bool, shape=S.shape, true if the pixel is suppressed
    """
    # pts in N_range
    pts = mask & (N >= N_range[0]) & (N < N_range[1])

    # find pixels that are local max or suppressed
    # max_plus/minus: true on pixels > its adjacent
    # supp_plus/minus: true on pixels being suppressed
    def init():
        return np.zeros(S.shape, dtype=bool)

    # plus direction
    sub_I, sub_pts = slice_adjacent(x=x, y=y)
    max_plus = init()
    max_plus[pts] = S[pts] > S[sub_I][pts[sub_pts]]
    supp_plus = init()
    supp_plus[sub_I][pts[sub_pts]] = max_plus[pts]

    # minus direction
    invert = {"0": "0", "+": "-", "-": "+"}
    sub_I, sub_pts = slice_adjacent(x=invert[x], y=invert[y])
    max_minus = init()
    max_minus[pts] = S[pts] > S[sub_I][pts[sub_pts]]
    supp_minus = init()
    supp_minus[sub_I][pts[sub_pts]] = max_minus[pts]

    # combine both directions
    # max: in both; supp: in either
    local_max = np.logical_and(max_plus, max_minus)
    local_supp = np.logical_or(supp_plus, supp_minus)

    return local_max, local_supp

def set_mask_boundary(I):
    """ mask out grids on the boundary """
    mask_boundary = np.ones(I.shape, dtype=bool)
    for i in [0, -1]:
        mask_boundary[i, :] = 0  # y=0,ny
        mask_boundary[:, i] = 0  # x=0,nx
    return mask_boundary


#=========================
# non-max suppression
#=========================

def nms2d(S, O, S_threshold=1e-6, suppress=True):
    """ non-maximum suppresion of S[ny, nx]
    :param S: saliency
    :param O: orientation (tangent)
    :param S_threshold: threshold on S, usually a small value
    :param suppress: if exclude suppressed pixels
    :return: local_max
        local_max: int, shape=S.shape, 1 if the pixel is local max
    """
    # global masks
    mask_boundary = set_mask_boundary(S)
    mask_S = S > S_threshold
    mask = np.logical_and(mask_boundary, mask_S)

    # normal direction, mod pi
    N = np.mod(O+np.pi/2, np.pi)

    # local max in each direction
    directions = [
        dict(N_range=(0, np.pi/8), x="+", y="0"),
        dict(N_range=(np.pi/8, np.pi*3/8), x="+", y="+"),
        dict(N_range=(np.pi*3/8, np.pi*5/8), x="0", y="+"),
        dict(N_range=(np.pi*5/8, np.pi*7/8), x="-", y="+"),
        dict(N_range=(np.pi*7/8, np.pi), x="-", y="0")
    ]
    max_arr = []
    supp_arr = []
    kwargs = dict(S=S, N=N, mask=mask)
    for d in directions:
        max_d, supp_d = compare_local(d["N_range"], d["x"], d["y"], **kwargs)
        max_arr.append(max_d)
        supp_arr.append(supp_d)

    # combine all directions
    local_max = reduce(np.logical_or, max_arr)
    local_supp = reduce(np.logical_or, supp_arr)

    # exclude suppressed pixels
    if suppress:
        local_max = np.logical_and(
            local_max, np.logical_not(local_supp)
        )
    
    # return int-type array
    return local_max.astype(np.int_)

@numba.njit(parallel=True)
def nms3d(S, O, S_threshold=1e-6, suppress=True):
    """ non-maximum suppresion of S[nz, ny, nx], python version
    :param S: assumed already gaussian-smoothed
    """
    # find local max for each slice
    local_max = np.zeros(S.shape, dtype=np.int64)
    nz = S.shape[0]
    for i in numba.prange(nz):
        with numba.objmode(local_max_i="int64[:,:]"):
            local_max_i = nms2d(
                S[i], O[i],
                S_threshold=S_threshold,
                suppress=suppress
            )
        local_max[i] = local_max_i
    return local_max


# #=========================
# # NMS using skimage-canny (deprecated)
# #=========================

# def values_xminus(I, pts):
#     """ values of grids that are xminus to I[pts] """
#     # pts[:, 1:] - new loc = (y, x-1)
#     # I[:, :-1] - match shape of pts
#     return I[:, :-1][pts[:, 1:]]

# def values_xplus(I, pts):
#     """ values of grids that are xplus to I[pts] """
#     # I[:, 1:] - new loc = (y, x+1)
#     # pts[:, :-1] - match shape of I
#     return I[:, 1:][pts[:, :-1]]

# def values_yminus(I, pts):
#     """ values of grids that are yminus to I[pts] """
#     # pts[1:, :] - new loc = (y-1, x)
#     # I[:-1, :] - match shape of pts
#     return I[:-1, :][pts[1:, :]]

# def values_yplus(I, pts):
#     """ values of grids that are yplus to I[pts] """
#     # I[1:, :] - new loc = (y+1, x)
#     # pts[:-1, :] - match shape of I
#     return I[1:, :][pts[:-1, :]]

# def set_local_max(local_max, N_range, S, N, mask):
#     """ set local_max of pts with N in N_range
#     :param local_max: bool array, shape=S.shape
#     :param N_range: (N_min, N_max), in rad, in (0, pi)
#     :param S: values of saliency, shape=(ny, nx)
#     :param N: values of normal direction, in (0, pi)
#     :param mask: bool array, mask according to other criteria, shape=S.shape
#     :return: None, local_max[pts] is filled
#     """
#     # pts in N_range
#     pts = mask & (N >= N_range[0]) & (N < N_range[1])

#     # weights
#     weight_x = np.cos(N[pts])**2
#     weight_y = np.sin(N[pts])**2

#     # values in plus/minus directions
#     N_mid = np.mean(N_range)
#     # x - direction
#     if np.cos(N_mid) >= 0:  # N in (0, pi/2), Nplus_x = xplus
#         S_Nplus_x = values_xplus(S, pts)
#         S_Nminus_x = values_xminus(S, pts)
#     else:  # N in (pi/2, pi), Nplus_x = xminus
#         S_Nplus_x = values_xminus(S, pts)
#         S_Nminus_x = values_xplus(S, pts)
#     # y - direction
#     # since N in (0, pi), Nplus_y = yplus
#     S_Nplus_y = values_yplus(S, pts)
#     S_Nminus_y = values_yminus(S, pts)

#     # compare S with S_Nplus/minus
#     gt_plus = S[pts] > (weight_x*S_Nplus_x + weight_y*S_Nplus_y)
#     gt_minus = S[pts] > (weight_x*S_Nminus_x + weight_y*S_Nminus_y)

#     # set local max
#     local_max[pts] = gt_plus & gt_minus


# #=========================
# # non-max suppression
# #=========================

# def nms2d(S, O, sigma=1, qthreshold=0.5):
#     """ non-maximum suppresion of S[ny, nx]
#     :param S: saliency
#     :param O: orientation (tangent)
#     :param sigma: gaussian smoothing
#     :param qthreshold: threshold on quantile of S
#     :return: local_max (int array, for easier multiplication between masks)
#     """
#     # global masks
#     mask_boundary = set_mask_boundary(S)
#     mask_S = S > np.quantile(S, qthreshold)
#     mask = mask_boundary & mask_S

#     # smoothing
#     if sigma > 0:
#         S = gaussian(S, sigma)

#     # normal direction, mod pi
#     N = np.mod(O+np.pi/2, np.pi)

#     # local max in four directions
#     local_max = np.zeros(S.shape, dtype=bool)
#     kwargs = dict(S=S, N=N, mask=mask)
#     set_local_max(local_max, N_range=(0, np.pi/4), **kwargs)
#     set_local_max(local_max, N_range=(np.pi/4, np.pi/2), **kwargs)
#     set_local_max(local_max, N_range=(np.pi/2, np.pi*3/4), **kwargs)
#     set_local_max(local_max, N_range=(np.pi*3/4, np.pi), **kwargs)

#     # return int-type array
#     return local_max.astype(np.int64)


# def nms3d_python(S, O, qthreshold=0.5):
#     """ non-maximum suppresion of S[nz, ny, nx], python version
#     :param S: assumed already gaussian-smoothed
#     """
#     # find local max for each slice
#     local_max = np.zeros(S.shape, dtype=np.int64)
#     nz = S.shape[0]
#     for i in range(nz):
#         local_max[i] = nms2d(
#             S[i], O[i],
#             sigma=0,  # set to 0 because gaussian is already done
#             qthreshold=qthreshold
#         )
#     return local_max

