""" nonmaxsup: non-maximum suppression
"""

from functools import reduce
import multiprocessing.dummy
import numpy as np

__all__ = [
    # Gonzalez-Woods method (default): 8-neighbors, no interpolation
    "nms2d", "nms3d",
]


def set_mask_boundary(I):
    """ mask out grids on the boundary """
    mask_boundary = np.ones(I.shape, dtype=bool)
    for i in [0, -1]:
        mask_boundary[i, :] = 0  # y=0,ny
        mask_boundary[:, i] = 0  # x=0,nx
    return mask_boundary

#=========================
# Gonzalez-Woods method
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

def nms2d(S, O, suppress=True):
    """ non-maximum suppresion of S[ny, nx]
    :param S: saliency
    :param O: orientation (tangent)
    :param suppress: if exclude suppressed pixels
    :return: local_max
        local_max: int, shape=S.shape, 1 if the pixel is local max
    """
    # global masks
    mask = set_mask_boundary(S)

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

def nms3d(S, O, suppress=True):
    """ non-maximum suppresion of S[nz, ny, nx]
    :param S: saliency
    :param O: orientation (tangent)
    :param suppress: if exclude suppressed pixels
    :return: local_max
        local_max: int, shape=S.shape, 1 if the pixel is local max
    """
    # find local max for each slice
    local_max = np.zeros(S.shape, dtype=np.int64)
    nz = S.shape[0]

    # parallel computing
    def calc_one(i):
        local_max[i] = nms2d(
            S[i], O[i],
            suppress=suppress
        )
        
    pool = multiprocessing.dummy.Pool()
    pool.map(calc_one, range(nz))
    pool.close()

    return local_max

